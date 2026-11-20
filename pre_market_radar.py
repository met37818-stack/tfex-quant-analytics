#!/usr/bin/env python3
"""
Pre-Market Quant Radar (Phase 1)
Automated SET50 Index Futures & Options Pre-Market Analysis Engine.

Generates:
1. Black-76 Implied Volatility (IV) & Greeks (Delta, Gamma, Vega, Theta)
2. Net GEX (Gamma Exposure) Profile by Strike (200 THB/point)
3. Structural Levels: Gamma Flip, Call Wall, Put Wall, Max Pain
4. 1-Day Expected Move (1-SD) & Nearest Active Tactical Strikes (Spot ± 25)
5. Smart Money Flow Decomposition (Foreign Net vs Total Futures OI Change)
6. Rule-Based Actionable Thai Daily Trading Plan (แผนการเทรด SET50 ประจำวัน)
7. Interactive 3-Panel Plotly HTML Dashboard
8. Automated Telegram Pre-Market Briefing & HTML File Dispatch
9. Data-Driven Holiday & Duplicate Execution Guard
"""

import os
import sys
import math
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple, Set
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# SHARED HELPERS (tfex_common)
# ==============================================================================
# Calendar, THOR unit handling, the Turso client and Telegram dispatch live in the
# tfex_common package so the four TFEX repos stop drifting apart. Pinned to a tag in
# requirements.txt. The names are imported directly (not wrapped) so every existing
# call site - and the tests that patch `execute_turso_sql` on this module - keep
# working against the single shared implementation.
from tfex_common.calendar import (  # noqa: E402
    get_penultimate_business_day,
    get_series_expiry_date,
)
from tfex_common.constants import (  # noqa: E402
    BKK_TZ,
    CHARM_CLAMP,
    CONTRACT_MULTIPLIER,
    SET_HOLIDAYS,
    THOR_MAX_PCT,
)
from tfex_common.telegram import send_alert as send_telegram_alert  # noqa: E402
from tfex_common.thor import select_dynamic_thor_rate  # noqa: E402
from tfex_common.thor import to_decimal_rate as _to_decimal_rate  # noqa: E402
from tfex_common.turso import execute_sql as execute_turso_sql  # noqa: E402
from tfex_common.turso import normalize_url as normalize_turso_url  # noqa: E402

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Volume-profile price bucket in index points (SET50 futures tick is 0.1).
VOLUME_PROFILE_BUCKET = 0.1



def init_analytics_tables(pipeline_url: str, auth_token: str):
    """Ensure analytics_premarket_log and daily_realized_vol tables exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS analytics_premarket_log (
        trade_date TEXT PRIMARY KEY,
        futures_ref REAL,
        atm_iv REAL,
        gamma_flip REAL,
        call_wall REAL,
        put_wall REAL,
        max_pain REAL,
        foreign_regime TEXT,
        sent_at_bkk TEXT
    );
    """
    execute_turso_sql(pipeline_url, auth_token, ddl)

    ddl_rv = """
    CREATE TABLE IF NOT EXISTS daily_realized_vol (
        trade_date TEXT PRIMARY KEY,
        rv_annualized REAL,
        bar_count INTEGER,
        updated_at_bkk TEXT
    );
    """
    execute_turso_sql(pipeline_url, auth_token, ddl_rv)

# ==============================================================================
# QUANT & DERIVATIVES MATHEMATICS (BLACK-76 & GREEKS)
# ==============================================================================

def black76_price(F: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Black-76 European futures option pricing model.
    F: Futures settlement price
    K: Strike price
    T: Time to expiration in years
    r: Annualized risk-free rate
    sigma: Volatility (annualized)
    option_type: 'C' (Call) or 'P' (Put)
    """
    if sigma <= 0.0 or T <= 0.0:
        disc = math.exp(-r * T)
        return max(0.0, disc * ((F - K) if option_type == 'C' else (K - F)))

    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)

    if option_type == 'C':
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def calculate_implied_vol(price: float, F: float, K: float, T: float, r: float, option_type: str) -> float | None:
    """Solve for Black-76 Implied Volatility via Brent's method, capped at 1.50 (150%)."""
    if price is None or price <= 0.0 or T <= 0.0 or F <= 0.0 or K <= 0.0:
        return None

    disc = math.exp(-r * T)
    intrinsic = max(0.0, disc * ((F - K) if option_type == 'C' else (K - F)))
    if price <= intrinsic + 1e-4:
        return None

    MAX_IV = 1.50
    p_high = black76_price(F, K, T, r, MAX_IV, option_type)
    if price >= p_high:
        return None

    def obj(sigma):
        return black76_price(F, K, T, r, sigma, option_type) - price

    try:
        iv = float(brentq(obj, 1e-4, MAX_IV, maxiter=100))
        return iv if 1e-4 <= iv <= MAX_IV else None
    except Exception:
        return None

def calculate_black76_greeks(F: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    """
    Compute Delta, Gamma, Vega, Theta, Vanna, and Charm under Black-76.
    """
    if sigma <= 0.0 or T <= 0.0 or F <= 0.0 or K <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "vanna": 0.0, "charm": 0.0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = (disc * pdf_d1) / (F * sigma * sqrt_T)
    vega_1pct = (F * disc * pdf_d1 * sqrt_T) * 0.01

    if option_type == 'C':
        delta = disc * norm.cdf(d1)
        # theta = -dPrice/dT. dPrice/dT = +[F·disc·n(d1)·sigma/(2√T)] - r·Price,
        # so the discount term carries a PLUS sign here (checked against a
        # finite-difference of black76_price: the old -r·Price was 0.4-9.8% off).
        theta_call = - (F * disc * pdf_d1 * sigma) / (2.0 * sqrt_T) + r * disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        theta_1d = theta_call / 365.0
        charm_1d = (r * disc * norm.cdf(d1) + disc * pdf_d1 * (d2 / (2.0 * T))) / 365.0
    else:
        delta = -disc * norm.cdf(-d1)
        theta_put = - (F * disc * pdf_d1 * sigma) / (2.0 * sqrt_T) + r * disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        theta_1d = theta_put / 365.0
        charm_1d = (-r * disc * norm.cdf(-d1) + disc * pdf_d1 * (d2 / (2.0 * T))) / 365.0

    # Dampen / cap Charm when DTE <= 2 to prevent near-expiry asymptotic explosion
    if T <= (2.0 / 365.0) or math.isnan(charm_1d) or math.isinf(charm_1d):
        charm_1d = max(-5.0, min(5.0, charm_1d)) if not (math.isnan(charm_1d) or math.isinf(charm_1d)) else 0.0

    vanna = -disc * pdf_d1 * (d2 / sigma)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega_1pct),
        "theta": float(theta_1d),
        "vanna": float(vanna),
        "charm": float(charm_1d)
    }

def analyze_options_greeks_edge(
    pipeline_url: str,
    auth_token: str,
    trade_date: str,
    front_series: str,
    futures_ref: float,
    thor_rate: float,
    T_front: float,
    front_opts_df: pd.DataFrame = None
) -> dict:
    """
    Analyzes 3 Dimensions of Options Greeks Edge from eod_options_greeks:
    1. 25-Delta Skew (IV_25DP - IV_25DC)
    2. ATM Theta-to-Gamma Ratio and Daily Breakeven Move (sqrt(2*|Theta|/Gamma))
    3. Market Maker / Dealer Flow (Net Dealer Vanna & Charm Exposure)
    """
    greeks_records = []
    underlying = front_series if front_series.startswith("S50") else f"S50{front_series}"

    # 1. Query pre-computed Greeks from eod_options_greeks in Turso
    if pipeline_url and auth_token:
        sql = """
        SELECT symbol, option_type, strike, iv, delta, gamma, theta, vega, vanna, charm, open_interest
        FROM eod_options_greeks
        WHERE trade_date = ? AND (underlying_futures = ? OR series = ?) AND iv IS NOT NULL;
        """
        try:
            _, rows = execute_turso_sql(pipeline_url, auth_token, sql, [trade_date, underlying, front_series])
            for r in rows:
                if r[3] is not None and r[4] is not None:
                    greeks_records.append({
                        "symbol": str(r[0]),
                        "option_type": str(r[1]).upper(),
                        "strike": float(r[2]),
                        "iv": float(r[3]),
                        "delta": float(r[4]),
                        "gamma": float(r[5]),
                        "theta": float(r[6]),
                        "vega": float(r[7]),
                        "vanna": float(r[8]),
                        "charm": float(r[9]),
                        "oi": int(r[10]) if r[10] else 0
                    })
        except Exception as e:
            print(f"[GREEKS EDGE] Note checking eod_options_greeks: {e}")

    # Fallback to computing on-the-fly from front_opts_df (for test mode or if table not yet populated)
    if not greeks_records and front_opts_df is not None and not front_opts_df.empty:
        for _, row in front_opts_df.iterrows():
            k = float(row["strike"])
            otype = str(row["option_type"]).upper()
            oi = int(row.get("oi", 0)) if row.get("oi") else 0
            price = float(row["settlement_price"]) if row.get("settlement_price") and row["settlement_price"] > 0 else float(row.get("close", 0) or 0)
            if price > 0:
                iv = calculate_implied_vol(price, futures_ref, k, T_front, thor_rate, otype)
                if iv and iv > 0.0001:
                    g = calculate_black76_greeks(futures_ref, k, T_front, thor_rate, iv, otype)
                    greeks_records.append({
                        "symbol": row.get("symbol", f"{front_series}{otype}{k:.0f}"),
                        "option_type": otype,
                        "strike": k,
                        "iv": iv,
                        "delta": g["delta"],
                        "gamma": g["gamma"],
                        "theta": g["theta"],
                        "vega": g["vega"],
                        "vanna": g["vanna"],
                        "charm": g["charm"],
                        "oi": oi
                    })

    if not greeks_records:
        return {
            "skew_25d": None, "c_iv_pct": None, "p_iv_pct": None,
            "skew_desc": "ข้อมูล Greeks ไม่เพียงพอ", "atm_strike": futures_ref,
            "atm_gamma": 0.0, "atm_theta": 0.0, "theta_gamma_ratio": 0.0,
            "daily_breakeven_move": 0.0, "dealer_vanna": 0.0, "dealer_charm": 0.0,
            "vanna_bias_desc": "N/A", "records_count": 0
        }

    # 1. 25-Delta Skew Calculation (Interpolated)
    # Filter out illiquid (OI == 0), extreme delta (< 0.10 or > 0.90), and exploded IV (> 1.50)
    valid_calls = [
        r for r in greeks_records 
        if r["option_type"] == "C" and r.get("oi", 0) > 0 and 0.10 <= r["delta"] <= 0.90 and (r.get("iv") is not None and r["iv"] <= 1.50)
    ]
    valid_puts = [
        r for r in greeks_records 
        if r["option_type"] == "P" and r.get("oi", 0) > 0 and 0.10 <= abs(r["delta"]) <= 0.90 and (r.get("iv") is not None and r["iv"] <= 1.50)
    ]

    calls = valid_calls if len(valid_calls) >= 2 else [r for r in greeks_records if r["option_type"] == "C" and (r.get("iv") is not None and r["iv"] <= 1.50)]
    puts = valid_puts if len(valid_puts) >= 2 else [r for r in greeks_records if r["option_type"] == "P" and (r.get("iv") is not None and r["iv"] <= 1.50)]

    calls.sort(key=lambda x: x["delta"], reverse=True)
    puts.sort(key=lambda x: abs(x["delta"]))

    c_iv_25 = None
    for i in range(len(calls) - 1):
        d1, d2 = calls[i]["delta"], calls[i + 1]["delta"]
        if (d1 >= 0.25 >= d2) or (d1 <= 0.25 <= d2):
            iv1, iv2 = calls[i]["iv"], calls[i + 1]["iv"]
            c_iv_25 = iv1 + (0.25 - d1) * (iv2 - iv1) / (d2 - d1) if d2 != d1 else iv1
            break
    if c_iv_25 is None and calls:
        c_iv_25 = min(calls, key=lambda x: abs(x["delta"] - 0.25))["iv"]

    p_iv_25 = None
    for i in range(len(puts) - 1):
        d1, d2 = abs(puts[i]["delta"]), abs(puts[i + 1]["delta"])
        if (d1 <= 0.25 <= d2) or (d1 >= 0.25 >= d2):
            iv1, iv2 = puts[i]["iv"], puts[i + 1]["iv"]
            p_iv_25 = iv1 + (0.25 - d1) * (iv2 - iv1) / (d2 - d1) if d2 != d1 else iv1
            break
    if p_iv_25 is None and puts:
        p_iv_25 = min(puts, key=lambda x: abs(abs(x["delta"]) - 0.25))["iv"]

    # W3: เดิมฝั่งที่ไม่มีข้อมูลถูกแทนด้วย 0.0 ทำให้ skew_25d = ผลต่างจริงข้างเดียว
    # (เช่น 0 - 12.5 = -12.5) แล้วสรุปทิศทางจากข้อมูลที่ไม่มีอยู่จริง
    c_iv_pct = (c_iv_25 * 100.0) if c_iv_25 is not None else None
    p_iv_pct = (p_iv_25 * 100.0) if p_iv_25 is not None else None

    if c_iv_pct is None or p_iv_pct is None:
        skew_25d = None
        missing = "Call" if c_iv_pct is None else "Put"
        skew_desc = f"ไม่มีข้อมูล 25-Delta IV ฝั่ง {missing} — งดสรุปทิศทาง skew"
    else:
        skew_25d = p_iv_pct - c_iv_pct
        if skew_25d > 2.0:
            skew_desc = "Put แพงกว่า Call (ตลาดกังวลความเสี่ยงขาลง / พรีเมียมประกันพอร์ตสูง)"
        elif skew_25d < -1.0:
            skew_desc = "Call แพงกว่า Put (ตลาดไล่ซื้อ Call เก็งกำไร / Upside Call Skew)"
        else:
            skew_desc = "ความกลัว-โลภสองฝั่งสมดุล (Normal Skew)"

    # 2. ATM Theta-to-Gamma Edge & Daily Breakeven Move
    if calls and puts:
        atm_c = min(calls, key=lambda x: abs(x["strike"] - futures_ref))
        atm_p = min(puts, key=lambda x: abs(x["strike"] - futures_ref))
        atm_strike = atm_c["strike"]
        atm_gamma = (atm_c["gamma"] + atm_p["gamma"]) / 2.0
        atm_theta = (abs(atm_c["theta"]) + abs(atm_p["theta"])) / 2.0
        theta_gamma_ratio = (atm_theta / atm_gamma) if atm_gamma > 0 else 0.0
        daily_breakeven_move = math.sqrt(2.0 * atm_theta / atm_gamma) if atm_gamma > 0 else 0.0
    else:
        atm_strike = futures_ref
        atm_gamma = 0.0
        atm_theta = 0.0
        theta_gamma_ratio = 0.0
        daily_breakeven_move = 0.0

    # 3. Market Maker Dealer Flow (Net Dealer Vanna & Charm Exposure)
    # Dealer is short customer OI. Filter extreme delta / illiquid strikes to prevent distorted flow.
    flow_records = [
        r for r in greeks_records 
        if r.get("oi", 0) > 0 and 0.10 <= abs(r.get("delta", 0.0)) <= 0.90 and (r.get("iv") is None or r["iv"] <= 1.50)
    ]
    if len(flow_records) < 3:
        flow_records = [r for r in greeks_records if r.get("oi", 0) > 0]

    dealer_vanna = -sum(r["vanna"] * r["oi"] for r in flow_records) * 0.01
    dealer_charm = -sum(r["charm"] * r["oi"] for r in flow_records)
    if T_front <= (2.0 / 365.0):
        dealer_charm = max(-CHARM_CLAMP, min(CHARM_CLAMP, dealer_charm))

    if dealer_vanna > 50:
        vanna_bias_desc = "หาก IV Crush ดีลเลอร์จะซื้อ Futures พยุงตลาด"
    elif dealer_vanna < -50:
        vanna_bias_desc = "หาก IV Crush ดีลเลอร์จะขาย Futures กดดันตลาด"
    else:
        vanna_bias_desc = "พอร์ตดีลเลอร์สมดุลต่อการเปลี่ยนแปลง IV"

    return {
        "skew_25d": skew_25d,
        "c_iv_pct": c_iv_pct,
        "p_iv_pct": p_iv_pct,
        "skew_desc": skew_desc,
        "atm_strike": atm_strike,
        "atm_gamma": atm_gamma,
        "atm_theta": atm_theta,
        "theta_gamma_ratio": theta_gamma_ratio,
        "daily_breakeven_move": daily_breakeven_move,
        "dealer_vanna": dealer_vanna,
        "dealer_charm": dealer_charm,
        "vanna_bias_desc": vanna_bias_desc,
        "records_count": len(greeks_records)
    }

# ==============================================================================
# SMART MONEY FLOW DECOMPOSITION & TACTICAL STRIKE RESOLUTION
# ==============================================================================

def decompose_smart_money(foreign_net: int, delta_oi: int) -> tuple[str, str, str]:
    """
    Decomposes Foreign Institutional Flow and Futures Open Interest Change
    into canonical 4-quadrant derivative regimes.
    """
    if foreign_net > 0 and delta_oi > 0:
        regime = "New Long Build-up"
        emoji = "🟢"
        th_desc = "เปิดสถานะ Long ใหม่เพื่อเก็งกำไรขาขึ้นอย่างมีนัยสำคัญ"
    elif foreign_net > 0 and delta_oi < 0:
        regime = "Short Covering"
        emoji = "🟡"
        th_desc = "ปิดสถานะ Short เดิมเพื่อทำกำไร/ลดความเสี่ยง (แรงซื้อคืน)"
    elif foreign_net < 0 and delta_oi > 0:
        regime = "New Short Build-up"
        emoji = "🔴"
        th_desc = "เปิดสถานะ Short ใหม่เพื่อเก็งกำไรขาลงอย่างมีนัยสำคัญ"
    elif foreign_net < 0 and delta_oi < 0:
        regime = "Long Liquidation"
        emoji = "🟠"
        th_desc = "ปิดสถานะ Long เดิมเพื่อลดพอร์ต/ตัดความเสี่ยง (แรงขายออก)"
    else:
        regime = "Neutral / Rebalancing"
        emoji = "⚪"
        th_desc = "การปรับสมดุลพอร์ตโฟลิโอหรือสถานะยังไม่เด่นชัด"

    return regime, emoji, th_desc

def resolve_tactical_levels(res_df: pd.DataFrame, futures_ref: float, call_wall: float, put_wall: float) -> dict:
    """
    Identifies intraday nearest active strikes around current price (Spot ± 25 points).
    Resolves:
    - Resistance: res_1 (near), res_2 (strong) where res_2 > res_1 > futures_ref
    - Support: sup_1 (near), sup_2 (strong) where futures_ref > sup_1 > sup_2
    """
    # 1. Calls above futures_ref
    calls_above = res_df[res_df["strike"] > futures_ref].copy()
    calls_tactical = calls_above[calls_above["strike"] <= futures_ref + 25.0].copy()
    if len(calls_tactical) < 2:
        calls_tactical = calls_above[calls_above["strike"] <= futures_ref + 35.0].copy()
    if len(calls_tactical) < 2:
        calls_tactical = calls_above.head(2).copy()

    if not calls_tactical.empty:
        top_c = calls_tactical.sort_values(by=["call_oi", "strike"], ascending=[False, True]).head(2)
        top_c_sorted = top_c.sort_values(by="strike", ascending=True)
        res_1 = float(top_c_sorted.iloc[0]["strike"])
        res_1_oi = int(top_c_sorted.iloc[0]["call_oi"])
        res_1_gex = float(top_c_sorted.iloc[0]["net_gex_mb"])

        if len(top_c_sorted) > 1:
            res_2 = float(top_c_sorted.iloc[1]["strike"])
            res_2_oi = int(top_c_sorted.iloc[1]["call_oi"])
            res_2_gex = float(top_c_sorted.iloc[1]["net_gex_mb"])
        else:
            next_c = calls_above[calls_above["strike"] > res_1]
            if not next_c.empty:
                res_2 = float(next_c.iloc[0]["strike"])
                res_2_oi = int(next_c.iloc[0]["call_oi"])
                res_2_gex = float(next_c.iloc[0]["net_gex_mb"])
            else:
                res_2 = call_wall
                res_2_oi = int(res_df.loc[res_df["strike"] == call_wall, "call_oi"].values[0]) if not res_df[res_df["strike"] == call_wall].empty else 0
                res_2_gex = float(res_df.loc[res_df["strike"] == call_wall, "net_gex_mb"].values[0]) if not res_df[res_df["strike"] == call_wall].empty else 0.0
    else:
        res_1 = call_wall
        res_1_oi = 0
        res_1_gex = 0.0
        res_2 = call_wall
        res_2_oi = 0
        res_2_gex = 0.0

    # 2. Puts below futures_ref
    puts_below = res_df[res_df["strike"] < futures_ref].copy()
    puts_tactical = puts_below[puts_below["strike"] >= futures_ref - 25.0].copy()
    if len(puts_tactical) < 2:
        puts_tactical = puts_below[puts_below["strike"] >= futures_ref - 35.0].copy()
    if len(puts_tactical) < 2:
        puts_tactical = puts_below.tail(2).copy()

    if not puts_tactical.empty:
        top_p = puts_tactical.sort_values(by=["put_oi", "strike"], ascending=[False, False]).head(2)
        top_p_sorted = top_p.sort_values(by="strike", ascending=False)
        sup_1 = float(top_p_sorted.iloc[0]["strike"])
        sup_1_oi = int(top_p_sorted.iloc[0]["put_oi"])
        sup_1_gex = float(top_p_sorted.iloc[0]["net_gex_mb"])

        if len(top_p_sorted) > 1:
            sup_2 = float(top_p_sorted.iloc[1]["strike"])
            sup_2_oi = int(top_p_sorted.iloc[1]["put_oi"])
            sup_2_gex = float(top_p_sorted.iloc[1]["net_gex_mb"])
        else:
            prev_p = puts_below[puts_below["strike"] < sup_1]
            if not prev_p.empty:
                sup_2 = float(prev_p.iloc[-1]["strike"])
                sup_2_oi = int(prev_p.iloc[-1]["put_oi"])
                sup_2_gex = float(prev_p.iloc[-1]["net_gex_mb"])
            else:
                sup_2 = put_wall
                sup_2_oi = int(res_df.loc[res_df["strike"] == put_wall, "put_oi"].values[0]) if not res_df[res_df["strike"] == put_wall].empty else 0
                sup_2_gex = float(res_df.loc[res_df["strike"] == put_wall, "net_gex_mb"].values[0]) if not res_df[res_df["strike"] == put_wall].empty else 0.0
    else:
        sup_1 = put_wall
        sup_1_oi = 0
        sup_1_gex = 0.0
        sup_2 = put_wall
        sup_2_oi = 0
        sup_2_gex = 0.0

    return {
        "res_2": res_2, "res_2_oi": res_2_oi, "res_2_gex": res_2_gex,
        "res_1": res_1, "res_1_oi": res_1_oi, "res_1_gex": res_1_gex,
        "sup_1": sup_1, "sup_1_oi": sup_1_oi, "sup_1_gex": sup_1_gex,
        "sup_2": sup_2, "sup_2_oi": sup_2_oi, "sup_2_gex": sup_2_gex,
    }

# ==============================================================================
# SMART MONEY & OPTIONS STRIKE FOOTPRINT MODULES
# ==============================================================================

def analyze_foreign_footprint(
    pipeline_url: str,
    auth_token: str,
    target_date: str,
    front_series: str,
    futures_ref: float,
    lookback_days: int = 60
) -> dict:
    """
    Analyzes historical Foreign Investor Flow and cost basis on SET50 Index Futures.
    Tracks running inventory-weighted cost basis (resets when position flips side),
    calculates 20-day rolling Flow Z-Score, and checks 5-day price-flow divergence.
    """
    try:
        # 1. Fetch Foreign Net Futures history
        sql_inv = """
        SELECT trade_date, foreign_net 
        FROM eod_investor_types 
        WHERE category = 'Equity Index Futures' AND trade_date <= ? 
        ORDER BY trade_date DESC 
        LIMIT ?;
        """
        _, rows_inv = execute_turso_sql(pipeline_url, auth_token, sql_inv, [target_date, lookback_days])

        # 2. Fetch settlement prices for front series
        sql_px = """
        SELECT trade_date, settlement_price 
        FROM eod_futures_series 
        WHERE series = ? AND trade_date <= ? 
        ORDER BY trade_date DESC 
        LIMIT ?;
        """
        _, rows_px = execute_turso_sql(pipeline_url, auth_token, sql_px, [front_series, target_date, lookback_days])
        px_dict = {r[0]: float(r[1]) for r in rows_px if r[1] is not None}

        # Fallback if front series was not active on older dates in lookback
        if len(px_dict) < len(rows_inv):
            sql_fb = """
            SELECT trade_date, settlement_price 
            FROM eod_futures_series 
            WHERE trade_date <= ? 
            GROUP BY trade_date 
            ORDER BY trade_date DESC 
            LIMIT ?;
            """
            _, rows_fb = execute_turso_sql(pipeline_url, auth_token, sql_fb, [target_date, lookback_days])
            for r in rows_fb:
                if r[0] not in px_dict and r[1] is not None:
                    px_dict[r[0]] = float(r[1])

        # Align chronological order (oldest to newest)
        records = []
        for r in sorted(rows_inv, key=lambda x: x[0]):
            dt = r[0]
            f_net = int(r[1] or 0)
            px = px_dict.get(dt, futures_ref)
            records.append({"date": dt, "trade_date": dt, "foreign_net": f_net, "settlement_price": px})

        if not records:
            return {
                "cum_foreign_net": 0,
                "cost_basis": futures_ref,
                "pnl_pts": 0.0,
                "pnl_status": "ไม่มีข้อมูลย้อนหลัง",
                "z_score": 0.0,
                "z_desc": "ข้อมูลไม่เพียงพอ",
                "is_outlier": False,
                "divergence_text": "ไม่มีข้อมูล",
                "history_days": 0
            }

        # Inventory-weighted cost basis calculation
        cum_net = 0
        cost_basis = futures_ref

        for rec in records:
            net_i = rec["foreign_net"]
            px_i = rec["settlement_price"]
            if net_i == 0:
                rec["cost_basis"] = float(cost_basis)
                rec["cum_net"] = int(cum_net)
                continue

            if cum_net == 0:
                cost_basis = px_i
                cum_net = net_i
            elif (cum_net > 0 and net_i > 0) or (cum_net < 0 and net_i < 0):
                # Expanding existing position
                new_net = cum_net + net_i
                cost_basis = (cum_net * cost_basis + net_i * px_i) / new_net
                cum_net = new_net
            else:
                # Reducing or reversing position
                if abs(net_i) < abs(cum_net):
                    # Partial unwinding: cost basis remains unchanged
                    cum_net += net_i
                elif abs(net_i) == abs(cum_net):
                    # Flat
                    cum_net = 0
                    cost_basis = px_i
                else:
                    # Position flipped to opposite side
                    cum_net += net_i
                    cost_basis = px_i

            rec["cost_basis"] = float(cost_basis)
            rec["cum_net"] = int(cum_net)

        if cum_net == 0:
            cost_basis = futures_ref

        if cum_net > 0:
            pnl_pts = futures_ref - cost_basis
            pnl_status = f"กำไร (+{pnl_pts:.1f} จุด)" if pnl_pts >= 0 else f"ขาดทุน ({pnl_pts:.1f} จุด)"
        elif cum_net < 0:
            pnl_pts = cost_basis - futures_ref
            pnl_status = f"กำไร (+{pnl_pts:.1f} จุด)" if pnl_pts >= 0 else f"ขาดทุน ({pnl_pts:.1f} จุด)"
        else:
            pnl_pts = 0.0
            pnl_status = "พอร์ตว่าง / เท่าทุน"

        # 20-day rolling Flow Z-Score
        recent_flows = [r["foreign_net"] for r in records[-20:]]
        latest_flow = recent_flows[-1] if recent_flows else 0
        if len(recent_flows) >= 3:
            m = float(np.mean(recent_flows))
            s = float(np.std(recent_flows))
            z_score = float((latest_flow - m) / s) if s > 1e-4 else 0.0
        else:
            z_score = 0.0

        is_outlier = abs(z_score) >= 1.5
        if is_outlier:
            tag = "ซื้อหนัก" if z_score > 0 else "ขายหนัก"
            z_desc = f"🚨 Outlier {tag} (Z = {z_score:+.2f} SD)"
        else:
            z_desc = f"ปกติ (Z = {z_score:+.2f} SD)"

        # 5-day Price-Flow Divergence
        n_div = min(5, len(records))
        if n_div >= 2:
            p_latest = records[-1]["settlement_price"]
            p_prev5 = records[-n_div]["settlement_price"]
            price_diff = p_latest - p_prev5
            flow_sum_5d = sum(r["foreign_net"] for r in records[-n_div:])

            if price_diff >= 3.0 and flow_sum_5d <= -5000:
                divergence_text = f"⚠️ Bearish (ราคาขึ้น {price_diff:+.1f} แต่ฝรั่งขาย {flow_sum_5d:+,})"
            elif price_diff <= -3.0 and flow_sum_5d >= 5000:
                divergence_text = f"💡 Bullish (ราคาย่อ {price_diff:+.1f} แต่ฝรั่งซื้อ {flow_sum_5d:+,})"
            elif (price_diff >= 0 and flow_sum_5d >= 0) or (price_diff <= 0 and flow_sum_5d <= 0):
                divergence_text = "สอดคล้องกับราคา (Confluent)"
            else:
                divergence_text = "ไม่มี Divergence เด่นชัด (Neutral)"
        else:
            divergence_text = "ข้อมูลไม่พอวิเคราะห์ Divergence"

        return {
            "cum_foreign_net": int(cum_net),
            "cost_basis": float(cost_basis),
            "pnl_pts": float(pnl_pts),
            "pnl_status": pnl_status,
            "z_score": float(z_score),
            "z_desc": z_desc,
            "is_outlier": is_outlier,
            "divergence_text": divergence_text,
            "history_days": len(records),
            "series_data": records
        }
    except Exception as e:
        print(f"[FOOTPRINT] Warning: analyze_foreign_footprint failed: {e}")
        return {
            "cum_foreign_net": 0,
            "cost_basis": futures_ref,
            "pnl_pts": 0.0,
            "pnl_status": "Error",
            "z_score": 0.0,
            "z_desc": "เกิดข้อผิดพลาดในการดึงข้อมูล",
            "is_outlier": False,
            "divergence_text": "N/A",
            "history_days": 0,
            "series_data": []
        }

def analyze_options_footprint(
    pipeline_url: str,
    auth_token: str,
    t_0: str,
    t_minus_1: str,
    front_series: str,
    futures_ref_t0: float,
    futures_ref_tprev: float,
    T_front_t0: float,
    T_front_tprev: float,
    r_rf: float
) -> dict:
    """
    Pairs Foreign Call/Put flow with strike-level delta OI and delta IV
    between T_0 and T_-1 to uncover institutional options positioning intent.
    """
    try:
        # 1. Fetch Foreign Options Net from eod_investor_types
        sql_inv = """
        SELECT category, foreign_net 
        FROM eod_investor_types 
        WHERE trade_date = ? AND (category LIKE '%Option%' OR category LIKE '%Call%' OR category LIKE '%Put%');
        """
        _, rows_inv = execute_turso_sql(pipeline_url, auth_token, sql_inv, [t_0])
        f_call_net = 0
        f_put_net = 0
        for r in rows_inv:
            cat = str(r[0])
            val = int(r[1] or 0)
            if "Call" in cat:
                f_call_net += val
            elif "Put" in cat:
                f_put_net += val

        # 2. Fetch Options Series at T_0 and T_-1 for front_series
        sql_opts = """
        SELECT option_type, strike, settlement_price, oi
        FROM eod_options_series 
        WHERE trade_date = ? AND (series = ? OR symbol LIKE ?);
        """
        _, rows_o0 = execute_turso_sql(pipeline_url, auth_token, sql_opts, [t_0, front_series, f"{front_series}%"])
        _, rows_oprev = execute_turso_sql(pipeline_url, auth_token, sql_opts, [t_minus_1, front_series, f"{front_series}%"])

        prev_map = {}
        for r in rows_oprev:
            otype = str(r[0]).upper()
            strike = float(r[1])
            px = float(r[2]) if r[2] is not None else 0.0
            oi = int(r[3] or 0)
            prev_map[(otype, strike)] = {"px": px, "oi": oi}

        call_candidates = []
        put_candidates = []

        for r in rows_o0:
            otype = str(r[0]).upper()
            strike = float(r[1])
            px_0 = float(r[2]) if r[2] is not None else 0.0
            oi_0 = int(r[3] or 0)

            prev_info = prev_map.get((otype, strike), {"px": px_0, "oi": oi_0})
            px_prev = prev_info["px"]
            oi_prev = prev_info["oi"]
            doi = oi_0 - oi_prev

            # Calculate IV at T_0 and T_-1
            iv_0 = calculate_implied_vol(px_0, futures_ref_t0, strike, T_front_t0, r_rf, otype)
            iv_prev = calculate_implied_vol(px_prev, futures_ref_tprev, strike, T_front_tprev, r_rf, otype)

            is_valid_0 = (iv_0 is not None and not np.isnan(iv_0))
            is_valid_prev = (iv_prev is not None and not np.isnan(iv_prev))
            if is_valid_0 and is_valid_prev:
                div = (iv_0 - iv_prev) * 100.0
            else:
                # W4: ห้ามใช้ 0.0 แทน "ไม่รู้" เพราะ 0.0 ถูกตีความเป็น div <= 0
                # = "Call Writing / Cap Resistance" ซึ่งเป็นการสรุปจากอากาศ
                div = None

            item = {
                "strike": strike,
                "oi_0": oi_0,
                "doi": doi,
                "iv_0": iv_0,
                "div": div
            }

            if otype == "C":
                call_candidates.append(item)
            elif otype == "P":
                put_candidates.append(item)

        # Top Call Strike with positive delta OI
        calls_pos = [c for c in call_candidates if c["doi"] > 0]
        if calls_pos:
            top_call = max(calls_pos, key=lambda x: x["doi"])
            top_call_strike = float(top_call["strike"])
            top_call_doi = int(top_call["doi"])
            top_call_div = top_call["div"]  # None = IV เปลี่ยนไม่ได้
            if top_call_div is None:
                call_intent = "IV เปลี่ยนไม่ได้ (ข้อมูลไม่พอ) — งดสรุปเจตนา"
            elif top_call_div > 0.5:
                call_intent = "Aggressive Call Buying (เก็งกำไรทะลุต้าน)"
            elif top_call_div <= 0.0:
                call_intent = "Call Writing / Cap Resistance (ล็อกแนวต้าน / เก็บพรีเมียม)"
            else:
                call_intent = "Call Accumulation (สะสมสัญญา Call)"
        else:
            top_call_strike = None
            top_call_doi = 0
            top_call_div = None
            call_intent = "ไม่มีการเปลี่ยนแปลง OI ที่มีนัยสำคัญ"

        # Top Put Strike with positive delta OI
        puts_pos = [p for p in put_candidates if p["doi"] > 0]
        if puts_pos:
            top_put = max(puts_pos, key=lambda x: x["doi"])
            top_put_strike = float(top_put["strike"])
            top_put_doi = int(top_put["doi"])
            top_put_div = top_put["div"]  # None = IV เปลี่ยนไม่ได้
            if top_put_div is None:
                put_intent = "IV เปลี่ยนไม่ได้ (ข้อมูลไม่พอ) — งดสรุปเจตนา"
            elif top_put_div > 0.5:
                put_intent = "Aggressive Put Buying / Downside Hedging (ประกันความเสี่ยงขาลงเร่งด่วน)"
            elif top_put_div <= 0.0:
                put_intent = "Put Writing / Floor Support (ขาย Put ล็อกฐานแนวรับ)"
            else:
                put_intent = "Put Accumulation (สะสมสัญญา Put)"
        else:
            top_put_strike = None
            top_put_doi = 0
            top_put_div = None
            put_intent = "ไม่มีการเปลี่ยนแปลง OI ที่มีนัยสำคัญ"

        return {
            "foreign_call_net": f_call_net,
            "foreign_put_net": f_put_net,
            "top_call_strike": top_call_strike,
            "top_call_doi": top_call_doi,
            "top_call_div": top_call_div,
            "call_intent": call_intent,
            "top_put_strike": top_put_strike,
            "top_put_doi": top_put_doi,
            "top_put_div": top_put_div,
            "put_intent": put_intent
        }
    except Exception as e:
        print(f"[OPTIONS_FOOTPRINT] Warning: analyze_options_footprint failed: {e}")
        return {
            "foreign_call_net": 0,
            "foreign_put_net": 0,
            "top_call_strike": None,
            "top_call_doi": 0,
            "top_call_div": None,
            "call_intent": "ไม่มีการเปลี่ยนแปลง OI ที่มีนัยสำคัญ",
            "top_put_strike": None,
            "top_put_doi": 0,
            "top_put_div": None,
            "put_intent": "ไม่มีการเปลี่ยนแปลง OI ที่มีนัยสำคัญ"
        }

# ==============================================================================
# INTRADAY WARM-UP ANALYTICS (REALIZED VOLATILITY, VOLUME PROFILE, LATE SESSION)
# ==============================================================================

def resample_cdc_to_1min_prices(raw_rows: List[Any], trade_date: str) -> List[float]:
    """
    Resamples irregular CDC ticks into continuous 1-minute bars using forward fill (ffill).
    Generates standard 355-minute TFEX trading grid (09:45-12:30, 13:45-16:55)
    to prevent Realized Volatility (RV) distortion from irregular event capture.
    """
    if not raw_rows:
        return []

    # Map observed last price to minute buckets (last tick of that minute)
    observed_by_minute: Dict[str, float] = {}
    for r in raw_rows:
        ts_str = str(r[0])
        px = float(r[1]) if (r[1] is not None and float(r[1]) > 0) else None
        if px is not None:
            m_k = ts_str[:16] + ":00" if len(ts_str) >= 16 else ts_str
            # Clamp session close snapshots (12:30 -> 12:29, 16:55 -> 16:54) to bar grid
            if m_k == f"{trade_date} 12:30:00":
                m_k = f"{trade_date} 12:29:00"
            elif m_k == f"{trade_date} 16:55:00":
                m_k = f"{trade_date} 16:54:00"
            observed_by_minute[m_k] = px

    if not observed_by_minute:
        return []

    # Generate TFEX regular 1-minute grid (exactly 355 bars: 165 morning + 190 afternoon)
    grid = []
    # Morning: 09:45:00 to 12:29:00 (165 bars)
    cur = datetime.datetime.strptime(f"{trade_date} 09:45:00", "%Y-%m-%d %H:%M:%S")
    end_m = datetime.datetime.strptime(f"{trade_date} 12:30:00", "%Y-%m-%d %H:%M:%S")
    while cur < end_m:
        grid.append(cur.strftime("%Y-%m-%d %H:%M:00"))
        cur += datetime.timedelta(minutes=1)
    # Afternoon: 13:45:00 to 16:54:00 (190 bars)
    cur = datetime.datetime.strptime(f"{trade_date} 13:45:00", "%Y-%m-%d %H:%M:%S")
    end_a = datetime.datetime.strptime(f"{trade_date} 16:55:00", "%Y-%m-%d %H:%M:%S")
    while cur < end_a:
        grid.append(cur.strftime("%Y-%m-%d %H:%M:00"))
        cur += datetime.timedelta(minutes=1)

    # Forward fill prices across the continuous 355-minute grid
    resampled_prices: List[float] = []
    last_known = None
    for m_ts in grid:
        if m_ts in observed_by_minute:
            last_known = observed_by_minute[m_ts]
        if last_known is not None:
            resampled_prices.append(last_known)

    return resampled_prices


def resample_cdc_to_5min_prices(raw_rows: List[Any], trade_date: str) -> List[float]:
    """
    Resamples CDC ticks to 5-minute frequency bars (09:45, 09:50, ..., 16:50, 16:54)
    to dampen microstructure noise / bid-ask bounce in Realized Volatility (RV) calculations.
    """
    prices_1m = resample_cdc_to_1min_prices(raw_rows, trade_date)
    if not prices_1m:
        return []
    # Sample every 5th minute: index 0, 5, 10, ...
    prices_5m = prices_1m[::5]
    if (len(prices_1m) - 1) % 5 != 0:
        prices_5m.append(prices_1m[-1])
    return prices_5m


def compute_intraday_rv(
    pipeline_url: str,
    auth_token: str,
    trade_date: str,
    front_series: str,
    atm_iv: float
) -> dict:
    """
    Fetches 1-minute candlestick data from Turso (futures_snapshots / spot_snapshots),
    resamples to 5-minute bars to eliminate microstructure noise,
    incorporates overnight gap ln(O_t / C_t-1)^2 for full-day close-to-close RV,
    persists/upserts into daily_realized_vol table, and compares RV vs ATM IV.
    """
    now_bkk = datetime.datetime.now(BKK_TZ)
    updated_at_bkk = now_bkk.strftime("%Y-%m-%d %H:%M:%S")

    try:
        # 1. Fetch 1-minute futures bars for the given trade_date
        sql_fut = """
        SELECT captured_at_bkk, last 
        FROM futures_snapshots 
        WHERE symbol = ? AND captured_at_bkk LIKE ? AND last IS NOT NULL AND last > 0 
        ORDER BY captured_at_bkk ASC;
        """
        _, rows_fut = execute_turso_sql(pipeline_url, auth_token, sql_fut, [front_series, f"{trade_date}%"])
        prices = resample_cdc_to_1min_prices(rows_fut, trade_date)

        bar_count = len(prices)
        if bar_count < 10:
            return {
                "rv_annualized": None,
                "bar_count": bar_count,
                "is_valid_rv": False,
                "iv_rv_spread": None,
                "vol_regime_desc": f"[INSUFFICIENT FUTURES DATA] มีเพียง {bar_count}/10 บาร์ (ต้องการอย่างน้อย 10 บาร์)"
            }

        is_valid_rv = (bar_count >= 310)

        # Resample to 5-minute bars to eliminate microstructure noise & bid-ask bounce
        prices_5m = resample_cdc_to_5min_prices(rows_fut, trade_date)
        log_returns_5m = [
            math.log(prices_5m[i] / prices_5m[i - 1]) 
            for i in range(1, len(prices_5m)) 
            if prices_5m[i] > 0 and prices_5m[i - 1] > 0
        ]
        intraday_var = sum(r * r for r in log_returns_5m)

        # Close-to-Close RV incorporating Overnight Gap: ln(O_t / C_t_minus_1)^2
        overnight_var = 0.0
        try:
            sql_prev = """
            SELECT last FROM futures_snapshots 
            WHERE symbol = ? AND captured_at_bkk < ? AND last IS NOT NULL AND last > 0 
            ORDER BY captured_at_bkk DESC LIMIT 1;
            """
            _, rows_prev = execute_turso_sql(pipeline_url, auth_token, sql_prev, [front_series, f"{trade_date} 09:00:00"])
            if rows_prev and rows_prev[0][0] is not None and float(rows_prev[0][0]) > 0:
                prev_close = float(rows_prev[0][0])
                if prices_5m and prices_5m[0] > 0 and prev_close > 0:
                    r_overnight = math.log(prices_5m[0] / prev_close)
                    overnight_var = r_overnight * r_overnight
        except Exception as e:
            print(f"[RV] Note fetching prev close for overnight gap: {e}")

        total_daily_var = overnight_var + intraday_var
        rv_annualized = math.sqrt(252.0 * total_daily_var)

        # 2. Persist / UPSERT to daily_realized_vol
        upsert_sql = """
        INSERT INTO daily_realized_vol (trade_date, rv_annualized, bar_count, updated_at_bkk)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(trade_date) DO UPDATE SET
            rv_annualized = excluded.rv_annualized,
            bar_count = excluded.bar_count,
            updated_at_bkk = excluded.updated_at_bkk;
        """
        execute_turso_sql(pipeline_url, auth_token, upsert_sql, [trade_date, rv_annualized, bar_count, updated_at_bkk])

        # 3. Compare RV vs ATM IV
        if atm_iv is not None and atm_iv > 0:
            spread = (atm_iv - rv_annualized) * 100.0
            if not is_valid_rv:
                vol_regime_desc = f"ข้อมูล Intraday วันล่าสุดไม่เต็มวัน (มี {bar_count}/355 บาร์) — RV {rv_annualized*100:.2f}% (เพื่อการอ้างอิง)"
            elif atm_iv > rv_annualized + 0.02:
                vol_regime_desc = f"IV Premium สูง (Options แพงกว่าแกว่งจริง +{spread:.1f}% เหมาะเทรดในกรอบ)"
            elif rv_annualized >= atm_iv:
                vol_regime_desc = f"RV ทะลุ IV (ตลาดแกว่งจริงแรงกว่าคาดการณ์ {rv_annualized*100:.1f}% vs {atm_iv*100:.1f}% ระวัง Breakout)"
            else:
                vol_regime_desc = f"สมดุล (IV {atm_iv*100:.1f}% สอดคล้องกับ RV {rv_annualized*100:.1f}%)"
        else:
            spread = None
            vol_regime_desc = f"RV คำนวณได้ {rv_annualized*100:.2f}% ({bar_count} บาร์) แต่ไม่มีค่า ATM IV เปรียบเทียบ"

        return {
            "rv_annualized": float(rv_annualized),
            "bar_count": bar_count,
            "is_valid_rv": is_valid_rv,
            "iv_rv_spread": float(spread) if spread is not None else None,
            "vol_regime_desc": vol_regime_desc
        }
    except Exception as e:
        print(f"[RV] Warning: compute_intraday_rv failed: {e}")
        return {
            "rv_annualized": None,
            "bar_count": 0,
            "is_valid_rv": False,
            "iv_rv_spread": None,
            "vol_regime_desc": "[INSUFFICIENT FUTURES DATA] ไม่สามารถดึงข้อมูลได้"
        }

def compute_volume_profile(
    pipeline_url: str,
    auth_token: str,
    target_date: str,
    front_series: str,
    tactical_levels: dict,
    call_wall: float,
    put_wall: float,
    max_pain: float,
    lookback_days: int = 20
) -> dict:
    """
    Computes rolling up to 20-day Intraday Volume Profile for front quarter futures,
    extracting Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
    Cross-checks POC/VAH/VAL with Options Walls and Max Pain for Confluence Zones.
    """
    try:
        # 1. Fetch available trade dates and bar counts in futures_snapshots (up to lookback_days)
        sql_counts = """
        SELECT SUBSTR(captured_at_bkk, 1, 10) as dt, COUNT(DISTINCT SUBSTR(captured_at_bkk, 1, 16)) as cnt 
        FROM futures_snapshots 
        WHERE captured_at_bkk <= ? AND symbol = ?
        GROUP BY dt
        ORDER BY dt DESC 
        LIMIT ?;
        """
        _, rows_counts = execute_turso_sql(pipeline_url, auth_token, sql_counts, [f"{target_date} 23:59:59", front_series, lookback_days])

        if not rows_counts:
            sql_counts_all = """
            SELECT SUBSTR(captured_at_bkk, 1, 10) as dt, COUNT(DISTINCT SUBSTR(captured_at_bkk, 1, 16)) as cnt 
            FROM futures_snapshots 
            WHERE captured_at_bkk <= ? 
            GROUP BY dt
            ORDER BY dt DESC 
            LIMIT ?;
            """
            _, rows_counts = execute_turso_sql(pipeline_url, auth_token, sql_counts_all, [f"{target_date} 23:59:59", lookback_days])

        warmup_days = len(rows_counts)
        valid_days = len([r for r in rows_counts if int(r[1] or 0) >= 310])

        if warmup_days == 0:
            return {
                "poc": None,
                "vah": None,
                "val": None,
                "warmup_days": 0,
                "valid_days": 0,
                "confluence_text": "[NO INTRADAY VOLUME DATA]"
            }

        sorted_counts = sorted(rows_counts, key=lambda x: x[0])
        min_date = sorted_counts[0][0]
        max_date = sorted_counts[-1][0]

        # 2. Fetch 1-minute records across these dates
        sql_bars = """
        SELECT captured_at_bkk, last, volume 
        FROM futures_snapshots 
        WHERE symbol = ? AND SUBSTR(captured_at_bkk, 1, 10) >= ? AND SUBSTR(captured_at_bkk, 1, 10) <= ?
        ORDER BY captured_at_bkk ASC;
        """
        _, rows_bars = execute_turso_sql(pipeline_url, auth_token, sql_bars, [front_series, min_date, max_date])

        if not rows_bars or len(rows_bars) < 10:
            return {
                "poc": None,
                "vah": None,
                "val": None,
                "warmup_days": warmup_days,
                "valid_days": valid_days,
                "confluence_text": "[NO INTRADAY VOLUME DATA]"
            }

        # 3. Aggregate volume by price
        vol_by_price = {}
        prev_vol = None
        prev_day = None

        for r in rows_bars:
            ts = str(r[0])
            day_str = ts[:10]
            px = float(r[1]) if r[1] is not None else 0.0
            tot_v = float(r[2]) if r[2] is not None else 0.0

            if px <= 0:
                continue

            if prev_day != day_str:
                prev_day = day_str
                delta_v = 1.0
                prev_vol = tot_v
            else:
                if prev_vol is not None and tot_v >= prev_vol:
                    delta_v = tot_v - prev_vol
                else:
                    delta_v = 1.0
                prev_vol = tot_v

            # S3: SET50 futures tick = 0.1 point. Bucketing at 1.0 point collapsed
            # distinct price levels and made POC/VAH/VAL 10x coarser than the tick.
            k = round(round(px / VOLUME_PROFILE_BUCKET) * VOLUME_PROFILE_BUCKET, 1)
            vol_by_price[k] = vol_by_price.get(k, 0.0) + max(1.0, delta_v)

        total_v = sum(vol_by_price.values()) if vol_by_price else 0.0
        if not vol_by_price or total_v <= 0:
            return {
                "poc": None,
                "vah": None,
                "val": None,
                "warmup_days": warmup_days,
                "valid_days": valid_days,
                "confluence_text": "[NO INTRADAY VOLUME DATA]"
            }

        total_v = sum(vol_by_price.values())
        poc = max(vol_by_price.keys(), key=lambda k: vol_by_price[k])

        sorted_prices = sorted(vol_by_price.keys())
        poc_idx = sorted_prices.index(poc)
        va_prices = {poc}
        current_v = vol_by_price[poc]
        target_v = 0.70 * total_v

        up_idx = poc_idx + 1
        down_idx = poc_idx - 1

        while current_v < target_v and (up_idx < len(sorted_prices) or down_idx >= 0):
            vol_up = vol_by_price[sorted_prices[up_idx]] if up_idx < len(sorted_prices) else -1
            vol_down = vol_by_price[sorted_prices[down_idx]] if down_idx >= 0 else -1

            if vol_up >= vol_down and up_idx < len(sorted_prices):
                va_prices.add(sorted_prices[up_idx])
                current_v += vol_up
                up_idx += 1
            elif down_idx >= 0:
                va_prices.add(sorted_prices[down_idx])
                current_v += vol_down
                down_idx -= 1
            else:
                break

        vah = max(va_prices)
        val = min(va_prices)

        # 4. Cross-check Confluence Zone with Options levels
        opts_levels = [
            ("Max Pain", max_pain),
            ("Call Wall", call_wall),
            ("Put Wall", put_wall),
            ("ต้าน 1", tactical_levels.get("res_1", 930)),
            ("รับ 1", tactical_levels.get("sup_1", 910)),
        ]

        confluences = []
        for name, lvl in opts_levels:
            if abs(poc - lvl) <= 3.0:
                confluences.append(f"POC ({poc:.0f}) ชน {name} ({lvl:.0f}) ➔ ปะทะหลัก")
            elif abs(vah - lvl) <= 3.0:
                confluences.append(f"VAH ({vah:.0f}) ชน {name} ({lvl:.0f}) ➔ ต้านคอนกรีต")
            elif abs(val - lvl) <= 3.0:
                confluences.append(f"VAL ({val:.0f}) ชน {name} ({lvl:.0f}) ➔ รับคอนกรีต")

        confluence_text = " | ".join(confluences[:2]) if confluences else "กรอบ Volume Profile สอดคล้องกับการเคลื่อนไหวในวัน"

        return {
            "poc": float(poc),
            "vah": float(vah),
            "val": float(val),
            "warmup_days": warmup_days,
            "valid_days": valid_days,
            "confluence_text": confluence_text
        }
    except Exception as e:
        print(f"[VOL_PROFILE] Warning: compute_volume_profile failed: {e}")
        return {
            "poc": None,
            "vah": None,
            "val": None,
            "warmup_days": 0,
            "valid_days": 0,
            "confluence_text": "[NO INTRADAY VOLUME DATA]"
        }

def analyze_late_session_flow(
    pipeline_url: str,
    auth_token: str,
    trade_date: str,
    front_series: str,
    foreign_net: int
) -> dict:
    """
    Examines the last 25 minutes (16:30 - 16:55) of trading on trade_date.
    Analyzes price trajectory, Futures-Spot Basis expansion/contraction,
    and interprets market closing pressure combined with Foreign Net flow.
    """
    try:
        start_ts = f"{trade_date} 16:30:00"
        end_ts = f"{trade_date} 16:55:00"

        sql_fut = """
        SELECT captured_at_bkk, last 
        FROM futures_snapshots 
        WHERE symbol = ? AND captured_at_bkk >= ? AND captured_at_bkk <= ? AND last IS NOT NULL AND last > 0 
        ORDER BY captured_at_bkk ASC;
        """
        _, rows_fut = execute_turso_sql(pipeline_url, auth_token, sql_fut, [front_series, start_ts, end_ts])

        sql_spot = """
        SELECT captured_at_bkk, last 
        FROM spot_snapshots 
        WHERE symbol = 'SET50' AND captured_at_bkk >= ? AND captured_at_bkk <= ? AND last IS NOT NULL AND last > 0 
        ORDER BY captured_at_bkk ASC;
        """
        _, rows_spot = execute_turso_sql(pipeline_url, auth_token, sql_spot, [start_ts, end_ts])

        if not rows_fut or len(rows_fut) < 2:
            return {
                "late_fut_delta": 0.0,
                "late_basis_close": 0.0,
                "late_basis_delta": 0.0,
                "late_session_desc": "⚪ ข้อมูลท้ายตลาดยังไม่เพียงพอในการตรวจจับแรงปิด"
            }

        f_start = float(rows_fut[0][1])
        f_close = float(rows_fut[-1][1])
        late_fut_delta = f_close - f_start

        spot_dict = {str(r[0]): float(r[1]) for r in rows_spot if r[1] is not None}
        s_start = spot_dict.get(str(rows_fut[0][0]), f_start)
        s_close = spot_dict.get(str(rows_fut[-1][0]), f_close)

        basis_start = f_start - s_start
        basis_close = f_close - s_close
        late_basis_delta = basis_close - basis_start

        if foreign_net > 0 and late_fut_delta > 0.3 and late_basis_delta >= 0:
            late_session_desc = "🟢 ต่างชาติซื้อ + ท้ายตลาดไล่ราคา Basis ขยับขึ้น ปิดใกล้ High มีโอกาสเปิดบวกต่อ"
        elif foreign_net < 0 and late_fut_delta < -0.3 and late_basis_delta <= 0:
            late_session_desc = "🔴 ต่างชาติขาย + มีแรงทิ้งสัญญาท้ายตลาด Basis หดตัว ระวังแรงกดดันต่อเนื่อง"
        elif late_fut_delta > 0.5:
            late_session_desc = "🟢 มีแรงซื้อเร่งดึงราคาท้ายตลาด ดันดัชนีปิดโซนบน"
        elif late_fut_delta < -0.5:
            late_session_desc = "🔴 มีแรงขายกดราคาท้ายตลาด ดัชนีปิดโซนล่าง"
        else:
            late_session_desc = "⚪ ท้ายตลาดทรงตัวในกรอบ Basis ปกติ ไม่มีแรงเร่งผิดปกติ"

        return {
            "late_fut_delta": float(late_fut_delta),
            "late_basis_close": float(basis_close),
            "late_basis_delta": float(late_basis_delta),
            "late_session_desc": late_session_desc
        }
    except Exception as e:
        print(f"[LATE_SESSION] Warning: analyze_late_session_flow failed: {e}")
        return {
            "late_fut_delta": 0.0,
            "late_basis_close": 0.0,
            "late_basis_delta": 0.0,
            "late_session_desc": "⚪ ไม่สามารถดึงข้อมูลช่วงท้ายตลาดได้"
        }

def fetch_volatility_time_series(
    pipeline_url: str,
    auth_token: str,
    current_date: str,
    current_atm_iv: float,
    current_rv: float,
    is_valid_rv: bool,
    lookback_days: int = 60
) -> list[dict]:
    """
    Fetches historical ATM IV (from analytics_premarket_log) and valid Daily RV (bar_count >= 310 from daily_realized_vol).
    Merges with current session's data for the Volatility Regime Time-Series chart.
    """
    iv_map = {}
    rv_map = {}
    try:
        sql_log = """
        SELECT trade_date, atm_iv 
        FROM analytics_premarket_log 
        WHERE trade_date <= ? 
        ORDER BY trade_date ASC 
        LIMIT ?;
        """
        _, rows_log = execute_turso_sql(pipeline_url, auth_token, sql_log, [current_date, lookback_days])
        for r in rows_log:
            if r[0] and r[1] is not None:
                iv_map[str(r[0])] = float(r[1])
    except Exception as e:
        print(f"[VOL_SERIES] Note: fetch analytics_premarket_log: {e}")

    try:
        sql_rv = """
        SELECT trade_date, rv_annualized, bar_count 
        FROM daily_realized_vol 
        WHERE bar_count >= 310 AND trade_date <= ? 
        ORDER BY trade_date ASC 
        LIMIT ?;
        """
        _, rows_rv = execute_turso_sql(pipeline_url, auth_token, sql_rv, [current_date, lookback_days])
        for r in rows_rv:
            if r[0] and r[1] is not None:
                rv_map[str(r[0])] = float(r[1])
    except Exception as e:
        print(f"[VOL_SERIES] Note: fetch daily_realized_vol: {e}")

    # Add or update current session
    if current_atm_iv is not None:
        iv_map[current_date] = float(current_atm_iv)
    if is_valid_rv and current_rv is not None:
        rv_map[current_date] = float(current_rv)

    all_dates = sorted(set(list(iv_map.keys()) + list(rv_map.keys())))
    vol_series = []
    for dt in all_dates:
        vol_series.append({
            "date": dt,
            "trade_date": dt,
            "atm_iv": iv_map.get(dt),
            "rv_annualized": rv_map.get(dt)
        })
    return vol_series

# ==============================================================================
# RULE-BASED THAI TRADING PLAYBOOK GENERATOR
# ==============================================================================

def generate_thai_trading_playbook(
    trade_date: str,
    as_of_time_bkk: str,
    futures_ref: float,
    front_series: str,
    atm_iv: float,
    expected_move_1d: float,
    upper_1sd: float,
    lower_1sd: float,
    tactical_levels: dict,
    max_pain: float,
    call_wall: float,
    put_wall: float,
    gamma_flip_val: float,
    is_positive_gamma: bool,
    gamma_zone_title: str,
    gamma_desc: str,
    total_net_gex: float,
    foreign_net: int,
    delta_oi: int,
    inst_net: int,
    local_net: int,
    flow_regime: str,
    flow_emoji: str,
    foreign_footprint: dict = None,
    options_footprint: dict = None,
    intraday_rv: dict = None,
    volume_profile: dict = None,
    late_session: dict = None,
    greeks_edge: dict = None
) -> str:
    """
    Constructs an actionable, rule-based frontline trading plan in Thai.
    5 Core Pillars:
    1. 1-Day Expected Move (1-SD)
    2. Tactical Key Levels & Quarter Walls
    3. Intraday Structure & Realized Volatility (Warm-up 20D)
    4. Smart Money Flow & Footprint Breakdown
    5. Daily Action Plan & Institutional Defense Level
    """
    res_1 = tactical_levels["res_1"]
    res_1_oi = tactical_levels["res_1_oi"]
    res_2 = tactical_levels["res_2"]
    res_2_oi = tactical_levels["res_2_oi"]
    sup_1 = tactical_levels["sup_1"]
    sup_1_oi = tactical_levels["sup_1_oi"]
    sup_2 = tactical_levels["sup_2"]
    sup_2_oi = tactical_levels["sup_2_oi"]

    # 1. IV interpretation
    if atm_iv is not None and atm_iv > 0:
        if atm_iv < 0.10:
            iv_desc = "ความผันผวนต่ำมาก ตลาดมักบีบกรอบแคบ"
        elif atm_iv <= 0.15:
            iv_desc = "ความผันผวนปกติ เหมาะเทรดในกรอบ"
        else:
            iv_desc = "ความผันผวนสูง แกว่งกว้าง ควรกระชับจุดคัต"
        atm_iv_text = f"<code>{atm_iv * 100:.2f}%</code> <i>({iv_desc})</i>"
    else:
        atm_iv_text = "<i>[UNAVAILABLE - ข้อมูลราคา Options ไม่เพียงพอ]</i>"

    if expected_move_1d is not None and upper_1sd is not None and lower_1sd is not None:
        expected_move_text = f"<code>{lower_1sd:.1f}</code> – <code>{upper_1sd:.1f}</code> จุด <i>(ระยะแกว่ง ±{expected_move_1d:.1f} จุด)</i>"
    else:
        expected_move_text = "<i>[UNAVAILABLE - ข้อมูลราคา Options ไม่เพียงพอ]</i>"

    f_cost = foreign_footprint.get("cost_basis", futures_ref) if foreign_footprint else futures_ref

    # 2. Smart Money Translation & Actionable Playbook Logic
    if foreign_net > 0 and delta_oi > 0:
        flow_title = "New Long Build-up (ต่างชาติเปิดสะสม Long ใหม่)"
        flow_detail = "ซื้อสุทธิพร้อม OI ขยายตัว เป็นการเปิด Long ใหม่หนุนตลาดขึ้น"
        action_strategy = "Follow-Through ฝั่ง Long ตามโมเมนตัม ไม่ดักสวนเทรนด์"
        long_plan = f"ย่อแถว รับ 1 (<code>{sup_1:.0f}</code>) ยืนได้เปิด Long เป้า <code>{res_1:.0f}</code> และ <code>{upper_1sd:.1f}</code>"
        short_plan = f"ชะลอ Short จนกว่าจะขึ้นทดสอบ ต้าน 2 (<code>{res_2:.0f}</code>) หรือ Upper 1-SD แล้ว Rejection ไม่ผ่าน"
        stop_loss_plan = f"ฝั่ง Long วาง Stop Loss เมื่อหลุด <code>{min(sup_2, f_cost):.0f}</code> จุด (แนว Defense ฝรั่ง)"
    elif foreign_net > 0 and delta_oi < 0:
        flow_title = "Short Covering (ต่างชาติปิดทำกำไร Short)"
        flow_detail = "ซื้อสุทธิแต่ OI ลดลง เป็นแรงซื้อคืนปิด Short เดิมเพื่อลดพอร์ต"
        action_strategy = "เน้นเล่นรอบในกรอบ (Mean Reversion) ดักย่อซื้อ-เด้งขาย ไม่ไล่ราคา"
        long_plan = f"ตั้งรับ Lower 1-SD (<code>{lower_1sd:.1f}</code>) หรือ รับ 1 (<code>{sup_1:.0f}</code>) เป้า Max Pain (<code>{max_pain:.0f}</code>) หรือ ต้าน 1"
        short_plan = f"เปิดกระโดดชน ต้าน 1 (<code>{res_1:.0f}</code>) หรือ Upper 1-SD (<code>{upper_1sd:.1f}</code>) ไม่ผ่าน ดัก Short เล่นรอบย่อ"
        stop_loss_plan = f"ฝั่ง Long คัตเมื่อหลุด <code>{sup_2:.0f}</code> | ฝั่ง Short คัตเมื่อทะลุ <code>{res_1 + 3.0:.1f}</code>"
    elif foreign_net < 0 and delta_oi > 0:
        flow_title = "New Short Build-up (ต่างชาติเปิดสะสม Short ใหม่)"
        flow_detail = "ขายสุทธิพร้อม OI ขยายตัว สะท้อนการเปิด Short เพิ่มเก็งกำไรขาลง"
        action_strategy = "เน้นฝั่ง Short (Trend Expansion Down) ตามน้ำ ระวังการหลุดแนวรับแล้วรูดแรง"
        long_plan = f"เลี่ยงดักสวนรับเร็ว จนกว่าราคาจะแตะ รับ 2 (<code>{sup_2:.0f}</code>) หรือ Lower 1-SD แล้วเกิดสัญญาณหยุดลง"
        short_plan = f"เด้งชน ต้าน 1 (<code>{res_1:.0f}</code>) หรือ Max Pain (<code>{max_pain:.0f}</code>) ไม่ผ่าน เปิด Short เป้า <code>{sup_1:.0f}</code> และ <code>{lower_1sd:.1f}</code>"
        stop_loss_plan = f"ฝั่ง Short วาง Stop Loss เหนือ <code>{max(res_1 + 2.5, f_cost):.1f}</code> จุด"
    elif foreign_net < 0 and delta_oi < 0:
        flow_title = "Long Liquidation (ต่างชาติปิดลดพอร์ต Long)"
        flow_detail = "ขายสุทธิควบคู่ OI ลดลง เป็นการปิดสัญญาลดความเสี่ยง"
        action_strategy = "เทรดรอบสั้นอย่างระมัดระวัง เน้นตั้งรับต่ำและไม่ไล่ราคา"
        long_plan = f"ดักซื้อบริเวณ รับ 1 (<code>{sup_1:.0f}</code>) หรือ Lower 1-SD (<code>{lower_1sd:.1f}</code>) เล่นรีบาวด์สั้นๆ"
        short_plan = f"เด้งไม่ผ่าน ต้าน 1 (<code>{res_1:.0f}</code>) เปิด Short เล่นรอบย่อ"
        stop_loss_plan = f"ฝั่ง Long คัตเมื่อหลุด <code>{sup_1 - 2.0:.1f}</code> | ฝั่ง Short คัตเมื่อทะลุเหนือ <code>{res_2:.0f}</code>"
    else:
        flow_title = "Neutral / Rebalancing (ปรับสมดุลพอร์ต)"
        flow_detail = "ทิศทางฟันด์โฟลว์ยังไม่ชี้นำเด่นชัด เคลื่อนไหวตามกรอบเทคนิค"
        action_strategy = "เล่นในกรอบ รับ-ต้าน ตามปกติ"
        long_plan = f"รับแถว <code>{sup_1:.0f}</code> จุด"
        short_plan = f"ต้านแถว <code>{res_1:.0f}</code> จุด"
        stop_loss_plan = f"คัตเมื่อหลุด <code>{sup_2:.0f}</code> หรือทะลุ <code>{res_2:.0f}</code>"

    gamma_flip_str = f"{gamma_flip_val:.1f}" if not math.isnan(gamma_flip_val) else "N/A"
    if not math.isnan(gamma_flip_val) and abs(futures_ref - gamma_flip_val) <= 3.0:
        gamma_emoji = "⚖️"
        gamma_zone_title = "Transitional Gamma (ใกล้จุดเปลี่ยนโซน)"
        gamma_desc_compact = f"ราคาประชิด Gamma Flip ({gamma_flip_val:.1f}) เป็นจุดหมุนความผันผวน ตลาดอยู่ในช่วงเปลี่ยนผ่าน"
    elif is_positive_gamma:
        gamma_emoji = "🛡️"
        gamma_zone_title = "Positive Gamma (โซนลดความผันผวน / Mean Reversion)"
        gamma_desc_compact = f"ราคาเหนือ Gamma Flip ({gamma_flip_val:.1f}) ดีลเลอร์ Long Gamma ซับแรงเหวี่ยง พยุงตลาดแกว่งในกรอบ"
    else:
        gamma_emoji = "⚡"
        gamma_zone_title = "Negative Gamma (โซนเร่งความผันผวน / Trend Expansion)"
        gamma_desc_compact = f"ราคาใต้ Gamma Flip ({gamma_flip_val:.1f}) ดีลเลอร์ Short Gamma หลุดรับจะเร่งแรงขาย ระวังผันผวนขยายตัว"

    # Format Options Greeks & Skew Edge Block
    greeks_edge_lines = []
    if greeks_edge and greeks_edge.get("records_count", 0) > 0:
        skew_val = greeks_edge.get("skew_25d")
        p_iv = greeks_edge.get("p_iv_pct")
        c_iv = greeks_edge.get("c_iv_pct")
        skew_desc = greeks_edge.get("skew_desc", "สมดุล")
        be_move = greeks_edge.get("daily_breakeven_move", 0.0)
        tg_ratio = greeks_edge.get("theta_gamma_ratio", 0.0)
        d_vanna = greeks_edge.get("dealer_vanna", 0.0)
        d_charm = greeks_edge.get("dealer_charm", 0.0)
        vanna_desc = greeks_edge.get("vanna_bias_desc", "")
        be_rule = "แกว่ง > ค่านี้เน้น Long / แคบกว่าเน้น Short รับ Theta"

        # W3: skew/IV อาจเป็น None เมื่อข้อมูลฝั่งใดฝั่งหนึ่งขาด -> ห้าม format ด้วย :.2f
        if skew_val is None or p_iv is None or c_iv is None:
            skew_line = f"  - <b>25Δ Skew:</b> <code>N/A</code> ➔ <i>{skew_desc}</i>"
        else:
            skew_line = (f"  - <b>25Δ Skew:</b> <code>{skew_val:+.2f}%</code> "
                         f"(Put <code>{p_iv:.1f}%</code> vs Call <code>{c_iv:.1f}%</code>) ➔ <i>{skew_desc}</i>")

        greeks_edge_lines = [
            f"• <b>Options Greeks & Skew Edge ({front_series}):</b>",
            skew_line,
            f"  - <b>ATM Breakeven Move:</b> <code>{be_move:.1f}</code> จุด/วัน (θ/γ: <code>{tg_ratio:.1f}</code>) ➔ <i>{be_rule}</i>",
            f"  - <b>Dealer Flow:</b> Vanna <code>{d_vanna:+,.0f}</code> สัญญา/1% IV | Charm <code>{d_charm:+,.0f}</code>/วัน ➔ <i>{vanna_desc}</i>"
        ]

    # Format Intraday & Realized Volatility Block
    if intraday_rv and volume_profile and late_session:
        rv_ann = intraday_rv.get("rv_annualized")
        bar_count = intraday_rv.get("bar_count", 0)
        is_valid_rv = intraday_rv.get("is_valid_rv", bar_count >= 310)
        vol_regime_desc = intraday_rv.get("vol_regime_desc", "N/A")

        poc = volume_profile.get("poc")
        vah = volume_profile.get("vah")
        val = volume_profile.get("val")
        warmup_days = volume_profile.get("warmup_days", 0)
        valid_days = volume_profile.get("valid_days", 0)
        confluence_text = volume_profile.get("confluence_text", "N/A")

        late_fut_delta = late_session.get("late_fut_delta", 0.0)
        late_basis_close = late_session.get("late_basis_close", 0.0)
        late_basis_delta = late_session.get("late_basis_delta", 0.0)
        late_session_desc = late_session.get("late_session_desc", "N/A")

        if rv_ann is not None:
            if atm_iv is not None:
                rv_section_line = f"• <b>RV vs IV:</b> RV <code>{rv_ann * 100:.2f}%</code> vs ATM IV <code>{atm_iv * 100:.2f}%</code> ({bar_count} บาร์) ➔ <i>{vol_regime_desc}</i>"
            else:
                rv_section_line = f"• <b>RV vs IV:</b> RV <code>{rv_ann * 100:.2f}%</code> ({bar_count} บาร์) ➔ <i>{vol_regime_desc}</i>"
        else:
            rv_section_line = f"• <b>RV vs IV:</b> ⏳ <i>[INSUFFICIENT FUTURES DATA] ({bar_count}/10 บาร์) — งดคำนวณ RV</i>"

        if poc is None or vah is None or val is None:
            profile_line = f"• <b>Volume Profile:</b> <i>[NO INTRADAY VOLUME DATA]</i>"
        elif valid_days < 5:
            profile_line = f"• <b>Volume Profile:</b> POC <code>{poc:.0f}</code> | VA: <code>{val:.0f}</code>–<code>{vah:.0f}</code> (<i>⚠️ สะสม {valid_days}/20 วัน</i>)"
        else:
            profile_line = f"• <b>Volume Profile:</b> POC <code>{poc:.0f}</code> | VA: <code>{val:.0f}</code>–<code>{vah:.0f}</code> ➔ <i>{confluence_text}</i>"

        late_line = f"• <b>ท้ายตลาด (16:30–16:55):</b> Fut <code>{late_fut_delta:+.1f}</code> | Basis <code>{late_basis_close:+.2f}</code> (Δ<code>{late_basis_delta:+.2f}</code>) ➔ <i>{late_session_desc}</i>"

        sec3_lines = [
            f"📊 <b>3. โครงสร้างความผันผวน & Intraday (Warm-up: {valid_days}/20 วัน)</b>"
        ]
        if greeks_edge_lines:
            sec3_lines.extend(greeks_edge_lines)
        sec3_lines.append(rv_section_line)
        sec3_lines.append(profile_line)
        sec3_lines.append(late_line)
        intraday_block = "\n".join(sec3_lines)
    else:
        if greeks_edge_lines:
            intraday_block = f"📊 <b>3. โครงสร้าง Options Greeks ({front_series})</b>\n" + "\n".join(greeks_edge_lines)
        else:
            intraday_block = ""

    # Format Foreign Footprint
    if foreign_footprint:
        f_cum = foreign_footprint.get("cum_foreign_net", 0)
        f_pnl_status = foreign_footprint.get("pnl_status", "N/A")
        f_z_desc = foreign_footprint.get("z_desc", "N/A")
        f_div_text = foreign_footprint.get("divergence_text", "N/A")
        f_days = foreign_footprint.get("history_days", 60)
        foreign_footprint_block = f"""• <b>ต่างชาติสะสม ({f_days}D):</b> <code>{f_cum:+,}</code> สัญญา | ทุนเฉลี่ย: <code>{f_cost:.1f}</code> จุด <i>({f_pnl_status})</i>
  - Z-Score: {f_z_desc} | {f_div_text}"""
    else:
        foreign_footprint_block = ""

    # Format Options Footprint
    if options_footprint:
        o_c_net = options_footprint.get("foreign_call_net", 0)
        o_p_net = options_footprint.get("foreign_put_net", 0)
        top_c_k = options_footprint.get("top_call_strike")
        top_c_doi = options_footprint.get("top_call_doi", 0)
        c_intent = options_footprint.get("call_intent", "ไม่มีนัยสำคัญ")

        top_p_k = options_footprint.get("top_put_strike")
        top_p_doi = options_footprint.get("top_put_doi", 0)
        p_intent = options_footprint.get("put_intent", "ไม่มีนัยสำคัญ")

        if top_c_k is not None and top_c_doi > 0:
            call_line = f"Call <code>{top_c_k:.0f}</code> (ΔOI: <code>{top_c_doi:+,}</code>) ➔ <i>{c_intent}</i>"
        else:
            call_line = f"Call: <i>{c_intent}</i>"

        if top_p_k is not None and top_p_doi > 0:
            put_line = f"Put <code>{top_p_k:.0f}</code> (ΔOI: <code>{top_p_doi:+,}</code>) ➔ <i>{p_intent}</i>"
        else:
            put_line = f"Put: <i>{p_intent}</i>"

        options_footprint_block = f"""• <b>Options Footprint:</b> Call <code>{o_c_net:+,}</code> | Put <code>{o_p_net:+,}</code> สัญญา
  - {call_line}
  - {put_line}"""
    else:
        options_footprint_block = ""

    # Format Institutional Cost Narrative for Action Plan
    if foreign_footprint and f_cost > 0:
        cum_f = foreign_footprint.get("cum_foreign_net", 0)
        if cum_f < 0:
            if futures_ref < f_cost:
                pnl = f_cost - futures_ref
                cost_relation = f"ต่างชาติกำไร Short (+{pnl:.1f} จุด) แนว {f_cost:.1f} เป็นต้านต้นทุน ไม่ผ่านระวังแรงกดซ้ำ ทะลุระวัง Squeeze"
            else:
                loss = futures_ref - f_cost
                cost_relation = f"ต่างชาติติดลบ Short (-{loss:.1f} จุด) เหนือแนวต้าน {f_cost:.1f} ระวัง Short Squeeze เร่งแรงดีด"
        elif cum_f > 0:
            if futures_ref >= f_cost:
                pnl = futures_ref - f_cost
                cost_relation = f"ต่างชาติกำไร Long (+{pnl:.1f} จุด) แนว {f_cost:.1f} เป็นแนวรับจิตวิทยา (Defense Line)"
            else:
                loss = f_cost - futures_ref
                cost_relation = f"ต่ำกว่าทุนฝรั่ง ({f_cost:.1f}) ต่างชาติขาดทุน Long (-{loss:.1f} จุด) ระวังแรงขายลดเสี่ยงเมื่อชนทุน"
        else:
            cost_relation = f"ต้นทุนเฉลี่ยฝรั่ง {f_cost:.1f} จุด (สะสมใกล้เคียงศูนย์)"

        cost_narrative = f"""• <b>แนวปะทะต้นทุนสถาบัน (Cost Line):</b> <code>{f_cost:.1f}</code> จุด
  <i>({cost_relation})</i>
"""
    else:
        cost_narrative = ""

    # Assemble Sections 3 and 4 cleanly
    sec4_lines = [
        "🌊 <b>4. พฤติกรรมรายใหญ่ (Smart Money Flow & Footprint)</b>",
        f"• <b>สถานะ Flow:</b> {flow_emoji} <b>{flow_title}</b>",
        f"  <i>({flow_detail})</i>",
        f"• <b>ยอดซื้อขายรายกลุ่ม:</b> ต่างชาติ <code>{foreign_net:+,}</code> (ΔOI: <code>{delta_oi:+,}</code>) | กองทุน <code>{inst_net:+,}</code> | รายย่อย <code>{local_net:+,}</code>"
    ]
    if foreign_footprint_block:
        sec4_lines.append(foreign_footprint_block)
    if options_footprint_block:
        sec4_lines.append(options_footprint_block)
    sec4_text = "\n".join(sec4_lines)

    body_blocks = []
    if intraday_block:
        body_blocks.append(intraday_block.strip())
    if sec4_text:
        body_blocks.append(sec4_text.strip())
    body_content = "\n\n".join(body_blocks)

    msg = f"""🎯 <b>แผนการเทรด SET50 ประจำวัน | TFEX QUANT RADAR</b>
📅 <i>ข้อมูลสิ้นวัน {trade_date} | ออกรายงานเช้า {as_of_time_bkk} น.</i>
━━━━━━━━━━━━━━━━━━━━━━
🎯 <b>1. กรอบวิ่งคาดการณ์ (1-Day Expected Move)</b>
• <b>กรอบราคา 1-SD:</b> {expected_move_text}
• <b>ระดับความผันผวน (ATM IV):</b> {atm_iv_text}

🧱 <b>2. แนวรับ - แนวต้านสำคัญ (Tactical Key Levels)</b>
🔴 <b>ต้าน 2:</b> <code>{res_2:.0f}</code> <i>({res_2_oi:,})</i> | <b>ต้าน 1:</b> <code>{res_1:.0f}</code> <i>({res_1_oi:,})</i>
📍 <b>ราคาปิดล่าสุด ({front_series}):</b> <code>{futures_ref:.1f}</code> จุด <i>(Max Pain: <code>{max_pain:.0f}</code>)</i>
🟢 <b>รับ 1:</b> <code>{sup_1:.0f}</code> <i>({sup_1_oi:,})</i> | <b>รับ 2:</b> <code>{sup_2:.0f}</code> <i>({sup_2_oi:,})</i>
───────────────
🏛️ <i>กรอบไตรมาส: Put Wall <code>{put_wall:.0f}</code> – Call Wall <code>{call_wall:.0f}</code> | Gamma Flip: <code>{gamma_flip_str}</code></i>

{body_content}

⚔️ <b>5. กลยุทธ์และแผนการเทรด (Daily Action Plan)</b>
• <b>สภาวะตลาด (Dealer Gamma):</b> {gamma_emoji} <b>{gamma_zone_title}</b>
  <i>({gamma_desc_compact})</i>
{cost_narrative}• <b>แผนการปฏิบัติหน้างาน:</b>
  💡 <b>กลยุทธ์หลัก:</b> {action_strategy}
  🟢 <b>ฝั่ง Long:</b> {long_plan}
  🔴 <b>ฝั่ง Short:</b> {short_plan}
  ⚠️ <b>จุดตัดขาดทุน (Stop Loss):</b> {stop_loss_plan}
━━━━━━━━━━━━━━━━━━━━━━
🌐 <b>เปิดดูกราฟ Interactive:</b> <a href="https://met37818-stack.github.io/tfex-quant-analytics/">คลิกเปิดใน Chrome ทันที</a>"""
    return msg

# ==============================================================================
# VISUALIZATION & CHART GENERATION (PLOTLY INTERACTIVE DASHBOARD)
# ==============================================================================

def build_interactive_dashboard(
    trade_date: str,
    futures_ref: float,
    front_series: str,
    next_series: str,
    res_df: pd.DataFrame,
    gamma_flip_cum: float,
    call_wall: float,
    put_wall: float,
    max_pain: float,
    iv_df_front: pd.DataFrame,
    iv_df_next: pd.DataFrame,
    foreign_regime: str,
    upper_1sd: float = None,
    lower_1sd: float = None,
    foreign_footprint: dict = None,
    vol_time_series: list = None
) -> str:
    """
    Generates a standalone, responsive, mobile-friendly HTML report with 5 interactive subplots.
    """
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        subplot_titles=(
            f"<b>1. Foreign Futures Flow & Running Cost Basis vs S50 Price ({front_series})</b>",
            "<b>2. Volatility Regime (ATM IV vs Valid Daily Realized Volatility)</b>",
            f"<b>3. Net Gamma Exposure (GEX) Profile by Strike — {front_series} (200 THB/pt)</b>",
            f"<b>4. Implied Volatility Smile & Skew Curve — Front ({front_series}) vs Next ({next_series})</b>",
            f"<b>5. Open Interest Distribution (Call vs Put) & Max Pain ({max_pain:.0f})</b>"
        ),
        row_heights=[0.23, 0.18, 0.21, 0.19, 0.19]
    )

    # ----------------------------------------------------
    # Subplot 1: Foreign Flow & Cost Basis vs S50 Price
    # ----------------------------------------------------
    f_series = foreign_footprint.get("series_data", []) if foreign_footprint else []
    if f_series:
        f_dates = [r.get("trade_date") or r.get("date") for r in f_series]
        f_px = [float(r.get("settlement_price", futures_ref)) for r in f_series]
        f_cost = [float(r.get("cost_basis", r.get("settlement_price", futures_ref))) for r in f_series]
        f_net = [int(r.get("foreign_net", 0)) for r in f_series]
    else:
        f_dates = [trade_date]
        f_px = [futures_ref]
        f_cost = [foreign_footprint.get("cost_basis", futures_ref) if foreign_footprint else futures_ref]
        f_net = [foreign_footprint.get("cum_foreign_net", 0) if foreign_footprint else 0]

    f_bar_colors = ["#00E676" if x >= 0 else "#FF1744" for x in f_net]

    fig.add_trace(
        go.Scatter(
            x=f_dates, y=f_px,
            mode="lines+markers", name=f"{front_series} S50 Price",
            line=dict(color="#00E5FF", width=2),
            marker=dict(size=4),
            hovertemplate="<b>%{x}</b><br>S50 Close: %{y:.1f} pts<extra></extra>"
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=f_dates, y=f_cost,
            mode="lines", name="Foreign Running Cost Basis",
            line=dict(color="#FFD600", width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Foreign Cost: %{y:.1f} pts<extra></extra>"
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Bar(
            x=f_dates, y=f_net,
            name="Daily Foreign Net",
            marker_color=f_bar_colors, opacity=0.4,
            hovertemplate="<b>%{x}</b><br>Foreign Net: %{y:+,} contracts<extra></extra>"
        ),
        row=1, col=1, secondary_y=True
    )

    # ----------------------------------------------------
    # Subplot 2: Volatility Regime Time-Series
    # ----------------------------------------------------
    if vol_time_series:
        v_dates = [v.get("trade_date") or v.get("date") for v in vol_time_series]
        v_iv = [v["atm_iv"] * 100.0 if v.get("atm_iv") is not None else None for v in vol_time_series]
        v_rv = [v["rv_annualized"] * 100.0 if v.get("rv_annualized") is not None else None for v in vol_time_series]

        fig.add_trace(
            go.Scatter(
                x=v_dates, y=v_iv,
                mode="lines+markers", name="ATM IV (%)",
                line=dict(color="#BA68C8", width=2),
                marker=dict(size=5),
                connectgaps=True,
                hovertemplate="<b>%{x}</b><br>ATM IV: %{y:.2f}%<extra></extra>"
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=v_dates, y=v_rv,
                mode="lines+markers", name="Daily RV (%) [≥310 bars]",
                line=dict(color="#00E676", width=2),
                marker=dict(size=6, symbol="diamond"),
                connectgaps=False,
                hovertemplate="<b>%{x}</b><br>Daily RV: %{y:.2f}%<extra></extra>"
            ),
            row=2, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[trade_date], y=[0.0],
                mode="markers", name="ATM IV (%)",
                marker=dict(size=5, color="#BA68C8")
            ),
            row=2, col=1
        )

    # ----------------------------------------------------
    # Subplot 3: Net GEX Profile
    # ----------------------------------------------------
    bar_colors = ["#00E676" if x >= 0 else "#FF1744" for x in res_df["net_gex_mb"]]
    fig.add_trace(
        go.Bar(
            x=res_df["strike"],
            y=res_df["net_gex_mb"],
            marker_color=bar_colors,
            name="Net GEX (MB/pt)",
            hovertemplate="<b>Strike %{x}</b><br>Net GEX: %{y:.2f} MB/pt<extra></extra>"
        ),
        row=3, col=1
    )

    # Key Level Lines for Row 3
    # Futures Ref line
    fig.add_vline(
        x=futures_ref, line_width=2, line_dash="dash", line_color="#00E5FF",
        annotation_text=f"F: {futures_ref:.1f}", annotation_position="top left",
        annotation_font=dict(color="#00E5FF", size=11),
        row=3, col=1
    )
    # Expected Move 1-SD Lines
    if upper_1sd is not None:
        fig.add_vline(
            x=upper_1sd, line_width=1.5, line_dash="dot", line_color="#B388FF",
            annotation_text=f"Upper 1-SD: {upper_1sd:.1f}", annotation_position="top right",
            annotation_font=dict(color="#B388FF", size=9),
            row=3, col=1
        )
    if lower_1sd is not None:
        fig.add_vline(
            x=lower_1sd, line_width=1.5, line_dash="dot", line_color="#B388FF",
            annotation_text=f"Lower 1-SD: {lower_1sd:.1f}", annotation_position="bottom left",
            annotation_font=dict(color="#B388FF", size=9),
            row=3, col=1
        )

    # Gamma Flip Line
    if not math.isnan(gamma_flip_cum):
        fig.add_vline(
            x=gamma_flip_cum, line_width=2.5, line_dash="dot", line_color="#FFD600",
            annotation_text=f"Flip: {gamma_flip_cum:.1f}", annotation_position="bottom right",
            annotation_font=dict(color="#FFD600", size=11),
            row=3, col=1
        )
    # Call Wall
    fig.add_vline(
        x=call_wall, line_width=1.5, line_dash="dashdot", line_color="#69F0AE",
        annotation_text=f"Call Wall: {call_wall:.0f}", annotation_position="top right",
        annotation_font=dict(color="#69F0AE", size=10),
        row=3, col=1
    )
    # Put Wall
    fig.add_vline(
        x=put_wall, line_width=1.5, line_dash="dashdot", line_color="#FF5252",
        annotation_text=f"Put Wall: {put_wall:.0f}", annotation_position="bottom left",
        annotation_font=dict(color="#FF5252", size=10),
        row=3, col=1
    )

    # ----------------------------------------------------
    # Subplot 4: IV Smile & Skew Curve
    # ----------------------------------------------------
    if not iv_df_front.empty:
        c_front = iv_df_front[(iv_df_front["option_type"] == "C") & (iv_df_front["iv"].notnull())].sort_values("strike")
        p_front = iv_df_front[(iv_df_front["option_type"] == "P") & (iv_df_front["iv"].notnull())].sort_values("strike")
        fig.add_trace(
            go.Scatter(
                x=c_front["strike"], y=c_front["iv"] * 100.0,
                mode="lines+markers", name=f"{front_series} Call IV (%)",
                line=dict(color="#00E5FF", width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{x} Call</b><br>IV: %{y:.2f}%<extra></extra>"
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=p_front["strike"], y=p_front["iv"] * 100.0,
                mode="lines+markers", name=f"{front_series} Put IV (%)",
                line=dict(color="#FF4081", width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{x} Put</b><br>IV: %{y:.2f}%<extra></extra>"
            ),
            row=4, col=1
        )

    if not iv_df_next.empty:
        c_next = iv_df_next[(iv_df_next["option_type"] == "C") & (iv_df_next["iv"].notnull())].sort_values("strike")
        fig.add_trace(
            go.Scatter(
                x=c_next["strike"], y=c_next["iv"] * 100.0,
                mode="lines", name=f"{next_series} Term Skew (%)",
                line=dict(color="#B388FF", width=1.5, dash="dot"),
                hovertemplate="<b>%{x} Next Q</b><br>IV: %{y:.2f}%<extra></extra>"
            ),
            row=4, col=1
        )

    fig.add_vline(
        x=futures_ref, line_width=1.5, line_dash="dash", line_color="#78909C",
        row=4, col=1
    )

    # ----------------------------------------------------
    # Subplot 5: Open Interest Distribution
    # ----------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=res_df["strike"], y=res_df["call_oi"],
            name="Call OI", marker_color="#29B6F6", opacity=0.85,
            hovertemplate="<b>Strike %{x} Call OI</b>: %{y:,}<extra></extra>"
        ),
        row=5, col=1
    )
    fig.add_trace(
        go.Bar(
            x=res_df["strike"], y=res_df["put_oi"],
            name="Put OI", marker_color="#FF7043", opacity=0.85,
            hovertemplate="<b>Strike %{x} Put OI</b>: %{y:,}<extra></extra>"
        ),
        row=5, col=1
    )

    fig.add_vline(
        x=max_pain, line_width=2, line_dash="dash", line_color="#FFD600",
        annotation_text=f"Max Pain: {max_pain:.0f}", annotation_position="top left",
        annotation_font=dict(color="#FFD600", size=10),
        row=5, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>SET50 PRE-MARKET QUANT RADAR</b> | As of EOD {trade_date} (Flow Regime: {foreign_regime})",
            font=dict(size=18, color="#FFFFFF"),
            x=0.03, y=0.99
        ),
        height=1750,
        dragmode=False,
        margin=dict(l=60, r=60, t=80, b=50),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="right",
            x=0.98,
            font=dict(size=11)
        ),
        paper_bgcolor="#121824",
        plot_bgcolor="#182232"
    )

    fig.update_xaxes(gridcolor="#26354A", fixedrange=True, row=1, col=1)
    fig.update_yaxes(title_text="Price (pts)", gridcolor="#26354A", fixedrange=True, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Foreign Net", showgrid=False, fixedrange=True, row=1, col=1, secondary_y=True)

    fig.update_xaxes(gridcolor="#26354A", fixedrange=True, row=2, col=1)
    fig.update_yaxes(title_text="Volatility (%)", gridcolor="#26354A", fixedrange=True, row=2, col=1)

    fig.update_xaxes(gridcolor="#26354A", fixedrange=True, row=3, col=1)
    fig.update_yaxes(title_text="Net GEX (MB/pt)", gridcolor="#26354A", fixedrange=True, row=3, col=1)

    fig.update_xaxes(gridcolor="#26354A", fixedrange=True, row=4, col=1)
    fig.update_yaxes(title_text="Implied Volatility (%)", gridcolor="#26354A", fixedrange=True, row=4, col=1)

    fig.update_xaxes(title_text="Strike Price", gridcolor="#26354A", fixedrange=True, row=5, col=1)
    fig.update_yaxes(title_text="Contracts (OI)", gridcolor="#26354A", fixedrange=True, row=5, col=1)

    plotly_config = {
        "displayModeBar": False,
        "scrollZoom": False,
        "responsive": True
    }

    raw_html = fig.to_html(include_plotlyjs=True, full_html=True, config=plotly_config)

    custom_injections = """
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <style>
        .js-plotly-plot, .plotly, .plot-container {
            pointer-events: none !important;
            touch-action: pan-x pan-y pinch-zoom !important;
        }
    </style>
"""

    if "</head>" in raw_html:
        final_html = raw_html.replace("</head>", f"{custom_injections}\n</head>", 1)
    else:
        final_html = custom_injections + "\n" + raw_html

    html_path = f"premarket_report_{trade_date}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    # Save copy to public/index.html for GitHub Pages deployment
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        public_dir = os.path.join(script_dir, "public")
        os.makedirs(public_dir, exist_ok=True)
        public_index_path = os.path.join(public_dir, "index.html")
        with open(public_index_path, "w", encoding="utf-8") as f:
            f.write(final_html)
        print(f"[PAGES] Dashboard copy successfully saved to {public_index_path}")
    except Exception as e:
        print(f"[PAGES] Warning: Failed to save dashboard to public/index.html: {e}")

    return html_path

# ==============================================================================
# CORE WORKFLOW ORCHESTRATOR
# ==============================================================================

def run_premarket_radar(force: bool = False, no_telegram: bool = False, test_mode: bool = False):
    """Main execution pipeline."""
    start_time = datetime.datetime.now(BKK_TZ)
    print("=" * 75)
    print("TFEX PRE-MARKET QUANT RADAR (PHASE 1) - THAI DAILY PLAYBOOK")
    print(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 75)

    # 1. Load Environment Variables
    load_dotenv()
    if not os.environ.get("TURSO_DATABASE_URL"):
        load_dotenv("../Pre-Trade-Github/.env")
        load_dotenv("../tfex-eod-collector/.env")

    turso_url = os.environ.get("TURSO_DATABASE_URL")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN")
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    # If test mode is requested, execute with realistic mock data
    if test_mode:
        print("[TEST] Running with mock SET50 quant dataset to verify Thai trading plan formatting...")
        t_0 = (datetime.datetime.now(BKK_TZ) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        futures_ref = 918.5
        front_series = "S50Z26"
        next_series = "S50H27"
        atm_iv = 0.1180
        expected_move_1d = futures_ref * (atm_iv / math.sqrt(252.0))
        upper_1sd = futures_ref + expected_move_1d
        lower_1sd = futures_ref - expected_move_1d

        # Mock strike data
        strikes = list(range(850, 1000, 10))
        mock_data = []
        for k in strikes:
            c_oi = 8200 if k == 930 else (4500 if k == 920 else 1200)
            p_oi = 9500 if k == 900 else (6000 if k == 910 else 1100)
            net_gex = 2.5 if k > 915 else -1.8
            mock_data.append({
                "strike": k, "call_oi": c_oi, "put_oi": p_oi, "iv": 0.118, "gamma": 0.005,
                "call_gex_mb": 3.0, "put_gex_mb": -1.5, "net_gex_mb": net_gex
            })
        res_df = pd.DataFrame(mock_data)
        call_wall = 960.0
        put_wall = 880.0
        max_pain = 920.0
        gamma_flip_val = 915.2
        is_positive_gamma = (futures_ref >= gamma_flip_val)
        gamma_zone_title = "Positive Gamma (โซนลดความผันผวน / Mean Reversion)" if is_positive_gamma else "Negative Gamma (โซนเร่งความผันผวน / Trend Expansion)"
        gamma_desc = (
            f"ราคาปัจจุบัน ({futures_ref:.1f}) อยู่เหนือ Gamma Flip ({gamma_flip_val:.1f}) ดีลเลอร์อยู่ในสถานะ Long Gamma (ซื้อแนวรับ-ขายแนวต้าน) ช่วยซับแรงเหวี่ยง ทำให้ตลาดมักแกว่งตัวในกรอบ"
            if is_positive_gamma else
            f"ราคาปัจจุบัน ({futures_ref:.1f}) อยู่ใต้ Gamma Flip ({gamma_flip_val:.1f}) ดีลเลอร์อยู่ในสถานะ Short Gamma หากหลุดแนวรับจะเร่งแรงขายซ้ำเติม"
        )
        total_net_gex = float(res_df["net_gex_mb"].sum())
        tactical_levels = resolve_tactical_levels(res_df, futures_ref, call_wall, put_wall)
        foreign_net = 3500
        delta_oi = -4200
        inst_net = -1200
        local_net = -2300
        flow_regime, flow_emoji, flow_desc = decompose_smart_money(foreign_net, delta_oi)

        # Mock Footprint Analytics for Phase 1
        mock_series_data = []
        base_date = datetime.date.today() - datetime.timedelta(days=15)
        for i in range(15):
            d_str = (base_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            mock_series_data.append({
                "trade_date": d_str,
                "foreign_net": 1200 if i % 2 == 0 else -800,
                "settlement_price": 910.0 + i * 0.6,
                "cost_basis": 912.0 + i * 0.2,
                "cum_net": 1000 + i * 400
            })

        mock_foreign_footprint = {
            "cum_foreign_net": 18450,
            "cost_basis": 914.8,
            "pnl_pts": 3.7,
            "pnl_status": "กำไร (+3.7 จุด)",
            "z_score": 1.68,
            "z_desc": "🚨 Outlier ซื้อหนัก (Z = +1.68 SD)",
            "is_outlier": True,
            "divergence_text": "สอดคล้องกับราคา (Confluent)",
            "history_days": 60,
            "series_data": mock_series_data
        }
        mock_options_footprint = {
            "foreign_call_net": 3250,
            "foreign_put_net": -1420,
            "top_call_strike": 930.0,
            "top_call_doi": 1850,
            "top_call_div": 0.85,
            "call_intent": "Aggressive Call Buying (เก็งกำไรทะลุต้าน)",
            "top_put_strike": 900.0,
            "top_put_doi": 2400,
            "top_put_div": -0.35,
            "put_intent": "Put Writing / Floor Support (ขาย Put ล็อกฐานรับ)"
        }

        # Mock Intraday Warm-up Analytics for Phase 2
        mock_vol_time_series = []
        for i in range(15):
            d_str = (base_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            mock_vol_time_series.append({
                "date": d_str,
                "atm_iv": 0.115 + (i % 3) * 0.003,
                "rv_annualized": 0.100 + (i % 4) * 0.004 if i % 2 == 0 else None
            })

        mock_intraday_rv = {
            "rv_annualized": 0.1045,
            "bar_count": 320,
            "is_valid_rv": True,
            "iv_rv_spread": (atm_iv - 0.1045) * 100.0,
            "vol_regime_desc": f"IV Premium สูง (Options แพงกว่าแกว่งจริง +{(atm_iv - 0.1045) * 100.0:.1f}% เหมาะเทรดในกรอบ)"
        }
        mock_volume_profile = {
            "poc": 917.0,
            "vah": 922.0,
            "val": 912.0,
            "warmup_days": 20,
            "valid_days": 20,
            "confluence_text": f"POC (917) รับราคาปิด | VAH (922) ชน Max Pain ({max_pain:.0f}) ➔ ต้านคอนกรีต"
        }
        mock_late_session = {
            "late_fut_delta": 1.2,
            "late_basis_close": -1.85,
            "late_basis_delta": 0.45,
            "late_session_desc": "🟢 ต่างชาติซื้อ + ท้ายตลาดไล่ราคา Basis ขยับขึ้น ปิดใกล้ High มีโอกาสเปิดบวกต่อ"
        }
        mock_greeks_edge = {
            "skew_25d": 1.60,
            "c_iv_pct": 12.4,
            "p_iv_pct": 14.0,
            "skew_desc": "ความกลัว-โลภสองฝั่งสมดุล (Normal)",
            "atm_strike": 920.0,
            "atm_gamma": 0.0035,
            "atm_theta": 0.076,
            "theta_gamma_ratio": 21.7,
            "daily_breakeven_move": 6.6,
            "dealer_vanna": -195.0,
            "dealer_charm": 27.0,
            "vanna_bias_desc": "หาก IV Crush ดีลเลอร์จะขาย Futures กดดันตลาด",
            "records_count": 78
        }

        html_file = build_interactive_dashboard(
            trade_date=t_0, futures_ref=futures_ref, front_series=front_series, next_series=next_series,
            res_df=res_df, gamma_flip_cum=gamma_flip_val, call_wall=call_wall, put_wall=put_wall,
            max_pain=max_pain, iv_df_front=pd.DataFrame(), iv_df_next=pd.DataFrame(),
            foreign_regime=flow_regime, upper_1sd=upper_1sd, lower_1sd=lower_1sd,
            foreign_footprint=mock_foreign_footprint, vol_time_series=mock_vol_time_series
        )

        now_bkk = datetime.datetime.now(BKK_TZ)
        telegram_text = generate_thai_trading_playbook(
            trade_date=t_0, as_of_time_bkk=now_bkk.strftime("%H:%M"),
            futures_ref=futures_ref, front_series=front_series, atm_iv=atm_iv,
            expected_move_1d=expected_move_1d, upper_1sd=upper_1sd, lower_1sd=lower_1sd,
            tactical_levels=tactical_levels, max_pain=max_pain, call_wall=call_wall, put_wall=put_wall,
            gamma_flip_val=gamma_flip_val, is_positive_gamma=is_positive_gamma,
            gamma_zone_title=gamma_zone_title, gamma_desc=gamma_desc, total_net_gex=total_net_gex,
            foreign_net=foreign_net, delta_oi=delta_oi, inst_net=inst_net, local_net=local_net,
            flow_regime=flow_regime, flow_emoji=flow_emoji,
            foreign_footprint=mock_foreign_footprint,
            options_footprint=mock_options_footprint,
            intraday_rv=mock_intraday_rv,
            volume_profile=mock_volume_profile,
            late_session=mock_late_session,
            greeks_edge=mock_greeks_edge
        )

        print("\n" + "=" * 75)
        print("GENERATED THAI TRADING PLAYBOOK (TELEGRAM PREVIEW)")
        print("=" * 75)
        print(telegram_text)
        print("=" * 75)
        print(f"Message Length: {len(telegram_text)} characters (Limit: 4096)")

        if tg_token and tg_chat_id and not no_telegram:
            print(f"[TELEGRAM] Dispatching test notification to Chat ID {tg_chat_id}...")
            send_telegram_alert(tg_token, tg_chat_id, telegram_text)
        else:
            print("[TELEGRAM] Telegram credentials not configured or --no-telegram enabled. Test output displayed locally.")

        # S5: --test ต้องเป็น smoke test จริง ไม่ใช่แค่พิมพ์ demo
        # ตรวจว่าผลลัพธ์ที่ประกอบขึ้นมีส่วนสำคัญครบ ถ้าไม่ครบให้ fail (exit 1)
        # เพื่อให้ job ใน CI จับ regression ของ formatter ได้
        required_sections = ["Gamma Flip", "Max Pain", "Stop Loss"]
        missing = [sec for sec in required_sections if sec.lower() not in telegram_text.lower()]
        if missing:
            print(f"[TEST] FAILED: telegram output missing sections: {missing}")
            sys.exit(1)
        if not os.path.exists(html_file):
            print(f"[TEST] FAILED: dashboard file was not written ({html_file})")
            sys.exit(1)
        print(f"[TEST] OK: playbook sections present and dashboard written ({os.path.getsize(html_file):,} bytes)")

        if os.path.exists(html_file):
            try:
                os.remove(html_file)
            except Exception:
                pass
        return

    if not turso_url or not turso_token:
        print("[ERROR] TURSO_DATABASE_URL or TURSO_AUTH_TOKEN not configured.")
        print("Tip: Use --test to run with mock data and verify the Thai trading plan format.")
        sys.exit(1)

    pipeline_url = normalize_turso_url(turso_url)

    # 2. Initialize Analytics Tables
    init_analytics_tables(pipeline_url, turso_token)

    # 3. Data-Driven Holiday & Duplicate Guard
    print("[GUARD] Checking latest trading date in Turso database...")
    _, max_date_rows = execute_turso_sql(pipeline_url, turso_token, "SELECT MAX(trade_date) FROM eod_futures_series;")
    if not max_date_rows or not max_date_rows[0][0]:
        print("[ERROR] No data found in eod_futures_series table.")
        sys.exit(1)

    latest_trade_date = max_date_rows[0][0]
    print(f"[GUARD] Latest recorded EOD session: {latest_trade_date}")

    _, log_rows = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT sent_at_bkk FROM analytics_premarket_log WHERE trade_date = ?;",
        [latest_trade_date]
    )

    if log_rows and not force:
        sent_time = log_rows[0][0]
        print(f"[GUARD] Trade date {latest_trade_date} was already processed and sent at {sent_time}.")
        print("[GUARD] Halting execution to prevent duplicate notifications during holidays/weekends. (Use --force to override).")
        return

    # 4. Fetch EOD Data for 2 Latest Trading Sessions
    _, date_rows = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT DISTINCT trade_date FROM eod_futures_series WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 2;",
        [latest_trade_date]
    )
    t_0 = date_rows[0][0]
    t_minus_1 = date_rows[1][0] if len(date_rows) > 1 else t_0
    print(f"[DATA] Target Session: T_0 = {t_0} | Benchmark Previous: T_-1 = {t_minus_1}")

    # 4.1 Futures Data
    cols_f, rows_f = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT series, settlement_price, volume, oi FROM eod_futures_series WHERE trade_date = ?;",
        [t_0]
    )
    futures_dict = {r[0]: {"settlement": float(r[1]), "vol": int(r[2] or 0), "oi": int(r[3] or 0)} for r in rows_f}

    # 4.2 Determine Front & Next Active Quarter Series
    quarterly_series = [s for s in futures_dict.keys() if len(s) == 6 and s[3] in ("H", "M", "U", "Z")]
    quarterly_series.sort(key=lambda s: futures_dict[s]["oi"], reverse=True)
    if not quarterly_series:
        print("[ERROR] No quarterly SET50 futures series found.")
        sys.exit(1)

    front_series = quarterly_series[0]
    next_series = quarterly_series[1] if len(quarterly_series) > 1 else quarterly_series[0]
    futures_ref = futures_dict[front_series]["settlement"]
    print(f"[STRUCTURE] Front Quarter Series: {front_series} (Settlement: {futures_ref:.1f}, OI: {futures_dict[front_series]['oi']:,})")
    print(f"[STRUCTURE] Next Quarter Series : {next_series} (Settlement: {futures_dict[next_series]['settlement']:.1f}, OI: {futures_dict[next_series]['oi']:,})")

    # 4.3 THOR Interest Rate (All Tenors for Dynamic Matching)
    _, thor_rows = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT thor_rate, thor_1m_avg, thor_3m_avg, thor_6m_avg FROM eod_thor_rates WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 1;",
        [t_0]
    )
    thor_dict = {"overnight": None, "1m": None, "3m": None, "6m": None}
    if thor_rows and thor_rows[0]:
        r_th = thor_rows[0]
        thor_dict["overnight"] = r_th[0] if len(r_th) > 0 else None
        thor_dict["1m"] = r_th[1] if len(r_th) > 1 else None
        thor_dict["3m"] = r_th[2] if len(r_th) > 2 else None
        thor_dict["6m"] = r_th[3] if len(r_th) > 3 else None

    # 4.4 Expiry Calculations (Penultimate Business Day)
    t_0_dt = datetime.datetime.strptime(t_0, "%Y-%m-%d").date()
    expiry_front = get_series_expiry_date(front_series, t_0_dt)
    expiry_next = get_series_expiry_date(next_series, t_0_dt)

    days_to_exp_front = max(0.5, (expiry_front - t_0_dt).days)
    days_to_exp_next = max(0.5, (expiry_next - t_0_dt).days)
    T_front = days_to_exp_front / 365.0
    T_next = days_to_exp_next / 365.0
    print(f"[EXPIRY] {front_series} Expiry: {expiry_front} ({days_to_exp_front:.0f} calendar days, T = {T_front:.4f} yrs)")
    print(f"[EXPIRY] {next_series} Expiry: {expiry_next} ({days_to_exp_next:.0f} calendar days, T = {T_next:.4f} yrs)")

    # Dynamic Tenor Selection for Front & Next series
    r_rf_front, tenor_front = select_dynamic_thor_rate(thor_dict, int(days_to_exp_front))
    r_rf_next, tenor_next = select_dynamic_thor_rate(thor_dict, int(days_to_exp_next))
    r_rf = r_rf_front
    thor_rate = r_rf_front
    print(f"[STRUCTURE] Front Risk-free Rate (THOR {tenor_front}): {r_rf_front*100.0:.4f}% (r = {r_rf_front:.6f})")
    print(f"[STRUCTURE] Next Risk-free Rate (THOR {tenor_next}): {r_rf_next*100.0:.4f}% (r = {r_rf_next:.6f})")

    # 4.5 Options Series Data
    cols_o, rows_o = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT symbol, series, option_type, strike, settlement_price, oi, volume FROM eod_options_series WHERE trade_date = ?;",
        [t_0]
    )
    opts_df = pd.DataFrame(rows_o, columns=cols_o)
    print(f"[DATA] Total Options Series Loaded: {len(opts_df)}")

    # 5. Black-76 IV and Greeks Calculations
    # Front Quarter
    front_opts = opts_df[opts_df["series"] == front_series].copy()
    front_iv_list = []
    for _, opt in front_opts.iterrows():
        p = float(opt["settlement_price"]) if opt["settlement_price"] is not None else 0.0
        k = float(opt["strike"])
        otype = str(opt["option_type"]).upper()
        iv = calculate_implied_vol(p, futures_ref, k, T_front, r_rf_front, otype)
        front_iv_list.append(iv)
    front_opts["iv"] = front_iv_list

    # Next Quarter
    next_opts = opts_df[opts_df["series"] == next_series].copy()
    next_ref = futures_dict[next_series]["settlement"]
    next_iv_list = []
    for _, opt in next_opts.iterrows():
        p = float(opt["settlement_price"]) if opt["settlement_price"] is not None else 0.0
        k = float(opt["strike"])
        otype = str(opt["option_type"]).upper()
        iv = calculate_implied_vol(p, next_ref, k, T_next, r_rf_next, otype)
        next_iv_list.append(iv)
    next_opts["iv"] = next_iv_list

    # Calculate ATM IV (Closest strike to futures_ref)
    strikes_front = sorted(front_opts["strike"].unique())
    atm_strike = min(strikes_front, key=lambda k: abs(k - futures_ref))
    atm_records = front_opts[(front_opts["strike"] == atm_strike) & (front_opts["iv"].notnull())]
    if not atm_records.empty:
        atm_iv = float(atm_records["iv"].mean())
    else:
        valid_ivs = front_opts[front_opts["iv"].notnull()]["iv"]
        atm_iv = float(valid_ivs.mean()) if not valid_ivs.empty else None

    # 5.1 Calculate 1-Day Expected Move (1-SD)
    if atm_iv is not None and atm_iv > 0:
        expected_move_1d = futures_ref * (atm_iv / math.sqrt(252.0))
        upper_1sd = futures_ref + expected_move_1d
        lower_1sd = futures_ref - expected_move_1d
    else:
        expected_move_1d = None
        upper_1sd = None
        lower_1sd = None

    # 6. Strike-Level GEX Profile, Walls, and Max Pain
    gex_records = []
    skipped_gex_strikes = 0
    for k in strikes_front:
        c_item = front_opts[(front_opts["strike"] == k) & (front_opts["option_type"] == "C")]
        p_item = front_opts[(front_opts["strike"] == k) & (front_opts["option_type"] == "P")]

        c_oi = int(c_item["oi"].values[0]) if len(c_item) and c_item["oi"].values[0] else 0
        p_oi = int(p_item["oi"].values[0]) if len(p_item) and p_item["oi"].values[0] else 0

        c_iv = c_item["iv"].values[0] if len(c_item) else None
        p_iv = p_item["iv"].values[0] if len(p_item) else None

        valid_k_ivs = [x for x in [c_iv, p_iv] if x is not None and not np.isnan(x)]
        if not valid_k_ivs:
            skipped_gex_strikes += 1
            continue
        sigma_k = float(np.mean(valid_k_ivs))

        greeks = calculate_black76_greeks(futures_ref, k, T_front, r_rf, sigma_k, "C")
        gamma_k = greeks["gamma"]

        # GEX (Million THB per 1-point move)
        call_gex = gamma_k * c_oi * CONTRACT_MULTIPLIER * futures_ref / 1e6
        put_gex = -gamma_k * p_oi * CONTRACT_MULTIPLIER * futures_ref / 1e6
        net_gex = call_gex + put_gex

        gex_records.append({
            "strike": k,
            "call_oi": c_oi,
            "put_oi": p_oi,
            "iv": sigma_k,
            "gamma": gamma_k,
            "call_gex_mb": call_gex,
            "put_gex_mb": put_gex,
            "net_gex_mb": net_gex
        })

    res_df = pd.DataFrame(gex_records)
    res_df["cum_gex"] = res_df["net_gex_mb"].cumsum()

    # Structural Metrics
    call_wall = float(res_df.loc[res_df["call_oi"].idxmax()]["strike"])
    put_wall = float(res_df.loc[res_df["put_oi"].idxmax()]["strike"])

    # Max Pain Strike
    pain_payouts = []
    for s_eval in strikes_front:
        total_payout = 0.0
        for _, r_k in res_df.iterrows():
            total_payout += max(0.0, s_eval - r_k["strike"]) * r_k["call_oi"] * CONTRACT_MULTIPLIER
            total_payout += max(0.0, r_k["strike"] - s_eval) * r_k["put_oi"] * CONTRACT_MULTIPLIER
        pain_payouts.append((s_eval, total_payout))
    max_pain = float(min(pain_payouts, key=lambda x: x[1])[0])

    # True Gamma Flip Calculation: Search for S where Total_Net_GEX(S) = sum_k(GEX_k(S)) = 0
    # over active grid S in [futures_ref - 75.0, futures_ref + 75.0] in 1.0 pt increments
    s_grid = np.arange(futures_ref - 75.0, futures_ref + 75.0 + 0.5, 1.0)
    net_gex_curve = []
    disc_front = math.exp(-r_rf * T_front)
    sqrt_T_front = math.sqrt(T_front)

    for s_eval in s_grid:
        total_gex_s = 0.0
        for _, r_k in res_df.iterrows():
            k_val = r_k["strike"]
            c_oi = r_k["call_oi"]
            p_oi = r_k["put_oi"]
            sig_k = r_k["iv"]
            if sig_k <= 0 or s_eval <= 0 or k_val <= 0:
                continue
            d1_s = (math.log(s_eval / k_val) + 0.5 * sig_k * sig_k * T_front) / (sig_k * sqrt_T_front)
            pdf_s = norm.pdf(d1_s)
            gamma_s = (disc_front * pdf_s) / (s_eval * sig_k * sqrt_T_front)
            # Net GEX(S) in Million THB: gamma_s * (c_oi - p_oi) * CONTRACT_MULTIPLIER * s_eval / 1e6
            gex_k_s = gamma_s * (c_oi - p_oi) * CONTRACT_MULTIPLIER * s_eval / 1e6
            total_gex_s += gex_k_s
        net_gex_curve.append((s_eval, total_gex_s))

    flip_candidates = []
    for i in range(len(net_gex_curve) - 1):
        s1, g1 = net_gex_curve[i]
        s2, g2 = net_gex_curve[i + 1]
        if (g1 <= 0.0 and g2 >= 0.0) or (g1 >= 0.0 and g2 <= 0.0):
            if abs(g2 - g1) > 1e-9:
                s_zero = s1 - g1 * (s2 - s1) / (g2 - g1)
                flip_candidates.append(float(s_zero))
            else:
                flip_candidates.append(float(s1))

    if flip_candidates:
        # Pick zero-crossing closest to futures_ref
        gamma_flip_val = float(min(flip_candidates, key=lambda x: abs(x - futures_ref)))
    else:
        # Fallback to max_pain if no zero-crossing in active range
        gamma_flip_val = float(max_pain)

    total_net_gex = float(res_df["net_gex_mb"].sum())

    # Consistent Gamma Regime: Aligned with current price position relative to Gamma Flip
    if not math.isnan(gamma_flip_val) and abs(futures_ref - gamma_flip_val) <= 3.0:
        is_positive_gamma = (futures_ref >= gamma_flip_val)
        gamma_zone_title = "Neutral / Transitional Gamma (ใกล้จุดเปลี่ยนโซน)"
        gamma_desc = f"ราคาประชิด Gamma Flip ({gamma_flip_val:.1f}) เป็นจุดหมุนความผันผวน ตลาดอยู่ในช่วงเปลี่ยนผ่าน เหมาะเทรดในกรอบ (Mean Reversion)"
    elif futures_ref >= gamma_flip_val:
        is_positive_gamma = True
        gamma_zone_title = "Positive Gamma (โซนลดความผันผวน / Mean Reversion)"
        gamma_desc = f"ราคาปัจจุบัน ({futures_ref:.1f}) อยู่เหนือ Gamma Flip ({gamma_flip_val:.1f}) ดีลเลอร์อยู่ในสถานะ Long Gamma (ซื้อแนวรับ-ขายแนวต้าน) ช่วยซับแรงเหวี่ยง ทำให้ตลาดมักแกว่งตัวในกรอบ"
    else:
        is_positive_gamma = False
        gamma_zone_title = "Negative Gamma (โซนเร่งความผันผวน / Trend Expansion)"
        gamma_desc = f"ราคาปัจจุบัน ({futures_ref:.1f}) อยู่ใต้ Gamma Flip ({gamma_flip_val:.1f}) ดีลเลอร์อยู่ในสถานะ Short Gamma (ต้องเปิด Hedging ตามแนวโน้ม) หากหลุดแนวรับจะเร่งแรงขายซ้ำเติม ทำให้เกิดความผันผวนสูง"

    gamma_regime = "Positive Gamma" if is_positive_gamma else "Negative Gamma"

    # 6.1 Resolve Nearest Tactical Levels (Spot ± 25)
    tactical_levels = resolve_tactical_levels(res_df, futures_ref, call_wall, put_wall)

    if atm_iv is not None:
        print(f"[QUANT] ATM IV            : {atm_iv * 100.0:.2f}% (ATM Strike: {atm_strike:.0f})")
    else:
        print(f"[QUANT] ATM IV            : [UNAVAILABLE] (ATM Strike: {atm_strike:.0f})")
    if expected_move_1d is not None:
        print(f"[QUANT] 1-Day Expected Move: ±{expected_move_1d:.1f} pts (Range: {lower_1sd:.1f} - {upper_1sd:.1f})")
    else:
        print(f"[QUANT] 1-Day Expected Move: [UNAVAILABLE]")
    print(f"[QUANT] Tactical Levels   : Sup2={tactical_levels['sup_2']:.0f}, Sup1={tactical_levels['sup_1']:.0f} | Res1={tactical_levels['res_1']:.0f}, Res2={tactical_levels['res_2']:.0f}")
    print(f"[QUANT] Call Wall         : {call_wall:.0f} (OI: {int(res_df.loc[res_df['strike'] == call_wall, 'call_oi'].values[0]):,})")
    print(f"[QUANT] Put Wall          : {put_wall:.0f} (OI: {int(res_df.loc[res_df['strike'] == put_wall, 'put_oi'].values[0]):,})")
    print(f"[QUANT] Max Pain          : {max_pain:.0f}")
    print(f"[QUANT] Gamma Flip        : {gamma_flip_val:.1f} | Regime: {gamma_zone_title}")
    print(f"[QUANT] Total Net GEX     : {total_net_gex:+.2f} MB/pt")

    # 7. Smart Money Flow Decomposition
    _, inv_rows = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT category, foreign_net, inst_net, local_net FROM eod_investor_types WHERE trade_date = ? AND category = 'Equity Index Futures';",
        [t_0]
    )
    if inv_rows:
        foreign_net = int(inv_rows[0][1] or 0)
        inst_net = int(inv_rows[0][2] or 0)
        local_net = int(inv_rows[0][3] or 0)
    else:
        foreign_net, inst_net, local_net = 0, 0, 0

    _, mkt_oi_0 = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT oi, volume FROM eod_market_stats WHERE trade_date = ? AND instrument_id = 'SET50_FC';",
        [t_0]
    )
    _, mkt_oi_prev = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT oi, volume FROM eod_market_stats WHERE trade_date = ? AND instrument_id = 'SET50_FC';",
        [t_minus_1]
    )

    if mkt_oi_0 and mkt_oi_prev:
        tot_oi_0 = int(mkt_oi_0[0][0] or 0)
        tot_oi_prev = int(mkt_oi_prev[0][0] or 0)
        delta_oi = tot_oi_0 - tot_oi_prev
        tot_vol_0 = int(mkt_oi_0[0][1] or 0)
    else:
        _, sum_oi_0 = execute_turso_sql(pipeline_url, turso_token, "SELECT SUM(oi) FROM eod_futures_series WHERE trade_date = ?;", [t_0])
        _, sum_oi_prev = execute_turso_sql(pipeline_url, turso_token, "SELECT SUM(oi) FROM eod_futures_series WHERE trade_date = ?;", [t_minus_1])
        tot_oi_0 = int(sum_oi_0[0][0] or 0)
        tot_oi_prev = int(sum_oi_prev[0][0] or 0)
        delta_oi = tot_oi_0 - tot_oi_prev
        tot_vol_0 = 0

    flow_regime, flow_emoji, flow_desc = decompose_smart_money(foreign_net, delta_oi)
    print(f"[FLOW] Foreign Net Flow : {foreign_net:+,} contracts")
    print(f"[FLOW] Total Futures OI : {tot_oi_0:,} (Change: {delta_oi:+,} contracts)")
    print(f"[FLOW] Smart Money State: {flow_emoji} {flow_regime} ({flow_desc})")

    # 7.1 Deep Footprint Analytics (Foreign Cost Basis & Options Strike Footprint)
    # Fetch T_-1 settlement for front_series
    _, rows_f_prev = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT settlement_price FROM eod_futures_series WHERE trade_date = ? AND series = ?;",
        [t_minus_1, front_series]
    )
    futures_ref_tprev = float(rows_f_prev[0][0]) if rows_f_prev and rows_f_prev[0][0] is not None else futures_ref
    t_minus_1_dt = datetime.datetime.strptime(t_minus_1, "%Y-%m-%d").date()
    days_to_exp_tprev = max(0.5, (expiry_front - t_minus_1_dt).days)
    T_front_tprev = days_to_exp_tprev / 365.0

    print("[FOOTPRINT] Analyzing Foreign 60-day cumulative flow and inventory cost basis...")
    foreign_footprint = analyze_foreign_footprint(
        pipeline_url=pipeline_url,
        auth_token=turso_token,
        target_date=t_0,
        front_series=front_series,
        futures_ref=futures_ref,
        lookback_days=60
    )
    print(f"[FOOTPRINT] Foreign 60D Cum Net: {foreign_footprint['cum_foreign_net']:+,} contracts | Cost Basis: {foreign_footprint['cost_basis']:.1f} ({foreign_footprint['pnl_status']})")
    print(f"[FOOTPRINT] Foreign Flow Z-Score: {foreign_footprint['z_score']:+.2f} ({foreign_footprint['z_desc']})")
    print(f"[FOOTPRINT] Price-Flow Divergence: {foreign_footprint['divergence_text']}")

    print("[FOOTPRINT] Analyzing Options Strike Footprint (ΔOI & ΔIV)...")
    options_footprint = analyze_options_footprint(
        pipeline_url=pipeline_url,
        auth_token=turso_token,
        t_0=t_0,
        t_minus_1=t_minus_1,
        front_series=front_series,
        futures_ref_t0=futures_ref,
        futures_ref_tprev=futures_ref_tprev,
        T_front_t0=T_front,
        T_front_tprev=T_front_tprev,
        r_rf=r_rf
    )
    print(f"[FOOTPRINT] Options Foreign Net: Call {options_footprint['foreign_call_net']:+,} | Put {options_footprint['foreign_put_net']:+,}")
    if options_footprint.get("top_call_strike") is not None:
        print(f"[FOOTPRINT] Top Call Expansion: Strike {options_footprint['top_call_strike']:.0f} (ΔOI: {options_footprint['top_call_doi']:+,}, ΔIV: {options_footprint['top_call_div']:+.2f}%) -> {options_footprint['call_intent']}")
    else:
        print(f"[FOOTPRINT] Top Call Expansion: {options_footprint['call_intent']}")

    if options_footprint.get("top_put_strike") is not None:
        print(f"[FOOTPRINT] Top Put Expansion : Strike {options_footprint['top_put_strike']:.0f} (ΔOI: {options_footprint['top_put_doi']:+,}, ΔIV: {options_footprint['top_put_div']:+.2f}%) -> {options_footprint['put_intent']}")
    else:
        print(f"[FOOTPRINT] Top Put Expansion : {options_footprint['put_intent']}")

    # 7.2 Intraday Warm-up Analytics (Realized Volatility, Volume Profile, Late-Session Flow)
    print("[INTRADAY] Computing Annualized Realized Volatility (RV) from 1-min bars...")
    intraday_rv = compute_intraday_rv(
        pipeline_url=pipeline_url,
        auth_token=turso_token,
        trade_date=t_0,
        front_series=front_series,
        atm_iv=atm_iv
    )
    rv_disp = f"{intraday_rv['rv_annualized']*100:.2f}%" if intraday_rv.get('rv_annualized') is not None else "[INSUFFICIENT DATA]"
    iv_disp = f"{atm_iv*100:.2f}%" if atm_iv is not None else "[UNAVAILABLE]"
    spread_disp = f"{intraday_rv['iv_rv_spread']:+.2f}%" if intraday_rv.get('iv_rv_spread') is not None else "N/A"
    print(f"[INTRADAY] Realized Vol: {rv_disp} (Bars: {intraday_rv['bar_count']}) vs ATM IV: {iv_disp} | Spread: {spread_disp}")

    vol_time_series = fetch_volatility_time_series(
        pipeline_url=pipeline_url,
        auth_token=turso_token,
        current_date=t_0,
        current_atm_iv=atm_iv,
        current_rv=intraday_rv["rv_annualized"],
        is_valid_rv=intraday_rv["is_valid_rv"],
        lookback_days=60
    )

    print("[INTRADAY] Computing 20-Day Rolling Volume Profile (POC, VAH, VAL)...")
    volume_profile = compute_volume_profile(
        pipeline_url=pipeline_url,
        auth_token=turso_token,
        target_date=t_0,
        front_series=front_series,
        tactical_levels=tactical_levels,
        call_wall=call_wall,
        put_wall=put_wall,
        max_pain=max_pain,
        lookback_days=20
    )
    print(f"[INTRADAY] Volume Profile ({volume_profile['warmup_days']}/20 days): POC={volume_profile['poc']:.0f}, VAH={volume_profile['vah']:.0f}, VAL={volume_profile['val']:.0f}")
    print(f"[INTRADAY] Confluence: {volume_profile['confluence_text']}")

    print("[INTRADAY] Analyzing Late-Session (16:30 - 16:55) Futures & Basis Flow...")
    late_session = analyze_late_session_flow(
        pipeline_url=pipeline_url,
        auth_token=turso_token,
        trade_date=t_0,
        front_series=front_series,
        foreign_net=foreign_net
    )
    print(f"[INTRADAY] Late Session: ΔFutures={late_session['late_fut_delta']:+.1f}, Basis Close={late_session['late_basis_close']:+.2f} (ΔBasis={late_session['late_basis_delta']:+.2f})")
    print(f"[INTRADAY] Late Session Signal: {late_session['late_session_desc']}")

    # 8. Interactive HTML Dashboard Generation
    html_file = build_interactive_dashboard(
        trade_date=t_0,
        futures_ref=futures_ref,
        front_series=front_series,
        next_series=next_series,
        res_df=res_df,
        gamma_flip_cum=gamma_flip_val,
        call_wall=call_wall,
        put_wall=put_wall,
        max_pain=max_pain,
        iv_df_front=front_opts,
        iv_df_next=next_opts,
        foreign_regime=flow_regime,
        upper_1sd=upper_1sd,
        lower_1sd=lower_1sd,
        foreign_footprint=foreign_footprint,
        vol_time_series=vol_time_series
    )
    print(f"[CHART] Interactive Plotly report generated: {html_file}")

    # 8.1 Options Greeks & Skew Edge Analysis (Dealer Vanna/Charm, 25-Delta Skew, Breakeven Move)
    print("[GREEKS EDGE] Analyzing Options Greeks, 25-Delta Skew, and Dealer Exposure...")
    greeks_edge = analyze_options_greeks_edge(
        pipeline_url=pipeline_url,
        auth_token=turso_token,
        trade_date=t_0,
        front_series=front_series,
        futures_ref=futures_ref,
        thor_rate=thor_rate,
        T_front=T_front,
        front_opts_df=front_opts
    )
    _skew_dbg = greeks_edge.get("skew_25d")
    _skew_txt = f"{_skew_dbg:+.2f}%" if _skew_dbg is not None else "N/A"
    print(f"[GREEKS EDGE] Skew 25Δ: {_skew_txt} | ATM Breakeven: {greeks_edge['daily_breakeven_move']:.1f} pts | Dealer Vanna: {greeks_edge['dealer_vanna']:+,.0f}")

    # 9. Format Telegram Notification: Actionable Thai Trading Playbook
    now_bkk = datetime.datetime.now(BKK_TZ)
    as_of_time_bkk = now_bkk.strftime("%H:%M")

    telegram_text = generate_thai_trading_playbook(
        trade_date=t_0,
        as_of_time_bkk=as_of_time_bkk,
        futures_ref=futures_ref,
        front_series=front_series,
        atm_iv=atm_iv,
        expected_move_1d=expected_move_1d,
        upper_1sd=upper_1sd,
        lower_1sd=lower_1sd,
        tactical_levels=tactical_levels,
        max_pain=max_pain,
        call_wall=call_wall,
        put_wall=put_wall,
        gamma_flip_val=gamma_flip_val,
        is_positive_gamma=is_positive_gamma,
        gamma_zone_title=gamma_zone_title,
        gamma_desc=gamma_desc,
        total_net_gex=total_net_gex,
        foreign_net=foreign_net,
        delta_oi=delta_oi,
        inst_net=inst_net,
        local_net=local_net,
        flow_regime=flow_regime,
        flow_emoji=flow_emoji,
        foreign_footprint=foreign_footprint,
        options_footprint=options_footprint,
        intraday_rv=intraday_rv,
        volume_profile=volume_profile,
        late_session=late_session,
        greeks_edge=greeks_edge
    )

    # 10. Persist Log to Turso Database
    print("[DATABASE] Logging pre-market execution record to analytics_premarket_log...")
    upsert_sql = """
    INSERT INTO analytics_premarket_log (
        trade_date, futures_ref, atm_iv, gamma_flip, call_wall, put_wall, max_pain, foreign_regime, sent_at_bkk
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(trade_date) DO UPDATE SET
        futures_ref = excluded.futures_ref,
        atm_iv = excluded.atm_iv,
        gamma_flip = excluded.gamma_flip,
        call_wall = excluded.call_wall,
        put_wall = excluded.put_wall,
        max_pain = excluded.max_pain,
        foreign_regime = excluded.foreign_regime,
        sent_at_bkk = excluded.sent_at_bkk;
    """
    execute_turso_sql(
        pipeline_url, turso_token, upsert_sql,
        [
            t_0,
            futures_ref,
            atm_iv,
            gamma_flip_val if not math.isnan(gamma_flip_val) else None,
            call_wall,
            put_wall,
            max_pain,
            flow_regime,
            now_bkk.strftime("%Y-%m-%d %H:%M:%S")
        ]
    )
    print(f"[DATABASE] Execution record successfully committed for trade_date = {t_0}.")

    # 11. Dispatch to Telegram
    if no_telegram:
        print("[TELEGRAM] --no-telegram flag supplied. Skipping Telegram delivery.")
    elif tg_token and tg_chat_id:
        print(f"[TELEGRAM] Dispatching alert to Chat ID {tg_chat_id}...")
        send_telegram_alert(
            bot_token=tg_token,
            chat_id=tg_chat_id,
            message_html=telegram_text
        )
    else:
        print("[TELEGRAM] Warning: Telegram credentials not set. Delivery skipped.")

    # 12. Cleanup Local HTML File
    if os.path.exists(html_file):
        try:
            os.remove(html_file)
            print(f"[CLEANUP] Temporary interactive report {html_file} deleted.")
        except Exception as e:
            print(f"[CLEANUP] Warning: Failed to remove {html_file}: {e}")

    duration = (datetime.datetime.now(BKK_TZ) - start_time).total_seconds()
    print("=" * 75)
    print(f"PRE-MARKET QUANT RADAR COMPLETED SUCCESSFULLY ({duration:.2f}s)")
    print("=" * 75)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-Market Quant Radar Engine (Phase 1)")
    parser.add_argument("--force", action="store_true", help="Force run ignoring duplicate trade_date guard")
    parser.add_argument("--no-telegram", action="store_true", help="Skip sending Telegram notification")
    parser.add_argument("--test", action="store_true", help="Run in test mode with sample SET50 quant data")
    args = parser.parse_args()

    run_premarket_radar(force=args.force, no_telegram=args.no_telegram, test_mode=args.test)
