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
import calendar
import datetime
import argparse
import requests
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

BKK_TZ = datetime.timezone(datetime.timedelta(hours=7))
CONTRACT_MULTIPLIER = 200.0  # TFEX SET50 Index: 200 THB per index point

# Official SET Holidays (2026 - 2028)
SET_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-02", "2026-03-03", "2026-04-06", "2026-04-13",
    "2026-04-14", "2026-04-15", "2026-05-01", "2026-05-04", "2026-06-01",
    "2026-06-03", "2026-07-28", "2026-07-29", "2026-08-12", "2026-10-13",
    "2026-10-23", "2026-12-07", "2026-12-10", "2026-12-31",
    # 2027
    "2027-01-01", "2027-02-22", "2027-04-06", "2027-04-13", "2027-04-14",
    "2027-04-15", "2027-05-03", "2027-05-04", "2027-05-20", "2027-06-03",
    "2027-07-19", "2027-07-28", "2027-08-12", "2027-10-13", "2027-10-25",
    "2027-12-06", "2027-12-10", "2027-12-31",
    # 2028
    "2028-01-03", "2028-02-10", "2028-04-06", "2028-04-13", "2028-04-14",
    "2028-05-01", "2028-05-05", "2028-06-05", "2028-07-17", "2028-07-28",
    "2028-08-14", "2028-10-13", "2028-10-23", "2028-12-05", "2028-12-11"
}

# ==============================================================================
# TURSO DATABASE CLIENT & HELPERS
# ==============================================================================

def normalize_turso_url(raw_url: str) -> str:
    """Normalize Turso URL to HTTP Pipeline endpoint (/v2/pipeline)."""
    url = raw_url.strip()
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://"):]
    elif not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    url = url.rstrip("/")
    if not url.endswith("/v2/pipeline"):
        url = url + "/v2/pipeline"
    return url

def to_turso_arg(val) -> dict:
    """Format Python values into libSQL / Turso typed arguments."""
    if val is None:
        return {"type": "null"}
    elif isinstance(val, bool):
        return {"type": "integer", "value": "1" if val else "0"}
    elif isinstance(val, int):
        return {"type": "integer", "value": str(val)}
    elif isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return {"type": "null"}
        return {"type": "float", "value": val}
    else:
        return {"type": "text", "value": str(val)}

def execute_turso_sql(pipeline_url: str, auth_token: str, sql: str, args: list = None) -> tuple[list[str], list[list]]:
    """Execute a single SQL statement on Turso and return (columns, rows)."""
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    turso_args = [to_turso_arg(a) for a in (args or [])]
    payload = {
        "requests": [
            {
                "type": "execute",
                "stmt": {
                    "sql": sql.strip(),
                    "args": turso_args
                }
            },
            {"type": "close"}
        ]
    }
    resp = requests.post(pipeline_url, headers=headers, json=payload, timeout=25)
    if resp.status_code != 200:
        raise RuntimeError(f"Turso query failed (HTTP {resp.status_code}): {resp.text}")
    data = resp.json()
    result_item = data.get("results", [])[0]
    if result_item.get("type") == "error":
        raise RuntimeError(f"Turso SQL execution error: {result_item.get('error')}")

    r = result_item.get("response", {}).get("result", {})
    cols = [c["name"] for c in r.get("cols", [])]
    rows = []
    for raw_row in r.get("rows", []):
        parsed_row = []
        for cell in raw_row:
            ctype = cell.get("type")
            cval = cell.get("value")
            if ctype == "null" or cval is None:
                parsed_row.append(None)
            elif ctype == "integer":
                parsed_row.append(int(cval))
            elif ctype == "float":
                parsed_row.append(float(cval))
            else:
                parsed_row.append(cval)
        rows.append(parsed_row)
    return cols, rows

def init_analytics_tables(pipeline_url: str, auth_token: str):
    """Ensure analytics_premarket_log table exists."""
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

# ==============================================================================
# CALENDAR & EXPIRY UTILITIES
# ==============================================================================

def get_penultimate_business_day(year: int, month: int, holidays: set = SET_HOLIDAYS) -> datetime.date:
    """
    Calculate the penultimate business day of a given month.
    (TFEX contract specification for SET50 Futures & Options expiration).
    """
    _, last_day = calendar.monthrange(year, month)
    biz_days = []
    for d in range(1, last_day + 1):
        dt = datetime.date(year, month, d)
        if dt.weekday() < 5 and dt.strftime("%Y-%m-%d") not in holidays:
            biz_days.append(dt)
    if len(biz_days) < 2:
        return biz_days[-1] if biz_days else datetime.date(year, month, last_day)
    return biz_days[-2]

def get_series_expiry_date(series_code: str, ref_date: datetime.date) -> datetime.date:
    """
    Determine expiration date for a quarterly series (e.g., S50Z26 -> Dec 2026).
    Quarter codes: H = March, M = June, U = September, Z = December.
    """
    code_map = {"H": 3, "M": 6, "U": 9, "Z": 12}
    quarter_char = series_code[3].upper()
    quarter_month = code_map.get(quarter_char, 12)
    year_suffix = int(series_code[4:6])
    full_year = 2000 + year_suffix

    expiry = get_penultimate_business_day(full_year, quarter_month)
    return expiry

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
    """Solve for Black-76 Implied Volatility via Brent's method."""
    if price is None or price <= 0.0 or T <= 0.0:
        return None

    disc = math.exp(-r * T)
    intrinsic = max(0.0, disc * ((F - K) if option_type == 'C' else (K - F)))
    if price <= intrinsic + 1e-4:
        return None

    def obj(sigma):
        return black76_price(F, K, T, r, sigma, option_type) - price

    try:
        return float(brentq(obj, 1e-4, 5.0, maxiter=100))
    except Exception:
        return None

def calculate_black76_greeks(F: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    """
    Compute Delta, Gamma, Vega, and Theta under Black-76.
    """
    if sigma <= 0.0 or T <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = (disc * pdf_d1) / (F * sigma * sqrt_T)
    vega_1pct = (F * disc * pdf_d1 * sqrt_T) * 0.01

    if option_type == 'C':
        delta = disc * norm.cdf(d1)
        theta_call = - (F * disc * pdf_d1 * sigma) / (2.0 * sqrt_T) - r * disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        theta_1d = theta_call / 365.0
    else:
        delta = -disc * norm.cdf(-d1)
        theta_put = - (F * disc * pdf_d1 * sigma) / (2.0 * sqrt_T) - r * disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        theta_1d = theta_put / 365.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega_1pct),
        "theta": float(theta_1d)
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
    flow_emoji: str
) -> str:
    """
    Constructs an actionable, rule-based frontline trading plan in Thai.
    4 Core Pillars:
    1. 1-Day Expected Move (1-SD)
    2. Tactical Key Levels (Ranked nearest to current price)
    3. Smart Money Flow Breakdown
    4. Daily Action Plan & Stop Loss
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
    if atm_iv < 0.10:
        iv_desc = "ความผันผวนต่ำมาก ตลาดมักบีบกรอบแคบ"
    elif atm_iv <= 0.15:
        iv_desc = "ความผันผวนปกติ เหมาะแก่การเทรดตามกรอบ"
    else:
        iv_desc = "ความผันผวนสูง แกว่งกว้าง ควรกระชับจุดคัต"

    # 2. Smart Money Translation & Actionable Playbook Logic
    if foreign_net > 0 and delta_oi > 0:
        flow_title = "New Long Build-up (ต่างชาติเปิดสะสม Long สัญญาใหม่)"
        flow_detail = f"ต่างชาติซื้อสุทธิ <code>{foreign_net:+,}</code> สัญญา พร้อม OI ขยายตัว <code>{delta_oi:+,}</code> สัญญา เป็นการเปิดสถานะ Long ใหม่หนุนตลาดขึ้นอย่างแท้จริง"
        flow_caution = f"หากผ่านต้าน 1 (<code>{res_1:.0f}</code>) ได้ โมเมนตัมจะพาไปทดสอบ Upper 1-SD หรือต้าน 2 (<code>{res_2:.0f}</code>)"
        action_strategy = "Follow-Through ฝั่ง Long ตามโมเมนตัม ไม่ดักสวนเทรนด์"
        long_plan = f"รอจังหวะย่อตัวแถว รับ 1 (<code>{sup_1:.0f}</code>) หรือราคาเปิด หากยืนได้เปิด Long เป้าหมาย <code>{res_1:.0f}</code> และ <code>{upper_1sd:.1f}</code>"
        short_plan = f"ชะลอการ Short จนกว่าจะขึ้นทดสอบ ต้าน 2 (<code>{res_2:.0f}</code>) หรือ Upper 1-SD แล้วเกิดสัญญาณ Rejection ไม่ผ่าน"
        stop_loss_plan = f"ฝั่ง Long วาง Stop Loss เมื่อราคาหลุด <code>{sup_2:.0f}</code> จุด"
    elif foreign_net > 0 and delta_oi < 0:
        flow_title = "Short Covering (ต่างชาติปิดทำกำไร/ลดพอร์ต Short)"
        flow_detail = f"ต่างชาติซื้อสุทธิ <code>{foreign_net:+,}</code> สัญญา แต่ OI หดตัว <code>{delta_oi:+,}</code> สัญญา เป็นแรงซื้อคืนเพื่อปิดสัญญา Short เดิม"
        flow_caution = f"การดีดตัวอาจเป็นเพียง Technical Rebound ระยะสั้น ระวังแรงขายทำกำไรบริเวณแนวต้าน ต้าน 1 หรือ Max Pain (<code>{max_pain:.0f}</code>)"
        action_strategy = "เน้นเล่นรอบในกรอบ (Range-Bound / Mean Reversion) ดักย่อซื้อ-เด้งขาย ไม่ไล่ราคา"
        long_plan = f"รอตั้งรับบริเวณ Lower 1-SD (<code>{lower_1sd:.1f}</code>) หรือ รับ 1 (<code>{sup_1:.0f}</code>) เป้าทำกำไรที่ Max Pain (<code>{max_pain:.0f}</code>) หรือ ต้าน 1"
        short_plan = f"หากเปิดกระโดดขึ้นทดสอบ ต้าน 1 (<code>{res_1:.0f}</code>) หรือ Upper 1-SD (<code>{upper_1sd:.1f}</code>) แล้วไม่ผ่าน พิจารณาดัก Short เล่นรอบย่อ"
        stop_loss_plan = f"ฝั่ง Long คัตเมื่อหลุด <code>{sup_2:.0f}</code> | ฝั่ง Short คัตเมื่อทะลุเหนือ <code>{res_1 + 3.0:.1f}</code>"
    elif foreign_net < 0 and delta_oi > 0:
        flow_title = "New Short Build-up (ต่างชาติเปิดสะสม Short สัญญาใหม่)"
        flow_detail = f"ต่างชาติขายสุทธิ <code>{foreign_net:+,}</code> สัญญา และ OI ขยายตัว <code>{delta_oi:+,}</code> สัญญา สะท้อนการเปิด Short เพิ่มเพื่อเก็งกำไรขาลง"
        flow_caution = f"แรงกดดันฝั่งขายสูง หากหลุด รับ 1 (<code>{sup_1:.0f}</code>) ดีลเลอร์จะเร่งขายตาม เสี่ยงไหลลงทดสอบ Lower 1-SD"
        action_strategy = "เน้นฝั่ง Short (Trend Expansion Down) ตามน้ำ ระวังการหลุดแนวรับแล้วรูดแรง"
        long_plan = f"หลีกเลี่ยงการดักสวนรับเร็ว (มีดตก) จนกว่าราคาจะลงแตะ รับ 2 (<code>{sup_2:.0f}</code>) หรือ Lower 1-SD แล้วเกิดสัญญาณหยุดลง"
        short_plan = f"เด้งขึ้นทดสอบ ต้าน 1 (<code>{res_1:.0f}</code>) หรือ Max Pain (<code>{max_pain:.0f}</code>) ไม่ผ่าน เปิด Short เป้าหมาย <code>{sup_1:.0f}</code> และ <code>{lower_1sd:.1f}</code>"
        stop_loss_plan = f"ฝั่ง Short วาง Stop Loss เหนือ <code>{res_1 + 2.5:.1f}</code> จุด"
    elif foreign_net < 0 and delta_oi < 0:
        flow_title = "Long Liquidation (ต่างชาติปิดทำกำไร/ลดพอร์ต Long)"
        flow_detail = f"ต่างชาติขายสุทธิ <code>{foreign_net:+,}</code> สัญญา ควบคู่กับ OI ลดลง <code>{delta_oi:+,}</code> สัญญา เป็นการปิดสัญญาลดความเสี่ยง"
        flow_caution = f"ตลาดขาดแรงซื้อหนุน ตราบใดที่ยังไม่ยืนเหนือต้าน 1 (<code>{res_1:.0f}</code>) ให้ระวังการซึมลง"
        action_strategy = "เทรดรอบสั้นอย่างระมัดระวัง เน้นตั้งรับต่ำและไม่ไล่ราคา"
        long_plan = f"ดักซื้อบริเวณ รับ 1 (<code>{sup_1:.0f}</code>) หรือ Lower 1-SD (<code>{lower_1sd:.1f}</code>) เพื่อเล่นรีบาวด์สั้นๆ"
        short_plan = f"เด้งไม่ผ่าน ต้าน 1 (<code>{res_1:.0f}</code>) เปิด Short เล่นรอบย่อ"
        stop_loss_plan = f"ฝั่ง Long คัตเมื่อหลุด <code>{sup_1 - 2.0:.1f}</code> | ฝั่ง Short คัตเมื่อทะลุเหนือ <code>{res_2:.0f}</code>"
    else:
        flow_title = "Neutral / Rebalancing (ปรับสมดุลพอร์ต)"
        flow_detail = f"ทิศทางฟันด์โฟลว์ยังไม่ชี้นำเด่นชัด (Foreign: <code>{foreign_net:+,}</code>, ΔOI: <code>{delta_oi:+,}</code>)"
        flow_caution = "ตลาดเคลื่อนไหวตามกรอบเทคนิคและบรรยากาศภายนอก"
        action_strategy = "เล่นในกรอบ รับ-ต้าน ตามปกติ"
        long_plan = f"รับแถว <code>{sup_1:.0f}</code> จุด"
        short_plan = f"ต้านแถว <code>{res_1:.0f}</code> จุด"
        stop_loss_plan = f"คัตเมื่อหลุด <code>{sup_2:.0f}</code> หรือทะลุ <code>{res_2:.0f}</code>"

    gamma_flip_str = f"{gamma_flip_val:.1f}" if not math.isnan(gamma_flip_val) else "N/A"
    gamma_emoji = "🛡️" if is_positive_gamma else "⚡"

    msg = f"""🎯 <b>แผนการเทรด SET50 ประจำวัน | TFEX QUANT RADAR</b>
📅 <i>ข้อมูลสิ้นวัน {trade_date} | ออกรายงานเช้า {as_of_time_bkk} น.</i>
━━━━━━━━━━━━━━━━━━━━━━
🎯 <b>1. กรอบวิ่งคาดการณ์วันนี้ (1-Day Expected Move)</b>
• <b>กรอบบน (Upper 1-SD):</b> <code>{upper_1sd:.1f}</code> จุด
• <b>กรอบล่าง (Lower 1-SD):</b> <code>{lower_1sd:.1f}</code> จุด
• <b>ระยะแกว่งตัวคาดการณ์ (1-SD):</b> <code>±{expected_move_1d:.1f}</code> จุด
• <b>ระดับความผันผวน (ATM IV):</b> <code>{atm_iv * 100:.2f}%</code> <i>({iv_desc})</i>

🧱 <b>2. แนวรับ - แนวต้านสำคัญรายวัน (Tactical Key Levels)</b>
🔴 <b>ต้าน 2:</b> <code>{res_2:.0f}</code> จุด <i>(Call OI: {res_2_oi:,} สัญญา)</i>
🔴 <b>ต้าน 1:</b> <code>{res_1:.0f}</code> จุด <i>(Call OI: {res_1_oi:,} สัญญา)</i>
🎯 <b>Max Pain:</b> <code>{max_pain:.0f}</code> จุด <i>(ระดับดึงดูดราคาหมดอายุ)</i>
📍 <b>ราคาปิดล่าสุด ({front_series}):</b> <code>{futures_ref:.1f}</code> จุด
🟢 <b>รับ 1:</b> <code>{sup_1:.0f}</code> จุด <i>(Put OI: {sup_1_oi:,} สัญญา)</i>
🟢 <b>รับ 2:</b> <code>{sup_2:.0f}</code> จุด <i>(Put OI: {sup_2_oi:,} สัญญา)</i>
───────────────
🏛️ <i>กรอบใหญ่ไตรมาส: Put Wall <code>{put_wall:.0f}</code> – Call Wall <code>{call_wall:.0f}</code> | Gamma Flip: <code>{gamma_flip_str}</code></i>

🌊 <b>3. พฤติกรรมรายใหญ่ (Smart Money Flow)</b>
• <b>สถานะ Flow:</b> {flow_emoji} <b>{flow_title}</b>
• <b>การวิเคราะห์:</b> {flow_detail}
• <b>ข้อควรระวัง:</b> {flow_caution}
• <b>ยอดซื้อขายสุทธิรายกลุ่ม:</b>
  - ต่างชาติ (Foreign): <code>{foreign_net:+,}</code> สัญญา (ΔOI: <code>{delta_oi:+,}</code>)
  - กองทุน (Institution): <code>{inst_net:+,}</code> | รายย่อย (Local): <code>{local_net:+,}</code>

⚔️ <b>4. กลยุทธ์และแผนการเล่นวันนี้ (Daily Action Plan)</b>
• <b>สภาวะตลาด (Dealer Gamma):</b> {gamma_emoji} <b>{gamma_zone_title}</b>
  <i>({gamma_desc})</i>
• <b>แผนการปฏิบัติหน้างาน:</b>
  💡 <b>กลยุทธ์หลัก:</b> {action_strategy}
  🟢 <b>ฝั่ง Long:</b> {long_plan}
  🔴 <b>ฝั่ง Short:</b> {short_plan}
  ⚠️ <b>จุดตัดขาดทุน (Stop Loss):</b> {stop_loss_plan}
━━━━━━━━━━━━━━━━━━━━━━
🕒 <i>แนบไฟล์กราฟอินเทอร์แอคทีฟ (.html) เปิดดูโครงสร้าง GEX & IV Curve ได้ทันที</i>"""
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
    lower_1sd: float = None
) -> str:
    """
    Generates a standalone, responsive, high-performance HTML report with 3 interactive subplots.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=(
            f"<b>1. Net Gamma Exposure (GEX) Profile by Strike — {front_series} (200 THB/pt)</b>",
            f"<b>2. Implied Volatility Smile & Skew Curve — Front ({front_series}) vs Next ({next_series})</b>",
            f"<b>3. Open Interest Distribution (Call vs Put) & Max Pain ({max_pain:.0f})</b>"
        ),
        row_heights=[0.36, 0.32, 0.32]
    )

    # ----------------------------------------------------
    # Subplot 1: Net GEX Profile
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
        row=1, col=1
    )

    # Key Level Lines for Row 1
    # Futures Ref line
    fig.add_vline(
        x=futures_ref, line_width=2, line_dash="dash", line_color="#00E5FF",
        annotation_text=f"F: {futures_ref:.1f}", annotation_position="top left",
        annotation_font=dict(color="#00E5FF", size=11),
        row=1, col=1
    )
    # Expected Move 1-SD Lines
    if upper_1sd is not None:
        fig.add_vline(
            x=upper_1sd, line_width=1.5, line_dash="dot", line_color="#B388FF",
            annotation_text=f"Upper 1-SD: {upper_1sd:.1f}", annotation_position="top right",
            annotation_font=dict(color="#B388FF", size=9),
            row=1, col=1
        )
    if lower_1sd is not None:
        fig.add_vline(
            x=lower_1sd, line_width=1.5, line_dash="dot", line_color="#B388FF",
            annotation_text=f"Lower 1-SD: {lower_1sd:.1f}", annotation_position="bottom left",
            annotation_font=dict(color="#B388FF", size=9),
            row=1, col=1
        )

    # Gamma Flip Line
    if not math.isnan(gamma_flip_cum):
        fig.add_vline(
            x=gamma_flip_cum, line_width=2.5, line_dash="dot", line_color="#FFD600",
            annotation_text=f"Flip: {gamma_flip_cum:.1f}", annotation_position="bottom right",
            annotation_font=dict(color="#FFD600", size=11),
            row=1, col=1
        )
    # Call Wall
    fig.add_vline(
        x=call_wall, line_width=1.5, line_dash="dashdot", line_color="#69F0AE",
        annotation_text=f"Call Wall: {call_wall:.0f}", annotation_position="top right",
        annotation_font=dict(color="#69F0AE", size=10),
        row=1, col=1
    )
    # Put Wall
    fig.add_vline(
        x=put_wall, line_width=1.5, line_dash="dashdot", line_color="#FF5252",
        annotation_text=f"Put Wall: {put_wall:.0f}", annotation_position="bottom left",
        annotation_font=dict(color="#FF5252", size=10),
        row=1, col=1
    )

    # ----------------------------------------------------
    # Subplot 2: IV Smile & Skew Curve
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
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=p_front["strike"], y=p_front["iv"] * 100.0,
                mode="lines+markers", name=f"{front_series} Put IV (%)",
                line=dict(color="#FF4081", width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{x} Put</b><br>IV: %{y:.2f}%<extra></extra>"
            ),
            row=2, col=1
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
            row=2, col=1
        )

    fig.add_vline(
        x=futures_ref, line_width=1.5, line_dash="dash", line_color="#78909C",
        row=2, col=1
    )

    # ----------------------------------------------------
    # Subplot 3: Open Interest Distribution
    # ----------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=res_df["strike"], y=res_df["call_oi"],
            name="Call OI", marker_color="#29B6F6", opacity=0.85,
            hovertemplate="<b>Strike %{x} Call OI</b>: %{y:,}<extra></extra>"
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(
            x=res_df["strike"], y=res_df["put_oi"],
            name="Put OI", marker_color="#FF7043", opacity=0.85,
            hovertemplate="<b>Strike %{x} Put OI</b>: %{y:,}<extra></extra>"
        ),
        row=3, col=1
    )

    fig.add_vline(
        x=max_pain, line_width=2, line_dash="dash", line_color="#FFD600",
        annotation_text=f"Max Pain: {max_pain:.0f}", annotation_position="top left",
        annotation_font=dict(color="#FFD600", size=10),
        row=3, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>SET50 PRE-MARKET QUANT RADAR</b> | As of EOD {trade_date} (Flow Regime: {foreign_regime})",
            font=dict(size=18, color="#FFFFFF"),
            x=0.03, y=0.98
        ),
        height=1000,
        margin=dict(l=60, r=40, t=90, b=50),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=0.98,
            font=dict(size=11)
        ),
        paper_bgcolor="#121824",
        plot_bgcolor="#182232"
    )

    fig.update_xaxes(title_text="Strike Price", gridcolor="#26354A", row=3, col=1)
    fig.update_xaxes(gridcolor="#26354A", row=1, col=1)
    fig.update_xaxes(gridcolor="#26354A", row=2, col=1)
    fig.update_yaxes(title_text="Net GEX (MB/pt)", gridcolor="#26354A", row=1, col=1)
    fig.update_yaxes(title_text="Implied Volatility (%)", gridcolor="#26354A", row=2, col=1)
    fig.update_yaxes(title_text="Contracts (OI)", gridcolor="#26354A", row=3, col=1)

    html_path = f"premarket_report_{trade_date}.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    return html_path

# ==============================================================================
# TELEGRAM DISPATCHER
# ==============================================================================

def send_telegram_alert(
    bot_token: str,
    chat_id: str,
    message_html: str,
    html_file_path: str = None
) -> bool:
    """
    Sends rich HTML summary message and uploads interactive Plotly HTML report.
    """
    base_url = f"https://api.telegram.org/bot{bot_token}"

    # 1. Send Text Briefing
    msg_url = f"{base_url}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    r = requests.post(msg_url, json=payload, timeout=20)
    if r.status_code != 200:
        print(f"[TELEGRAM] Warning: sendMessage failed ({r.status_code}): {r.text}")
        return False
    print("[TELEGRAM] Pre-market summary text delivered successfully.")

    # 2. Send Document (Plotly HTML)
    if html_file_path and os.path.exists(html_file_path):
        doc_url = f"{base_url}/sendDocument"
        caption = f"📊 <b>SET50 Quant Radar Interactive Dashboard</b> ({os.path.basename(html_file_path)})"
        with open(html_file_path, "rb") as f:
            file_bytes = f.read()

        for attempt in range(1, 4):
            try:
                files = {"document": (os.path.basename(html_file_path), file_bytes, "text/html")}
                doc_data = {
                    "chat_id": chat_id,
                    "caption": caption,
                    "parse_mode": "HTML"
                }
                r_doc = requests.post(doc_url, data=doc_data, files=files, timeout=(20, 120))
                if r_doc.status_code == 200:
                    print(f"[TELEGRAM] Interactive HTML chart delivered successfully on attempt {attempt}.")
                    break
                else:
                    print(f"[TELEGRAM] Attempt {attempt} warning: sendDocument status {r_doc.status_code}: {r_doc.text}")
            except Exception as e:
                print(f"[TELEGRAM] Attempt {attempt} failed to upload document: {e}")
                if attempt == 3:
                    print("[TELEGRAM] Max attempts reached for document upload.")

    return True

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

    # If test mode is requested without Turso DB credentials, execute with realistic mock data
    if test_mode and (not turso_url or not turso_token):
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

        html_file = build_interactive_dashboard(
            trade_date=t_0, futures_ref=futures_ref, front_series=front_series, next_series=next_series,
            res_df=res_df, gamma_flip_cum=gamma_flip_val, call_wall=call_wall, put_wall=put_wall,
            max_pain=max_pain, iv_df_front=pd.DataFrame(), iv_df_next=pd.DataFrame(),
            foreign_regime=flow_regime, upper_1sd=upper_1sd, lower_1sd=lower_1sd
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
            flow_regime=flow_regime, flow_emoji=flow_emoji
        )

        print("\n" + "=" * 75)
        print("GENERATED THAI TRADING PLAYBOOK (TELEGRAM PREVIEW)")
        print("=" * 75)
        print(telegram_text)
        print("=" * 75)
        print(f"Message Length: {len(telegram_text)} characters (Limit: 4096)")

        if tg_token and tg_chat_id and not no_telegram:
            print(f"[TELEGRAM] Dispatching test notification to Chat ID {tg_chat_id}...")
            send_telegram_alert(tg_token, tg_chat_id, telegram_text, html_file)
        else:
            print("[TELEGRAM] Telegram credentials not configured or --no-telegram enabled. Test output displayed locally.")

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

    # 4.3 THOR Interest Rate
    _, thor_rows = execute_turso_sql(
        pipeline_url, turso_token,
        "SELECT thor_rate FROM eod_thor_rates WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 1;",
        [t_0]
    )
    thor_rate_pct = float(thor_rows[0][0]) if thor_rows and thor_rows[0][0] is not None else 1.00
    r_rf = thor_rate_pct / 100.0
    print(f"[STRUCTURE] Risk-free Rate (THOR): {thor_rate_pct:.4f}% (r = {r_rf:.6f})")

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
        iv = calculate_implied_vol(p, futures_ref, k, T_front, r_rf, otype)
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
        iv = calculate_implied_vol(p, next_ref, k, T_next, r_rf, otype)
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
        atm_iv = float(valid_ivs.mean()) if not valid_ivs.empty else 0.12

    # 5.1 Calculate 1-Day Expected Move (1-SD)
    expected_move_1d = futures_ref * (atm_iv / math.sqrt(252.0))
    upper_1sd = futures_ref + expected_move_1d
    lower_1sd = futures_ref - expected_move_1d

    # 6. Strike-Level GEX Profile, Walls, and Max Pain
    gex_records = []
    for k in strikes_front:
        c_item = front_opts[(front_opts["strike"] == k) & (front_opts["option_type"] == "C")]
        p_item = front_opts[(front_opts["strike"] == k) & (front_opts["option_type"] == "P")]

        c_oi = int(c_item["oi"].values[0]) if len(c_item) and c_item["oi"].values[0] else 0
        p_oi = int(p_item["oi"].values[0]) if len(p_item) and p_item["oi"].values[0] else 0

        c_iv = c_item["iv"].values[0] if len(c_item) else None
        p_iv = p_item["iv"].values[0] if len(p_item) else None

        valid_k_ivs = [x for x in [c_iv, p_iv] if x is not None and not np.isnan(x)]
        sigma_k = np.mean(valid_k_ivs) if valid_k_ivs else atm_iv

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

    # Gamma Flip (Zero Crossing of Cumulative GEX)
    gamma_flip_cum = np.nan
    for i in range(len(res_df) - 1):
        c1 = res_df.iloc[i]["cum_gex"]
        c2 = res_df.iloc[i + 1]["cum_gex"]
        k1 = res_df.iloc[i]["strike"]
        k2 = res_df.iloc[i + 1]["strike"]
        if (c1 <= 0 and c2 >= 0) or (c1 >= 0 and c2 <= 0):
            if c2 != c1:
                gamma_flip_cum = float(k1 + (0 - c1) * (k2 - k1) / (c2 - c1))
            else:
                gamma_flip_cum = float(k1)
            break

    # Net GEX Zero Crossing
    net_flip = np.nan
    for i in range(len(res_df) - 1):
        g1 = res_df.iloc[i]["net_gex_mb"]
        g2 = res_df.iloc[i + 1]["net_gex_mb"]
        k1 = res_df.iloc[i]["strike"]
        k2 = res_df.iloc[i + 1]["strike"]
        if (g1 <= 0 and g2 >= 0) or (g1 >= 0 and g2 <= 0):
            if g2 != g1:
                net_flip = float(k1 + (0 - g1) * (k2 - k1) / (g2 - g1))
            else:
                net_flip = float(k1)
            break

    total_net_gex = float(res_df["net_gex_mb"].sum())
    gamma_flip_val = gamma_flip_cum if not math.isnan(gamma_flip_cum) else net_flip

    # Consistent Gamma Regime: Aligned with current price position relative to Gamma Flip
    if not math.isnan(gamma_flip_val):
        if futures_ref >= gamma_flip_val:
            is_positive_gamma = True
            gamma_zone_title = "Positive Gamma (โซนลดความผันผวน / Mean Reversion)"
            gamma_desc = f"ราคาปัจจุบัน ({futures_ref:.1f}) อยู่เหนือ Gamma Flip ({gamma_flip_val:.1f}) ดีลเลอร์อยู่ในสถานะ Long Gamma (ซื้อแนวรับ-ขายแนวต้าน) ช่วยซับแรงเหวี่ยง ทำให้ตลาดมักแกว่งตัวในกรอบ"
        else:
            is_positive_gamma = False
            gamma_zone_title = "Negative Gamma (โซนเร่งความผันผวน / Trend Expansion)"
            gamma_desc = f"ราคาปัจจุบัน ({futures_ref:.1f}) อยู่ใต้ Gamma Flip ({gamma_flip_val:.1f}) ดีลเลอร์อยู่ในสถานะ Short Gamma (ต้องเปิด Hedging ตามแนวโน้ม) หากหลุดแนวรับจะเร่งแรงขายซ้ำเติม ทำให้เกิดความผันผวนสูง"
    else:
        if total_net_gex >= 0:
            is_positive_gamma = True
            gamma_zone_title = "Positive Gamma (โซนลดความผันผวน)"
            gamma_desc = "Net GEX รวมเป็นบวก ดีลเลอร์ช่วยซับแรงกระแทก ตลาดมีแนวโน้มแกว่งตัวในกรอบ"
        else:
            is_positive_gamma = False
            gamma_zone_title = "Negative Gamma (โซนเร่งความผันผวน)"
            gamma_desc = "Net GEX รวมเป็นลบ ดีลเลอร์เร่งโมเมนตัมตามแนวโน้ม ระวังความผันผวนขยายตัว"

    gamma_regime = "Positive Gamma" if is_positive_gamma else "Negative Gamma"

    # 6.1 Resolve Nearest Tactical Levels (Spot ± 25)
    tactical_levels = resolve_tactical_levels(res_df, futures_ref, call_wall, put_wall)

    print(f"[QUANT] ATM IV            : {atm_iv * 100.0:.2f}% (ATM Strike: {atm_strike:.0f})")
    print(f"[QUANT] 1-Day Expected Move: ±{expected_move_1d:.1f} pts (Range: {lower_1sd:.1f} - {upper_1sd:.1f})")
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
        lower_1sd=lower_1sd
    )
    print(f"[CHART] Interactive Plotly report generated: {html_file}")

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
        flow_emoji=flow_emoji
    )

    # 10. Persist Log to Turso Database
    print("[DATABASE] Logging pre-market execution record to analytics_premarket_log...")
    upsert_sql = """
    INSERT OR REPLACE INTO analytics_premarket_log (
        trade_date, futures_ref, atm_iv, gamma_flip, call_wall, put_wall, max_pain, foreign_regime, sent_at_bkk
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            message_html=telegram_text,
            html_file_path=html_file
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
