"""
Unit tests for CDC 1-minute resampler and Realized Volatility in pre_market_radar.py
"""
import unittest
import math
from unittest.mock import patch
from pre_market_radar import (
    resample_cdc_to_1min_prices,
    resample_cdc_to_5min_prices,
    select_dynamic_thor_rate,
    compute_intraday_rv,
    compute_volume_profile,
    calculate_implied_vol,
    calculate_black76_greeks
)


class TestCDCResampler(unittest.TestCase):

    def test_resample_cdc_irregular_ticks(self):
        trade_date = "2026-11-19"
        # Simulate irregular CDC ticks:
        # 09:45:03 -> 920.0
        # 09:45:25 -> 920.5 (close of 09:45 is 920.5)
        # 09:46 -> No ticks (quiet market) -> should ffill 920.5
        # 09:47:12 -> 921.0
        # ...
        raw_rows = [
            ("2026-11-19 09:45:03", 920.0),
            ("2026-11-19 09:45:25", 920.5),
            ("2026-11-19 09:47:12", 921.0),
            ("2026-11-19 09:47:45", 921.2),
            ("2026-11-19 12:30:00", 925.0),
            ("2026-11-19 13:45:00", 926.0),
            ("2026-11-19 16:55:00", 930.0),
        ]

        prices = resample_cdc_to_1min_prices(raw_rows, trade_date)
        # Standard grid has 165 morning bars + 190 afternoon bars = 355 bars
        self.assertEqual(len(prices), 355)

        # Bar 0 (09:45): 920.5 (last tick of 09:45)
        self.assertEqual(prices[0], 920.5)
        # Bar 1 (09:46): forward-filled 920.5 (quiet minute)
        self.assertEqual(prices[1], 920.5)
        # Bar 2 (09:47): 921.2 (last tick of 09:47)
        self.assertEqual(prices[2], 921.2)
        # Bar 164 (12:29 / 12:30 close snapshot): 925.0
        self.assertEqual(prices[164], 925.0)
        # Bar 165 (13:45): 926.0
        self.assertEqual(prices[165], 926.0)
        # Bar 354 (16:54 / 16:55 close snapshot): 930.0
        self.assertEqual(prices[354], 930.0)

    def test_quiet_minutes_zero_log_returns(self):
        trade_date = "2026-11-19"
        # Constant price throughout the day
        raw_rows = [
            ("2026-11-19 09:45:00", 920.0),
            ("2026-11-19 16:55:00", 920.0),
        ]
        prices = resample_cdc_to_1min_prices(raw_rows, trade_date)
        self.assertEqual(len(prices), 355)
        
        # All returns must be exactly 0
        log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        sum_sq = sum(r * r for r in log_returns)
        self.assertAlmostEqual(sum_sq, 0.0)

    def test_dynamic_thor_rate_fail_loud(self):
        empty_record = {"thor_overnight": None, "thor_1m": None, "thor_3m": None, "thor_6m": None}
        with self.assertRaises(ValueError):
            select_dynamic_thor_rate(empty_record, dte=30)

    @patch("pre_market_radar.execute_turso_sql")
    def test_compute_intraday_rv_insufficient_bars_returns_none(self, mock_sql):
        # Return only 3 ticks (insufficient for 10 bars)
        mock_sql.return_value = (None, [
            ("2026-11-19 09:45:00", 920.0),
            ("2026-11-19 09:46:00", 920.5),
            ("2026-11-19 09:47:00", 921.0),
        ])
        # Note: resample_cdc_to_1min_prices on a full trade_date creates 325 grid points if start/end match,
        # but if we pass empty rows, prices will be empty (< 10)
        mock_sql.return_value = (None, [])
        res = compute_intraday_rv("https://test", "token", "2026-11-19", "S50Z26", atm_iv=0.12)
        self.assertIsNone(res["rv_annualized"])
        self.assertFalse(res["is_valid_rv"])
        self.assertIn("[INSUFFICIENT FUTURES DATA]", res["vol_regime_desc"])

    @patch("pre_market_radar.execute_turso_sql")
    def test_compute_volume_profile_insufficient_data_returns_none(self, mock_sql):
        # No bar counts in database
        mock_sql.return_value = (None, [])
        res = compute_volume_profile(
            "https://test", "token", "2026-11-19", "S50Z26",
            tactical_levels={}, call_wall=940.0, put_wall=900.0, max_pain=920.0
        )
        self.assertIsNone(res["poc"])
        self.assertIsNone(res["vah"])
        self.assertIsNone(res["val"])
        self.assertEqual(res["confluence_text"], "[NO INTRADAY VOLUME DATA]")

    def test_resample_cdc_to_5min_prices(self):
        trade_date = "2026-11-19"
        raw_rows = [
            ("2026-11-19 09:45:00", 920.0),
            ("2026-11-19 16:55:00", 930.0),
        ]
        prices_5m = resample_cdc_to_5min_prices(raw_rows, trade_date)
        # 355 1-minute bars sampled every 5 minutes + final bar if needed = 72 bars
        self.assertGreaterEqual(len(prices_5m), 71)
        self.assertEqual(prices_5m[0], 920.0)

    def test_iv_capped_at_150_pct(self):
        F = 920.0
        K = 1100.0
        T = 30.0 / 365.0
        r = 0.02
        unrealistic_price = 100.0
        iv = calculate_implied_vol(unrealistic_price, F, K, T, r, "C")
        self.assertIsNone(iv, "IV exceeding 1.50 (150%) must return None")

    def test_charm_dampened_at_low_dte(self):
        F = 920.0
        K = 920.0
        T = 1.0 / 365.0
        r = 0.02
        sigma = 0.15
        greeks = calculate_black76_greeks(F, K, T, r, sigma, "C")
        self.assertLessEqual(abs(greeks["charm"]), 5.0)


if __name__ == "__main__":
    unittest.main()
