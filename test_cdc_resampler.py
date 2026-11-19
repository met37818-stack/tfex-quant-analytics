"""
Unit tests for CDC 1-minute resampler and Realized Volatility in pre_market_radar.py
"""
import unittest
import math
from pre_market_radar import resample_cdc_to_1min_prices


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
            ("2026-11-19 14:15:00", 926.0),
            ("2026-11-19 16:55:00", 930.0),
        ]

        prices = resample_cdc_to_1min_prices(raw_rows, trade_date)
        # Standard grid has 165 morning bars + 160 afternoon bars = 325 bars
        self.assertEqual(len(prices), 325)

        # Bar 0 (09:45): 920.5 (last tick of 09:45)
        self.assertEqual(prices[0], 920.5)
        # Bar 1 (09:46): forward-filled 920.5 (quiet minute)
        self.assertEqual(prices[1], 920.5)
        # Bar 2 (09:47): 921.2 (last tick of 09:47)
        self.assertEqual(prices[2], 921.2)
        # Bar 164 (12:29 / 12:30 close snapshot): 925.0
        self.assertEqual(prices[164], 925.0)
        # Bar 165 (14:15): 926.0
        self.assertEqual(prices[165], 926.0)
        # Bar 324 (16:54 / 16:55 close snapshot): 930.0
        self.assertEqual(prices[324], 930.0)

    def test_quiet_minutes_zero_log_returns(self):
        trade_date = "2026-11-19"
        # Constant price throughout the day
        raw_rows = [
            ("2026-11-19 09:45:00", 920.0),
            ("2026-11-19 16:55:00", 920.0),
        ]
        prices = resample_cdc_to_1min_prices(raw_rows, trade_date)
        self.assertEqual(len(prices), 325)
        
        # All returns must be exactly 0
        log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        sum_sq = sum(r * r for r in log_returns)
        self.assertAlmostEqual(sum_sq, 0.0)


if __name__ == "__main__":
    unittest.main()
