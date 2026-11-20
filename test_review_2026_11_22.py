"""
Regression tests for the 2026-11-22 review of pre_market_radar.py

C1  THOR unit handling (0.45% must stay 0.45%, not 45%) and its effect on solved IV
C2  Telegram failures must not leak the bot token into logs and must not raise
W1  Turso responses without a `results` envelope must not raise IndexError
W2  Transient Turso failures retry; SQL errors do not
W3  25-delta skew must be None (not 0.0) when one side has no IV
W4  Options footprint must not turn missing IV into "Call Writing"
W5  theta must match a finite-difference of black76_price
W8  SET holidays cover 2026-10-16 and warn outside the table
"""
import io
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pre_market_radar as pmr


class TestTHORUnitAndIV(unittest.TestCase):

    def test_below_half_percent_stays_percent(self):
        record = {"overnight": 1.50, "1m": 0.45, "3m": 0.90, "6m": 1.20}
        rate, tenor = pmr.select_dynamic_thor_rate(record, 30)
        self.assertEqual(tenor, "1M")
        self.assertAlmostEqual(rate, 0.0045, places=9)

    def test_absurd_rate_raises(self):
        with self.assertRaises(ValueError):
            pmr.select_dynamic_thor_rate({"overnight": 1.0, "1m": 45.0}, 30)

    def test_wrong_rate_shifts_solved_iv(self):
        """ยืนยันขนาดความเสียหาย: r ผิดทำให้ IV ที่แก้ได้เปลี่ยนจริง"""
        F, K, T, true_r, true_iv = 920.0, 850.0, 30 / 365.0, 0.0045, 0.18
        price = pmr.black76_price(F, K, T, true_r, true_iv, "C")
        iv_ok = pmr.calculate_implied_vol(price, F, K, T, true_r, "C")
        iv_bad = pmr.calculate_implied_vol(price, F, K, T, 0.45, "C")
        self.assertAlmostEqual(iv_ok, true_iv, places=4)
        self.assertGreater(abs(iv_bad - iv_ok) * 100.0, 1.0)


class TestThetaMatchesFiniteDifference(unittest.TestCase):

    def test_theta_matches_fd_all_moneyness(self):
        h = 1e-6
        for F, K, T, sig, otype in [
            (920, 920, 0.10, 0.15, "C"), (920, 920, 1.00, 0.15, "P"),
            (920, 850, 1.00, 0.18, "C"), (920, 1050, 1.00, 0.22, "P"),
        ]:
            r = 0.01
            theta = pmr.calculate_black76_greeks(F, K, T, r, sig, otype)["theta"]
            fd = -(pmr.black76_price(F, K, T + h, r, sig, otype)
                   - pmr.black76_price(F, K, T - h, r, sig, otype)) / (2 * h) / 365.0
            self.assertAlmostEqual(theta, fd, delta=abs(fd) * 0.005 + 1e-6,
                                   msg=f"theta mismatch at {F}/{K} T={T} {otype}")


class TestTursoClient(unittest.TestCase):

    def test_missing_results_envelope_raises_clear_error(self):
        class R:
            status_code = 200
            headers = {}
            text = "{}"
            def json(self):
                return {"foo": "bar"}
        with patch.object(pmr.requests, "post", return_value=R()):
            with self.assertRaises(RuntimeError) as ctx:
                pmr.execute_turso_sql("http://x", "t", "SELECT 1")
        self.assertIn("no results envelope", str(ctx.exception))

    def test_retries_transient_then_succeeds(self):
        calls = {"n": 0}

        class Bad:
            status_code = 503
            headers = {}
            text = "unavailable"
            def json(self):
                return {}

        class Good:
            status_code = 200
            headers = {}
            text = ""
            def json(self):
                return {"results": [{"type": "ok", "response": {"result": {
                    "cols": [{"name": "x"}], "rows": [[{"type": "integer", "value": "7"}]]}}}]}

        def fake_post(*a, **k):
            calls["n"] += 1
            return Bad() if calls["n"] < 3 else Good()

        with patch.object(pmr.requests, "post", side_effect=fake_post), \
             patch.object(pmr.time, "sleep", lambda *_: None):
            cols, rows = pmr.execute_turso_sql("http://x", "t", "SELECT 1")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(rows, [[7]])

    def test_sql_error_is_not_retried(self):
        class Err:
            status_code = 200
            headers = {}
            text = ""
            def json(self):
                return {"results": [{"type": "error", "error": "no such table"}]}
        calls = {"n": 0}
        def fake_post(*a, **k):
            calls["n"] += 1
            return Err()
        with patch.object(pmr.requests, "post", side_effect=fake_post):
            with self.assertRaises(RuntimeError):
                pmr.execute_turso_sql("http://x", "t", "SELECT 1")
        self.assertEqual(calls["n"], 1, "SQL error ต้องไม่ retry")


class TestTelegramRedaction(unittest.TestCase):

    def test_connection_error_does_not_leak_token(self):
        token = "123456789:AAFakeTokenForTestingOnly_abcdefghijklm"
        buf = io.StringIO()

        def boom(*a, **k):
            raise pmr.requests.ConnectionError(
                f"HTTPSConnectionPool(host='api.telegram.org'): Max retries exceeded "
                f"with url: /bot{token}/sendMessage")

        with patch.object(pmr.requests, "post", side_effect=boom), redirect_stdout(buf):
            ok = pmr.send_telegram_alert(token, "12345", "<b>hi</b>")

        out = buf.getvalue()
        self.assertFalse(ok)
        self.assertNotIn(token, out, "token ต้องไม่หลุดลง log")
        self.assertIn("ConnectionError", out)


class TestSkewAndFootprintHonesty(unittest.TestCase):
    """
    W3/W4: ข้อมูลที่ขาดต้องไม่ถูกแปลงเป็นข้อสรุปเชิงทิศทาง
    """

    def _greeks_edge_with_calls_only(self):
        rows = [
            ["S50Z26C900", "C", 900.0, 0.16, 0.62, 0.0021, -0.9, 12.0, -0.4, 0.02, 5000],
            ["S50Z26C930", "C", 930.0, 0.15, 0.34, 0.0024, -0.8, 11.0, -0.3, 0.01, 6000],
        ]
        with patch.object(pmr, "execute_turso_sql", return_value=(["x"] * 11, rows)):
            return pmr.analyze_options_greeks_edge(
                pipeline_url="http://x", auth_token="t", trade_date="2026-11-19",
                front_series="Z26", futures_ref=920.0, thor_rate=0.0099,
                T_front=30 / 365.0, front_opts_df=None)

    def test_skew_is_none_when_put_side_missing(self):
        edge = self._greeks_edge_with_calls_only()
        self.assertIsNone(edge["skew_25d"], "ขาด Put IV -> skew ต้องเป็น None ไม่ใช่ค่าจริงข้างเดียว")
        self.assertIsNone(edge["p_iv_pct"])
        self.assertIn("งดสรุปทิศทาง", edge["skew_desc"])

    def test_footprint_does_not_call_missing_iv_call_writing(self):
        inv_rows = [["Call Total", 1200], ["Put Total", -800]]
        # t_0 มี OI สูงกว่า t_-1 -> doi > 0 (มี candidate) แต่ราคา 0 -> IV แก้ไม่ได้
        rows_t0 = [["C", 900.0, 0.0, 5000], ["C", 930.0, 0.0, 6000]]
        rows_prev = [["C", 900.0, 0.0, 4000], ["C", 930.0, 0.0, 5000]]
        state = {"opt_calls": 0}

        def fake_sql(url, tok, sql, args=None):
            if "eod_investor_types" in sql:
                return (["a", "b"], inv_rows)
            state["opt_calls"] += 1
            return (["a", "b", "c", "d"], rows_t0 if state["opt_calls"] == 1 else rows_prev)

        with patch.object(pmr, "execute_turso_sql", side_effect=fake_sql):
            fp = pmr.analyze_options_footprint(
                pipeline_url="http://x", auth_token="t", t_0="2026-11-19", t_minus_1="2026-11-18",
                front_series="Z26", futures_ref_t0=920.0, futures_ref_tprev=918.0,
                T_front_t0=30 / 365.0, T_front_tprev=31 / 365.0, r_rf=0.0099)

        self.assertNotIn("Cap Resistance", fp["call_intent"],
                         "ราคา 0 -> IV แก้ไม่ได้ ต้องไม่สรุปว่าเป็น Call Writing")
        self.assertIn("งดสรุปเจตนา", fp["call_intent"])
        self.assertIsNone(fp["top_call_div"])


class TestCalendarTable(unittest.TestCase):

    def test_2026_10_16_is_a_holiday(self):
        self.assertIn("2026-10-16", pmr.SET_HOLIDAYS)

    def test_holiday_table_matches_sibling_repos_counts(self):
        y2026 = [h for h in pmr.SET_HOLIDAYS if str(h).startswith("2026")]
        self.assertEqual(len(y2026), 20)

    def test_year_outside_table_warns(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            pmr.get_penultimate_business_day(2031, 3)
        self.assertIn("no data for 2031", buf.getvalue())


class TestVolumeProfileBucket(unittest.TestCase):
    def test_bucket_constant_matches_tfex_tick(self):
        self.assertAlmostEqual(pmr.VOLUME_PROFILE_BUCKET, 0.1)


if __name__ == "__main__":
    unittest.main()
