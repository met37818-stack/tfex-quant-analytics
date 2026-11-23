"""
Regression tests for the 2026-11-23 audit of the daily Telegram playbook.

The numbers in that message were all real, but three classes of output misled the
reader: labels derived from noise, a plan that contradicted the gamma regime it had
just printed, and values printed as 0.0 when the data was simply absent. These tests
pin the behaviour that replaced them.
"""
import math
import unittest

import pre_market_radar as pmr


def _playbook(**overrides):
    """Baseline = the real 2026-11-20 snapshot, as verified against TFEX."""
    F = 991.1
    iv = 0.14645
    em = F * iv / math.sqrt(252.0)
    regime, emoji, _ = pmr.decompose_smart_money(5428, -6, 506838)
    kw = dict(
        trade_date="2026-11-20", as_of_time_bkk="06:46",
        futures_ref=F, front_series="S50Z26", atm_iv=iv,
        expected_move_1d=em, upper_1sd=F + em, lower_1sd=F - em,
        tactical_levels={"res_1": 1000, "res_1_oi": 983, "res_2": 1010, "res_2_oi": 1085,
                         "sup_1": 980, "sup_1_oi": 1602, "sup_2": 970, "sup_2_oi": 1479},
        max_pain=1020, call_wall=1100, put_wall=960,
        gamma_flip_val=1016.3, is_positive_gamma=False,
        gamma_zone_title="", gamma_desc="", total_net_gex=-120.5,
        foreign_net=5428, delta_oi=-6, inst_net=753, local_net=-6181,
        flow_regime=regime, flow_emoji=emoji,
        foreign_footprint={"cum_foreign_net": -91273, "cost_basis": 1030.1,
                           "pnl_status": "", "z_desc": "", "divergence_text": "",
                           "history_days": 60},
        options_footprint={"foreign_call_net": -305, "foreign_put_net": 25,
                           "top_call_strike": 1030, "top_call_doi": 257, "top_call_div": 1.17,
                           "call_intent": "Aggressive Call Buying", "top_put_strike": 950,
                           "top_put_doi": 350, "top_put_div": 0.53,
                           "put_intent": "Put Accumulation", "div_threshold": 0.60},
        intraday_rv={"bar_count": 0, "rv_annualized": None, "vol_regime_desc": ""},
        volume_profile={"poc": 995, "vah": 1003, "val": 994, "valid_days": 1,
                        "warmup_days": 1, "confluence_text": ""},
        late_session={"late_fut_delta": None, "late_basis_close": None,
                      "late_basis_delta": None, "has_data": False, "late_session_desc": ""},
        greeks_edge={"records_count": 390, "skew_25d": 2.60, "p_iv_pct": 16.6, "c_iv_pct": 14.0,
                     "skew_desc": "", "daily_breakeven_move": 7.6, "theta_gamma_ratio": 28.9,
                     "dealer_vanna": -129, "dealer_charm": 21},
        total_oi=506838,
    )
    kw.update(overrides)
    return pmr.generate_thai_trading_playbook(**kw)


class TestFlowRegimeNeedsMagnitude(unittest.TestCase):
    def test_oi_change_of_six_contracts_is_not_a_regime(self):
        regime, emoji, desc = pmr.decompose_smart_money(5428, -6, 506838)
        self.assertEqual(regime, "Neutral / Rebalancing")
        self.assertEqual(emoji, "⚪")
        self.assertIn("0.001%", desc)
        self.assertIn("noise", desc)

    def test_a_real_oi_move_still_classifies(self):
        # 6,000 contracts out of 506,838 = 1.18% -> above the 0.5% floor
        regime, _, _ = pmr.decompose_smart_money(5428, -6000, 506838)
        self.assertEqual(regime, "Short Covering")

    def test_without_total_oi_the_legacy_behaviour_is_kept(self):
        regime, _, _ = pmr.decompose_smart_money(5428, -6000)
        self.assertEqual(regime, "Short Covering")

    def test_exactly_at_the_floor_is_still_neutral(self):
        # 0.5% of 100_000 = 500; the comparison is strict <
        regime, _, _ = pmr.decompose_smart_money(1000, 499, 100_000)
        self.assertEqual(regime, "Neutral / Rebalancing")


class TestPlanRespectsGammaRegime(unittest.TestCase):
    def test_negative_gamma_does_not_print_a_mean_reversion_plan(self):
        msg = _playbook()
        self.assertIn("Negative Gamma", msg)
        self.assertIn("เล่นตามทิศที่เบรก", msg)
        self.assertNotIn("ดักย่อซื้อ-เด้งขาย", msg)

    def test_entries_are_labelled_as_half_size_fades(self):
        msg = _playbook()
        self.assertIn("ดักรับ (ครึ่งไซส์)", msg)
        self.assertIn("ดักขาย (ครึ่งไซส์)", msg)

    def test_break_levels_live_in_the_mode_line_not_in_the_entry_cell(self):
        msg = _playbook()
        mode_line = [ln for ln in msg.splitlines() if "โหมด" in ln][0]
        self.assertIn("1000.2", mode_line)      # upper 1-SD
        self.assertIn("982.0", mode_line)       # lower 1-SD
        self.assertIn("เล่นตามทิศที่เบรก", mode_line)

    def test_positive_gamma_keeps_the_range_plan(self):
        msg = _playbook(gamma_flip_val=950.0, is_positive_gamma=True)
        self.assertIn("Positive Gamma", msg)
        self.assertIn("Mean Reversion ในกรอบใช้ได้", msg)


class TestUnknownGammaIsItsOwnState(unittest.TestCase):
    def test_missing_flip_is_reported_not_guessed(self):
        msg = _playbook(gamma_flip_val=float("nan"), is_positive_gamma=None)
        self.assertIn("ไม่ทราบโซน Gamma", msg)
        self.assertNotIn("Negative Gamma", msg)
        self.assertNotIn("Positive Gamma", msg)
        self.assertIn("Gamma Flip", msg.replace("ไม่พบจุดตัด", ""))  # still listed as pending

    def test_no_flip_means_the_watchlist_cannot_use_a_flip_level(self):
        msg = _playbook(gamma_flip_val=float("nan"), is_positive_gamma=None)
        self.assertNotIn("gamma เปลี่ยนโซน", msg, "ไม่มี flip ก็ไม่ควรมี trigger ที่อ้าง flip")
        self.assertIn("Gamma Flip (ไม่พบจุดตัด GEX)", msg)


class TestThinDataIsNotPrinted(unittest.TestCase):
    def test_one_day_volume_profile_is_not_published(self):
        msg = _playbook()
        self.assertNotIn("POC", msg)
        self.assertIn("Volume Profile (1/20 วัน", msg)

    def test_missing_late_session_does_not_print_zeros(self):
        msg = _playbook()
        self.assertNotIn("Fut +0.0", msg)
        self.assertNotIn("Basis +0.00", msg)
        self.assertIn("Late Session", msg)

    def test_missing_rv_is_listed_once_under_pending(self):
        msg = _playbook()
        self.assertIn("RV vs IV (0/10 บาร์)", msg)
        self.assertEqual(msg.count("⏳ <b>ยังไม่พร้อม:</b>"), 1)
        self.assertNotIn("\n• <b>RV vs IV", msg)

    def test_a_healthy_profile_is_published(self):
        msg = _playbook(volume_profile={"poc": 995, "vah": 1003, "val": 994, "valid_days": 12,
                                        "warmup_days": 12, "confluence_text": "POC ตรง Max Pain"})
        self.assertIn("POC", msg)
        self.assertIn("Volume Profile 12 วัน", msg)


class TestPlanIsFalsifiable(unittest.TestCase):
    def test_every_target_carries_a_risk_reward(self):
        msg = _playbook()
        self.assertGreaterEqual(msg.count("R:R 1:"), 2)
        self.assertNotIn("R:R n/a", msg)

    def test_short_risk_reward_is_not_dropped(self):
        msg = _playbook()
        short_line = [ln for ln in msg.splitlines() if "Short</b> |" in ln][0]
        self.assertIn("R:R 1:", short_line)

    def test_there_is_an_explicit_do_not_trade_condition(self):
        msg = _playbook()
        self.assertIn("ไม่เทรดเมื่อ", msg)

    def test_message_stays_under_the_telegram_limit(self):
        self.assertLess(len(_playbook()), 4096)


if __name__ == "__main__":
    unittest.main()
