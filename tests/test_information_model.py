"""Unit tests for agentevac.agents.information_model."""

import random

import pytest

from agentevac.agents.information_model import (
    apply_signal_delay,
    build_social_signal,
    inject_signal_noise,
    sample_environment_signal,
)


class TestInjectSignalNoise:
    def _base_signal(self, margin=500.0):
        return {"base_margin_m": margin}

    def test_zero_sigma_returns_unchanged_margin(self):
        sig = inject_signal_noise(self._base_signal(500.0), sigma_info=0.0)
        assert sig["observed_margin_m"] == pytest.approx(500.0)
        assert sig["noise_delta_m"] == pytest.approx(0.0)

    def test_nonzero_sigma_changes_margin_with_seeded_rng(self):
        rng = random.Random(42)
        sig = inject_signal_noise(self._base_signal(500.0), sigma_info=50.0, rng=rng)
        # With seed 42 the noise won't be exactly 0.
        assert sig["observed_margin_m"] != pytest.approx(500.0)

    def test_deterministic_with_same_seed(self):
        rng1 = random.Random(7)
        rng2 = random.Random(7)
        s1 = inject_signal_noise(self._base_signal(300.0), sigma_info=30.0, rng=rng1)
        s2 = inject_signal_noise(self._base_signal(300.0), sigma_info=30.0, rng=rng2)
        assert s1["observed_margin_m"] == pytest.approx(s2["observed_margin_m"])

    def test_none_base_margin_produces_none_observed(self):
        sig = inject_signal_noise({"base_margin_m": None}, sigma_info=20.0)
        assert sig["observed_margin_m"] is None
        assert sig["observed_state"] == "unknown"

    def test_observed_state_danger_when_margin_small(self):
        rng = random.Random(0)
        sig = inject_signal_noise({"base_margin_m": 50.0}, sigma_info=0.0, rng=rng)
        assert sig["observed_state"] == "danger"

    def test_observed_state_safe_when_margin_large(self):
        sig = inject_signal_noise({"base_margin_m": 1000.0}, sigma_info=0.0)
        assert sig["observed_state"] == "safe"

    def test_output_is_shallow_copy(self):
        original = {"base_margin_m": 100.0, "extra": "keep"}
        sig = inject_signal_noise(original, sigma_info=0.0)
        assert sig["extra"] == "keep"
        # Modifying the copy does not affect original.
        sig["extra"] = "changed"
        assert original["extra"] == "keep"


class TestApplySignalDelay:
    def _make_signal(self, round_n):
        return {"decision_round": round_n, "observed_margin_m": float(round_n * 10)}

    def test_zero_delay_returns_current_signal(self):
        current = self._make_signal(5)
        out = apply_signal_delay(current, history=[], delay_rounds=0)
        assert out["decision_round"] == 5
        assert out["is_delayed"] is False
        assert out["delay_rounds_applied"] == 0

    def test_delay_within_history_returns_stale_signal(self):
        history = [self._make_signal(1), self._make_signal(2), self._make_signal(3)]
        current = self._make_signal(4)
        out = apply_signal_delay(current, history=history, delay_rounds=2)
        assert out["is_delayed"] is True
        assert out["decision_round"] == 2

    def test_delay_exceeds_history_returns_current(self):
        history = [self._make_signal(1)]
        current = self._make_signal(5)
        out = apply_signal_delay(current, history=history, delay_rounds=10)
        assert out["is_delayed"] is False
        assert out["decision_round"] == 5

    def test_delay_of_one_returns_most_recent_history(self):
        history = [self._make_signal(3), self._make_signal(4)]
        current = self._make_signal(5)
        out = apply_signal_delay(current, history=history, delay_rounds=1)
        assert out["is_delayed"] is True
        assert out["decision_round"] == 4


class TestSampleEnvironmentSignal:
    def test_returns_required_keys(self):
        sig = sample_environment_signal(
            agent_id="v1",
            sim_t_s=10.0,
            current_edge="edge_a",
            current_edge_margin_m=500.0,
            route_head_min_margin_m=None,
            decision_round=1,
            sigma_info=0.0,
        )
        for key in ("agent_id", "base_margin_m", "observed_margin_m", "observed_state"):
            assert key in sig

    def test_prefers_route_head_margin_over_current_edge(self):
        sig = sample_environment_signal(
            agent_id="v1",
            sim_t_s=0.0,
            current_edge="e",
            current_edge_margin_m=1000.0,
            route_head_min_margin_m=50.0,
            decision_round=1,
            sigma_info=0.0,
        )
        assert sig["base_margin_m"] == pytest.approx(50.0)
        assert sig["source_metric"] == "route_head_min_margin_m"

    def test_falls_back_to_current_edge_when_route_head_none(self):
        sig = sample_environment_signal(
            agent_id="v1",
            sim_t_s=0.0,
            current_edge="e",
            current_edge_margin_m=250.0,
            route_head_min_margin_m=None,
            decision_round=1,
            sigma_info=0.0,
        )
        assert sig["base_margin_m"] == pytest.approx(250.0)
        assert sig["source_metric"] == "current_edge_margin_m"

    def test_none_margins_produces_unknown_state(self):
        sig = sample_environment_signal(
            agent_id="v1",
            sim_t_s=0.0,
            current_edge="e",
            current_edge_margin_m=None,
            route_head_min_margin_m=None,
            decision_round=1,
            sigma_info=0.0,
        )
        assert sig["observed_state"] == "unknown"
        assert sig["observed_margin_m"] is None


class TestBuildSocialSignal:
    def test_empty_inbox_returns_uniform_prior(self):
        sig = build_social_signal("v1", inbox=[])
        b = sig["social_belief"]
        assert abs(b["p_safe"] - 1 / 3) < 1e-9
        assert sig["dominant_state"] == "none"
        assert sig["message_count"] == 0

    def test_danger_keyword_detected(self):
        inbox = [{"message": "There is fire on main street"}]
        sig = build_social_signal("v1", inbox=inbox)
        assert sig["votes"]["danger"] == 1
        assert sig["dominant_state"] == "danger"
        assert sig["social_belief"]["p_danger"] == pytest.approx(1.0)

    def test_safe_keyword_detected(self):
        inbox = [{"message": "The road is clear and open"}]
        sig = build_social_signal("v1", inbox=inbox)
        assert sig["votes"]["safe"] == 1
        assert sig["dominant_state"] == "safe"

    def test_risky_keyword_detected(self):
        inbox = [{"message": "heavy traffic and crowded"}]
        sig = build_social_signal("v1", inbox=inbox)
        assert sig["votes"]["risky"] == 1

    def test_danger_takes_priority_over_safe(self):
        inbox = [{"message": "fire but clear road"}]
        sig = build_social_signal("v1", inbox=inbox)
        assert sig["votes"]["danger"] == 1
        assert sig["votes"]["safe"] == 0

    def test_max_messages_caps_inbox(self):
        inbox = [{"message": "fire"} for _ in range(20)]
        sig = build_social_signal("v1", inbox=inbox, max_messages=3)
        assert sig["message_count"] == 3
        assert sig["votes"]["danger"] == 3

    def test_mixed_inbox_normalizes_to_one(self):
        inbox = [
            {"message": "fire near here"},
            {"message": "road is clear"},
            {"message": "some traffic"},
        ]
        sig = build_social_signal("v1", inbox=inbox)
        b = sig["social_belief"]
        total = b["p_safe"] + b["p_risky"] + b["p_danger"]
        assert abs(total - 1.0) < 1e-9

    def test_unmatched_messages_ignored(self):
        inbox = [{"message": "hello world"}, {"message": "weather is nice"}]
        sig = build_social_signal("v1", inbox=inbox)
        assert sig["dominant_state"] == "none"
        assert sig["votes"] == {"safe": 0, "risky": 0, "danger": 0}
