"""Unit tests for agentevac.agents.belief_model."""

import math

import pytest

from agentevac.agents.belief_model import (
    bucket_uncertainty,
    categorize_hazard_state,
    compute_belief_entropy,
    compute_signal_conflict,
    fuse_env_and_social_beliefs,
    normalize_entropy,
    smooth_belief,
    update_agent_belief,
)


class TestCategorizeHazardState:
    def test_safe_margin_gives_high_p_safe(self):
        result = categorize_hazard_state({"observed_margin_m": 8000.0})
        assert result["p_safe"] > result["p_danger"]

    def test_danger_margin_gives_high_p_danger(self):
        result = categorize_hazard_state({"observed_margin_m": 0.0})
        assert result["p_danger"] > result["p_safe"]

    def test_probabilities_sum_to_one(self):
        for m in [0.0, 50.0, 200.0, 500.0, 1000.0]:
            r = categorize_hazard_state({"observed_margin_m": m})
            total = r["p_safe"] + r["p_risky"] + r["p_danger"]
            assert abs(total - 1.0) < 1e-9, f"margin={m}: sum={total}"

    def test_falls_back_to_base_margin_m(self):
        result = categorize_hazard_state({"base_margin_m": 8000.0})
        assert result["p_safe"] > result["p_danger"]

    def test_missing_margin_returns_uniform(self):
        result = categorize_hazard_state({})
        for key in ("p_safe", "p_risky", "p_danger"):
            assert abs(result[key] - 1.0 / 3.0) < 1e-9

    def test_closer_margin_implies_higher_danger(self):
        close = categorize_hazard_state({"observed_margin_m": 50.0})
        far = categorize_hazard_state({"observed_margin_m": 8000.0})
        assert close["p_danger"] > far["p_danger"]


class TestFuseBeliefs:
    def test_zero_trust_uses_env_belief(self):
        env = {"p_safe": 0.8, "p_risky": 0.1, "p_danger": 0.1}
        soc = {"p_safe": 0.2, "p_risky": 0.3, "p_danger": 0.5}
        fused = fuse_env_and_social_beliefs(env, soc, theta_trust=0.0)
        assert abs(fused["p_safe"] - env["p_safe"]) < 1e-9

    def test_full_trust_uses_social_belief(self):
        env = {"p_safe": 0.8, "p_risky": 0.1, "p_danger": 0.1}
        soc = {"p_safe": 0.2, "p_risky": 0.3, "p_danger": 0.5}
        fused = fuse_env_and_social_beliefs(env, soc, theta_trust=1.0)
        assert abs(fused["p_safe"] - soc["p_safe"]) < 1e-9

    def test_fused_probabilities_sum_to_one(self):
        env = {"p_safe": 0.6, "p_risky": 0.3, "p_danger": 0.1}
        soc = {"p_safe": 0.1, "p_risky": 0.2, "p_danger": 0.7}
        fused = fuse_env_and_social_beliefs(env, soc, theta_trust=0.4)
        total = fused["p_safe"] + fused["p_risky"] + fused["p_danger"]
        assert abs(total - 1.0) < 1e-9


class TestComputeEntropy:
    def test_uniform_max_entropy(self):
        belief = {"p_safe": 1 / 3, "p_risky": 1 / 3, "p_danger": 1 / 3}
        h = compute_belief_entropy(belief)
        assert h == pytest.approx(math.log(3), rel=1e-6)

    def test_certain_zero_entropy(self):
        belief = {"p_safe": 1.0, "p_risky": 0.0, "p_danger": 0.0}
        h = compute_belief_entropy(belief)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_entropy_is_non_negative(self):
        belief = {"p_safe": 0.7, "p_risky": 0.2, "p_danger": 0.1}
        assert compute_belief_entropy(belief) >= 0.0


class TestNormalizeEntropy:
    def test_zero_entropy_maps_to_zero(self):
        assert normalize_entropy(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_max_entropy_maps_to_one(self):
        assert normalize_entropy(math.log(3)) == pytest.approx(1.0, rel=1e-6)

    def test_values_bounded_zero_to_one(self):
        for raw in [0.0, 0.5, math.log(3)]:
            norm = normalize_entropy(raw)
            assert 0.0 <= norm <= 1.0


class TestBucketUncertainty:
    def test_low_entropy_norm_gives_low_bucket(self):
        assert bucket_uncertainty(0.1) == "Low"

    def test_medium_entropy_norm_gives_medium_bucket(self):
        assert bucket_uncertainty(0.5) == "Medium"

    def test_high_entropy_norm_gives_high_bucket(self):
        assert bucket_uncertainty(1.0) == "High"

    def test_uniform_entropy_maps_to_high(self):
        entropy_norm = normalize_entropy(math.log(3))
        assert bucket_uncertainty(entropy_norm) == "High"

    def test_boundary_at_0_33_is_low(self):
        assert bucket_uncertainty(0.33) == "Low"

    def test_just_above_0_33_is_medium(self):
        assert bucket_uncertainty(0.34) == "Medium"


class TestSmoothBelief:
    def test_zero_inertia_returns_new_belief(self):
        prev = {"p_safe": 1.0, "p_risky": 0.0, "p_danger": 0.0}
        new = {"p_safe": 0.0, "p_risky": 0.0, "p_danger": 1.0}
        result = smooth_belief(prev, new, inertia=0.0)
        assert abs(result["p_danger"] - 1.0) < 1e-9

    def test_high_inertia_stays_close_to_prev(self):
        prev = {"p_safe": 1.0, "p_risky": 0.0, "p_danger": 0.0}
        new = {"p_safe": 0.0, "p_risky": 0.0, "p_danger": 1.0}
        result = smooth_belief(prev, new, inertia=0.9)
        assert result["p_safe"] > 0.5

    def test_output_sums_to_one(self):
        prev = {"p_safe": 0.8, "p_risky": 0.1, "p_danger": 0.1}
        new = {"p_safe": 0.1, "p_risky": 0.2, "p_danger": 0.7}
        result = smooth_belief(prev, new)
        total = result["p_safe"] + result["p_risky"] + result["p_danger"]
        assert abs(total - 1.0) < 1e-9

    def test_default_inertia_blends_beliefs(self):
        prev = {"p_safe": 1.0, "p_risky": 0.0, "p_danger": 0.0}
        new = {"p_safe": 0.0, "p_risky": 0.0, "p_danger": 1.0}
        result = smooth_belief(prev, new)  # default inertia=0.35
        # Neither fully prev nor fully new.
        assert 0.0 < result["p_safe"] < 1.0
        assert 0.0 < result["p_danger"] < 1.0


class TestUpdateAgentBelief:
    def _prev_belief(self):
        return {"p_safe": 1 / 3, "p_risky": 1 / 3, "p_danger": 1 / 3}

    def _safe_env(self):
        return {"observed_margin_m": 8000.0}

    def _danger_env(self):
        return {"observed_margin_m": 0.0}

    def _no_messages(self):
        return {"message_count": 0, "social_belief": {}}

    def _danger_messages(self):
        return {
            "message_count": 3,
            "social_belief": {"p_safe": 0.05, "p_risky": 0.05, "p_danger": 0.90},
        }

    def test_returns_all_required_keys(self):
        result = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._no_messages(), theta_trust=0.5
        )
        for key in ("p_safe", "p_risky", "p_danger", "entropy", "entropy_norm",
                    "uncertainty_bucket", "env_weight", "social_weight"):
            assert key in result

    def test_probabilities_sum_to_one(self):
        result = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._no_messages(), theta_trust=0.5
        )
        total = result["p_safe"] + result["p_risky"] + result["p_danger"]
        assert abs(total - 1.0) < 1e-9

    def test_no_messages_uses_zero_social_weight(self):
        result = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._no_messages(), theta_trust=0.8
        )
        assert result["social_weight"] == pytest.approx(0.0)
        assert result["env_weight"] == pytest.approx(1.0)

    def test_danger_env_increases_p_danger(self):
        safe_result = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._no_messages(), theta_trust=0.5
        )
        danger_result = update_agent_belief(
            self._prev_belief(), self._danger_env(), self._no_messages(), theta_trust=0.5
        )
        assert danger_result["p_danger"] > safe_result["p_danger"]

    def test_danger_messages_with_high_trust_increase_p_danger(self):
        no_msg = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._no_messages(), theta_trust=0.9
        )
        with_msg = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._danger_messages(), theta_trust=0.9
        )
        assert with_msg["p_danger"] > no_msg["p_danger"]

    def test_uncertainty_bucket_is_valid_label(self):
        result = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._no_messages(), theta_trust=0.5
        )
        assert result["uncertainty_bucket"] in ("Low", "Medium", "High")

    def test_signal_conflict_present_in_result(self):
        result = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._no_messages(), theta_trust=0.5
        )
        assert "signal_conflict" in result

    def test_no_messages_gives_zero_conflict(self):
        result = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._no_messages(), theta_trust=0.5
        )
        assert result["signal_conflict"] == pytest.approx(0.0)

    def test_conflicting_sources_give_high_conflict(self):
        # env says safe (margin 900m), social says danger
        result = update_agent_belief(
            self._prev_belief(), self._safe_env(), self._danger_messages(), theta_trust=0.5
        )
        assert result["signal_conflict"] > 0.3


class TestComputeSignalConflict:
    def test_identical_beliefs_give_zero(self):
        b = {"p_safe": 0.8, "p_risky": 0.15, "p_danger": 0.05}
        assert compute_signal_conflict(b, b) == pytest.approx(0.0, abs=1e-6)

    def test_maximally_opposed_gives_near_one(self):
        env = {"p_safe": 0.98, "p_risky": 0.01, "p_danger": 0.01}
        soc = {"p_safe": 0.01, "p_risky": 0.01, "p_danger": 0.98}
        assert compute_signal_conflict(env, soc) > 0.85

    def test_moderate_disagreement(self):
        env = {"p_safe": 0.75, "p_risky": 0.20, "p_danger": 0.05}
        soc = {"p_safe": 0.10, "p_risky": 0.30, "p_danger": 0.60}
        conflict = compute_signal_conflict(env, soc)
        assert 0.2 < conflict < 0.7

    def test_symmetry(self):
        env = {"p_safe": 0.9, "p_risky": 0.05, "p_danger": 0.05}
        soc = {"p_safe": 0.1, "p_risky": 0.1, "p_danger": 0.8}
        assert compute_signal_conflict(env, soc) == pytest.approx(
            compute_signal_conflict(soc, env), abs=1e-9
        )

    def test_result_bounded_zero_to_one(self):
        for env, soc in [
            ({"p_safe": 1.0, "p_risky": 0.0, "p_danger": 0.0},
             {"p_safe": 0.0, "p_risky": 0.0, "p_danger": 1.0}),
            ({"p_safe": 0.5, "p_risky": 0.3, "p_danger": 0.2},
             {"p_safe": 0.5, "p_risky": 0.3, "p_danger": 0.2}),
        ]:
            c = compute_signal_conflict(env, soc)
            assert 0.0 <= c <= 1.0
