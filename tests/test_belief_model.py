"""Unit tests for agentevac.agents.belief_model."""

import math

import pytest

from agentevac.agents.belief_model import (
    bucket_uncertainty,
    categorize_hazard_state,
    compute_belief_entropy,
    fuse_env_and_social_beliefs,
    smooth_belief,
    update_agent_belief,
)


class TestCategorizeHazardState:
    def test_safe_margin(self):
        result = categorize_hazard_state(margin_m=800.0)
        assert result["p_safe"] > result["p_danger"]

    def test_danger_margin(self):
        result = categorize_hazard_state(margin_m=0.0)
        assert result["p_danger"] > result["p_safe"]

    def test_probabilities_sum_to_one(self):
        for m in [0.0, 50.0, 200.0, 500.0, 1000.0]:
            r = categorize_hazard_state(m)
            total = r["p_safe"] + r["p_risky"] + r["p_danger"]
            assert abs(total - 1.0) < 1e-9, f"margin={m}: sum={total}"


class TestFuseBeliefs:
    def test_weights_sum_effect(self):
        env = {"p_safe": 0.8, "p_risky": 0.1, "p_danger": 0.1}
        soc = {"p_safe": 0.2, "p_risky": 0.3, "p_danger": 0.5}
        fused = fuse_env_and_social_beliefs(env, soc, theta_trust=0.0)
        # zero trust → fused == env
        assert abs(fused["p_safe"] - env["p_safe"]) < 1e-9

    def test_full_trust(self):
        env = {"p_safe": 0.8, "p_risky": 0.1, "p_danger": 0.1}
        soc = {"p_safe": 0.2, "p_risky": 0.3, "p_danger": 0.5}
        fused = fuse_env_and_social_beliefs(env, soc, theta_trust=1.0)
        assert abs(fused["p_safe"] - soc["p_safe"]) < 1e-9


class TestComputeEntropy:
    def test_uniform_max_entropy(self):
        belief = {"p_safe": 1 / 3, "p_risky": 1 / 3, "p_danger": 1 / 3}
        h = compute_belief_entropy(belief)
        assert h == pytest.approx(math.log(3), rel=1e-6)

    def test_certain_zero_entropy(self):
        belief = {"p_safe": 1.0, "p_risky": 0.0, "p_danger": 0.0}
        h = compute_belief_entropy(belief)
        assert h == pytest.approx(0.0, abs=1e-9)


class TestBucketUncertainty:
    def test_low_entropy_bucket(self):
        assert bucket_uncertainty(0.1) == "low"

    def test_high_entropy_bucket(self):
        assert bucket_uncertainty(math.log(3)) == "high"
