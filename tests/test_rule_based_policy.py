"""Unit tests for agentevac.agents.rule_based_policy."""

import random
from unittest.mock import patch

import pytest

from agentevac.agents.agent_state import AgentRuntimeState
from agentevac.agents.rule_based_policy import (
    _softmax_sample,
    rule_based_destination_choice,
    rule_based_predeparture,
    rule_based_routing_choice,
)


# ---------------------------------------------------------------------------
# _softmax_sample
# ---------------------------------------------------------------------------

class TestSoftmaxSample:
    def test_argmax_at_zero_tau(self):
        """tau=0 should deterministically pick the highest-utility option."""
        utilities = [-5.0, -1.0, -3.0]
        rng = random.Random(42)
        for _ in range(20):
            assert _softmax_sample(utilities, 0.0, rng) == 1

    def test_argmax_at_negative_tau(self):
        utilities = [-5.0, -1.0, -3.0]
        rng = random.Random(42)
        assert _softmax_sample(utilities, -1.0, rng) == 1

    def test_uniform_at_high_tau(self):
        """Very high tau should produce near-uniform distribution."""
        utilities = [-5.0, -1.0, -3.0]
        rng = random.Random(0)
        counts = [0, 0, 0]
        n = 3000
        for _ in range(n):
            idx = _softmax_sample(utilities, 1000.0, rng)
            counts[idx] += 1
        # Each option should get roughly 1/3 of the samples.
        for c in counts:
            assert c > n * 0.25, f"Expected near-uniform, got counts={counts}"

    def test_prefers_higher_utility(self):
        """Higher-utility option should be selected more often at moderate tau."""
        utilities = [-10.0, -1.0, -8.0]
        rng = random.Random(7)
        counts = [0, 0, 0]
        n = 2000
        for _ in range(n):
            idx = _softmax_sample(utilities, 1.0, rng)
            counts[idx] += 1
        assert counts[1] > counts[0]
        assert counts[1] > counts[2]

    def test_single_option(self):
        assert _softmax_sample([42.0], 1.0, random.Random(0)) == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _softmax_sample([], 1.0, random.Random(0))


# ---------------------------------------------------------------------------
# rule_based_predeparture
# ---------------------------------------------------------------------------

class TestRuleBasedPredeparture:
    def _make_agent(self, theta_r=0.45, theta_u=0.30, gamma=0.995):
        return AgentRuntimeState(
            agent_id="v0",
            created_sim_t_s=0.0,
            last_sim_t_s=0.0,
            profile={"theta_r": theta_r, "theta_u": theta_u, "gamma": gamma,
                     "social_trigger": 0.5, "social_min_danger": 0.15},
        )

    def test_departs_when_danger_exceeds_threshold(self):
        agent = self._make_agent(theta_r=0.3)
        belief = {"p_danger": 0.5, "p_safe": 0.3}
        psychology = {"confidence": 0.5}
        result = rule_based_predeparture(agent, belief, psychology, sim_t_s=10.0)
        assert result["action"] == "depart"
        assert result["reason"] == "risk_threshold"
        assert result["situation_summary"] == "rule_based"

    def test_waits_when_safe(self):
        agent = self._make_agent(theta_r=0.8)
        belief = {"p_danger": 0.1, "p_safe": 0.8}
        psychology = {"confidence": 0.9}
        result = rule_based_predeparture(agent, belief, psychology, sim_t_s=1.0)
        assert result["action"] == "wait"
        assert result["reason"] == "wait"

    def test_delegates_to_departure_model(self):
        """Verify it actually calls should_depart_now under the hood."""
        agent = self._make_agent()
        belief = {"p_danger": 0.1, "p_safe": 0.8}
        psychology = {"confidence": 0.9}
        with patch("agentevac.agents.rule_based_policy.should_depart_now",
                    return_value=(True, "urgency_threshold")) as mock_fn:
            result = rule_based_predeparture(agent, belief, psychology, sim_t_s=100.0)
            mock_fn.assert_called_once()
        assert result["action"] == "depart"
        assert result["reason"] == "urgency_threshold"


# ---------------------------------------------------------------------------
# rule_based_destination_choice
# ---------------------------------------------------------------------------

class TestRuleBasedDestinationChoice:
    def _menu(self):
        return [
            {"idx": 0, "name": "A", "expected_utility": -5.0, "reachable": True},
            {"idx": 1, "name": "B", "expected_utility": -1.0, "reachable": True},
            {"idx": 2, "name": "C", "expected_utility": -8.0, "reachable": False},
        ]

    def test_chooses_from_reachable(self):
        menu = self._menu()
        idx, reason = rule_based_destination_choice(menu, [0, 1], 0.0, random.Random(0))
        assert idx == 1  # highest utility among reachable
        assert reason == "softmax_utility"

    def test_no_reachable_returns_neg1(self):
        menu = self._menu()
        idx, reason = rule_based_destination_choice(menu, [], 1.0, random.Random(0))
        assert idx == -1
        assert reason == "no_reachable"

    def test_excludes_unreachable(self):
        menu = self._menu()
        # Only idx=0 is reachable in this call.
        idx, reason = rule_based_destination_choice(menu, [0], 0.0, random.Random(0))
        assert idx == 0

    def test_deterministic_with_seed(self):
        menu = self._menu()
        results = []
        for _ in range(10):
            idx, _ = rule_based_destination_choice(menu, [0, 1], 1.0, random.Random(999))
            results.append(idx)
        assert len(set(results)) == 1  # all identical with same seed


# ---------------------------------------------------------------------------
# rule_based_routing_choice
# ---------------------------------------------------------------------------

class TestRuleBasedRoutingChoice:
    def _menu(self):
        return [
            {"idx": 0, "name": "A", "expected_utility": -5.0, "reachable": True},
            {"idx": 1, "name": "B", "expected_utility": -1.0, "reachable": True},
            {"idx": 2, "name": "C", "expected_utility": -3.0, "reachable": True},
        ]

    def test_keep_when_same_as_current(self):
        """When softmax picks the same as current → KEEP."""
        menu = self._menu()
        # tau=0 → argmax → idx=1.  current_choice_idx=1 → KEEP.
        idx, reason = rule_based_routing_choice(menu, [0, 1, 2], 1, 0.0, random.Random(0))
        assert idx == -1
        assert reason == "softmax_keep"

    def test_returns_new_choice_when_different(self):
        menu = self._menu()
        # tau=0 → argmax → idx=1.  current_choice_idx=0 → new choice.
        idx, reason = rule_based_routing_choice(menu, [0, 1, 2], 0, 0.0, random.Random(0))
        assert idx == 1
        assert reason == "softmax_utility"

    def test_no_reachable(self):
        menu = self._menu()
        idx, reason = rule_based_routing_choice(menu, [], None, 1.0, random.Random(0))
        assert idx == -1
        assert reason == "no_reachable"

    def test_current_none_never_keeps(self):
        menu = self._menu()
        idx, reason = rule_based_routing_choice(menu, [0, 1, 2], None, 0.0, random.Random(0))
        assert idx == 1  # argmax
        assert reason == "softmax_utility"
