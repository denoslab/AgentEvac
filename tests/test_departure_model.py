"""Unit tests for agentevac.agents.departure_model."""

import pytest

from agentevac.agents.agent_state import AGENT_STATES, ensure_agent_state
from agentevac.agents.departure_model import should_depart_now


@pytest.fixture(autouse=True)
def clear_agent_states():
    """Wipe the global registry before and after each test to avoid cross-test pollution."""
    AGENT_STATES.clear()
    yield
    AGENT_STATES.clear()


def _belief(p_safe=0.33, p_risky=0.33, p_danger=0.33):
    return {"p_safe": p_safe, "p_risky": p_risky, "p_danger": p_danger}


def _psychology(confidence=0.9):
    return {"confidence": confidence}


class TestShouldDepartNow:
    def test_returns_two_tuple_bool_and_str(self):
        state = ensure_agent_state("v0", 0.0)
        result = should_depart_now(state, _belief(), _psychology(), sim_t_s=0.0)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_risk_threshold_triggers(self):
        state = ensure_agent_state("v1", 0.0, default_theta_r=0.5)
        belief = _belief(p_safe=0.1, p_risky=0.1, p_danger=0.8)
        departed, clause = should_depart_now(state, belief, _psychology(), sim_t_s=0.0)
        assert departed is True
        assert clause == "risk_threshold"

    def test_no_departure_when_safe(self):
        # p_safe high enough that urgency_term = 1 * p_safe > theta_u; p_danger << theta_r
        state = ensure_agent_state("v2", 0.0, default_theta_r=0.5, default_theta_u=0.30)
        belief = _belief(p_safe=0.85, p_risky=0.10, p_danger=0.05)
        departed, _ = should_depart_now(state, belief, _psychology(confidence=0.9), sim_t_s=0.0)
        assert departed is False

    def test_low_confidence_precaution(self):
        # Very uncertain agent (confidence < 0.15) with p_danger >= 0.60 * theta_r
        state = ensure_agent_state("v3", 0.0, default_theta_r=0.5)
        belief = _belief(p_safe=0.50, p_risky=0.15, p_danger=0.35)
        departed, clause = should_depart_now(state, belief, _psychology(confidence=0.10), sim_t_s=0.0)
        assert departed is True
        assert clause == "low_confidence_precaution"

    def test_urgency_threshold_triggers_when_p_safe_low(self):
        # With elapsed_s=0, urgency_term = gamma^0 * p_safe = p_safe.
        # If p_safe < theta_u, clause 2 fires (assuming clause 1 does not fire).
        state = ensure_agent_state("v4", 0.0, default_theta_r=0.5, default_theta_u=0.30, default_gamma=0.99)
        belief = _belief(p_safe=0.10, p_risky=0.40, p_danger=0.05)
        departed, clause = should_depart_now(state, belief, _psychology(confidence=0.9), sim_t_s=0.0)
        assert departed is True
        assert clause == "urgency_threshold"

    def test_wait_returned_when_no_clauses_fire(self):
        state = ensure_agent_state("v5", 0.0, default_theta_r=0.5, default_theta_u=0.30)
        belief = _belief(p_safe=0.90, p_risky=0.08, p_danger=0.02)
        departed, clause = should_depart_now(state, belief, _psychology(confidence=0.9), sim_t_s=0.0)
        assert departed is False
        assert clause == "wait"

    def test_danger_exactly_at_threshold_does_not_trigger(self):
        # p_danger == theta_r is NOT > theta_r, so clause 1 does not fire.
        state = ensure_agent_state("v6", 0.0, default_theta_r=0.5)
        belief = _belief(p_safe=0.8, p_risky=0.1, p_danger=0.5)
        departed, clause = should_depart_now(state, belief, _psychology(confidence=0.9), sim_t_s=0.0)
        # Clause 1 does not fire (not strictly greater); outcome may be wait or urgency clause.
        assert clause != "risk_threshold"

    def test_profile_theta_r_respected(self):
        # Agent with theta_r=0.9 should not trigger on p_danger=0.6.
        state = ensure_agent_state("v7", 0.0, default_theta_r=0.9)
        belief = _belief(p_safe=0.35, p_risky=0.05, p_danger=0.60)
        departed, clause = should_depart_now(state, belief, _psychology(confidence=0.9), sim_t_s=0.0)
        assert clause != "risk_threshold"
