"""Unit tests for agentevac.agents.departure_model."""

import pytest

from agentevac.agents.departure_model import should_depart_now


def _make_state(
    p_safe=0.8,
    p_risky=0.1,
    p_danger=0.1,
    theta_r=0.5,
    theta_u=0.1,
    gamma=0.99,
    confidence=0.9,
    elapsed_s=0.0,
):
    """Build a minimal agent-state-like dict for testing."""
    return {
        "belief": {"p_safe": p_safe, "p_risky": p_risky, "p_danger": p_danger},
        "profile": {
            "theta_r": theta_r,
            "theta_u": theta_u,
            "gamma": gamma,
        },
        "confidence": confidence,
        "elapsed_s": elapsed_s,
    }


class TestShouldDepartNow:
    def test_risk_threshold_triggers(self):
        state = _make_state(p_danger=0.8, theta_r=0.5)
        departed, clause, _ = should_depart_now(state)
        assert departed is True
        assert clause == "risk_threshold"

    def test_no_departure_when_safe(self):
        state = _make_state(p_danger=0.05, theta_r=0.5, elapsed_s=1.0)
        departed, _, _ = should_depart_now(state)
        assert departed is False

    def test_low_confidence_precaution(self):
        # Very uncertain agent with moderate danger should trigger precaution.
        state = _make_state(
            p_danger=0.35,
            theta_r=0.5,
            confidence=0.10,
        )
        departed, clause, _ = should_depart_now(state)
        assert departed is True
        assert clause == "low_confidence_precaution"
