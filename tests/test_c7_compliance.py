"""Tests for the C.7 belief-only compliance channel and theta_auth sampling.

The institutional channel blends an active evacuation order into the fused belief with
weight ``a = order_weight`` (which the caller sets to theta_auth * channel_factor).  It is
inert at weight 0, so every non-order update is bit-identical to the legacy pipeline.
"""

import pytest

from agentevac.agents.belief_model import update_agent_belief
from agentevac.agents.agent_state import (
    sample_profile_params,
    ensure_agent_state,
    AGENT_STATES,
)


ORDER = {"p_safe": 0.05, "p_risky": 0.15, "p_danger": 0.80}
SAFE_ENV = {"observed_margin_m": 8000.0}
NO_MSG = {"message_count": 0, "social_belief": {}}
PREV = {"p_safe": 1 / 3, "p_risky": 1 / 3, "p_danger": 1 / 3}


def test_order_weight_zero_is_identical_to_no_order():
    base = update_agent_belief(PREV, SAFE_ENV, NO_MSG, theta_trust=0.5)
    with_zero = update_agent_belief(
        PREV, SAFE_ENV, NO_MSG, theta_trust=0.5, order_weight=0.0, order_belief=ORDER
    )
    for k in ("p_safe", "p_risky", "p_danger"):
        assert with_zero[k] == pytest.approx(base[k])


def test_active_order_raises_p_danger():
    base = update_agent_belief(PREV, SAFE_ENV, NO_MSG, theta_trust=0.5)
    ordered = update_agent_belief(
        PREV, SAFE_ENV, NO_MSG, theta_trust=0.5, order_weight=0.6, order_belief=ORDER
    )
    assert ordered["p_danger"] > base["p_danger"]


def test_full_weight_no_inertia_pulls_to_order():
    # a = 1 with no temporal smoothing means the belief equals the order triplet.
    ordered = update_agent_belief(
        PREV, SAFE_ENV, NO_MSG, theta_trust=0.5, inertia=0.0,
        order_weight=1.0, order_belief=ORDER,
    )
    for k in ("p_safe", "p_risky", "p_danger"):
        assert ordered[k] == pytest.approx(ORDER[k], abs=1e-6)


def test_p_danger_monotonic_in_order_weight():
    p = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = update_agent_belief(
            PREV, SAFE_ENV, NO_MSG, theta_trust=0.5, inertia=0.0,
            order_weight=w, order_belief=ORDER,
        )
        p.append(r["p_danger"])
    assert all(p[i] <= p[i + 1] + 1e-9 for i in range(len(p) - 1))
    assert p[-1] > p[0]


def test_probabilities_sum_to_one_under_order():
    r = update_agent_belief(
        PREV, SAFE_ENV, NO_MSG, theta_trust=0.5, order_weight=0.6, order_belief=ORDER
    )
    assert r["p_safe"] + r["p_risky"] + r["p_danger"] == pytest.approx(1.0)


def test_theta_auth_sampled_within_bounds_and_deterministic():
    means = {"theta_auth": 0.5}
    spreads = {"theta_auth": 0.2}
    bounds = {"theta_auth": (0.0, 1.0)}
    a1 = sample_profile_params("veh_7", means, spreads, bounds, master_seed=42)
    a2 = sample_profile_params("veh_7", means, spreads, bounds, master_seed=42)
    assert a1["theta_auth"] == a2["theta_auth"]     # deterministic per (seed, id)
    assert 0.0 <= a1["theta_auth"] <= 1.0           # respects bounds


def test_ensure_agent_state_has_theta_auth():
    AGENT_STATES.pop("c7_test_agent", None)
    state = ensure_agent_state("c7_test_agent", 0.0)
    assert "theta_auth" in state.profile
    assert 0.0 <= state.profile["theta_auth"] <= 1.0
    AGENT_STATES.pop("c7_test_agent", None)
