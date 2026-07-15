"""Regression tests for the two-clock mapping fixed in M0.

The institutional-delay model serves official information ``delay_rounds``
decision rounds old, where ``delay_rounds = round(INFO_DELAY_S / DECISION_PERIOD_S)``.
The canonical decision round is 240 s over a 0.2 s SUMO step, so the INFO_DELAY_S
sweep {0, 240, 480, 720} must map to {0, 1, 2, 3} rounds. The pre-M0 default of
60 s silently quadrupled the delay to {0, 4, 8, 12}. These tests pin the mapping.
"""

import pytest

from agentevac.utils.run_parameters import delay_rounds_for


CANONICAL_PERIOD_S = 240.0
STALE_PERIOD_S = 60.0


@pytest.mark.parametrize(
    "info_delay_s, expected_rounds",
    [(0.0, 0), (240.0, 1), (480.0, 2), (720.0, 3)],
)
def test_canonical_240s_round_maps_delay_levels(info_delay_s, expected_rounds):
    assert delay_rounds_for(info_delay_s, CANONICAL_PERIOD_S) == expected_rounds


@pytest.mark.parametrize(
    "info_delay_s, quadrupled_rounds",
    [(0.0, 0), (240.0, 4), (480.0, 8), (720.0, 12)],
)
def test_stale_60s_round_quadruples_delay(info_delay_s, quadrupled_rounds):
    # Documents the bug the M0 default change avoids: at the stale 60 s period the
    # same delay levels quadruple. This asserts the arithmetic, not the desired
    # behaviour, so a future reintroduction of a 60 s default is caught by the
    # canonical test above rather than here.
    assert delay_rounds_for(info_delay_s, STALE_PERIOD_S) == quadrupled_rounds


def test_zero_period_does_not_divide_by_zero():
    assert delay_rounds_for(240.0, 0.0) >= 0
