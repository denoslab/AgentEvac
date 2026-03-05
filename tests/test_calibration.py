"""Unit tests for agentevac.analysis.calibration."""

import pytest

from agentevac.analysis.calibration import score_run_against_reference


def _ref():
    return {
        "departure_time_variance_s2": 100.0,
        "route_choice_entropy": 1.0,
        "decision_instability": 0.2,
        "average_hazard_exposure": 0.3,
        "average_travel_time_s": 300.0,
    }


def _run(overrides=None):
    base = dict(_ref())
    if overrides:
        base.update(overrides)
    return base


class TestScoreRunAgainstReference:
    def test_perfect_match_gives_fit_score_one(self):
        result = score_run_against_reference(_run(), _ref())
        assert result["fit_score"] == pytest.approx(1.0, rel=1e-6)
        assert result["normalized_loss"] == pytest.approx(0.0, abs=1e-9)

    def test_worse_run_lower_fit_score(self):
        bad_run = _run({"departure_time_variance_s2": 1000.0})
        perfect = score_run_against_reference(_run(), _ref())
        worse = score_run_against_reference(bad_run, _ref())
        assert worse["fit_score"] < perfect["fit_score"]

    def test_fit_score_bounded(self):
        extreme_run = _run({
            "departure_time_variance_s2": 1e9,
            "average_travel_time_s": 1e9,
        })
        result = score_run_against_reference(extreme_run, _ref())
        assert 0.0 <= result["fit_score"] <= 1.0
