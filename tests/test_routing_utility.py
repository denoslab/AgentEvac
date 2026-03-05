"""Unit tests for agentevac.agents.routing_utility."""

import pytest

from agentevac.agents.routing_utility import (
    annotate_menu_with_expected_utility,
    score_destination_utility,
    score_route_utility,
)


def _neutral_belief():
    return {"p_safe": 1 / 3, "p_risky": 1 / 3, "p_danger": 1 / 3}


def _safe_belief():
    return {"p_safe": 0.9, "p_risky": 0.05, "p_danger": 0.05}


def _danger_belief():
    return {"p_safe": 0.05, "p_risky": 0.05, "p_danger": 0.9}


def _psychology(confidence=0.8, perceived_risk=0.1):
    return {"confidence": confidence, "perceived_risk": perceived_risk}


def _profile(lambda_e=1.0, lambda_t=0.1):
    return {"lambda_e": lambda_e, "lambda_t": lambda_t}


def _menu_item(
    risk_sum=0.0,
    blocked_edges=0,
    min_margin_m=None,
    travel_time_s=300.0,
    reachable=True,
):
    return {
        "risk_sum": risk_sum,
        "blocked_edges": blocked_edges,
        "min_margin_m": min_margin_m,
        "travel_time_s_fastest_path": travel_time_s,
        "reachable": reachable,
    }


class TestScoreDestinationUtility:
    def test_returns_negative_float(self):
        score = score_destination_utility(
            _menu_item(), _neutral_belief(), _psychology(), _profile()
        )
        assert isinstance(score, float)
        assert score <= 0.0

    def test_higher_risk_sum_lowers_score(self):
        low_risk = score_destination_utility(
            _menu_item(risk_sum=0.1), _danger_belief(), _psychology(), _profile()
        )
        high_risk = score_destination_utility(
            _menu_item(risk_sum=5.0), _danger_belief(), _psychology(), _profile()
        )
        assert high_risk < low_risk

    def test_blocked_edges_heavily_penalise(self):
        no_block = score_destination_utility(
            _menu_item(blocked_edges=0), _neutral_belief(), _psychology(), _profile()
        )
        blocked = score_destination_utility(
            _menu_item(blocked_edges=2), _neutral_belief(), _psychology(), _profile()
        )
        assert blocked < no_block

    def test_higher_travel_time_lowers_score(self):
        fast = score_destination_utility(
            _menu_item(travel_time_s=60.0), _neutral_belief(), _psychology(), _profile()
        )
        slow = score_destination_utility(
            _menu_item(travel_time_s=600.0), _neutral_belief(), _psychology(), _profile()
        )
        assert slow < fast

    def test_lambda_e_zero_ignores_exposure(self):
        # With lambda_e=0, only travel cost matters; adding risk_sum should not change score.
        item_low = _menu_item(risk_sum=0.0, travel_time_s=300.0)
        item_high = _menu_item(risk_sum=10.0, travel_time_s=300.0)
        s_low = score_destination_utility(item_low, _neutral_belief(), _psychology(), _profile(lambda_e=0.0))
        s_high = score_destination_utility(item_high, _neutral_belief(), _psychology(), _profile(lambda_e=0.0))
        # Scores will still differ due to margin_penalty and uncertainty_penalty,
        # but blocked_edges=0 so risk_sum contribution is zero.
        # Actually min_margin_m=None → margin_penalty=0.25 both times → equal from that term.
        assert s_low == pytest.approx(s_high, rel=1e-6)

    def test_danger_belief_scores_worse_than_safe_belief(self):
        item = _menu_item(risk_sum=1.0)
        safe_score = score_destination_utility(item, _safe_belief(), _psychology(), _profile())
        danger_score = score_destination_utility(item, _danger_belief(), _psychology(), _profile())
        assert danger_score < safe_score


class TestScoreRouteUtility:
    def test_same_formula_as_destination(self):
        item = _menu_item(risk_sum=0.5, blocked_edges=1, travel_time_s=120.0)
        dest_score = score_destination_utility(item, _neutral_belief(), _psychology(), _profile())
        route_score = score_route_utility(item, _neutral_belief(), _psychology(), _profile())
        assert dest_score == pytest.approx(route_score, rel=1e-9)


class TestAnnotateMenuWithExpectedUtility:
    def _make_menu(self, n=3, reachable=True):
        return [
            {
                "idx": i,
                "name": f"shelter_{i}",
                "risk_sum": float(i),
                "blocked_edges": 0,
                "min_margin_m": 500.0,
                "travel_time_s_fastest_path": 300.0,
                "reachable": reachable,
            }
            for i in range(n)
        ]

    def test_each_item_gains_expected_utility_key(self):
        menu = self._make_menu(3)
        annotate_menu_with_expected_utility(
            menu, mode="destination", belief=_neutral_belief(),
            psychology=_psychology(), profile=_profile()
        )
        for item in menu:
            assert "expected_utility" in item

    def test_each_item_gains_utility_components_key(self):
        menu = self._make_menu(3)
        annotate_menu_with_expected_utility(
            menu, mode="destination", belief=_neutral_belief(),
            psychology=_psychology(), profile=_profile()
        )
        for item in menu:
            assert "utility_components" in item
            assert "expected_exposure" in item["utility_components"]

    def test_unreachable_destination_gets_none_utility(self):
        menu = [{"idx": 0, "name": "far", "reachable": False}]
        annotate_menu_with_expected_utility(
            menu, mode="destination", belief=_neutral_belief(),
            psychology=_psychology(), profile=_profile()
        )
        assert menu[0]["expected_utility"] is None
        assert menu[0]["utility_components"]["reachable"] is False

    def test_route_mode_scores_all_items(self):
        menu = [
            {"idx": 0, "name": "r0", "risk_sum": 0.0, "blocked_edges": 0,
             "min_margin_m": None, "travel_time_s_fastest_path": 120.0},
        ]
        annotate_menu_with_expected_utility(
            menu, mode="route", belief=_neutral_belief(),
            psychology=_psychology(), profile=_profile()
        )
        assert menu[0]["expected_utility"] is not None

    def test_higher_risk_gets_lower_utility(self):
        menu = [
            {"idx": 0, "name": "safe", "risk_sum": 0.0, "blocked_edges": 0,
             "min_margin_m": 1000.0, "travel_time_s_fastest_path": 300.0, "reachable": True},
            {"idx": 1, "name": "risky", "risk_sum": 5.0, "blocked_edges": 2,
             "min_margin_m": 10.0, "travel_time_s_fastest_path": 300.0, "reachable": True},
        ]
        annotate_menu_with_expected_utility(
            menu, mode="destination", belief=_danger_belief(),
            psychology=_psychology(confidence=0.1), profile=_profile()
        )
        assert menu[0]["expected_utility"] > menu[1]["expected_utility"]
