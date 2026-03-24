"""Unit tests for agentevac.utils.forecast_layer."""

import pytest

from agentevac.utils.forecast_layer import (
    build_fire_forecast,
    estimate_edge_forecast_risk,
    render_forecast_briefing,
    summarize_route_forecast,
)


def _safe_edge_risk_fn(edge_id):
    """Always-safe: not blocked, zero risk, large margin."""
    return (False, 0.0, 8000.0)


def _dangerous_edge_risk_fn(edge_id):
    """Always dangerous: blocked, high risk, negative margin."""
    return (True, 0.95, -10.0)


class TestBuildFireForecast:
    def test_no_fires_returns_zeros(self):
        result = build_fire_forecast(0.0, [], [], 300.0)
        assert result["current_fire_count"] == 0
        assert result["projected_fire_count"] == 0
        assert result["max_radius_growth_m"] == pytest.approx(0.0)
        assert result["avg_radius_growth_m"] == pytest.approx(0.0)
        assert result["new_projected_fire_ids"] == []

    def test_growing_fire_computes_growth(self):
        current = [{"id": "f1", "r": 100.0}]
        projected = [{"id": "f1", "r": 150.0}]
        result = build_fire_forecast(0.0, current, projected, 300.0)
        assert result["max_radius_growth_m"] == pytest.approx(50.0, rel=1e-6)
        assert result["avg_radius_growth_m"] == pytest.approx(50.0, rel=1e-6)

    def test_new_fire_in_projection_counted_as_growth(self):
        current = []
        projected = [{"id": "f2", "r": 80.0}]
        result = build_fire_forecast(0.0, current, projected, 300.0)
        assert "f2" in result["new_projected_fire_ids"]
        assert result["max_radius_growth_m"] == pytest.approx(80.0, rel=1e-6)

    def test_multiple_fires_picks_max_growth(self):
        current = [{"id": "f1", "r": 50.0}, {"id": "f2", "r": 50.0}]
        projected = [{"id": "f1", "r": 100.0}, {"id": "f2", "r": 200.0}]
        result = build_fire_forecast(0.0, current, projected, 300.0)
        assert result["max_radius_growth_m"] == pytest.approx(150.0, rel=1e-6)

    def test_multiple_fires_averages_growth(self):
        current = [{"id": "f1", "r": 50.0}, {"id": "f2", "r": 50.0}]
        projected = [{"id": "f1", "r": 100.0}, {"id": "f2", "r": 200.0}]
        result = build_fire_forecast(0.0, current, projected, 300.0)
        # growth: 50 and 150; avg = 100
        assert result["avg_radius_growth_m"] == pytest.approx(100.0, rel=1e-6)

    def test_generated_at_matches_sim_t(self):
        result = build_fire_forecast(42.5, [], [], 60.0)
        assert result["generated_at_s"] == pytest.approx(42.5, rel=1e-6)

    def test_horizon_stored_correctly(self):
        result = build_fire_forecast(0.0, [], [], 600.0)
        assert result["horizon_s"] == pytest.approx(600.0, rel=1e-6)

    def test_max_projected_radius_is_largest(self):
        current = [{"id": "f1", "r": 50.0}, {"id": "f2", "r": 30.0}]
        projected = [{"id": "f1", "r": 200.0}, {"id": "f2", "r": 100.0}]
        result = build_fire_forecast(0.0, current, projected, 300.0)
        assert result["max_projected_radius_m"] == pytest.approx(200.0, rel=1e-6)

    def test_fire_counts_match_inputs(self):
        current = [{"id": "f1", "r": 50.0}]
        projected = [{"id": "f1", "r": 80.0}, {"id": "f2", "r": 20.0}]
        result = build_fire_forecast(0.0, current, projected, 300.0)
        assert result["current_fire_count"] == 1
        assert result["projected_fire_count"] == 2


class TestEstimateEdgeForecastRisk:
    def test_safe_edge_returns_correct_structure(self):
        result = estimate_edge_forecast_risk("edge_1", _safe_edge_risk_fn)
        assert result["edge_id"] == "edge_1"
        assert result["blocked"] is False
        assert result["risk_score"] == pytest.approx(0.0, abs=1e-6)
        assert result["band"] == "clear"

    def test_dangerous_edge_is_blocked(self):
        result = estimate_edge_forecast_risk("edge_2", _dangerous_edge_risk_fn)
        assert result["blocked"] is True
        assert result["band"] == "inside_predicted_fire"

    def test_very_close_band(self):
        result = estimate_edge_forecast_risk("e", lambda _: (False, 0.1, 50.0))
        assert result["band"] == "very_close"

    def test_near_band(self):
        result = estimate_edge_forecast_risk("e", lambda _: (False, 0.05, 1800.0))
        assert result["band"] == "near"

    def test_buffered_band(self):
        result = estimate_edge_forecast_risk("e", lambda _: (False, 0.01, 3500.0))
        assert result["band"] == "buffered"

    def test_edge_id_forwarded_to_risk_fn(self):
        received = []
        def capturing_fn(edge_id):
            received.append(edge_id)
            return (False, 0.0, 1000.0)
        estimate_edge_forecast_risk("specific_edge", capturing_fn)
        assert received == ["specific_edge"]


class TestSummarizeRouteForecast:
    def test_empty_route_returns_zero_counts(self):
        result = summarize_route_forecast([], _safe_edge_risk_fn, max_edges=3)
        assert result["head_edges_evaluated"] == 0
        assert result["blocked_edges"] == 0
        assert result["min_margin_m"] is None

    def test_skips_junction_edges(self):
        edges = [":junction_1", ":junction_2", "real_edge"]
        result = summarize_route_forecast(edges, _safe_edge_risk_fn, max_edges=2)
        assert result["head_edges_evaluated"] == 1

    def test_counts_blocked_edges(self):
        edges = ["e1", "e2", "e3"]
        result = summarize_route_forecast(edges, _dangerous_edge_risk_fn, max_edges=3)
        assert result["blocked_edges"] == 3

    def test_finds_min_margin_among_edges(self):
        margins = [500.0, 50.0, 1000.0]
        idx = [0]
        def fn(edge_id):
            m = margins[idx[0] % len(margins)]
            idx[0] += 1
            return (False, 0.1, m)
        result = summarize_route_forecast(["a", "b", "c"], fn, max_edges=3)
        assert result["min_margin_m"] == pytest.approx(50.0, rel=1e-6)

    def test_respects_max_edges_limit(self):
        edges = ["e1", "e2", "e3", "e4", "e5"]
        result = summarize_route_forecast(edges, _safe_edge_risk_fn, max_edges=2)
        assert result["head_edges_evaluated"] == 2

    def test_safe_route_has_clear_band(self):
        result = summarize_route_forecast(["e1", "e2"], _safe_edge_risk_fn, max_edges=5)
        assert result["band"] == "clear"

    def test_dangerous_route_has_danger_band(self):
        result = summarize_route_forecast(["e1"], _dangerous_edge_risk_fn, max_edges=5)
        assert result["band"] == "inside_predicted_fire"


class TestRenderForecastBriefing:
    def _forecast(self):
        return {"horizon_s": 300, "current_fire_count": 1}

    def _belief_high_danger(self):
        return {"p_danger": 0.7, "uncertainty_bucket": "Low"}

    def _belief_uncertain(self):
        return {"p_danger": 0.1, "uncertainty_bucket": "High"}

    def _edge_safe(self):
        return {"edge_id": "e1", "blocked": False, "band": "clear"}

    def _edge_blocked(self):
        return {"edge_id": "e1", "blocked": True, "band": "inside_predicted_fire"}

    def _route_safe(self):
        return {"head_edges_evaluated": 3, "blocked_edges": 0, "band": "buffered"}

    def _route_blocked(self):
        return {"head_edges_evaluated": 3, "blocked_edges": 2, "band": "very_close"}

    def test_returns_non_empty_string(self):
        result = render_forecast_briefing(
            "v1", self._forecast(), self._belief_high_danger(),
            self._edge_safe(), self._route_safe()
        )
        assert isinstance(result, str) and len(result) > 0

    def test_high_danger_uses_threat_tone(self):
        result = render_forecast_briefing(
            "v1", self._forecast(), self._belief_high_danger(),
            self._edge_safe(), self._route_safe()
        )
        assert "worsening" in result.lower()

    def test_high_uncertainty_low_danger_uses_uncertain_tone(self):
        result = render_forecast_briefing(
            "v1", self._forecast(), self._belief_uncertain(),
            self._edge_safe(), self._route_safe()
        )
        assert "uncertain" in result.lower()

    def test_blocked_current_edge_mentions_overtaken(self):
        result = render_forecast_briefing(
            "v1", self._forecast(), self._belief_uncertain(),
            self._edge_blocked(), self._route_safe()
        )
        assert "overtaken" in result.lower()

    def test_blocked_route_head_mentions_blocked(self):
        result = render_forecast_briefing(
            "v1", self._forecast(), self._belief_uncertain(),
            self._edge_safe(), self._route_blocked()
        )
        assert "blocked" in result.lower()

    def test_horizon_included_in_output(self):
        result = render_forecast_briefing(
            "v1", self._forecast(), self._belief_high_danger(),
            self._edge_safe(), self._route_safe()
        )
        assert "300" in result
