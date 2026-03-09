"""Unit tests for agentevac.agents.neighborhood_observation."""

import pytest

from agentevac.agents.neighborhood_observation import (
    build_neighbor_map,
    build_departure_observation_update,
    summarize_neighborhood_observation,
    render_neighborhood_summary,
    compute_social_departure_pressure,
)


def _spawn_events():
    return [
        ("veh1", "edge_a", "dest", 0.0, "first", "10", "max", (255, 0, 0, 255)),
        ("veh2", "edge_a", "dest", 0.0, "first", "10", "max", (0, 0, 255, 255)),
        ("veh3", "edge_a", "dest", 0.0, "first", "10", "max", (0, 255, 0, 255)),
        ("veh4", "edge_b", "dest", 0.0, "first", "10", "max", (255, 255, 0, 255)),
    ]


def _spawn_edge_by_agent():
    return {
        "veh1": "edge_a",
        "veh2": "edge_a",
        "veh3": "edge_a",
        "veh4": "edge_b",
    }


class TestBuildNeighborMap:
    def test_same_spawn_edge_groups_peers(self):
        mapping = build_neighbor_map(_spawn_events(), scope="same_spawn_edge")
        assert sorted(mapping["veh1"]) == ["veh2", "veh3"]
        assert sorted(mapping["veh4"]) == []

    def test_unsupported_scope_raises(self):
        with pytest.raises(ValueError):
            build_neighbor_map(_spawn_events(), scope="unknown")


class TestSummarizeNeighborhoodObservation:
    def test_counts_departed_and_still_staying(self):
        obs = summarize_neighborhood_observation(
            "veh1",
            100.0,
            build_neighbor_map(_spawn_events()),
            _spawn_edge_by_agent(),
            {"veh2": 40.0},
            window_s=120.0,
        )
        assert obs["neighbor_count"] == 2
        assert obs["departed_total_count"] == 1
        assert obs["still_staying_count"] == 1
        assert obs["recent_departures_count"] == 1

    def test_zero_neighbor_case(self):
        obs = summarize_neighborhood_observation(
            "veh4",
            100.0,
            build_neighbor_map(_spawn_events()),
            _spawn_edge_by_agent(),
            {},
            window_s=120.0,
        )
        assert obs["neighbor_count"] == 0
        assert "No neighbors" in obs["summary"]

    def test_recent_departure_respects_window(self):
        obs = summarize_neighborhood_observation(
            "veh1",
            200.0,
            build_neighbor_map(_spawn_events()),
            _spawn_edge_by_agent(),
            {"veh2": 10.0},
            window_s=60.0,
        )
        assert obs["departed_total_count"] == 1
        assert obs["recent_departures_count"] == 0


class TestRenderNeighborhoodSummary:
    def test_plural_summary_uses_still_staying(self):
        text = render_neighborhood_summary(
            {
                "neighbor_count": 5,
                "recent_departures_count": 2,
                "still_staying_count": 3,
            }
        )
        assert text == "2 neighbors have departed to evacuate. 3 neighbors are still staying."

    def test_singular_summary_uses_correct_grammar(self):
        text = render_neighborhood_summary(
            {
                "neighbor_count": 2,
                "recent_departures_count": 1,
                "still_staying_count": 1,
            }
        )
        assert text == "1 neighbor has departed to evacuate. 1 neighbor is still staying."


class TestComputeSocialDeparturePressure:
    def test_pressure_is_bounded(self):
        pressure = compute_social_departure_pressure(
            {
                "recent_departure_fraction": 1.0,
                "departed_total_fraction": 1.0,
            },
            w_recent=0.9,
            w_total=0.9,
        )
        assert pressure == pytest.approx(1.0)

    def test_recent_departures_weight_more_than_total(self):
        pressure = compute_social_departure_pressure(
            {
                "recent_departure_fraction": 0.5,
                "departed_total_fraction": 0.25,
            },
            w_recent=0.7,
            w_total=0.3,
        )
        assert pressure == pytest.approx(0.425, rel=1e-6)


class TestBuildDepartureObservationUpdate:
    def test_update_contains_departed_neighbor_id(self):
        update = build_departure_observation_update(
            "veh1",
            "veh2",
            35.0,
            build_neighbor_map(_spawn_events()),
            _spawn_edge_by_agent(),
            {"veh2": 35.0},
            window_s=120.0,
        )
        assert update["departed_neighbor_id"] == "veh2"
        assert update["kind"] == "neighbor_departure_observation"
