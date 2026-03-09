"""Unit tests for agentevac.agents.neighborhood_observation."""

import pytest

from agentevac.agents.neighborhood_observation import (
    build_departure_observation_update,
    build_neighbor_map,
    compute_social_departure_pressure,
    render_neighborhood_summary,
    summarize_neighborhood_observation,
)


def _spawn_events():
    return [
        ("a1", "edge_a", "dest", 0.0, "free", "base", "max", (255, 0, 0, 255)),
        ("a2", "edge_a", "dest", 0.0, "free", "base", "max", (255, 0, 0, 255)),
        ("a3", "edge_a", "dest", 0.0, "free", "base", "max", (255, 0, 0, 255)),
        ("b1", "edge_b", "dest", 0.0, "free", "base", "max", (255, 0, 0, 255)),
    ]


def _spawn_edge_by_agent():
    return {
        "a1": "edge_a",
        "a2": "edge_a",
        "a3": "edge_a",
        "b1": "edge_b",
    }


class TestBuildNeighborMap:
    def test_same_spawn_edge_groups_peers(self):
        neighbor_map = build_neighbor_map(_spawn_events(), scope="same_spawn_edge")
        assert sorted(neighbor_map["a1"]) == ["a2", "a3"]
        assert neighbor_map["b1"] == []

    def test_unsupported_scope_raises(self):
        with pytest.raises(ValueError, match="Unsupported neighborhood scope"):
            build_neighbor_map(_spawn_events(), scope="invalid")


class TestSummarizeNeighborhoodObservation:
    def test_counts_departed_and_still_staying(self):
        neighbor_map = build_neighbor_map(_spawn_events(), scope="same_spawn_edge")
        obs = summarize_neighborhood_observation(
            "a1",
            100.0,
            neighbor_map,
            _spawn_edge_by_agent(),
            {"a2": 70.0},
            window_s=120.0,
        )
        assert obs["neighbor_count"] == 2
        assert obs["departed_total_count"] == 1
        assert obs["still_staying_count"] == 1
        assert obs["recent_departures_count"] == 1

    def test_zero_neighbor_case(self):
        neighbor_map = build_neighbor_map(_spawn_events(), scope="same_spawn_edge")
        obs = summarize_neighborhood_observation(
            "b1",
            10.0,
            neighbor_map,
            _spawn_edge_by_agent(),
            {},
            window_s=120.0,
        )
        assert obs["neighbor_count"] == 0
        assert obs["summary"] == "No neighbors are associated with this observation scope."

    def test_recent_departure_respects_window(self):
        neighbor_map = build_neighbor_map(_spawn_events(), scope="same_spawn_edge")
        obs = summarize_neighborhood_observation(
            "a1",
            200.0,
            neighbor_map,
            _spawn_edge_by_agent(),
            {"a2": 10.0, "a3": 170.0},
            window_s=20.0,
        )
        assert obs["departed_total_count"] == 2
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
                "recent_departure_fraction": 2.0,
                "departed_total_fraction": 2.0,
            }
        )
        assert 0.0 <= pressure <= 1.0

    def test_recent_departures_weight_more_than_total(self):
        high_recent = compute_social_departure_pressure(
            {
                "recent_departure_fraction": 0.8,
                "departed_total_fraction": 0.2,
            },
            w_recent=0.7,
            w_total=0.3,
        )
        high_total = compute_social_departure_pressure(
            {
                "recent_departure_fraction": 0.2,
                "departed_total_fraction": 0.8,
            },
            w_recent=0.7,
            w_total=0.3,
        )
        assert high_recent > high_total


class TestBuildDepartureObservationUpdate:
    def test_update_contains_departed_neighbor_id(self):
        neighbor_map = build_neighbor_map(_spawn_events(), scope="same_spawn_edge")
        obs = build_departure_observation_update(
            focal_agent_id="a1",
            departed_agent_id="a2",
            sim_t_s=35.0,
            neighbor_map=neighbor_map,
            spawn_edge_by_agent=_spawn_edge_by_agent(),
            departure_times={"a2": 35.0},
            window_s=120.0,
        )
        assert obs["departed_neighbor_id"] == "a2"
        assert obs["summary"] == "1 neighbor has departed to evacuate. 1 neighbor is still staying."
