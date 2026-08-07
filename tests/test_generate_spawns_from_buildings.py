"""Unit tests for the building-centroid attachment in generate_spawns_from_buildings.

The attachment mode exists so a centroid config is the same household population as the
edge config it derives from.  These tests pin that invariant, since a drift in edges,
counts, or agent identity would silently make the two experiment batches incomparable.
"""

import math

import pytest

from scripts.generate_spawns_from_buildings import (
    _DENSIFY_STEP_M,
    _densify,
    attach_centroids_to_existing,
    generate_spawn_config,
)


def _match(building_id, edge_id, distance_m, x, y):
    return {
        "building_id": building_id,
        "edge_id": edge_id,
        "distance_m": distance_m,
        "home_x": x,
        "home_y": y,
    }


def _existing(*groups):
    return {"groups": [{"edge": e, "count": n} for e, n in groups]}


class TestAttachPreservesThePopulation:
    def test_edges_and_counts_are_untouched(self):
        existing = _existing(("e0", 2), ("e1", 1))
        cfg, _ = attach_centroids_to_existing(
            existing,
            [_match("b1", "e0", 10, 1.0, 1.0), _match("b2", "e0", 20, 2.0, 2.0),
             _match("b3", "e1", 5, 3.0, 3.0)],
            net=None,
        )
        assert [g["edge"] for g in cfg["groups"]] == ["e0", "e1"]
        assert [g["count"] for g in cfg["groups"]] == [2, 1]

    def test_source_config_is_not_mutated(self):
        existing = _existing(("e0", 1))
        attach_centroids_to_existing(existing, [_match("b1", "e0", 10, 1.0, 1.0)], net=None)
        assert existing["groups"][0] == {"edge": "e0", "count": 1}

    def test_home_xy_length_always_equals_count(self):
        cfg, _ = attach_centroids_to_existing(
            _existing(("e0", 3)), [_match("b1", "e0", 10, 1.0, 1.0)], net=None,
        )
        g = cfg["groups"][0]
        assert len(g["home_xy"]) == 3
        assert len(g["building_id"]) == 3

    def test_closest_buildings_are_assigned_first(self):
        cfg, _ = attach_centroids_to_existing(
            _existing(("e0", 2)),
            [_match("far", "e0", 90, 9.0, 9.0), _match("near", "e0", 5, 1.0, 1.0),
             _match("mid", "e0", 40, 4.0, 4.0)],
            net=None,
        )
        assert cfg["groups"][0]["building_id"] == ["near", "mid"]


class TestAttachNeverDoubleAssigns:
    def test_each_building_serves_one_household(self):
        cfg, report = attach_centroids_to_existing(
            _existing(("e0", 2), ("e1", 2)),
            [_match("b1", "e0", 10, 1.0, 1.0), _match("b2", "e0", 20, 2.0, 2.0),
             _match("b3", "e1", 10, 3.0, 3.0), _match("b4", "e1", 20, 4.0, 4.0)],
            net=None,
        )
        ids = [b for g in cfg["groups"] for b in g["building_id"]]
        assert ids == ["b1", "b2", "b3", "b4"]
        assert len(set(ids)) == len(ids)
        assert report["matched_direct"] == 4

    def test_home_points_are_distinct(self):
        cfg, _ = attach_centroids_to_existing(
            _existing(("e0", 2)),
            [_match("b1", "e0", 10, 1.0, 1.0), _match("b2", "e0", 20, 2.0, 2.0)],
            net=None,
        )
        points = [tuple(h) for h in cfg["groups"][0]["home_xy"]]
        assert len(set(points)) == 2


class TestAttachDeficitHandling:
    def test_surplus_households_get_null_and_are_reported(self):
        cfg, report = attach_centroids_to_existing(
            _existing(("e0", 3)), [_match("b1", "e0", 10, 1.0, 1.0)], net=None,
        )
        g = cfg["groups"][0]
        assert g["home_xy"] == [[1.0, 1.0], None, None]
        assert g["building_id"] == ["b1", None, None]
        assert report["edge_fallback"] == 2
        assert report["deficit_edges"] == [("e0", 3, 1)]

    def test_no_deficit_reports_none(self):
        _, report = attach_centroids_to_existing(
            _existing(("e0", 1)), [_match("b1", "e0", 10, 1.0, 1.0)], net=None,
        )
        assert report["deficit_edges"] == []
        assert report["edge_fallback"] == 0

    def test_report_totals_add_up(self):
        _, report = attach_centroids_to_existing(
            _existing(("e0", 3)), [_match("b1", "e0", 10, 1.0, 1.0)], net=None,
        )
        assert (
            report["matched_direct"] + report["matched_widened"] + report["edge_fallback"]
            == report["households"]
        )


class TestDensify:
    """Densification is what makes the exact snap reliable.

    A long straight edge carries only its two endpoints, so a radius query around a
    building beside its middle would find no vertex and the edge would never become a
    candidate. Inserting points caps that gap.
    """

    def test_endpoints_are_preserved(self):
        out = _densify([(0.0, 0.0), (200.0, 0.0)], step_m=50.0)
        assert out[0] == (0.0, 0.0)
        assert out[-1] == (200.0, 0.0)

    def test_no_gap_exceeds_the_step(self):
        out = _densify([(0.0, 0.0), (237.0, 0.0), (237.0, 119.0)], step_m=50.0)
        gaps = [
            math.dist(a, b) for a, b in zip(out, out[1:])
        ]
        assert max(gaps) <= 50.0 + 1e-9

    def test_short_segments_are_left_alone(self):
        out = _densify([(0.0, 0.0), (10.0, 0.0)], step_m=50.0)
        assert out == [(0.0, 0.0), (10.0, 0.0)]

    def test_points_stay_on_the_line(self):
        out = _densify([(0.0, 0.0), (300.0, 400.0)], step_m=50.0)
        for x, y in out:
            assert y == pytest.approx(x * 4.0 / 3.0)

    def test_default_step_is_used(self):
        out = _densify([(0.0, 0.0), (_DENSIFY_STEP_M * 3, 0.0)])
        assert len(out) == 4


class TestAttachInputValidation:
    def test_detailed_format_is_rejected(self):
        with pytest.raises(ValueError, match="compact"):
            attach_centroids_to_existing([{"veh_id": "v1"}], [], net=None)


class TestGeneratorEmitsHomePoints:
    def test_per_building_carries_one_home_per_agent(self):
        cfg = generate_spawn_config(
            [_match("b1", "e0", 10, 1.0, 1.0), _match("b2", "e0", 20, 2.0, 2.0)],
            mode="per-building", count=1, dest_edge=None, output_format="compact",
        )
        g = cfg["groups"][0]
        assert g["count"] == 2
        assert g["home_xy"] == [[1.0, 1.0], [2.0, 2.0]]

    def test_per_building_repeats_the_home_when_count_exceeds_one(self):
        cfg = generate_spawn_config(
            [_match("b1", "e0", 10, 1.0, 1.0)],
            mode="per-building", count=2, dest_edge=None, output_format="compact",
        )
        g = cfg["groups"][0]
        assert g["count"] == 2
        assert g["home_xy"] == [[1.0, 1.0], [1.0, 1.0]]

    def test_detailed_format_carries_home_xy_per_agent(self):
        agents = generate_spawn_config(
            [_match("b1", "e0", 10, 1.0, 1.0), _match("b2", "e0", 20, 2.0, 2.0)],
            mode="per-building", count=1, dest_edge="d0", output_format="detailed",
        )
        assert [a["home_xy"] for a in agents] == [[1.0, 1.0], [2.0, 2.0]]
        assert [a["building_id"] for a in agents] == ["b1", "b2"]
