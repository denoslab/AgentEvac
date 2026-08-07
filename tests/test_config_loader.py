"""Unit tests for agentevac.config_loader home-point threading.

The spawn tuple format is unchanged by the building-centroid work, so these tests pin
both halves of that contract: the tuples callers already depend on stay identical, and
the optional ``home_points_out`` side channel lines up with the generated agent IDs.
"""

from agentevac.config_loader import (
    expand_spawn_groups,
    load_spawns,
    spawns_to_tuples,
)

_DESTS = [{"name": "Shelter", "edge": "d0"}]


def _detailed(**extra):
    base = {
        "veh_id": "v1",
        "spawn_edge": "e0",
        "dest_edge": "d0",
        "depart_time": 0.0,
    }
    base.update(extra)
    return [base]


class TestTupleFormatUnchanged:
    def test_compact_tuple_shape_is_eight(self):
        tuples = expand_spawn_groups([{"edge": "e0", "count": 2}], "d0")
        assert all(len(t) == 8 for t in tuples)

    def test_home_xy_does_not_leak_into_tuples(self):
        group = {"edge": "e0", "count": 2, "home_xy": [[1.0, 2.0], [3.0, 4.0]]}
        with_home = expand_spawn_groups([group], "d0")
        without = expand_spawn_groups([{"edge": "e0", "count": 2}], "d0")
        assert with_home == without

    def test_detailed_tuple_shape_is_eight(self):
        assert all(len(t) == 8 for t in spawns_to_tuples(_detailed(home_xy=[1.0, 2.0])))


class TestCompactHomePoints:
    def test_home_points_match_generated_ids(self):
        out = {}
        expand_spawn_groups(
            [{"edge": "e0", "count": 2, "home_xy": [[1.0, 2.0], [3.0, 4.0]]}],
            "d0", home_points_out=out,
        )
        assert out == {"e0_1": (1.0, 2.0), "e0_2": (3.0, 4.0)}

    def test_group_without_home_xy_contributes_nothing(self):
        out = {}
        expand_spawn_groups([{"edge": "e0", "count": 2}], "d0", home_points_out=out)
        assert out == {}

    def test_short_home_xy_list_leaves_surplus_agents_unmapped(self):
        out = {}
        expand_spawn_groups(
            [{"edge": "e0", "count": 3, "home_xy": [[1.0, 2.0]]}],
            "d0", home_points_out=out,
        )
        assert out == {"e0_1": (1.0, 2.0)}

    def test_null_entry_falls_back_to_no_home_point(self):
        out = {}
        expand_spawn_groups(
            [{"edge": "e0", "count": 3, "home_xy": [[1.0, 2.0], None, [5.0, 6.0]]}],
            "d0", home_points_out=out,
        )
        assert out == {"e0_1": (1.0, 2.0), "e0_3": (5.0, 6.0)}

    def test_malformed_entry_is_ignored(self):
        out = {}
        expand_spawn_groups(
            [{"edge": "e0", "count": 3, "home_xy": [[1.0], "x,y", [5.0, 6.0]]}],
            "d0", home_points_out=out,
        )
        assert out == {"e0_3": (5.0, 6.0)}

    def test_duplicate_edge_suffix_keeps_points_aligned(self):
        out = {}
        expand_spawn_groups(
            [
                {"edge": "e0", "count": 1, "home_xy": [[1.0, 1.0]]},
                {"edge": "e0", "count": 1, "home_xy": [[2.0, 2.0]]},
            ],
            "d0", home_points_out=out,
        )
        assert out == {"e0_1": (1.0, 1.0), "e0_g2_1": (2.0, 2.0)}

    def test_omitting_the_out_dict_is_supported(self):
        expand_spawn_groups(
            [{"edge": "e0", "count": 1, "home_xy": [[1.0, 2.0]]}], "d0"
        )  # must not raise


class TestDetailedHomePoints:
    def test_home_xy_collected(self):
        out = {}
        spawns_to_tuples(_detailed(home_xy=[7.0, 8.0]), home_points_out=out)
        assert out == {"v1": (7.0, 8.0)}

    def test_missing_home_xy_collected_as_nothing(self):
        out = {}
        spawns_to_tuples(_detailed(), home_points_out=out)
        assert out == {}


class TestLoadSpawnsThreading:
    def test_compact_path_threads_home_points(self):
        out = {}
        load_spawns(
            {"groups": [{"edge": "e0", "count": 1, "home_xy": [[9.0, 9.0]]}]},
            _DESTS, home_points_out=out,
        )
        assert out == {"e0_1": (9.0, 9.0)}

    def test_detailed_path_threads_home_points(self):
        out = {}
        load_spawns(_detailed(home_xy=[4.0, 5.0]), _DESTS, home_points_out=out)
        assert out == {"v1": (4.0, 5.0)}

    def test_edge_only_config_yields_empty_map(self):
        out = {}
        load_spawns({"groups": [{"edge": "e0", "count": 3}]}, _DESTS, home_points_out=out)
        assert out == {}
