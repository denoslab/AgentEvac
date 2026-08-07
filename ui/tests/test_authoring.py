"""Composing a scenario package from a map selection.

The rule the whole module rests on is that a package is created or nothing happens, so
these tests pin the refusal to write over an existing package as hard as they pin the
composition itself.
"""

from __future__ import annotations

import json

import pytest

from ui.backend import authoring


def _index(*rows):
    """A stand-in building layer, keyed the way a preview bundle is."""
    out = {}
    for bid, edge, x, y in rows:
        out[bid] = {"id": bid, "edge": edge, "x": x, "y": y, "lon": 0.0, "lat": 0.0}
    return out


_INDEX = _index(
    ("b1", "e0", 100.0, 200.0),
    ("b2", "e0", 110.0, 210.0),
    ("b3", "e1", 300.0, 400.0),
    ("stranded", None, 900.0, 900.0),
)


def _draft(**over):
    draft = {
        "id": "authored_pkg",
        "source_package": "halifax_3town_e0",
        "households": [{"building_id": "b1", "count": 1}, {"building_id": "b3", "count": 2}],
        "fires": [{"id": "f1", "x": 1.0, "y": 2.0, "t0": 0.0, "r0": 30.0, "growth_m_per_s": 0.4}],
    }
    draft.update(over)
    return draft


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the module at a temporary configs tree holding one source package."""
    configs = tmp_path / "configs"
    source = configs / "halifax_3town_e0"
    source.mkdir(parents=True)
    for name in ("map.json", "destinations.json", "routes.json"):
        (source / name).write_text('{"inherited": true}' if name == "map.json" else "[]")
    monkeypatch.setattr(authoring, "CONFIGS_DIR", configs)
    monkeypatch.setattr(authoring, "REPO_ROOT", tmp_path)
    return configs


class TestSpawnComposition:
    def test_buildings_on_one_edge_merge_into_one_group(self):
        households = [
            {"building_id": "b1", "edge": "e0", "x": 1.0, "y": 2.0, "count": 1},
            {"building_id": "b2", "edge": "e0", "x": 3.0, "y": 4.0, "count": 1},
        ]
        groups = authoring.compose_spawns(households)["groups"]
        assert len(groups) == 1
        assert groups[0]["count"] == 2
        assert groups[0]["building_id"] == ["b1", "b2"]
        assert groups[0]["home_xy"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_count_above_one_repeats_the_home_point(self):
        households = [{"building_id": "b1", "edge": "e0", "x": 1.0, "y": 2.0, "count": 3}]
        group = authoring.compose_spawns(households)["groups"][0]
        assert group["count"] == 3
        assert group["home_xy"] == [[1.0, 2.0]] * 3
        assert group["building_id"] == ["b1"] * 3

    def test_home_lists_always_match_the_count(self):
        households = [
            {"building_id": "b1", "edge": "e0", "x": 1.0, "y": 2.0, "count": 2},
            {"building_id": "b3", "edge": "e1", "x": 5.0, "y": 6.0, "count": 1},
        ]
        for group in authoring.compose_spawns(households)["groups"]:
            assert len(group["home_xy"]) == group["count"]
            assert len(group["building_id"]) == group["count"]

    def test_group_order_is_stable(self):
        households = [
            {"building_id": "z", "edge": "e1", "x": 1.0, "y": 1.0, "count": 1},
            {"building_id": "a", "edge": "e0", "x": 2.0, "y": 2.0, "count": 1},
        ]
        edges = [g["edge"] for g in authoring.compose_spawns(households)["groups"]]
        assert edges == ["e0", "e1"]


# An area only covers households. Ordering a building that holds no agents changes
# nothing, and its road would still be pulled into the ordered area, which would order
# households that were never selected.
_HOUSEHOLDS = [
    {"building_id": "b1", "edge": "e0", "count": 1},
    {"building_id": "b3", "edge": "e1", "count": 2},
]


class TestAlertComposition:
    def test_households_resolve_to_their_edges(self):
        problems, warnings = [], []
        alerts = authoring.compose_alerts(
            [{"name": "area_a", "building_ids": ["b1", "b3"]}],
            [], _INDEX, _HOUSEHOLDS, problems, warnings,
        )
        assert alerts["areas"]["area_a"]["edges"] == ["e0", "e1"]
        assert alerts["areas"]["area_a"]["building_ids"] == ["b1", "b3"]
        assert problems == []

    def test_a_building_that_is_not_a_household_is_dropped(self):
        problems, warnings = [], []
        alerts = authoring.compose_alerts(
            [{"name": "area_a", "building_ids": ["b1", "b2"]}],
            [], _INDEX, _HOUSEHOLDS, problems, warnings,
        )
        # b2 sits on e0 as well, but it holds no agents, so it never enters the area.
        assert alerts["areas"]["area_a"]["building_ids"] == ["b1"]
        assert any("not households" in w["message"] for w in warnings)

    def test_a_stranded_building_is_dropped(self):
        problems, warnings = [], []
        alerts = authoring.compose_alerts(
            [{"name": "area_a", "building_ids": ["b1", "stranded"]}],
            [], _INDEX, _HOUSEHOLDS, problems, warnings,
        )
        assert alerts["areas"]["area_a"]["edges"] == ["e0"]

    def test_area_covering_no_household_is_a_problem(self):
        problems, warnings = [], []
        authoring.compose_alerts(
            [{"name": "area_a", "building_ids": ["b2"]}], [], _INDEX, _HOUSEHOLDS,
            problems, warnings,
        )
        assert any("covers no household" in p["message"] for p in problems)

    def test_an_area_with_no_households_at_all_is_a_problem(self):
        problems, warnings = [], []
        authoring.compose_alerts(
            [{"name": "area_a", "building_ids": ["b1"]}], [], _INDEX, [], problems, warnings,
        )
        assert any("covers no household" in p["message"] for p in problems)

    def test_unnamed_area_is_rejected(self):
        problems, warnings = [], []
        authoring.compose_alerts([{"building_ids": ["b1"]}], [], _INDEX, [], problems, warnings)
        assert any("no name" in p["message"] for p in problems)

    def test_duplicate_area_names_are_rejected(self):
        problems, warnings = [], []
        authoring.compose_alerts(
            [{"name": "a", "building_ids": ["b1"]}, {"name": "a", "building_ids": ["b3"]}],
            [], _INDEX, _HOUSEHOLDS, problems, warnings,
        )
        assert any("both named" in p["message"] for p in problems)

    def test_event_naming_an_unknown_area_is_rejected(self):
        problems, warnings = [], []
        authoring.compose_alerts(
            [{"name": "a", "building_ids": ["b1"]}],
            [{"id": "EA-1", "issue_time_s": 10.0, "areas": ["ghost"]}],
            _INDEX, _HOUSEHOLDS, problems, warnings,
        )
        assert any("do not exist" in p["message"] for p in problems)

    def test_unknown_instruction_is_rejected(self):
        problems, warnings = [], []
        authoring.compose_alerts(
            [{"name": "a", "building_ids": ["b1"]}],
            [{"id": "EA-1", "issue_time_s": 10.0, "areas": ["a"], "instruction": "panic"}],
            _INDEX, _HOUSEHOLDS, problems, warnings,
        )
        assert any("unknown instruction" in p["message"] for p in problems)

    def test_schedule_is_sorted_by_issue_time(self):
        problems, warnings = [], []
        alerts = authoring.compose_alerts(
            [{"name": "a", "building_ids": ["b1"]}],
            [{"id": "late", "issue_time_s": 900.0, "areas": ["a"]},
             {"id": "early", "issue_time_s": 100.0, "areas": ["a"]}],
            _INDEX, _HOUSEHOLDS, problems, warnings,
        )
        assert [e["id"] for e in alerts["schedule"]] == ["early", "late"]

    def test_no_areas_and_no_events_means_no_alert_file(self):
        assert authoring.compose_alerts([], [], _INDEX, _HOUSEHOLDS, [], []) is None


class TestFireComposition:
    def test_sources_are_sorted_by_ignition_time(self):
        problems = []
        fires = authoring.compose_fires(
            [{"id": "b", "x": 1, "y": 1, "t0": 500},
             {"id": "a", "x": 2, "y": 2, "t0": 100}],
            problems,
        )
        assert [s["id"] for s in fires["sources"]] == ["a", "b"]

    def test_defaults_are_applied(self):
        fires = authoring.compose_fires([{"x": 1, "y": 2}], [])
        source = fires["sources"][0]
        assert source["t0"] == 0.0
        assert source["r0"] == 30.0
        assert source["growth_m_per_s"] == 0.3
        assert "max_r_m" not in source

    def test_radius_cap_is_carried_when_given(self):
        fires = authoring.compose_fires([{"x": 1, "y": 2, "max_r_m": 700}], [])
        assert fires["sources"][0]["max_r_m"] == 700.0

    def test_missing_position_is_reported(self):
        problems = []
        fires = authoring.compose_fires([{"id": "bad", "t0": 0}], problems)
        assert fires["sources"] == []
        assert any("missing a position" in p["message"] for p in problems)

    def test_negative_growth_is_rejected(self):
        problems = []
        authoring.compose_fires([{"x": 1, "y": 2, "growth_m_per_s": -1}], problems)
        assert any("negative rate" in p["message"] for p in problems)


class TestDraftValidation:
    def test_a_complete_draft_validates(self, sandbox):
        validation, files = authoring.validate_draft(_draft(), _INDEX)
        assert validation.ok, validation.problems
        assert set(files) == {"spawns.json", "fires.json"}
        assert validation.summary["agents"] == 3
        assert validation.summary["households"] == 2

    def test_existing_package_name_is_refused(self, sandbox):
        (sandbox / "authored_pkg").mkdir()
        validation, files = authoring.validate_draft(_draft(), _INDEX)
        assert not validation.ok
        assert files is None
        assert any("already exists" in p["message"] for p in validation.problems)

    @pytest.mark.parametrize("bad", ["", "ab", "Has Caps", "has-dash", "9" * 70, "../escape"])
    def test_unusable_names_are_refused(self, sandbox, bad):
        validation, _ = authoring.validate_draft(_draft(id=bad), _INDEX)
        assert any(p["field"] == "id" for p in validation.problems)

    def test_missing_source_package_is_refused(self, sandbox):
        validation, _ = authoring.validate_draft(_draft(source_package="ghost"), _INDEX)
        assert any(p["field"] == "source_package" for p in validation.problems)

    def test_selection_with_no_spawnable_household_is_refused(self, sandbox):
        validation, _ = authoring.validate_draft(
            _draft(households=[{"building_id": "stranded", "count": 1}]), _INDEX,
        )
        assert any("no household that can spawn" in p["message"] for p in validation.problems)

    def test_building_outside_the_bundle_is_refused(self, sandbox):
        validation, _ = authoring.validate_draft(
            _draft(households=[{"building_id": "ghost", "count": 1}]), _INDEX,
        )
        assert any("not in the source package" in p["message"] for p in validation.problems)

    def test_duplicate_selection_is_warned_and_counted_once(self, sandbox):
        validation, files = authoring.validate_draft(
            _draft(households=[{"building_id": "b1", "count": 1}, {"building_id": "b1", "count": 1}]),
            _INDEX,
        )
        assert validation.ok
        assert validation.summary["agents"] == 1
        assert any("more than once" in w["message"] for w in validation.warnings)

    def test_a_package_with_no_fire_is_refused(self, sandbox):
        validation, _ = authoring.validate_draft(_draft(fires=[]), _INDEX)
        assert any("no fire source" in p["message"] for p in validation.problems)

    def test_absurd_per_building_count_is_refused(self, sandbox):
        validation, _ = authoring.validate_draft(
            _draft(households=[{"building_id": "b1", "count": 5000}]), _INDEX,
        )
        assert any(p["field"] == "households" for p in validation.problems)

    def test_missing_alert_schedule_is_only_a_warning(self, sandbox):
        validation, files = authoring.validate_draft(_draft(), _INDEX)
        assert validation.ok
        assert "alerts.json" not in files
        assert any("no alert schedule" in w["message"] for w in validation.warnings)


class TestWritePackage:
    def test_files_are_written_and_inherited(self, sandbox):
        validation, target = authoring.write_package(_draft(), _INDEX)
        assert validation.ok
        assert target == sandbox / "authored_pkg"
        written = {p.name for p in target.iterdir()}
        assert {"map.json", "destinations.json", "routes.json",
                "spawns.json", "fires.json", "README.md"} <= written
        assert json.loads((target / "map.json").read_text()) == {"inherited": True}

    def test_alerts_are_written_when_present(self, sandbox):
        draft = _draft(
            alert_areas=[{"name": "a", "building_ids": ["b1"]}],
            alert_events=[{"id": "EA-1", "issue_time_s": 100.0, "areas": ["a"]}],
        )
        _validation, target = authoring.write_package(draft, _INDEX)
        alerts = json.loads((target / "alerts.json").read_text())
        assert alerts["areas"]["a"]["edges"] == ["e0"]
        assert alerts["schedule"][0]["id"] == "EA-1"

    def test_existing_directory_is_never_touched(self, sandbox):
        existing = sandbox / "authored_pkg"
        existing.mkdir()
        (existing / "spawns.json").write_text('{"do_not": "overwrite"}')
        validation, target = authoring.write_package(_draft(), _INDEX)
        assert target is None
        assert not validation.ok
        assert json.loads((existing / "spawns.json").read_text()) == {"do_not": "overwrite"}

    def test_an_invalid_draft_creates_nothing(self, sandbox):
        validation, target = authoring.write_package(_draft(fires=[]), _INDEX)
        assert target is None
        assert not (sandbox / "authored_pkg").exists()

    def test_a_failure_part_way_leaves_no_directory(self, sandbox, monkeypatch):
        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(authoring.shutil, "copyfile", boom)
        validation, target = authoring.write_package(_draft(), _INDEX)
        assert target is None
        assert not validation.ok
        assert not (sandbox / "authored_pkg").exists()

    def test_readme_records_the_provenance(self, sandbox):
        _validation, target = authoring.write_package(_draft(), _INDEX)
        text = (target / "README.md").read_text()
        assert "operator console" in text
        assert "halifax_3town_e0" in text
