"""Unit tests for agentevac.utils.replay.

``replay.py`` imports ``traci`` at the module level.  Since SUMO/TraCI is not
available in the test environment we patch ``sys.modules`` before the first
import to substitute a ``MagicMock`` in its place.  All tests that exercise
replay-mode path live under ``TestRouteReplayReplayMode`` and use a real JSONL
fixture written to a tmp directory.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Patch traci before importing RouteReplay so the module-level `import traci`
# does not raise an ImportError in the test environment.
# ---------------------------------------------------------------------------
if "traci" not in sys.modules:
    _traci_mock = MagicMock()
    _traci_mock.TraCIException = Exception
    sys.modules["traci"] = _traci_mock

from agentevac.utils.replay import RouteReplay  # noqa: E402 (must come after mock)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_path(tmp_path: Path) -> str:
    return str(tmp_path / "routes_test.jsonl")


def _write_jsonl(path: str, records: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_route_change(step=1, veh_id="v1", route_edges=None):
    return {
        "event": "route_change",
        "step": step,
        "time_s": float(step * 5),
        "veh_id": veh_id,
        "control_mode": "destination",
        "choice_idx": 0,
        "chosen_name": "shelter_a",
        "chosen_edge": "edge_dest",
        "current_edge": "edge_start",
        "route_edges": route_edges or ["edge_start", "edge_mid", "edge_dest"],
        "reason": "test",
    }


# ---------------------------------------------------------------------------
# Tests: record mode
# ---------------------------------------------------------------------------

class TestRouteReplayRecordMode:
    def test_creates_jsonl_file(self, tmp_path):
        rr = RouteReplay(mode="record", path=_record_path(tmp_path))
        rr.close()
        assert any(p.suffix == ".jsonl" for p in tmp_path.iterdir())

    def test_creates_dialog_log_file(self, tmp_path):
        rr = RouteReplay(mode="record", path=_record_path(tmp_path))
        rr.close()
        assert any(".dialogs.log" in p.name for p in tmp_path.iterdir())

    def test_creates_dialog_csv_file(self, tmp_path):
        rr = RouteReplay(mode="record", path=_record_path(tmp_path))
        rr.close()
        assert any(".dialogs.csv" in p.name for p in tmp_path.iterdir())

    def test_record_route_change_writes_jsonl_record(self, tmp_path):
        rr = RouteReplay(mode="record", path=_record_path(tmp_path))
        rr.record_route_change(
            step=1, sim_t_s=5.0, veh_id="v1",
            control_mode="destination", choice_idx=0,
            chosen_name="shelter_a", chosen_edge="edge_dest",
            current_edge_before="edge_start",
            applied_route_edges=["edge_start", "edge_dest"],
        )
        rr.close()
        jsonl_files = [p for p in tmp_path.iterdir() if p.suffix == ".jsonl"]
        assert jsonl_files
        with open(jsonl_files[0]) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["event"] == "route_change"
        assert rec["veh_id"] == "v1"

    def test_route_trimmed_to_current_edge(self, tmp_path):
        rr = RouteReplay(mode="record", path=_record_path(tmp_path))
        rr.record_route_change(
            step=1, sim_t_s=5.0, veh_id="v1",
            control_mode="destination", choice_idx=0,
            chosen_name="shelter_a", chosen_edge="edge_dest",
            current_edge_before="edge_mid",
            applied_route_edges=["edge_start", "edge_mid", "edge_dest"],
        )
        rr.close()
        jsonl_files = [p for p in tmp_path.iterdir() if p.suffix == ".jsonl"]
        with open(jsonl_files[0]) as f:
            rec = json.loads(f.read().strip())
        # Route should start at "edge_mid", not "edge_start"
        assert rec["route_edges"][0] == "edge_mid"

    def test_close_is_idempotent(self, tmp_path):
        rr = RouteReplay(mode="record", path=_record_path(tmp_path))
        rr.close()
        rr.close()  # second call must not raise

    def test_record_in_replay_mode_is_noop(self, tmp_path):
        jsonl_path = str(tmp_path / "fixture.jsonl")
        _write_jsonl(jsonl_path, [_make_route_change()])
        rr = RouteReplay(mode="replay", path=jsonl_path)
        # record_route_change should be a no-op (does not write to file)
        rr.record_route_change(
            step=99, sim_t_s=0.0, veh_id="v_noop",
            control_mode="destination", choice_idx=0,
            chosen_name="n", chosen_edge=None,
            current_edge_before="e1", applied_route_edges=["e1"],
        )
        # No assertion needed — just verify it does not raise.


# ---------------------------------------------------------------------------
# Tests: replay mode
# ---------------------------------------------------------------------------

class TestRouteReplayReplayMode:
    def test_loads_schedule_from_jsonl(self, tmp_path):
        jsonl_path = str(tmp_path / "schedule.jsonl")
        _write_jsonl(jsonl_path, [_make_route_change(step=5, veh_id="v1")])
        rr = RouteReplay(mode="replay", path=jsonl_path)
        assert 5 in rr._schedule
        assert "v1" in rr._schedule[5]

    def test_non_route_change_events_ignored(self, tmp_path):
        jsonl_path = str(tmp_path / "schedule.jsonl")
        records = [
            _make_route_change(step=1),
            {"event": "agent_cognition", "step": 2, "veh_id": "v2", "time_s": 10.0},
            {"event": "metrics_snapshot", "step": 3, "time_s": 15.0},
        ]
        _write_jsonl(jsonl_path, records)
        rr = RouteReplay(mode="replay", path=jsonl_path)
        # Only step=1 (route_change) should be in the schedule.
        assert 1 in rr._schedule
        assert 2 not in rr._schedule
        assert 3 not in rr._schedule

    def test_multiple_vehicles_per_step(self, tmp_path):
        jsonl_path = str(tmp_path / "schedule.jsonl")
        records = [
            _make_route_change(step=1, veh_id="v1"),
            _make_route_change(step=1, veh_id="v2"),
        ]
        _write_jsonl(jsonl_path, records)
        rr = RouteReplay(mode="replay", path=jsonl_path)
        assert "v1" in rr._schedule[1]
        assert "v2" in rr._schedule[1]

    def test_empty_jsonl_produces_empty_schedule(self, tmp_path):
        jsonl_path = str(tmp_path / "empty.jsonl")
        _write_jsonl(jsonl_path, [])
        rr = RouteReplay(mode="replay", path=jsonl_path)
        assert rr._schedule == {}

    def test_apply_step_skips_unknown_step(self, tmp_path):
        jsonl_path = str(tmp_path / "schedule.jsonl")
        _write_jsonl(jsonl_path, [_make_route_change(step=1)])
        rr = RouteReplay(mode="replay", path=jsonl_path)
        # Step 99 has no actions; apply_step must not raise.
        rr.apply_step(99, ["v1"])


# ---------------------------------------------------------------------------
# Tests: invalid mode
# ---------------------------------------------------------------------------

class TestRouteReplayInvalidMode:
    def test_invalid_mode_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown RUN_MODE"):
            RouteReplay(mode="invalid_mode", path=str(tmp_path / "x.jsonl"))
