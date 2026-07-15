"""Unit tests for the per-run timeline export (``agentevac.utils.run_timeline``)."""

import json
from pathlib import Path

from agentevac.agents.alert_schedule import AlertSchedule
from agentevac.utils.run_timeline import RunTimeline


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _toy_alert_cfg():
    return {
        "areas": {"westwood": {"edges": ["w0", "w1"]}},
        "schedule": [
            {
                "id": "EA-1", "issue_time_s": 6300, "areas": ["westwood"],
                "instruction": "evacuate_now", "hazard_text": "Leave Westwood",
                "comfort_centre": "Black Point", "routing_text": None,
                "channel": "wireless_emergency_alert",
            }
        ],
        "door_to_door": {
            "start_time_s": 840,
            "sweep": [{"area": "westwood", "begin_s": 840, "cleared_by_s": 3600}],
        },
    }


class TestDisabled:
    def test_disabled_writes_nothing(self, tmp_path):
        tl = RunTimeline(False, str(tmp_path / "run_timeline.jsonl"))
        assert tl.path is None
        tl.emit(0.0, "fire", "ignition")  # no-op, must not raise
        tl.close()

    def test_none_path_disables(self):
        tl = RunTimeline(True, None)
        assert tl.enabled is False
        tl.emit(0.0, "fire", "ignition")
        tl.close()


class TestEmit:
    def test_row_schema(self, tmp_path):
        p = str(tmp_path / "run_timeline.jsonl")
        tl = RunTimeline(True, p)
        tl.emit(123.4, "awareness", "aware", agent_id="v1", area="westwood",
                source="alert", detail={"x": 1})
        tl.close()
        assert _read_rows(p) == [{
            "t_s": 123.4, "layer": "awareness", "type": "aware",
            "agent_id": "v1", "area": "westwood", "source": "alert", "detail": {"x": 1},
        }]

    def test_optional_fields_omitted_when_none(self, tmp_path):
        p = str(tmp_path / "run_timeline.jsonl")
        tl = RunTimeline(True, p)
        tl.emit(5, "arrival", "arrive", agent_id="v2")
        tl.close()
        row = _read_rows(p)[0]
        assert row == {"t_s": 5.0, "layer": "arrival", "type": "arrive", "agent_id": "v2"}
        assert "area" not in row and "source" not in row and "detail" not in row

    def test_t_s_rounded(self, tmp_path):
        p = str(tmp_path / "run_timeline.jsonl")
        tl = RunTimeline(True, p)
        tl.emit(1.23456, "fire", "ignition")
        tl.close()
        assert _read_rows(p)[0]["t_s"] == 1.23


class TestScripted:
    def test_scripted_fire_alert_door_rows(self, tmp_path):
        p = str(tmp_path / "run_timeline.jsonl")
        sched = AlertSchedule.from_config(_toy_alert_cfg())
        tl = RunTimeline(True, p)
        tl.emit_scripted(
            fire_sources=[{"id": "F1", "t0": 0.0, "x": 1, "y": 2, "r0": 30,
                           "growth_m_per_s": 0.3, "max_r_m": 700}],
            alert_events=sched.scheduled_events(),
            door_sweeps=sched.door_sweeps(),
        )
        tl.close()
        rows = _read_rows(p)
        triples = [(r["layer"], r["type"], r["t_s"]) for r in rows]
        assert ("fire", "ignition", 0.0) in triples
        assert ("alert", "evacuate_now", 6300.0) in triples
        assert ("door", "sweep_begin", 840.0) in triples
        assert ("door", "sweep_clear", 3600.0) in triples
        alert_row = next(r for r in rows if r["layer"] == "alert")
        assert alert_row["area"] == ["westwood"]
        assert alert_row["source"] == "wireless_emergency_alert"
        assert alert_row["detail"]["id"] == "EA-1"

    def test_scripted_alert_offset_applied(self, tmp_path):
        p = str(tmp_path / "run_timeline.jsonl")
        sched = AlertSchedule.from_config(_toy_alert_cfg(), time_offset_s=-1200)
        tl = RunTimeline(True, p)
        tl.emit_scripted(alert_events=sched.scheduled_events())
        tl.close()
        alert_row = next(r for r in _read_rows(p) if r["layer"] == "alert")
        assert alert_row["t_s"] == 6300 - 1200
