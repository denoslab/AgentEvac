"""Unit tests for the M1 alert-schedule resolver (``agentevac.agents.alert_schedule``)."""

import json
from pathlib import Path

import pytest

from agentevac.agents.alert_schedule import (
    AlertSchedule,
    NO_ALERT,
    effective_mode,
)


def _toy_config():
    """A two-area, two-event schedule mirroring the E0 shape at small scale."""
    return {
        "areas": {
            "westwood": {"edges": ["w0", "w1", "w2"]},
            "highland": {"edges": ["h0", "h1"]},
        },
        "schedule": [
            {
                "id": "EA-1", "issue_time_s": 6300, "areas": ["westwood"],
                "instruction": "evacuate_now", "hazard_text": "Leave Westwood",
                "comfort_centre": "Black Point", "routing_text": None,
                "channel": "wireless_emergency_alert",
            },
            {
                "id": "EA-2", "issue_time_s": 9660, "areas": ["highland"],
                "instruction": "evacuate_now", "hazard_text": "Leave Highland",
                "comfort_centre": None, "routing_text": None,
                "channel": "wireless_emergency_alert",
            },
        ],
        "door_to_door": {
            "start_time_s": 840,
            "initial_areas": ["westwood"],
            "sweep": [{"area": "westwood", "begin_s": 840, "cleared_by_s": 3600}],
        },
    }


def test_areas_for_edge_reverse_index():
    sched = AlertSchedule.from_config(_toy_config())
    assert sched.areas_for_edge("w1") == {"westwood"}
    assert sched.areas_for_edge("h0") == {"highland"}
    assert sched.areas_for_edge("unknown") == set()


def test_no_alert_before_first_order():
    sched = AlertSchedule.from_config(_toy_config())
    state = sched.active_for_edge(6299, "w0")
    assert state == NO_ALERT
    assert effective_mode(state) == "no_notice"


def test_westwood_ordered_after_ea1():
    sched = AlertSchedule.from_config(_toy_config())
    state = sched.active_for_edge(6300, "w0")
    assert state.received
    assert state.received_t_s == 6300
    assert state.instruction == "evacuate_now"
    assert state.order_text is not None
    assert state.order_text["comfort_centre"] == "Black Point"
    # routing_text is None, so this is the (1,1,0) view: alert_guided filter plus order.
    assert effective_mode(state) == "alert_guided"


def test_cumulative_extension_keeps_earlier_order():
    sched = AlertSchedule.from_config(_toy_config())
    # Highland becomes ordered only at EA-2.
    assert sched.active_for_edge(9659, "h0") == NO_ALERT
    assert sched.active_for_edge(9660, "h0").received
    # Westwood is still ordered from EA-1 after EA-2 fires.
    westwood = sched.active_for_edge(9660, "w0")
    assert westwood.received and westwood.received_t_s == 6300


def test_routing_visible_flips_to_advice():
    cfg = _toy_config()
    cfg["schedule"][0]["routing_text"] = "Take Hammonds Plains Rd west"
    sched = AlertSchedule.from_config(cfg)
    state = sched.active_for_edge(6300, "w0")
    assert state.routing_visible
    assert effective_mode(state) == "advice_guided"
    assert state.order_text["routing_text"] == "Take Hammonds Plains Rd west"


def test_empty_schedule_always_no_alert():
    sched = AlertSchedule.empty()
    assert sched.active_for_edge(99999, "w0") == NO_ALERT
    assert effective_mode(sched.active_for_edge(99999, "w0")) == "no_notice"


def test_time_offset_shifts_issue_times():
    sched = AlertSchedule.from_config(_toy_config(), time_offset_s=-3600)
    # EA-1 now issues an hour earlier, at 2700 s.
    assert sched.active_for_edge(2699, "w0") == NO_ALERT
    assert sched.active_for_edge(2700, "w0").received


def test_door_knock_time_spread_across_window():
    sched = AlertSchedule.from_config(_toy_config())
    # Three Westwood edges spread linearly across [840, 3600].
    assert sched.door_knock_time("w0") == 840
    assert sched.door_knock_time("w2") == 3600
    assert sched.door_knock_time("w1") == pytest.approx(840 + (3600 - 840) * 0.5)
    # Highland is not swept.
    assert sched.door_knock_time("h0") is None


def test_door_sweep_scale_stretches_knock_window():
    # Doubling the sweep duration widens [840, 3600] to [840, 6360]; the first edge
    # still knocks at the fixed start and the last at the new clear-by time.
    sched = AlertSchedule.from_config(_toy_config(), door_sweep_scale=2.0)
    assert sched.door_knock_time("w0") == 840
    assert sched.door_knock_time("w1") == pytest.approx(3600.0)
    assert sched.door_knock_time("w2") == pytest.approx(6360.0)
    assert sched.door_sweeps() == [("westwood", 840.0, 6360.0)]


def test_door_sweep_scale_default_is_identity():
    scaled = AlertSchedule.from_config(_toy_config(), door_sweep_scale=1.0)
    base = AlertSchedule.from_config(_toy_config())
    assert scaled.door_sweeps() == base.door_sweeps()
    assert scaled.door_knock_time("w2") == base.door_knock_time("w2")


def test_instruction_none_is_hazard_only():
    cfg = _toy_config()
    cfg["schedule"][0]["instruction"] = "none"
    sched = AlertSchedule.from_config(cfg)
    state = sched.active_for_edge(6300, "w0")
    assert state.received and state.instruction == "none"
    assert state.order_text is None  # an alert exists but carries no departure order
    assert effective_mode(state) == "alert_guided"


def test_scheduled_events_offset_applied():
    sched = AlertSchedule.from_config(_toy_config(), time_offset_s=-600)
    events = sched.scheduled_events()
    assert [e.id for e in events] == ["EA-1", "EA-2"]
    assert events[0].issue_time_s == 6300 - 600  # the E1 offset is baked in
    assert events[0].areas == ("westwood",)


def test_door_sweeps_returns_tuples():
    sched = AlertSchedule.from_config(_toy_config())
    assert sched.door_sweeps() == [("westwood", 840.0, 3600.0)]


def test_ordered_areas_channels_and_edges():
    ordered = AlertSchedule.from_config(_toy_config()).ordered_areas()
    assert set(ordered) == {"westwood", "highland"}
    # Westwood is door-swept at 840, before the 6300 broadcast, so it is tagged door.
    assert ordered["westwood"]["channel"] == "door"
    assert ordered["westwood"]["order_t_s"] == 840
    assert ordered["westwood"]["edges"] == ["w0", "w1", "w2"]
    # Highland has no door sweep, so it is tagged broadcast at its alert time.
    assert ordered["highland"]["channel"] == "broadcast"
    assert ordered["highland"]["order_t_s"] == 9660


def test_ordered_areas_broadcast_before_door_stays_broadcast():
    cfg = _toy_config()
    # Push the door sweep after the broadcast; the earlier broadcast sets the channel.
    cfg["door_to_door"]["sweep"][0]["begin_s"] = 7000
    cfg["door_to_door"]["sweep"][0]["cleared_by_s"] = 9000
    ordered = AlertSchedule.from_config(cfg).ordered_areas()
    assert ordered["westwood"]["channel"] == "broadcast"
    assert ordered["westwood"]["order_t_s"] == 6300


def test_empty_schedule_has_no_ordered_areas():
    assert AlertSchedule.empty().ordered_areas() == {}


def test_parses_real_e0_schedule():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs" / "halifax_3town" / "alerts.json"
    )
    cfg = json.loads(path.read_text())
    sched = AlertSchedule.from_config(cfg)
    westwood_edge = cfg["areas"]["westwood_hills"]["edges"][0]
    outer_edge = cfg["areas"]["outer_extension"]["edges"][0]
    # Westwood ordered at EA-1 (6300 s), not before.
    assert sched.active_for_edge(6299, westwood_edge) == NO_ALERT
    assert sched.active_for_edge(6300, westwood_edge).instruction == "evacuate_now"
    # Outer extension ordered only at EA-3 (15180 s).
    assert sched.active_for_edge(15179, outer_edge) == NO_ALERT
    assert sched.active_for_edge(15180, outer_edge).received
