"""Snapshot and preview assembly, exercised without SUMO or TraCI.

The collector reads a simulation module through plain attribute access, so a
stand-in namespace with the same field names is enough to test it. That is also
what keeps the collector honest: it must degrade rather than raise when the
simulator does not carry a field it hoped for.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ui.bridge import collector

# One metre of simulation coordinate maps to one unit of longitude here, which
# keeps the expected values in these tests readable.
IDENTITY = collector.GeoProjector(lambda x, y: (x / 1000.0, y / 1000.0))


class FakeSchedule:
    def __init__(self):
        self._areas = {"westwood_hills": ["edgeA", "edgeB"], "highland_park": ["edgeC"]}

    def areas_for_edge(self, edge_id):
        return {name for name, edges in self._areas.items() if edge_id in edges}

    def ordered_areas(self):
        return {
            "westwood_hills": {"order_t_s": 840.0, "channel": "door", "edges": self._areas["westwood_hills"]},
            "highland_park": {"order_t_s": 9660.0, "channel": "broadcast", "edges": self._areas["highland_park"]},
        }

    def scheduled_events(self):
        return [
            SimpleNamespace(event_id="EA-1", issue_time_s=6300.0, instruction="evacuate_now", areas=("westwood_hills",)),
            SimpleNamespace(event_id="EA-2", issue_time_s=9660.0, instruction="evacuate_now", areas=("highland_park",)),
        ]


def build_module(**overrides):
    spawns = [
        ("h1", "edgeA", "dest1", 0.0, "first", "10", "max", (255, 0, 0, 255)),
        ("h2", "edgeB", "dest1", 0.0, "first", "10", "max", (255, 0, 0, 255)),
        ("h3", "edgeC", "dest1", 0.0, "first", "10", "max", (255, 0, 0, 255)),
    ]
    module = SimpleNamespace(
        SPAWN_EVENTS=spawns,
        SPAWN_EDGE_BY_AGENT={"h1": "edgeA", "h2": "edgeB", "h3": "edgeC"},
        SPAWN_EDGE_MIDPOINT={"h1": (1000.0, 2000.0), "h2": (1100.0, 2000.0), "h3": (5000.0, 6000.0)},
        EDGE_SHAPE={
            "edgeA": [(990.0, 2000.0), (1010.0, 2000.0)],
            "edgeB": [(1090.0, 2000.0), (1110.0, 2000.0)],
            "edgeC": [(4990.0, 6000.0), (5010.0, 6000.0)],
            "dest1": [(9000.0, 9000.0), (9010.0, 9000.0)],
        },
        DESTINATION_LIBRARY=[{"name": "Black_Point", "edge": "dest1"}],
        FIRE_SOURCES=[{"id": "F1", "x": 1500.0, "y": 2500.0, "r0": 30.0, "growth_m_per_s": 0.5, "t0": 0.0}],
        NEW_FIRE_EVENTS=[],
        ALERT_SCHEDULE=FakeSchedule(),
        agent_live_status={},
        AWARENESS_LOG={},
        metrics=SimpleNamespace(_arrival_times={}, _depart_times={}, _fire_contact_agents=set()),
        active_fires=lambda t: [{"id": "F1", "x": 1500.0, "y": 2500.0, "r": 30.0 + 0.5 * t}],
        net=None,
    )
    for key, value in overrides.items():
        setattr(module, key, value)
    return module


def snapshot_for(module, sim_t_s=1000.0):
    return collector.build_snapshot(
        module,
        IDENTITY,
        sim_t_s=sim_t_s,
        step_idx=42,
        intent={"paused": False, "speed": 16.0},
        round_progress={"in_progress": False, "index": 4},
        area_members=collector.build_area_members(module),
    )


def test_every_household_is_accounted_for_exactly_once():
    module = build_module()
    module.agent_live_status = {"h2": {"active": True, "pos_xy": [2000.0, 3000.0], "current_edge": "edgeX"}}
    module.metrics._arrival_times = {"h3": 500.0}
    module.metrics._depart_times = {"h2": 300.0, "h3": 200.0}

    snapshot = snapshot_for(module)
    counts = snapshot["counts"]
    assert counts["total"] == 3
    assert counts["waiting"] + counts["evacuating"] + counts["arrived"] == counts["total"]
    assert (counts["waiting"], counts["evacuating"], counts["arrived"]) == (1, 1, 1)
    assert len(snapshot["agents"]) == 3


def test_an_arrived_household_is_never_also_evacuating():
    module = build_module()
    # The simulator can still hold a stale active flag on the step an agent
    # leaves the network, and arrival is the stronger fact.
    module.agent_live_status = {"h1": {"active": True, "pos_xy": [1.0, 1.0]}}
    module.metrics._arrival_times = {"h1": 100.0}
    snapshot = snapshot_for(module)
    statuses = {agent["id"]: agent["status"] for agent in snapshot["agents"]}
    assert statuses["h1"] == "arrived"
    assert snapshot["counts"]["evacuating"] == 0


def test_a_waiting_household_is_drawn_at_its_home_edge():
    snapshot = snapshot_for(build_module())
    positions = {agent["id"]: (agent["lon"], agent["lat"]) for agent in snapshot["agents"]}
    assert positions["h1"] == pytest.approx((1.0, 2.0))


def test_an_evacuating_household_is_drawn_at_its_live_position():
    module = build_module()
    module.agent_live_status = {"h1": {"active": True, "pos_xy": [7000.0, 8000.0], "current_edge": "edgeQ"}}
    snapshot = snapshot_for(module)
    agent = next(a for a in snapshot["agents"] if a["id"] == "h1")
    assert (agent["lon"], agent["lat"]) == pytest.approx((7.0, 8.0))
    assert agent["edge"] == "edgeQ"


def test_fire_radius_travels_in_metres_and_the_centre_in_degrees():
    snapshot = snapshot_for(build_module(), sim_t_s=200.0)
    fire = snapshot["fires"][0]
    assert (fire["lon"], fire["lat"]) == pytest.approx((1.5, 2.5))
    assert fire["r_m"] == pytest.approx(130.0)


def test_alerts_split_into_issued_and_pending_at_the_current_time():
    snapshot = snapshot_for(build_module(), sim_t_s=7000.0)
    alerts = snapshot["alerts"]
    assert [event["id"] for event in alerts["issued"]] == ["EA-1"]
    assert [event["id"] for event in alerts["pending"]] == ["EA-2"]
    assert alerts["next"]["id"] == "EA-2"


def test_area_progress_counts_departures_per_community():
    module = build_module()
    module.metrics._depart_times = {"h1": 900.0}
    snapshot = snapshot_for(module, sim_t_s=1000.0)
    by_name = {row["name"]: row for row in snapshot["areas"]}
    assert by_name["westwood_hills"]["households"] == 2
    assert by_name["westwood_hills"]["departed"] == 1
    assert by_name["westwood_hills"]["ordered"] is True
    # The second community's order is still in the future at this instant.
    assert by_name["highland_park"]["ordered"] is False


def test_a_missing_simulator_field_thins_the_payload_instead_of_raising():
    module = build_module()
    del module.AWARENESS_LOG
    del module.active_fires
    snapshot = snapshot_for(module)
    assert snapshot["counts"]["aware"] == 0
    assert snapshot["fires"] == []


def test_preview_carries_the_static_geography():
    preview = collector.build_preview(build_module(), IDENTITY)
    assert [h["id"] for h in preview["households"]] == ["h1", "h2", "h3"]
    assert preview["destinations"][0]["name"] == "Black_Point"
    assert preview["fire_sources"][0]["t0_s"] == 0.0
    assert [event["id"] for event in preview["alert_events"]] == ["EA-1", "EA-2"]
    assert {area["name"] for area in preview["areas"]} == {"westwood_hills", "highland_park"}
    assert preview["bbox"] is not None


def test_preview_hull_needs_three_distinct_points():
    # Two households cannot enclose an area, so the outline is left empty and the
    # console falls back to drawing the members.
    preview = collector.build_preview(build_module(), IDENTITY)
    westwood = next(area for area in preview["areas"] if area["name"] == "westwood_hills")
    assert westwood["hull"] == []
    assert westwood["households"] == 2


def test_convex_hull_closes_its_ring():
    ring = collector._convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert ring[0] == ring[-1]
    assert len(ring) == 5  # four corners plus the closing point


def test_points_a_projection_cannot_place_are_dropped_not_faked():
    failing = collector.GeoProjector(lambda x, y: (_ for _ in ()).throw(ValueError("no projection")))
    snapshot = snapshot_for(build_module())
    assert len(snapshot["agents"]) == 3
    empty = collector.build_snapshot(
        build_module(),
        failing,
        sim_t_s=10.0,
        step_idx=1,
        intent={"paused": False, "speed": 1.0},
        round_progress={},
        area_members={},
    )
    assert empty["agents"] == []
    # The count of households is still truthful even when none can be drawn.
    assert empty["counts"]["total"] == 3
