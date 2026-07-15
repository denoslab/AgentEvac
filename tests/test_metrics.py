"""Unit tests for agentevac.analysis.metrics."""

import json
import math
import os
import tempfile

import pytest

from agentevac.analysis.metrics import RunMetricsCollector


def _make_collector(enabled=True, tmp_dir=None):
    base = os.path.join(tmp_dir or tempfile.mkdtemp(), "metrics_test.json")
    return RunMetricsCollector(enabled=enabled, base_path=base, run_mode="record")


class TestDisabledCollector:
    def test_record_departure_is_noop(self, tmp_path):
        c = _make_collector(enabled=False, tmp_dir=str(tmp_path))
        c.record_departure("v1", 10.0)
        assert c._depart_times == {}

    def test_observe_active_vehicles_is_noop(self, tmp_path):
        c = _make_collector(enabled=False, tmp_dir=str(tmp_path))
        c.observe_active_vehicles(["v1"], 10.0)
        assert c._last_seen_active == set()

    def test_export_returns_none(self, tmp_path):
        c = _make_collector(enabled=False, tmp_dir=str(tmp_path))
        assert c.export_run_metrics() is None

    def test_close_returns_none(self, tmp_path):
        c = _make_collector(enabled=False, tmp_dir=str(tmp_path))
        assert c.close() is None


class TestRecordDeparture:
    def test_records_first_departure(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 100.0)
        assert c._depart_times["v1"] == pytest.approx(100.0)

    def test_second_departure_for_same_agent_is_ignored(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 100.0)
        c.record_departure("v1", 200.0)
        assert c._depart_times["v1"] == pytest.approx(100.0)

    def test_multiple_agents_recorded_independently(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 10.0)
        c.record_departure("v2", 20.0)
        assert len(c._depart_times) == 2


class TestObserveActiveVehicles:
    def test_does_not_infer_arrival_on_disappearance(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 0.0)
        c.observe_active_vehicles(["v1"], 10.0)
        c.observe_active_vehicles([], 20.0)
        assert "v1" not in c._arrival_times

    def test_tracks_last_seen_active_set(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.observe_active_vehicles(["v1"], 10.0)
        assert c._last_seen_active == {"v1"}
        assert c._last_seen_time["v1"] == pytest.approx(10.0)

    def test_observe_active_updates_last_seen_time(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 0.0)
        c.observe_active_vehicles(["v1"], 10.0)
        c.observe_active_vehicles(["v1"], 30.0)
        assert c._last_seen_time["v1"] == pytest.approx(30.0)


class TestRecordArrival:
    def test_records_explicit_arrival(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 0.0)
        c.record_arrival("v1", 20.0)
        assert c._arrival_times["v1"] == pytest.approx(20.0)

    def test_requires_prior_departure(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_arrival("v1", 20.0)
        assert "v1" not in c._arrival_times

    def test_second_arrival_for_same_agent_is_ignored(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 0.0)
        c.record_arrival("v1", 20.0)
        c.record_arrival("v1", 40.0)
        assert c._arrival_times["v1"] == pytest.approx(20.0)


class TestDepartureTimeVariability:
    def test_no_agents_returns_zero(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        assert c.compute_departure_time_variability() == pytest.approx(0.0)

    def test_single_agent_returns_zero(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 50.0)
        assert c.compute_departure_time_variability() == pytest.approx(0.0)

    def test_two_agents_variance(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 0.0)
        c.record_departure("v2", 10.0)
        # mean=5, variance = ((0-5)^2 + (10-5)^2) / 2 = 25
        assert c.compute_departure_time_variability() == pytest.approx(25.0, rel=1e-6)

    def test_identical_departure_times_zero_variance(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 30.0)
        c.record_departure("v2", 30.0)
        assert c.compute_departure_time_variability() == pytest.approx(0.0, abs=1e-9)


class TestRouteChoiceEntropy:
    def test_no_choices_returns_zero(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        assert c.compute_route_choice_entropy() == pytest.approx(0.0)

    def test_single_choice_zero_entropy(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c._choice_counts["destination::shelter_a"] = 5
        assert c.compute_route_choice_entropy() == pytest.approx(0.0, abs=1e-9)

    def test_uniform_two_choices_max_entropy(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c._choice_counts["destination::shelter_a"] = 5
        c._choice_counts["destination::shelter_b"] = 5
        assert c.compute_route_choice_entropy() == pytest.approx(math.log(2), rel=1e-6)

    def test_entropy_increases_with_more_equal_choices(self, tmp_path):
        c2 = _make_collector(tmp_dir=str(tmp_path))
        c3 = _make_collector(tmp_dir=str(tmp_path))
        c2._choice_counts = {"a": 1, "b": 1}
        c3._choice_counts = {"a": 1, "b": 1, "c": 1}
        assert c3.compute_route_choice_entropy() > c2.compute_route_choice_entropy()


class TestDecisionInstability:
    def test_no_decisions_returns_zeros(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        result = c.compute_decision_instability()
        assert result["average_changes"] == pytest.approx(0.0)
        assert result["max_changes"] == 0

    def test_no_change_counts_as_zero(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        state = {"control_mode": "destination", "selected_option": {"name": "shelter_a"}}
        c.record_decision_snapshot("v1", 0.0, 1, state, 0, "depart_now")
        c.record_decision_snapshot("v1", 5.0, 2, state, 0, "depart_now")
        result = c.compute_decision_instability()
        assert result["per_agent_changes"]["v1"] == 0

    def test_detects_one_change(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        state1 = {"control_mode": "destination", "selected_option": {"name": "shelter_a"}}
        state2 = {"control_mode": "destination", "selected_option": {"name": "shelter_b"}}
        c.record_decision_snapshot("v1", 0.0, 1, state1, 0, "depart_now")
        c.record_decision_snapshot("v1", 5.0, 2, state2, 1, "depart_now")
        result = c.compute_decision_instability()
        assert result["per_agent_changes"]["v1"] == 1


class TestHazardExposure:
    def test_no_samples_returns_zero(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        result = c.compute_average_hazard_exposure()
        assert result["global_average"] == pytest.approx(0.0)
        assert result["sample_count"] == 0

    def test_averages_risk_scores(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_exposure_sample("v1", 0.0, "e1", None, risk_score=0.5)
        c.record_exposure_sample("v1", 5.0, "e1", None, risk_score=1.0)
        result = c.compute_average_hazard_exposure()
        assert result["global_average"] == pytest.approx(0.75, rel=1e-6)

    def test_none_risk_score_treated_as_zero(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_exposure_sample("v1", 0.0, "e1", None, risk_score=None)
        result = c.compute_average_hazard_exposure()
        assert result["global_average"] == pytest.approx(0.0)

    def test_per_agent_exposure_computed(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_exposure_sample("v1", 0.0, "e1", None, risk_score=0.2)
        c.record_exposure_sample("v2", 0.0, "e2", None, risk_score=0.8)
        result = c.compute_average_hazard_exposure()
        assert "v1" in result["per_agent_average"]
        assert "v2" in result["per_agent_average"]


class TestAverageTravelTime:
    def test_no_arrivals_returns_zero_average(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 0.0)
        result = c.compute_average_travel_time()
        assert result["average"] == pytest.approx(0.0)
        assert result["completed_agents"] == 0

    def test_correct_travel_time_computed(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 10.0)
        c.record_arrival("v1", 70.0)
        result = c.compute_average_travel_time()
        assert result["average"] == pytest.approx(60.0, rel=1e-6)
        assert result["completed_agents"] == 1


class TestDestinationChoiceShare:
    def test_no_destination_choices_returns_empty_summary(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        result = c.compute_destination_choice_share()
        assert result["counts"] == {}
        assert result["fractions"] == {}
        assert result["total_agents_with_destination"] == 0

    def test_uses_latest_destination_per_agent(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        state_a = {"control_mode": "destination", "selected_option": {"name": "shelter_a"}}
        state_b = {"control_mode": "destination", "selected_option": {"name": "shelter_b"}}
        c.record_decision_snapshot("v1", 0.0, 1, state_a, 0, "depart_now")
        c.record_decision_snapshot("v1", 5.0, 2, state_b, 1, "depart_now")
        result = c.compute_destination_choice_share()
        assert result["counts"] == {"shelter_b": 1}
        assert result["fractions"]["shelter_b"] == pytest.approx(1.0)

    def test_aggregates_counts_and_fractions_across_agents(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        state_a = {"control_mode": "destination", "selected_option": {"name": "shelter_a"}}
        state_b = {"control_mode": "destination", "selected_option": {"name": "shelter_b"}}
        route_state = {"control_mode": "route", "selected_option": {"name": "route_1"}}
        c.record_decision_snapshot("v1", 0.0, 1, state_a, 0, "depart_now")
        c.record_decision_snapshot("v2", 0.0, 1, state_b, 1, "depart_now")
        c.record_decision_snapshot("v3", 0.0, 1, state_b, 1, "depart_now")
        c.record_decision_snapshot("v4", 0.0, 1, route_state, 0, "keep_route")
        result = c.compute_destination_choice_share()
        assert result["counts"] == {"shelter_a": 1, "shelter_b": 2}
        assert result["fractions"]["shelter_a"] == pytest.approx(1.0 / 3.0)
        assert result["fractions"]["shelter_b"] == pytest.approx(2.0 / 3.0)
        assert result["total_agents_with_destination"] == 3


class TestSignalConflict:
    def test_no_samples_returns_zero(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        result = c.compute_average_signal_conflict()
        assert result["global_average"] == pytest.approx(0.0)
        assert result["sample_count"] == 0

    def test_averages_conflict_scores(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_conflict_sample("v1", 0.4)
        c.record_conflict_sample("v1", 0.8)
        result = c.compute_average_signal_conflict()
        assert result["global_average"] == pytest.approx(0.6, rel=1e-6)
        assert result["sample_count"] == 2

    def test_per_agent_conflict_computed(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_conflict_sample("v1", 0.2)
        c.record_conflict_sample("v2", 0.6)
        result = c.compute_average_signal_conflict()
        assert result["per_agent_average"]["v1"] == pytest.approx(0.2)
        assert result["per_agent_average"]["v2"] == pytest.approx(0.6)

    def test_single_sample_returns_exact_value(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_conflict_sample("v1", 0.73)
        result = c.compute_average_signal_conflict()
        assert result["global_average"] == pytest.approx(0.73)
        assert result["sample_count"] == 1


class TestFireContact:
    def test_margin_zero_counts_as_contact(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.total_agents = 2
        c.record_exposure_sample("v1", 0.0, "e1", 0.0, risk_score=1.0)
        c.record_exposure_sample("v2", 0.0, "e2", 500.0, risk_score=0.1)
        s = c.summary()
        assert s["fire_contact"]["agents_ever_in_contact"] == 1
        assert "v1" in s["fire_contact"]["agent_ids"]
        assert "v2" not in s["fire_contact"]["agent_ids"]

    def test_negative_margin_counts_as_contact(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.total_agents = 1
        c.record_exposure_sample("v1", 0.0, "e1", -50.0, risk_score=1.0)
        assert "v1" in c._fire_contact_agents

    def test_positive_margin_not_contact(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_exposure_sample("v1", 0.0, "e1", 100.0, risk_score=0.5)
        assert "v1" not in c._fire_contact_agents

    def test_none_margin_not_contact(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_exposure_sample("v1", 0.0, "e1", None, risk_score=0.5)
        assert "v1" not in c._fire_contact_agents

    def test_fire_contact_fraction(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.total_agents = 4
        c.record_exposure_sample("v1", 0.0, "e1", -10.0, risk_score=1.0)
        c.record_exposure_sample("v2", 0.0, "e2", 0.0, risk_score=1.0)
        s = c.summary()
        assert s["fire_contact"]["agents_ever_in_contact"] == 2
        assert s["fire_contact"]["fraction_of_total"] == pytest.approx(0.5)

    def test_fire_contact_deduplicates(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.total_agents = 1
        c.record_exposure_sample("v1", 0.0, "e1", -10.0, risk_score=1.0)
        c.record_exposure_sample("v1", 5.0, "e2", -5.0, risk_score=1.0)
        assert len(c._fire_contact_agents) == 1


class TestSummaryAndExport:
    def test_summary_is_json_serializable(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 10.0)
        json.dumps(c.summary())  # must not raise

    def test_summary_contains_required_keys(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        s = c.summary()
        for key in (
            "run_mode", "departed_agents", "arrived_agents",
            "departure_time_variability", "route_choice_entropy",
            "decision_instability",
            "destination_choice_share",
            "average_signal_conflict",
            "fire_contact",
        ):
            assert key in s

    def test_export_writes_valid_json_file(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        output = os.path.join(str(tmp_path), "out.json")
        path = c.export_run_metrics(path=output)
        assert path == output
        with open(output) as f:
            data = json.load(f)
        assert "departure_time_variability" in data

    def test_close_exports_file(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        path = c.close()
        assert path is not None
        assert os.path.exists(path)


# --- Part J: awareness, order compliance, clearance, corridor flow, fire reach ---

def _ordered_collector(tmp_path):
    """A collector wired with a two-area order schedule and a corridor split.

    Westwood is tagged door and Highland broadcast.  Agents a,b sit in Westwood, c,d in
    Highland, and e is in no ordered area.
    """
    ordered_areas = {
        "westwood": {"channel": "door", "order_t_s": 840, "edges": ["w0", "w1"]},
        "highland": {"channel": "broadcast", "order_t_s": 9660, "edges": ["h0", "h1"]},
    }
    spawn_edge_by_agent = {
        "a": "w0", "b": "w1",
        "c": "h0", "d": "h1",
        "e": "x0",
    }
    corridor_edges = {"NS101": ["c1a", "c1b"], "NS103": ["c3a"]}
    base = os.path.join(str(tmp_path), "run_metrics.json")
    return RunMetricsCollector(
        enabled=True, base_path=base, run_mode="record",
        ordered_areas=ordered_areas, spawn_edge_by_agent=spawn_edge_by_agent,
        corridor_edges=corridor_edges,
    )


class TestDepartureReason:
    def test_stores_and_returns_reason(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 100.0, "order")
        assert c.departure_reason("v1") == "order"

    def test_reason_histogram_counts_order(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 10.0, "order")
        c.record_departure("v2", 20.0, "order")
        c.record_departure("v3", 30.0, "risk_threshold")
        assert c.compute_departure_reasons() == {"order": 2, "risk_threshold": 1}

    def test_first_reason_wins(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 10.0, "order")
        c.record_departure("v1", 20.0, "risk_threshold")
        assert c.departure_reason("v1") == "order"


class TestMobilizationDelay:
    def test_delay_awareness_to_departure(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_awareness("v1", 500.0, "alert")
        c.record_departure("v1", 1000.0, "order")
        result = c.compute_mobilization_delay()
        assert result["per_agent"]["v1"] == pytest.approx(500.0)
        assert result["average"] == pytest.approx(500.0)
        assert result["count"] == 1

    def test_aware_but_never_departs_excluded(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_awareness("v1", 200.0, "peer")
        result = c.compute_mobilization_delay()
        assert result["count"] == 0
        assert "v1" not in result["per_agent"]


class TestAwarenessSourceShare:
    def test_source_share_and_never_aware(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.total_agents = 5
        c.record_awareness("a", 100.0, "alert")
        c.record_awareness("b", 100.0, "peer")
        c.record_awareness("c", 100.0, "alert")
        s = c.summary()
        assert s["n_aware"] == 3
        assert s["n_never_aware"] == 2
        assert s["awareness_source_share"]["counts"] == {"alert": 2, "peer": 1}
        assert s["awareness_source_share"]["share"]["alert"] == pytest.approx(2 / 3)


class TestOrderCompliance:
    def test_split_by_channel_and_area(self, tmp_path):
        c = _ordered_collector(tmp_path)
        c.record_departure("a", 1000.0, "order")
        c.record_departure("c", 10000.0, "risk_threshold")
        comp = c.compute_order_compliance()
        assert comp["overall"] == {"ordered": 4, "evacuated": 2, "rate": pytest.approx(0.5)}
        assert comp["by_channel"]["door"]["ordered"] == 2
        assert comp["by_channel"]["door"]["evacuated"] == 1
        assert comp["by_channel"]["broadcast"]["evacuated"] == 1
        assert comp["by_area"]["westwood"]["rate"] == pytest.approx(0.5)

    def test_agent_outside_ordered_area_not_counted(self, tmp_path):
        c = _ordered_collector(tmp_path)
        c.record_departure("e", 500.0, "risk_threshold")  # e is in no ordered area
        comp = c.compute_order_compliance()
        assert comp["overall"]["ordered"] == 4

    def test_full_compliance_rate_one(self, tmp_path):
        c = _ordered_collector(tmp_path)
        for agent, t in [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]:
            c.record_departure(agent, t, "order")
        assert c.compute_order_compliance()["overall"]["rate"] == pytest.approx(1.0)


class TestAreaClearance:
    def test_latest_departure_per_area(self, tmp_path):
        c = _ordered_collector(tmp_path)
        c.record_departure("a", 1000.0, "order")
        c.record_departure("c", 9000.0, "order")
        c.record_departure("d", 12000.0, "order")
        clear = c.compute_area_clearance()
        assert clear["westwood"]["clearance_t_s"] == pytest.approx(1000.0)
        assert clear["westwood"]["fully_cleared"] is False  # b stayed
        assert clear["highland"]["clearance_t_s"] == pytest.approx(12000.0)
        assert clear["highland"]["fully_cleared"] is True

    def test_no_departures_gives_none_clearance(self, tmp_path):
        c = _ordered_collector(tmp_path)
        clear = c.compute_area_clearance()
        assert clear["westwood"]["clearance_t_s"] is None
        assert clear["westwood"]["departed"] == 0


class TestNonEvacueeFireReach:
    def test_stayer_reached_by_fire_counted(self, tmp_path):
        c = _ordered_collector(tmp_path)
        c.record_fire_reached_edge("x0", 5000.0)   # e's spawn edge
        c.record_departure("a", 100.0, "order")     # a evacuates
        c.record_fire_reached_edge("w0", 200.0)     # a's edge reached, but a is not a stayer
        result = c.compute_non_evacuated_reached_by_fire()
        assert result["count"] == 1
        assert result["agent_ids"] == ["e"]

    def test_conditional_exposure_excludes_stayers(self, tmp_path):
        c = _ordered_collector(tmp_path)
        c.record_departure("a", 100.0, "order")
        c.record_exposure_sample("a", 200.0, "w0", None, risk_score=0.5)
        exposure = c.compute_average_hazard_exposure()
        assert "a" in exposure["per_agent_average"]
        assert "e" not in exposure["per_agent_average"]


class TestCorridorFlow:
    def test_distinct_agents_per_corridor(self, tmp_path):
        c = _ordered_collector(tmp_path)
        c.record_exposure_sample("a", 0.0, "c1a", None, risk_score=0.1)
        c.record_exposure_sample("a", 5.0, "c1b", None, risk_score=0.1)  # dedup same agent
        c.record_exposure_sample("b", 0.0, "c1b", None, risk_score=0.1)
        c.record_exposure_sample("c", 0.0, "c3a", None, risk_score=0.1)
        flow = c.compute_corridor_flow()
        assert flow["NS101"]["agents"] == 2
        assert flow["NS101"]["agent_ids"] == ["a", "b"]
        assert flow["NS103"]["agents"] == 1

    def test_no_corridors_configured_reports_empty(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        assert c.compute_corridor_flow() == {}
