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
    def test_infers_arrival_on_disappearance(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 0.0)
        c.observe_active_vehicles(["v1"], 10.0)
        c.observe_active_vehicles([], 20.0)
        assert "v1" in c._arrival_times
        assert c._arrival_times["v1"] == pytest.approx(20.0)

    def test_no_arrival_without_departure(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.observe_active_vehicles(["v1"], 10.0)
        c.observe_active_vehicles([], 20.0)
        assert "v1" not in c._arrival_times

    def test_vehicle_not_re_recorded_if_already_arrived(self, tmp_path):
        c = _make_collector(tmp_dir=str(tmp_path))
        c.record_departure("v1", 0.0)
        c.observe_active_vehicles(["v1"], 10.0)
        c.observe_active_vehicles([], 20.0)  # v1 arrives at 20
        c.observe_active_vehicles(["v1"], 30.0)
        c.observe_active_vehicles([], 40.0)  # would arrive again at 40
        # First arrival timestamp must be preserved
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
        c.observe_active_vehicles(["v1"], 10.0)
        c.observe_active_vehicles([], 70.0)
        result = c.compute_average_travel_time()
        assert result["average"] == pytest.approx(60.0, rel=1e-6)
        assert result["completed_agents"] == 1


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
