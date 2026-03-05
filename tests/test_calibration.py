"""Unit tests for agentevac.analysis.calibration."""

import json
import os

import pytest

from agentevac.analysis.calibration import (
    export_calibration_report,
    fit_agent_parameters,
    load_reference_scenario,
    score_run_against_reference,
)


def _ref():
    return {
        "departure_time_variability": 100.0,
        "route_choice_entropy": 1.0,
        "decision_instability": {"average_changes": 0.2},
        "average_hazard_exposure": {"global_average": 0.3},
        "average_travel_time": {"average": 300.0},
        "arrived_agents": 10,
    }


def _run(overrides=None):
    base = dict(_ref())
    if overrides:
        base.update(overrides)
    return base


class TestScoreRunAgainstReference:
    def test_perfect_match_gives_fit_score_one(self):
        result = score_run_against_reference(_run(), _ref())
        assert result["fit_score"] == pytest.approx(1.0, rel=1e-6)
        assert result["normalized_loss"] == pytest.approx(0.0, abs=1e-9)

    def test_worse_run_lower_fit_score(self):
        bad_run = _run({"departure_time_variability": 1000.0})
        perfect = score_run_against_reference(_run(), _ref())
        worse = score_run_against_reference(bad_run, _ref())
        assert worse["fit_score"] < perfect["fit_score"]

    def test_fit_score_bounded_zero_to_one(self):
        extreme_run = _run({
            "departure_time_variability": 1e9,
            "average_travel_time": {"average": 1e9},
        })
        result = score_run_against_reference(extreme_run, _ref())
        assert 0.0 <= result["fit_score"] <= 1.0

    def test_metric_count_nonzero(self):
        result = score_run_against_reference(_run(), _ref())
        assert result["metric_count"] > 0

    def test_metric_details_contains_departure_variability(self):
        result = score_run_against_reference(_run(), _ref())
        assert "departure_time_variability" in result["metric_details"]

    def test_custom_weights_affect_score(self):
        run_a = _run({"departure_time_variability": 1000.0})
        run_b = _run({"route_choice_entropy": 5.0})
        # High weight on departure_variability makes run_a worse.
        weights = {"departure_time_variability": 10.0, "route_choice_entropy": 0.01}
        score_a = score_run_against_reference(run_a, _ref(), weights=weights)
        score_b = score_run_against_reference(run_b, _ref(), weights=weights)
        assert score_a["fit_score"] < score_b["fit_score"]


class TestLoadReferenceScenario:
    def test_loads_flat_dict(self, tmp_path):
        ref = {"departure_time_variability": 50.0}
        path = str(tmp_path / "ref.json")
        with open(path, "w") as f:
            json.dump(ref, f)
        loaded = load_reference_scenario(path)
        assert loaded["departure_time_variability"] == 50.0

    def test_unwraps_reference_metrics_key(self, tmp_path):
        wrapped = {"reference_metrics": {"departure_time_variability": 42.0}}
        path = str(tmp_path / "ref.json")
        with open(path, "w") as f:
            json.dump(wrapped, f)
        loaded = load_reference_scenario(path)
        assert loaded["departure_time_variability"] == 42.0


class TestExportCalibrationReport:
    def test_writes_file_and_returns_path(self, tmp_path):
        report = {"mode": "single_run", "score": {"fit_score": 0.9}}
        output = str(tmp_path / "subdir" / "report.json")
        result_path = export_calibration_report(report, output)
        assert os.path.exists(result_path)

    def test_written_file_is_valid_json(self, tmp_path):
        report = {"mode": "test", "value": 42}
        output = str(tmp_path / "report.json")
        export_calibration_report(report, output)
        with open(output) as f:
            loaded = json.load(f)
        assert loaded["value"] == 42

    def test_creates_parent_directories(self, tmp_path):
        report = {"x": 1}
        output = str(tmp_path / "deep" / "nested" / "report.json")
        export_calibration_report(report, output)
        assert os.path.exists(output)


class TestFitAgentParameters:
    def _write_metrics(self, tmp_path, name, metrics):
        path = str(tmp_path / name)
        with open(path, "w") as f:
            json.dump(metrics, f)
        return path

    def test_raises_without_candidates(self):
        with pytest.raises(ValueError):
            fit_agent_parameters({}, reference=_ref())

    def test_scores_in_memory_candidates(self, tmp_path):
        m_path = self._write_metrics(tmp_path, "m1.json", _run())
        candidates = [{"case_id": "c1", "case": {}, "status": "ok", "metrics_path": m_path}]
        result = fit_agent_parameters({}, reference=_ref(), experiments_results=candidates)
        assert result["candidate_count"] == 1
        assert result["best_case"] is not None

    def test_returns_best_case(self, tmp_path):
        m_good = self._write_metrics(tmp_path, "m_good.json", _run())
        m_bad = self._write_metrics(tmp_path, "m_bad.json", _run({"departure_time_variability": 1e8}))
        candidates = [
            {"case_id": "good", "case": {}, "status": "ok", "metrics_path": m_good},
            {"case_id": "bad", "case": {}, "status": "ok", "metrics_path": m_bad},
        ]
        result = fit_agent_parameters({}, reference=_ref(), experiments_results=candidates)
        assert result["best_case"]["case_id"] == "good"

    def test_skips_candidates_without_metrics_path(self, tmp_path):
        candidates = [{"case_id": "c1", "case": {}, "status": "ok"}]  # no metrics_path
        result = fit_agent_parameters({}, reference=_ref(), experiments_results=candidates)
        assert result["candidate_count"] == 0

    def test_top_k_limits_ranked_cases(self, tmp_path):
        metrics_paths = [
            self._write_metrics(tmp_path, f"m{i}.json", _run({"departure_time_variability": float(i * 10)}))
            for i in range(5)
        ]
        candidates = [
            {"case_id": f"c{i}", "case": {}, "status": "ok", "metrics_path": metrics_paths[i]}
            for i in range(5)
        ]
        result = fit_agent_parameters({"top_k": 2}, reference=_ref(), experiments_results=candidates)
        assert len(result["ranked_cases"]) <= 2
