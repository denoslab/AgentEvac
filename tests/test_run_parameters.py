"""Unit tests for agentevac.utils.run_parameters."""

from pathlib import Path

from agentevac.utils.run_parameters import (
    build_parameter_log_path,
    companion_parameter_path,
    reference_suffix,
    write_run_parameter_log,
)


class TestReferenceSuffix:
    def test_strips_known_metric_prefix(self):
        assert reference_suffix("outputs/run_metrics_20260311_012202.json") == "20260311_012202"

    def test_preserves_case_id_prefixes(self):
        assert (
            reference_suffix("outputs/experiments/metrics_sigma-40_delay-0_20260311_012202.json")
            == "sigma-40_delay-0_20260311_012202"
        )


class TestBuildParameterLogPath:
    def test_uses_reference_suffix_for_companion_names(self, tmp_path):
        path = build_parameter_log_path(
            str(tmp_path / "run_params.json"),
            reference_path=tmp_path / "run_metrics_20260311_012202.json",
        )
        assert Path(path).name == "run_params_20260311_012202.json"


class TestWriteRunParameterLog:
    def test_writes_json_using_reference_suffix(self, tmp_path):
        target = write_run_parameter_log(
            str(tmp_path / "run_params.json"),
            {"scenario": "advice_guided"},
            reference_path=tmp_path / "events_20260311_012202.jsonl",
        )
        path = Path(target)
        assert path.name == "run_params_20260311_012202.json"
        assert path.read_text(encoding="utf-8").strip().startswith("{")


class TestCompanionParameterPath:
    def test_matches_metrics_artifact_suffix(self):
        candidate = companion_parameter_path(Path("outputs/run_metrics_20260311_012202.json"))
        assert candidate == Path("outputs/run_params_20260311_012202.json")
