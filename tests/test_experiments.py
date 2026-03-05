"""Unit tests for agentevac.analysis.experiments."""

import csv
import json
import os

import pytest

from agentevac.analysis.experiments import (
    build_experiment_grid,
    export_experiment_results,
)


class TestBuildExperimentGrid:
    def test_defaults_produce_single_case(self):
        grid = build_experiment_grid()
        assert len(grid) == 1
        assert grid[0]["info_sigma"] == 40.0
        assert grid[0]["info_delay_s"] == 0.0
        assert grid[0]["theta_trust"] == 0.5
        assert grid[0]["scenario"] == "advice_guided"

    def test_cartesian_product_size(self):
        grid = build_experiment_grid(
            sigma_values=[10.0, 20.0],
            delay_values=[0.0, 5.0],
            trust_values=[0.3, 0.7],
            scenario_modes=["no_notice"],
        )
        assert len(grid) == 2 * 2 * 2 * 1

    def test_all_scenarios_included(self):
        grid = build_experiment_grid(
            scenario_modes=["no_notice", "alert_guided", "advice_guided"]
        )
        scenarios = {case["scenario"] for case in grid}
        assert scenarios == {"no_notice", "alert_guided", "advice_guided"}

    def test_each_case_has_required_keys(self):
        grid = build_experiment_grid(
            sigma_values=[20.0, 40.0],
            delay_values=[0.0],
            trust_values=[0.5],
        )
        for case in grid:
            for key in ("info_sigma", "info_delay_s", "theta_trust", "scenario"):
                assert key in case

    def test_base_overrides_merged_into_every_case(self):
        overrides = {"messaging_enabled": False, "custom_flag": 42}
        grid = build_experiment_grid(
            sigma_values=[10.0, 20.0],
            base_overrides=overrides,
        )
        for case in grid:
            assert case["messaging_enabled"] is False
            assert case["custom_flag"] == 42

    def test_values_stored_as_floats(self):
        grid = build_experiment_grid(sigma_values=[30], delay_values=[5], trust_values=[1])
        assert isinstance(grid[0]["info_sigma"], float)
        assert isinstance(grid[0]["info_delay_s"], float)
        assert isinstance(grid[0]["theta_trust"], float)

    def test_single_sigma_value(self):
        grid = build_experiment_grid(sigma_values=[99.0])
        assert all(case["info_sigma"] == 99.0 for case in grid)

    def test_none_params_use_defaults(self):
        grid = build_experiment_grid(sigma_values=None, delay_values=None)
        assert grid[0]["info_sigma"] == 40.0
        assert grid[0]["info_delay_s"] == 0.0


class TestExportExperimentResults:
    def _make_results(self, n=3):
        return [
            {
                "case_id": f"case_{i:03d}",
                "case": {
                    "scenario": "advice_guided",
                    "info_sigma": 40.0,
                    "info_delay_s": 0.0,
                    "theta_trust": 0.5,
                },
                "status": "ok",
                "returncode": 0,
                "timeout": False,
                "elapsed_s": float(10 + i),
                "replay_path": None,
                "metrics_path": None,
                "stdout_log": f"/tmp/stdout_{i}.log",
                "stdout_tail": [],
            }
            for i in range(n)
        ]

    def test_writes_json_file(self, tmp_path):
        results = self._make_results(2)
        paths = export_experiment_results(results, output_dir=str(tmp_path))
        assert os.path.exists(paths["json"])

    def test_json_file_contains_all_cases(self, tmp_path):
        results = self._make_results(3)
        paths = export_experiment_results(results, output_dir=str(tmp_path))
        with open(paths["json"]) as f:
            loaded = json.load(f)
        assert len(loaded) == 3

    def test_writes_csv_file(self, tmp_path):
        results = self._make_results(2)
        paths = export_experiment_results(results, output_dir=str(tmp_path))
        assert os.path.exists(paths["csv"])

    def test_csv_has_correct_row_count(self, tmp_path):
        results = self._make_results(4)
        paths = export_experiment_results(results, output_dir=str(tmp_path))
        with open(paths["csv"], newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

    def test_csv_contains_case_id_column(self, tmp_path):
        results = self._make_results(1)
        paths = export_experiment_results(results, output_dir=str(tmp_path))
        with open(paths["csv"], newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "case_id" in row

    def test_returns_dict_with_json_and_csv_keys(self, tmp_path):
        paths = export_experiment_results(self._make_results(1), output_dir=str(tmp_path))
        assert "json" in paths
        assert "csv" in paths

    def test_custom_stem_used_in_filenames(self, tmp_path):
        paths = export_experiment_results(
            self._make_results(1), output_dir=str(tmp_path), stem="my_results"
        )
        assert "my_results" in os.path.basename(paths["json"])
        assert "my_results" in os.path.basename(paths["csv"])

    def test_creates_output_directory_if_missing(self, tmp_path):
        out_dir = str(tmp_path / "new_subdir" / "results")
        export_experiment_results(self._make_results(1), output_dir=out_dir)
        assert os.path.isdir(out_dir)
