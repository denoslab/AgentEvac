"""Unit tests for agentevac.analysis.experiments."""

import csv
import json
import os

import pytest

from unittest.mock import MagicMock, patch

from agentevac.analysis.experiments import (
    _case_id,
    build_experiment_grid,
    export_experiment_results,
    load_resume_cases,
    run_experiment_case,
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

    def test_seed_values_expand_grid(self):
        grid = build_experiment_grid(seed_values=[42, 43])
        assert len(grid) == 2
        seeds = {case["sumo_seed"] for case in grid}
        assert seeds == {42, 43}

    def test_seed_values_multiply_grid(self):
        grid = build_experiment_grid(
            sigma_values=[10.0, 20.0],
            seed_values=[42, 43],
        )
        assert len(grid) == 2 * 2

    def test_case_id_includes_seed(self):
        case_cfg = {
            "info_sigma": 40.0,
            "info_delay_s": 0.0,
            "theta_trust": 0.5,
            "scenario": "advice_guided",
            "sumo_seed": 99,
        }
        cid = _case_id(case_cfg, 1)
        assert "_seed-99" in cid

    def test_default_seed_is_42(self):
        grid = build_experiment_grid()
        assert grid[0]["sumo_seed"] == 42

    def test_perception_range_values_expand_grid(self):
        grid = build_experiment_grid(perception_range_values=[800.0, 1200.0])
        assert len(grid) == 2
        fpr_values = {case.get("fire_perception_range_m") for case in grid}
        assert fpr_values == {800.0, 1200.0}

    def test_case_id_includes_fpr_when_non_default(self):
        case_cfg = {
            "info_sigma": 40.0,
            "info_delay_s": 0.0,
            "theta_trust": 0.5,
            "scenario": "advice_guided",
            "fire_perception_range_m": 800.0,
        }
        cid = _case_id(case_cfg, 1)
        assert "_fpr-800" in cid

    def test_case_id_omits_fpr_at_default(self):
        case_cfg = {
            "info_sigma": 40.0,
            "info_delay_s": 0.0,
            "theta_trust": 0.5,
            "scenario": "advice_guided",
            "fire_perception_range_m": 1200.0,
        }
        cid = _case_id(case_cfg, 1)
        assert "_fpr-" not in cid


class TestRunExperimentCaseEnv:
    """Regression tests for the env vars that ``run_experiment_case`` exports.

    The case-config key is named ``sumo_seed`` for legacy compatibility but is
    actually the master seed for the replicate.  Phase 5 of the master-seed
    refactor exports it as both ``MASTER_SEED`` (so main.py's master-keyed
    streams pick it up) and ``SUMO_SEED`` (so SUMO's --seed sees the same value
    verbatim).  This test catches a regression where someone reverts to setting
    only one of those vars.
    """

    def _fake_subprocess_run(self, captured: dict):
        def _run(cmd, **kwargs):
            captured["env"] = dict(kwargs["env"])
            captured["cmd"] = list(cmd)
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = (
                "[REPLAY] mode=record path=/tmp/r.jsonl\n"
                "[METRICS] summary_path=/tmp/m.json\n"
            )
            return proc
        return _run

    def test_master_and_sumo_seed_env_set_in_tandem(self, tmp_path):
        captured: dict = {}
        case = {
            "info_sigma": 40.0,
            "info_delay_s": 0.0,
            "theta_trust": 0.5,
            "scenario": "advice_guided",
            "agent_type": "llm",
            "sumo_seed": 99,
            "_case_index": 1,
        }
        with patch(
            "agentevac.analysis.experiments.subprocess.run",
            side_effect=self._fake_subprocess_run(captured),
        ):
            run_experiment_case(case, output_dir=str(tmp_path))

        env = captured["env"]
        assert env.get("MASTER_SEED") == "99", (
            "MASTER_SEED env var must be exported so main.py's master-keyed "
            "streams (agent_profile, rule_policy, info_noise) actually vary "
            "across multi-seed sweep replicates."
        )
        assert env.get("SUMO_SEED") == "99", (
            "SUMO_SEED env var must be exported in tandem so SUMO's --seed "
            "sees the swept value verbatim, preserving bit-level continuity "
            "of the traffic stream."
        )

    def test_seed_env_omitted_when_no_seed_set(self, tmp_path):
        captured: dict = {}
        case = {
            "info_sigma": 40.0,
            "info_delay_s": 0.0,
            "theta_trust": 0.5,
            "scenario": "advice_guided",
            "agent_type": "llm",
            "_case_index": 1,
        }
        with patch(
            "agentevac.analysis.experiments.subprocess.run",
            side_effect=self._fake_subprocess_run(captured),
        ):
            run_experiment_case(case, output_dir=str(tmp_path))

        env = captured["env"]
        # When neither case_cfg["sumo_seed"] nor sumo_seed kwarg is set, the
        # subprocess inherits the parent's seed env (or none) -- the
        # case-driver must not invent a value.
        parent_master = os.environ.get("MASTER_SEED")
        parent_sumo = os.environ.get("SUMO_SEED")
        assert env.get("MASTER_SEED") == parent_master
        assert env.get("SUMO_SEED") == parent_sumo


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

    def test_csv_contains_sumo_seed_column(self, tmp_path):
        results = self._make_results(1)
        results[0]["case"]["sumo_seed"] = 42
        paths = export_experiment_results(results, output_dir=str(tmp_path))
        with open(paths["csv"], newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "sumo_seed" in row
        assert row["sumo_seed"] == "42"

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


class TestLoadResumeCases:
    @staticmethod
    def _make_results_payload():
        return [
            {
                "case_id": "001_scn-no_notice_sigma-0_delay-0_trust-0.5",
                "case": {
                    "info_sigma": 0.0,
                    "info_delay_s": 0.0,
                    "theta_trust": 0.5,
                    "scenario": "no_notice",
                    "messaging_enabled": True,
                    "_case_index": 1,
                    "case_id": "001_scn-no_notice_sigma-0_delay-0_trust-0.5",
                },
                "status": "ok",
            },
            {
                "case_id": "002_scn-alert_guided_sigma-20_delay-0_trust-0.5",
                "case": {
                    "info_sigma": 20.0,
                    "info_delay_s": 0.0,
                    "theta_trust": 0.5,
                    "scenario": "alert_guided",
                    "messaging_enabled": True,
                },
                "status": "failed",
            },
            {
                "case_id": "003_scn-advice_guided_sigma-40_delay-15_trust-0.5",
                "case": {
                    "info_sigma": 40.0,
                    "info_delay_s": 15.0,
                    "theta_trust": 0.5,
                    "scenario": "advice_guided",
                    "messaging_enabled": True,
                },
                "status": "timeout",
            },
        ]

    def _write_results(self, tmp_path, name="experiment_results.json"):
        path = tmp_path / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._make_results_payload(), fh)
        return path

    def test_default_returns_only_failed_cases(self, tmp_path):
        path = self._write_results(tmp_path)
        cases = load_resume_cases(str(path))
        assert len(cases) == 1
        assert cases[0]["scenario"] == "alert_guided"
        assert cases[0]["info_sigma"] == 20.0

    def test_directory_path_resolves_to_default_stem(self, tmp_path):
        self._write_results(tmp_path)
        cases = load_resume_cases(str(tmp_path))
        assert len(cases) == 1
        assert cases[0]["scenario"] == "alert_guided"

    def test_custom_statuses_include_timeout(self, tmp_path):
        path = self._write_results(tmp_path)
        cases = load_resume_cases(str(path), statuses=["failed", "timeout"])
        assert len(cases) == 2
        scenarios = {c["scenario"] for c in cases}
        assert scenarios == {"alert_guided", "advice_guided"}

    def test_strips_case_index_but_preserves_case_id(self, tmp_path):
        path = self._write_results(tmp_path)
        cases = load_resume_cases(str(path), statuses=["ok"])
        assert "_case_index" not in cases[0]
        assert cases[0]["case_id"] == "001_scn-no_notice_sigma-0_delay-0_trust-0.5"

    def test_falls_back_to_row_level_case_id(self, tmp_path):
        """When the nested case dict lacks case_id, use the top-level row field."""
        payload = [
            {
                "case_id": "007_scn-alert_guided_sigma-40_delay-0_trust-0.5",
                "case": {
                    "info_sigma": 40.0,
                    "info_delay_s": 0.0,
                    "theta_trust": 0.5,
                    "scenario": "alert_guided",
                    "messaging_enabled": True,
                },
                "status": "failed",
            },
        ]
        path = tmp_path / "experiment_results.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        cases = load_resume_cases(str(path))
        assert cases[0]["case_id"] == "007_scn-alert_guided_sigma-40_delay-0_trust-0.5"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_resume_cases(str(tmp_path / "does_not_exist.json"))

    def test_non_list_payload_raises(self, tmp_path):
        path = tmp_path / "experiment_results.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            load_resume_cases(str(path))
