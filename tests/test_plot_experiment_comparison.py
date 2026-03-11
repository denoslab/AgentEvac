"""Unit tests for scripts.plot_experiment_comparison."""

import json
from pathlib import Path

from scripts.plot_experiment_comparison import load_cases


class TestLoadCases:
    def test_metrics_glob_uses_companion_run_params(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = Path("outputs")
        out.mkdir()
        metrics_path = out / "run_metrics_20260311_012202.json"
        params_path = out / "run_params_20260311_012202.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "departure_time_variability": 12.0,
                    "route_choice_entropy": 0.5,
                    "average_hazard_exposure": {"global_average": 0.1},
                    "average_travel_time": {"average": 42.0},
                    "arrived_agents": 3,
                    "departed_agents": 4,
                }
            ),
            encoding="utf-8",
        )
        params_path.write_text(
            json.dumps(
                {
                    "scenario": "alert_guided",
                    "cognition": {
                        "info_sigma": 40.0,
                        "info_delay_s": 5.0,
                        "theta_trust": 0.7,
                    },
                }
            ),
            encoding="utf-8",
        )

        rows, source_path = load_cases(None, "outputs/run_metrics_*.json")

        assert source_path == metrics_path
        assert len(rows) == 1
        assert rows[0]["scenario"] == "alert_guided"
        assert rows[0]["info_sigma"] == 40.0
        assert rows[0]["info_delay_s"] == 5.0
        assert rows[0]["theta_trust"] == 0.7
