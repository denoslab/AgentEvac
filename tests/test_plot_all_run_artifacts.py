"""Unit tests for scripts.plot_all_run_artifacts."""

from argparse import Namespace
from pathlib import Path

from scripts.plot_all_run_artifacts import _resolve_paths, _resolve_run_id


class TestResolveRunId:
    def test_prefers_explicit_run_id(self):
        args = Namespace(
            run_id="20260309_030340",
            events=None,
            metrics=None,
            replay=None,
            dialogs=None,
        )
        assert _resolve_run_id(args) == "20260309_030340"

    def test_extracts_run_id_from_explicit_path(self):
        args = Namespace(
            run_id=None,
            events="outputs/events_20260309_030340.jsonl",
            metrics=None,
            replay=None,
            dialogs=None,
        )
        assert _resolve_run_id(args) == "20260309_030340"


class TestResolvePaths:
    def test_prefers_matching_run_id_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = Path("outputs")
        out.mkdir()
        (out / "run_metrics_20260309_030340.json").write_text("{}", encoding="utf-8")
        (out / "events_20260309_030340.jsonl").write_text("", encoding="utf-8")
        (out / "llm_routes_20260309_030340.jsonl").write_text("", encoding="utf-8")
        (out / "llm_routes_20260309_030340.dialogs.csv").write_text(
            "step,time_s,veh_id,control_mode,model,system_prompt,user_prompt,response_text,parsed_json,error\n",
            encoding="utf-8",
        )
        args = Namespace(
            metrics=None,
            events=None,
            replay=None,
            dialogs=None,
        )
        paths = _resolve_paths(args, "20260309_030340")
        assert paths["metrics"] == out / "run_metrics_20260309_030340.json"
        assert paths["events"] == out / "events_20260309_030340.jsonl"
        assert paths["replay"] == out / "llm_routes_20260309_030340.jsonl"
        assert paths["dialogs"] == out / "llm_routes_20260309_030340.dialogs.csv"

    def test_missing_replay_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = Path("outputs")
        out.mkdir()
        (out / "run_metrics_20260309_030340.json").write_text("{}", encoding="utf-8")
        (out / "events_20260309_030340.jsonl").write_text("", encoding="utf-8")
        (out / "llm_routes_20260309_030340.dialogs.csv").write_text(
            "step,time_s,veh_id,control_mode,model,system_prompt,user_prompt,response_text,parsed_json,error\n",
            encoding="utf-8",
        )
        args = Namespace(
            metrics=None,
            events=None,
            replay=None,
            dialogs=None,
        )
        paths = _resolve_paths(args, "20260309_030340")
        assert paths["replay"] is None
