"""Scenario package reading and run-configuration validation."""

from __future__ import annotations

import pytest

from ui.backend import packages


@pytest.fixture(scope="module")
def flagship():
    return packages.load_package("halifax_3town_e0")


def test_the_flagship_package_reads_the_reconstruction(flagship):
    assert flagship.households == 182
    assert flagship.fire_sources == 13
    assert [wave["issue_time_s"] for wave in flagship.alert_waves] == [6300.0, 9660.0, 15180.0]
    assert flagship.anchor_clock == "15:28:00"
    assert flagship.recommended_horizon_s == 28800.0
    assert flagship.problems == []


def test_packages_are_listed_with_usable_ones_first():
    listed = packages.list_packages()
    assert listed, "no scenario packages were found under configs/"
    problems_seen = False
    for package in listed:
        if package.problems:
            problems_seen = True
        elif problems_seen:
            pytest.fail("a usable package was listed after an unusable one")


def test_an_unknown_package_reports_the_problem_rather_than_raising():
    package = packages.load_package("not_a_real_package")
    assert package.problems
    assert "could not be read" in package.problems[0]


def test_the_incident_schedule_merges_ignitions_and_alerts(flagship):
    rows = packages.incident_schedule(flagship)
    assert rows == sorted(rows, key=lambda row: row["t_s"])
    kinds = {row["kind"] for row in rows}
    assert kinds == {"ignition", "alert"}
    assert len([row for row in rows if row["kind"] == "alert"]) == 3


def base_config(**overrides):
    config = {
        "package": "halifax_3town_e0",
        "scenario": "advice_guided",
        "engine": "rule_based",
        "seed": 1024,
        "sim_end_time_s": 28800,
        "alert_minutes_earlier": 0,
        "messaging": True,
    }
    config.update(overrides)
    return config


def test_a_sound_configuration_passes():
    result = packages.validate_config(base_config())
    assert result.ok, result.problems
    assert result.normalized["seed"] == 1024


def test_every_problem_carries_a_field_and_a_fix():
    result = packages.validate_config(base_config(scenario="telepathy", engine="magic", seed="abc"))
    assert not result.ok
    fields = {problem["field"] for problem in result.problems}
    assert {"scenario", "engine", "seed"} <= fields
    for problem in result.problems:
        assert problem["message"] and problem["hint"]


def test_live_language_model_runs_need_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = packages.validate_config(base_config(engine="llm"))
    assert not result.ok
    problem = next(p for p in result.problems if p["field"] == "engine")
    assert "OPENAI_API_KEY" in problem["message"]
    assert "rule_based" in problem["hint"]


def test_replay_needs_a_recording_that_exists():
    blocked = packages.validate_config(base_config(engine="replay"), recordings=[])
    assert not blocked.ok
    assert blocked.problems[0]["field"] == "replay_run_id"

    missing = packages.validate_config(
        base_config(engine="replay", replay_run_id="19990101_000000"), recordings=[]
    )
    assert not missing.ok
    assert "no recording found" in missing.problems[0]["message"]


def test_replay_follows_the_recording_and_says_so():
    recordings = [
        {
            "run_id": "20260716_155238",
            "params": {"map_name": "halifax_3town", "scenario_mode": "no_notice"},
        }
    ]
    result = packages.validate_config(
        base_config(engine="replay", replay_run_id="20260716_155238"), recordings=recordings
    )
    assert result.ok
    # The recorded decisions belong to their own package and regime, so the run
    # is set to match rather than silently diverging from the log.
    assert result.normalized["package"] == "halifax_3town"
    assert result.normalized["scenario"] == "no_notice"
    assert len(result.warnings) == 2


def test_a_horizon_that_ends_before_the_last_order_warns_without_blocking():
    result = packages.validate_config(base_config(sim_end_time_s=7200))
    assert result.ok
    assert any(warning["field"] == "sim_end_time_s" for warning in result.warnings)


def test_shifting_alerts_under_no_notice_warns_that_it_changes_nothing():
    result = packages.validate_config(base_config(scenario="no_notice", alert_minutes_earlier=30))
    assert result.ok
    assert any("never receive" in warning["message"] for warning in result.warnings)


def test_a_package_without_a_schedule_warns_about_the_timing_slider():
    result = packages.validate_config(base_config(package="lytton", alert_minutes_earlier=15))
    assert result.ok
    assert any("no alert schedule" in warning["message"] for warning in result.warnings)


def test_a_non_positive_horizon_is_blocked():
    result = packages.validate_config(base_config(sim_end_time_s=0))
    assert not result.ok
    assert any(problem["field"] == "sim_end_time_s" for problem in result.problems)
