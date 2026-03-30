"""Unit tests for agentevac.agents.scenarios."""

import pytest

from agentevac.agents.scenarios import (
    SCENARIO_CHOICES,
    apply_scenario_to_signals,
    filter_history_for_scenario,
    filter_menu_for_scenario,
    load_scenario_config,
    scenario_prompt_suffix,
)


class TestLoadScenarioConfig:
    def test_no_notice_mode(self):
        cfg = load_scenario_config("no_notice")
        assert cfg["mode"] == "no_notice"
        assert cfg["forecast_visible"] is False
        assert cfg["expected_utility_visible"] is True

    def test_alert_guided_mode(self):
        cfg = load_scenario_config("alert_guided")
        assert cfg["mode"] == "alert_guided"
        assert cfg["forecast_visible"] is True
        assert cfg["route_head_forecast_visible"] is False
        assert cfg["expected_utility_visible"] is True

    def test_advice_guided_mode(self):
        cfg = load_scenario_config("advice_guided")
        assert cfg["mode"] == "advice_guided"
        assert cfg["forecast_visible"] is True
        assert cfg["route_head_forecast_visible"] is True
        assert cfg["official_route_guidance_visible"] is True
        assert cfg["expected_utility_visible"] is True

    def test_unknown_mode_falls_back_to_advice_guided(self):
        cfg = load_scenario_config("garbage_mode")
        assert cfg["mode"] == "advice_guided"

    def test_all_keys_present(self):
        for mode in SCENARIO_CHOICES:
            cfg = load_scenario_config(mode)
            for key in (
                "mode", "title", "description", "forecast_visible",
                "route_head_forecast_visible", "official_route_guidance_visible",
                "expected_utility_visible", "neighborhood_observation_visible",
            ):
                assert key in cfg, f"Missing key '{key}' for mode '{mode}'"

    def test_case_insensitive(self):
        cfg = load_scenario_config("NO_NOTICE")
        assert cfg["mode"] == "no_notice"


class TestApplyScenarioToSignals:
    def _full_env(self):
        return {
            "observed_state": "clear",
            "is_delayed": False,
            "delay_rounds_applied": 0,
            "extra_field": "value",
        }

    def _full_forecast(self):
        return {
            "available": True,
            "summary": {"fire_count": 2},
            "current_edge": {"margin_m": 500},
            "route_head": {"available": True, "head_edges_evaluated": 3},
            "briefing": "Looking good",
        }

    def test_no_notice_strips_env_to_minimal(self):
        env, _ = apply_scenario_to_signals("no_notice", self._full_env(), self._full_forecast())
        assert env["source"] == "self_observation_and_neighbors_only"
        assert "extra_field" not in env

    def test_no_notice_forecast_available_false(self):
        _, forecast = apply_scenario_to_signals("no_notice", self._full_env(), self._full_forecast())
        assert forecast["available"] is False

    def test_alert_guided_preserves_forecast_summary(self):
        _, forecast = apply_scenario_to_signals("alert_guided", self._full_env(), self._full_forecast())
        assert "summary" in forecast
        assert forecast["available"] is True

    def test_alert_guided_hides_route_head(self):
        _, forecast = apply_scenario_to_signals("alert_guided", self._full_env(), self._full_forecast())
        assert forecast["route_head"]["available"] is False

    def test_advice_guided_passes_env_through_unmodified(self):
        env, _ = apply_scenario_to_signals("advice_guided", self._full_env(), self._full_forecast())
        assert "extra_field" in env

    def test_advice_guided_passes_forecast_through_unmodified(self):
        _, forecast = apply_scenario_to_signals("advice_guided", self._full_env(), self._full_forecast())
        assert forecast["available"] is True
        assert forecast["route_head"]["available"] is True

    def test_handles_none_env_input(self):
        env, _ = apply_scenario_to_signals("no_notice", None, None)
        assert isinstance(env, dict)

    def test_handles_none_forecast_input(self):
        _, forecast = apply_scenario_to_signals("no_notice", None, None)
        assert isinstance(forecast, dict)


class TestFilterMenuForScenario:
    def _full_menu(self):
        return [
            {
                "idx": 0,
                "name": "shelter_a",
                "risk_sum": 1.0,
                "blocked_edges": 0,
                "min_margin_m": 500.0,
                "travel_time_s_fastest_path": 300.0,
                "reachable": True,
                "dest_edge": "edge_a",
                "advisory": "Recommended",
                "briefing": "Take this route",
                "reasons": ["low risk"],
                "expected_utility": -0.5,
                "utility_components": {"expected_exposure": 0.1},
            }
        ]

    def test_advice_guided_passes_through_unchanged(self):
        menu = self._full_menu()
        result = filter_menu_for_scenario("advice_guided", menu, control_mode="destination")
        assert result[0]["advisory"] == "Recommended"
        assert result[0]["expected_utility"] == -0.5

    def test_alert_guided_removes_advisory(self):
        menu = self._full_menu()
        result = filter_menu_for_scenario("alert_guided", menu, control_mode="destination")
        assert "advisory" not in result[0]
        assert "briefing" not in result[0]
        assert "reasons" not in result[0]

    def test_alert_guided_retains_expected_utility(self):
        menu = self._full_menu()
        result = filter_menu_for_scenario("alert_guided", menu, control_mode="destination")
        assert "expected_utility" in result[0]
        assert "utility_components" in result[0]

    def test_alert_guided_retains_risk_fields(self):
        menu = self._full_menu()
        result = filter_menu_for_scenario("alert_guided", menu, control_mode="destination")
        assert "risk_sum" in result[0]
        assert "blocked_edges" in result[0]

    def test_no_notice_destination_keeps_local_knowledge_and_utility(self):
        menu = self._full_menu()
        menu[0]["travel_time_s_fastest_path"] = 300.0
        menu[0]["len_edges_fastest_path"] = 8
        result = filter_menu_for_scenario("no_notice", menu, control_mode="destination")
        allowed = {
            "idx", "name", "dest_edge", "reachable", "note",
            "travel_time_s_fastest_path", "len_edges_fastest_path",
            "expected_utility", "utility_components",
        }
        assert set(result[0].keys()).issubset(allowed)
        assert "risk_sum" not in result[0]
        assert "expected_utility" in result[0]
        assert "travel_time_s_fastest_path" in result[0]

    def test_no_notice_route_mode_keeps_utility_and_length(self):
        menu = [{
            "idx": 0, "name": "r0", "len_edges": 5, "risk_sum": 2.0,
            "expected_utility": -0.3, "utility_components": {"expected_exposure": 0.1},
        }]
        result = filter_menu_for_scenario("no_notice", menu, control_mode="route")
        assert "risk_sum" not in result[0]
        assert "len_edges" in result[0]
        assert "expected_utility" in result[0]
        assert "utility_components" in result[0]

    def test_original_menu_list_not_mutated(self):
        menu = self._full_menu()
        filter_menu_for_scenario("alert_guided", menu, control_mode="destination")
        assert "advisory" in menu[0]

    def test_returns_list_same_length(self):
        menu = self._full_menu() * 3
        result = filter_menu_for_scenario("advice_guided", menu, control_mode="destination")
        assert len(result) == 3


class TestFilterHistoryForScenario:
    def _sample_record(self):
        return {
            "decision_round": 5,
            "current_edge": "e1",
            "current_edge_margin_m": 1500.0,
            "route_head_min_margin_m": 800.0,
            "trend_vs_last_round": "stable",
            "signals": {
                "environment": {
                    "observed_state": "risky",
                    "is_delayed": False,
                    "base_margin_m": 800.0,
                    "observed_margin_m": 750.0,
                    "sigma_info": 40.0,
                    "source_metric": "route_head_min_margin_m",
                },
                "social": {"observed_state": "danger", "sample_count": 2},
            },
            "forecast": {
                "summary": {"fire_count": 3},
                "current_edge": {"margin_m": 1500},
                "route_head": {"available": True, "head_edges_evaluated": 4},
                "briefing": "Fire spreading east",
            },
            "selected_option": {
                "name": "shelter_1",
                "dest_edge": "e_shelter",
                "advisory": "Recommended",
                "briefing": "Take this route",
                "blocked_edges": 0,
                "risk_sum": 1.2,
                "min_margin_m": 800.0,
                "travel_time_s": 300.0,
                "expected_utility": -0.5,
            },
        }

    # --- no_notice ---

    def test_no_notice_strips_forecast(self):
        result = filter_history_for_scenario("no_notice", [self._sample_record()])
        assert result[0]["forecast"] == {"available": False}

    def test_no_notice_strips_advisory_and_briefing_from_selected_option(self):
        result = filter_history_for_scenario("no_notice", [self._sample_record()])
        sel = result[0]["selected_option"]
        assert "advisory" not in sel
        assert "briefing" not in sel

    def test_no_notice_strips_fire_metrics_from_selected_option(self):
        result = filter_history_for_scenario("no_notice", [self._sample_record()])
        sel = result[0]["selected_option"]
        assert "blocked_edges" not in sel
        assert "risk_sum" not in sel
        assert "min_margin_m" not in sel

    def test_no_notice_keeps_local_knowledge_in_selected_option(self):
        result = filter_history_for_scenario("no_notice", [self._sample_record()])
        sel = result[0]["selected_option"]
        assert sel["name"] == "shelter_1"
        assert sel["dest_edge"] == "e_shelter"
        assert sel["expected_utility"] == -0.5
        assert sel["travel_time_s"] == 300.0

    def test_no_notice_strips_raw_margin_from_env_signal(self):
        result = filter_history_for_scenario("no_notice", [self._sample_record()])
        env = result[0]["signals"]["environment"]
        assert env["observed_state"] == "risky"
        assert "base_margin_m" not in env
        assert "observed_margin_m" not in env
        assert "sigma_info" not in env

    def test_no_notice_preserves_non_leaked_fields(self):
        result = filter_history_for_scenario("no_notice", [self._sample_record()])
        assert result[0]["current_edge"] == "e1"
        assert result[0]["current_edge_margin_m"] == 1500.0
        assert result[0]["decision_round"] == 5

    # --- alert_guided ---

    def test_alert_guided_keeps_forecast_summary_but_strips_route_head(self):
        result = filter_history_for_scenario("alert_guided", [self._sample_record()])
        fc = result[0]["forecast"]
        assert "summary" in fc
        assert "route_head" not in fc

    def test_alert_guided_strips_advisory_from_selected_option(self):
        result = filter_history_for_scenario("alert_guided", [self._sample_record()])
        sel = result[0]["selected_option"]
        assert "advisory" not in sel
        assert "briefing" not in sel
        # But fire metrics are kept in alert_guided
        assert "blocked_edges" in sel
        assert "risk_sum" in sel

    # --- advice_guided ---

    def test_advice_guided_returns_unmodified(self):
        history = [self._sample_record()]
        result = filter_history_for_scenario("advice_guided", history)
        assert result is history  # identity — no copy needed

    # --- general ---

    def test_original_record_not_mutated(self):
        original = self._sample_record()
        filter_history_for_scenario("no_notice", [original])
        assert "advisory" in original["selected_option"]
        assert "summary" in original["forecast"]

    def test_empty_history(self):
        assert filter_history_for_scenario("no_notice", []) == []

    def test_record_without_selected_option(self):
        rec = self._sample_record()
        del rec["selected_option"]
        result = filter_history_for_scenario("no_notice", [rec])
        assert "selected_option" not in result[0]

    def test_record_without_signals(self):
        rec = self._sample_record()
        del rec["signals"]
        result = filter_history_for_scenario("no_notice", [rec])
        assert result[0]["forecast"] == {"available": False}


class TestScenarioPromptSuffix:
    def test_no_notice_suffix_non_empty(self):
        s = scenario_prompt_suffix("no_notice")
        assert isinstance(s, str) and len(s) > 0

    def test_alert_guided_suffix_non_empty(self):
        s = scenario_prompt_suffix("alert_guided")
        assert isinstance(s, str) and len(s) > 0

    def test_advice_guided_suffix_non_empty(self):
        s = scenario_prompt_suffix("advice_guided")
        assert isinstance(s, str) and len(s) > 0

    def test_each_mode_has_distinct_suffix(self):
        suffixes = {scenario_prompt_suffix(m) for m in SCENARIO_CHOICES}
        assert len(suffixes) == len(SCENARIO_CHOICES)
