"""Unit tests for scripts.plot_agent_round_timeline."""

from scripts.plot_agent_round_timeline import (
    _round_for_time,
    _round_table,
    _timeline_rows,
)


class TestRoundTable:
    def test_extracts_and_sorts_rounds(self):
        rows = [
            {"event": "decision_round_start", "round": 2, "sim_t_s": 20.0},
            {"event": "ignored", "round": 99, "sim_t_s": 99.0},
            {"event": "decision_round_start", "round": 1, "sim_t_s": 10.0},
        ]
        assert _round_table(rows) == [(1, 10.0), (2, 20.0)]


class TestRoundForTime:
    def test_maps_to_latest_round_not_exceeding_time(self):
        rounds = [(1, 10.0), (2, 20.0), (3, 30.0)]
        assert _round_for_time(9.0, rounds) == 1
        assert _round_for_time(20.0, rounds) == 2
        assert _round_for_time(29.9, rounds) == 2
        assert _round_for_time(31.0, rounds) == 3


class TestTimelineRows:
    def _event_rows(self):
        return [
            {"event": "decision_round_start", "round": 1, "sim_t_s": 10.0},
            {"event": "decision_round_start", "round": 2, "sim_t_s": 20.0},
            {"event": "decision_round_start", "round": 3, "sim_t_s": 30.0},
            {"event": "decision_round_start", "round": 4, "sim_t_s": 40.0},
            {"event": "departure_release", "veh_id": "veh_a", "sim_t_s": 20.0},
            {"event": "departure_release", "veh_id": "veh_b", "sim_t_s": 20.0},
        ]

    def test_completed_agent_uses_departure_plus_travel_time(self):
        rows, final_round = _timeline_rows(
            self._event_rows(),
            [{"event": "route_change", "veh_id": "veh_a", "time_s": 30.0}],
            {"average_travel_time": {"per_agent": {"veh_a": 15.0}}},
            include_no_departure=False,
        )
        by_id = {row["veh_id"]: row for row in rows}
        assert final_round == 4
        assert by_id["veh_a"]["start_round"] == 2
        assert by_id["veh_a"]["end_round"] == 3
        assert by_id["veh_a"]["change_rounds"] == [3]
        assert by_id["veh_a"]["status"] == "completed"

    def test_incomplete_agent_extends_to_final_round(self):
        rows, _ = _timeline_rows(
            self._event_rows(),
            [],
            {"average_travel_time": {"per_agent": {}}},
            include_no_departure=False,
        )
        by_id = {row["veh_id"]: row for row in rows}
        assert by_id["veh_a"]["end_round"] == 4
        assert by_id["veh_a"]["status"] == "incomplete"

    def test_include_no_departure_uses_first_route_change_round(self):
        rows, _ = _timeline_rows(
            self._event_rows(),
            [{"event": "route_change", "veh_id": "veh_c", "time_s": 30.0}],
            {"average_travel_time": {"per_agent": {}}},
            include_no_departure=True,
        )
        by_id = {row["veh_id"]: row for row in rows}
        assert by_id["veh_c"]["start_round"] == 3
        assert by_id["veh_c"]["end_round"] == 4
        assert by_id["veh_c"]["status"] == "no_departure_event"
