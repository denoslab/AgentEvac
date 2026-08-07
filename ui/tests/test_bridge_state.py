"""Bridge state, control intent, and the pacing that must never move the clock."""

from __future__ import annotations

import threading
import time

import pytest

from ui.bridge.control import (
    PHASE_RUNNING,
    BridgeState,
    Pacer,
)


def test_control_actions_produce_the_expected_intent():
    state = BridgeState()
    assert state.intent()["paused"] is False

    assert state.apply_control("pause")["paused"] is True
    assert state.apply_control("resume")["paused"] is False
    assert state.apply_control("toggle_pause")["paused"] is True

    assert state.apply_control("set_speed", 60)["speed"] == 60.0
    assert state.apply_control("set_speed", 0)["speed"] == 0.0

    # Ending a run also releases a pause, so the loop can reach its exit.
    state.apply_control("pause")
    intent = state.apply_control("end")
    assert intent["end_requested"] is True
    assert intent["paused"] is False

    # Ending is not shutting down. The bridge stays up so the console can read
    # the closing state.
    assert intent["shutdown_requested"] is False
    assert state.apply_control("shutdown")["shutdown_requested"] is True


def test_unknown_actions_and_bad_speeds_are_rejected():
    state = BridgeState()
    with pytest.raises(ValueError, match="unknown control action"):
        state.apply_control("teleport")
    with pytest.raises(ValueError, match="needs a number"):
        state.apply_control("set_speed", "fast")


def test_status_reports_phase_and_detail():
    state = BridgeState(run_meta={"label": "demo"})
    state.set_phase(PHASE_RUNNING, "simulation running")
    status = state.status()
    assert status["phase"] == PHASE_RUNNING
    assert status["phase_detail"] == "simulation running"
    assert status["run"]["label"] == "demo"


def test_events_reach_late_subscribers_and_live_ones():
    state = BridgeState()
    state.publish_event({"event": "before"})
    subscriber = state.subscribe(backlog=10)
    assert subscriber.get_nowait()["event"] == "before"

    state.publish_event({"event": "after"})
    record = subscriber.get_nowait()
    assert record["event"] == "after"
    # Sequence numbers let a reconnecting reader tell what it missed.
    assert record["seq"] == 2

    state.unsubscribe(subscriber)
    state.publish_event({"event": "ignored"})
    assert subscriber.empty()


def test_round_progress_follows_the_simulator_event_stream():
    state = BridgeState()
    state.set_round_total(50)
    state.round_begin(index=3, dispatched=12)
    for _ in range(5):
        state.round_resolve_one()
    progress = state.round_progress()
    assert (progress["in_progress"], progress["index"], progress["resolved"]) == (True, 3, 5)

    state.round_end()
    progress = state.round_progress()
    assert progress["in_progress"] is False
    assert progress["completed"] == 1
    assert progress["total"] == 50

    # A second end without a begin must not inflate the count.
    state.round_end()
    assert state.round_progress()["completed"] == 1


def test_agent_requests_are_answered_on_the_simulation_thread():
    state = BridgeState()
    answers: dict[str, object] = {}

    def caller(agent_id: str) -> None:
        answers[agent_id] = state.request_agent(agent_id, timeout_s=3.0)

    threads = [threading.Thread(target=caller, args=(f"veh{i}",)) for i in range(3)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)

    resolved: list[str] = []

    def resolver(agent_id: str) -> dict:
        resolved.append(agent_id)
        return {"agent_id": agent_id, "belief": {"p_danger": 0.5}}

    state.fulfil_agent_requests(resolver)
    for thread in threads:
        thread.join(timeout=3.0)

    assert sorted(answers) == ["veh0", "veh1", "veh2"]
    assert all(answers[key]["belief"]["p_danger"] == 0.5 for key in answers)
    assert sorted(resolved) == ["veh0", "veh1", "veh2"]


def test_duplicate_agent_requests_resolve_once_and_both_get_the_answer():
    state = BridgeState()
    results: list[object] = []

    def caller() -> None:
        results.append(state.request_agent("veh1", timeout_s=3.0))

    threads = [threading.Thread(target=caller) for _ in range(2)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)

    calls = 0

    def resolver(agent_id: str) -> dict:
        nonlocal calls
        calls += 1
        return {"agent_id": agent_id}

    state.fulfil_agent_requests(resolver)
    for thread in threads:
        thread.join(timeout=3.0)

    assert calls == 1
    assert len(results) == 2
    assert all(result == {"agent_id": "veh1"} for result in results)


def test_a_resolver_that_raises_does_not_stop_the_run():
    state = BridgeState()
    answer: list[object] = []
    thread = threading.Thread(target=lambda: answer.append(state.request_agent("nope", timeout_s=3.0)))
    thread.start()
    time.sleep(0.1)

    def resolver(agent_id: str):
        raise KeyError(agent_id)

    state.fulfil_agent_requests(resolver)
    thread.join(timeout=3.0)
    assert answer[0]["error"] == "agent_lookup_failed"


def test_an_unanswered_request_times_out_without_blocking_forever():
    state = BridgeState()
    started = time.monotonic()
    assert state.request_agent("veh1", timeout_s=0.2) is None
    assert time.monotonic() - started < 2.0


class TestPacer:
    """The pacer may only ever sleep. It must never advance anything."""

    def test_no_throttle_at_speed_zero(self):
        pacer = Pacer()
        started = time.perf_counter()
        for _ in range(20):
            pacer.wait(step_length_s=0.2, speed=0.0)
        assert time.perf_counter() - started < 0.05

    def test_sleeps_roughly_the_step_budget(self):
        pacer = Pacer()
        pacer.wait(step_length_s=0.2, speed=20.0)  # primes the deadline
        started = time.perf_counter()
        for _ in range(5):
            pacer.wait(step_length_s=0.2, speed=20.0)
        elapsed = time.perf_counter() - started
        assert 0.03 < elapsed < 0.25

    def test_debt_beyond_a_second_is_forgiven(self):
        pacer = Pacer()
        pacer.wait(step_length_s=0.2, speed=1.0)
        time.sleep(0.05)
        pacer._deadline = time.perf_counter() - 5.0  # simulate a long decision round
        started = time.perf_counter()
        pacer.wait(step_length_s=0.2, speed=1.0)
        # A machine that fell far behind must not then race to catch up, nor
        # block while it repays a debt it never owed.
        assert time.perf_counter() - started < 0.05
