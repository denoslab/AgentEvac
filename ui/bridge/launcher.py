"""Runs the unmodified simulation script with the console bridge attached.

Usage::

    python -m ui.bridge.launcher --ui-bridge-port 0 -- \
        --map halifax_3town_e0 --scenario advice_guided --agent-type rule_based

Everything after the launcher's own ``--ui-*`` flags is handed to
``agentevac.simulation.main`` exactly as typed, so a run started here takes the
same code path as one started from the command line.

How the bridge attaches
-----------------------
``agentevac/simulation/main.py`` is a script whose simulation loop runs at import
time and drives SUMO through ``traci.simulationStep()``. This launcher replaces
that one TraCI entry point with a wrapper before importing the script, so the
wrapper runs on the simulation thread between steps. From there it publishes
snapshots, paces wall time, and honours pause and end.

The wrapper only ever reads simulation state and sleeps. It issues no TraCI
commands of its own and changes no simulation variable, so a paced or paused run
produces the same trajectory and the same output artifacts as an unattended one.
No file under ``agentevac/`` is read, written, or patched on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

from ui.bridge import collector, projection
from ui.bridge.control import (
    PHASE_COMPLETE,
    PHASE_FAILED,
    PHASE_FINISHING,
    PHASE_LOADING,
    PHASE_RUNNING,
    BridgeState,
    EndRunRequested,
    Pacer,
)
from ui.bridge.server import BridgeServer

SIM_MODULE = "agentevac.simulation.main"

#: Events that mark one household's decision as resolved inside a round.
_RESOLVED_EVENTS = frozenset({
    "llm_decision",
    "llm_error",
    "predeparture_llm_decision",
    "predeparture_llm_error",
    "route_applied",
    "route_apply_error",
    "route_skip",
})

#: Events the console never needs and that would otherwise dominate the feed.
_MUTED_EVENTS = frozenset({"inbox_snapshot", "system_observation_generated"})


def _say(message: str) -> None:
    """Print without letting a closed pipe end the run.

    The console backend owns this process's standard output. If the backend dies
    the pipe breaks, and a run that is mid-export must not be killed by a log
    line. The bridge keeps serving so the backend can reattach when it returns.
    """
    try:
        print(message, flush=True)
    except (BrokenPipeError, ValueError, OSError):
        pass


def _parse_args(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        prog="ui.bridge.launcher",
        description="Run agentevac.simulation.main with the operator-console bridge attached.",
        add_help=False,
    )
    parser.add_argument("--ui-bridge-host", default="127.0.0.1")
    parser.add_argument("--ui-bridge-port", type=int, default=0,
                        help="0 asks the operating system for a free port.")
    parser.add_argument("--ui-bridge-port-file", default=None,
                        help="File to write the bound port into, for the console backend.")
    parser.add_argument("--ui-snapshot-interval-s", type=float, default=0.4,
                        help="Wall-clock spacing between published snapshots.")
    parser.add_argument("--ui-speed", type=float, default=16.0,
                        help="Initial speed multiplier. 0 runs with no throttle.")
    parser.add_argument("--ui-start-paused", action="store_true")
    parser.add_argument("--ui-anchor-clock", default=None,
                        help="Local wall-clock time that simulation second 0 represents, HH:MM[:SS].")
    parser.add_argument("--ui-label", default=None, help="Operator-facing run label.")
    parser.add_argument("--ui-linger-s", type=float, default=900.0,
                        help="Seconds to keep serving after the run ends, so the console can read final state.")
    parser.add_argument("-h", "--help", action="help")
    known, rest = parser.parse_known_args(argv)
    # An explicit separator is supported so simulator flags can never be mistaken
    # for launcher flags.
    if rest and rest[0] == "--":
        rest = rest[1:]
    return known, rest


class BridgeRuntime:
    """Holds everything the patched step wrapper needs between calls."""

    def __init__(self, state: BridgeState, args: argparse.Namespace):
        self.state = state
        self.args = args
        self.pacer = Pacer()
        self.module: Any = None
        self.projector: Optional[collector.GeoProjector] = None
        self.area_members: Dict[str, List[str]] = {}
        self.step_idx = 0
        self.attached = False
        self.last_snapshot_wall = 0.0
        self.step_length_s = 0.2

    # -- one-time attachment ------------------------------------------------

    def attach(self) -> None:
        """Bind to the simulation module the first time a step is taken."""
        if self.attached:
            return
        self.attached = True
        module = sys.modules.get(SIM_MODULE)
        if module is None:
            return
        self.module = module

        try:
            self.step_length_s = float(module.traci.simulation.getDeltaT())
        except Exception:
            self.step_length_s = 0.2

        net = getattr(module, "net", None)
        if net is not None:
            try:
                convert, report = projection.build_projection(
                    net, verify=module.traci.simulation.convertGeo
                )
                self.projector = collector.GeoProjector(convert, report)
                self.state.update_run_meta(projection=report)
            except projection.ProjectionError as exc:
                self.state.publish_event({
                    "event": "ui_bridge_warning",
                    "summary": f"map geography unavailable: {exc}",
                })
                self.state.update_run_meta(projection={"error": str(exc)})

        events = getattr(module, "events", None)
        if events is not None and hasattr(events, "add_listener"):
            events.add_listener(self._on_sim_event)

        self.area_members = collector.build_area_members(module)

        horizon = float(getattr(module, "SIM_END_TIME_S", 0.0) or 0.0)
        period = float(getattr(module, "DECISION_PERIOD_S", 0.0) or 0.0)
        self.state.set_round_total(int(horizon // period) if period > 0 else None)
        self.state.update_run_meta(
            sim_end_time_s=horizon,
            decision_period_s=period,
            step_length_s=self.step_length_s,
            total_households=len(getattr(module, "SPAWN_EVENTS", []) or []),
            agent_type=getattr(module, "AGENT_TYPE", None),
            scenario=getattr(module, "SCENARIO", None) or getattr(module, "SCENARIO_MODE", None),
            run_mode=getattr(module, "RUN_MODE", None),
            metrics_path=getattr(getattr(module, "metrics", None), "path", None),
            events_path=getattr(getattr(module, "events", None), "path", None),
            timeline_path=getattr(getattr(module, "timeline", None), "path", None),
        )
        self.state.set_phase(PHASE_RUNNING, "simulation running")

        # The static geography needs the full edge table, which is immutable once
        # the network has loaded. Building it on a helper thread keeps the first
        # simulation steps from stalling behind a metropolitan network.
        threading.Thread(target=self._build_preview, daemon=True,
                         name="agentevac-ui-preview").start()

    def _build_preview(self) -> None:
        if self.module is None or self.projector is None:
            return
        try:
            preview = collector.build_preview(self.module, self.projector)
        except Exception as exc:
            self.state.publish_event({
                "event": "ui_bridge_warning",
                "summary": f"preview geometry unavailable: {exc}",
            })
            return
        self.state.publish_preview(preview)

    # -- simulator event stream --------------------------------------------

    def _on_sim_event(self, record: Dict[str, Any]) -> None:
        """Mirror one simulator event onto the console stream.

        Called synchronously on the simulation thread by ``LiveEventStream``, so
        it does the least possible work and never blocks.
        """
        event = str(record.get("event", ""))
        if event in _MUTED_EVENTS:
            return
        if event == "decision_round_start":
            self.state.round_begin(
                index=int(record.get("round", 0) or 0),
                dispatched=int(record.get("controlled_count", 0) or 0),
            )
        elif event in _RESOLVED_EVENTS:
            self.state.round_resolve_one()
        self.state.publish_event(record)

    # -- per-step work ------------------------------------------------------

    def before_step(self) -> None:
        self.step_idx += 1
        self.attach()
        # Reaching this point means the previous iteration's decision round, if
        # there was one, has finished and the clock is about to advance again.
        self.state.round_end()
        self._wait_while_paused()
        self.pacer.wait(self.step_length_s, float(self.state.intent().get("speed") or 0.0))

    def after_step(self) -> None:
        self.pump(force=False)

    def _wait_while_paused(self) -> None:
        intent = self.state.intent()
        if intent.get("end_requested"):
            self._begin_finishing()
        if not intent.get("paused"):
            return
        self.pump(force=True)
        while True:
            time.sleep(0.05)
            intent = self.state.intent()
            if intent.get("end_requested"):
                self._begin_finishing()
            if not intent.get("paused"):
                break
            self.pump(force=True)
        # Wall-clock debt built up while paused is not the simulation's to repay.
        self.pacer.reset()

    def _begin_finishing(self) -> None:
        """Leave the simulation loop so the simulator's own cleanup can run.

        The loop sits inside ``try/finally``, so raising here hands control to the
        simulator's export path, which writes metrics, the timeline, and the event
        log the same way a natural end does.
        """
        self.publish_snapshot()
        self.state.set_phase(PHASE_FINISHING, "stopping the run and exporting results")
        raise EndRunRequested()

    def pump(self, *, force: bool) -> None:
        """Publish state and answer parked requests. Simulation thread only."""
        if self.module is None:
            return
        self.state.fulfil_agent_requests(self._agent_detail)
        now = time.monotonic()
        if not force and (now - self.last_snapshot_wall) < self.args.ui_snapshot_interval_s:
            return
        self.last_snapshot_wall = now
        self.publish_snapshot()

    def publish_snapshot(self) -> None:
        if self.module is None or self.projector is None:
            return
        try:
            sim_t_s = float(self.module.traci.simulation.getTime())
        except Exception:
            return
        try:
            snapshot = collector.build_snapshot(
                self.module,
                self.projector,
                sim_t_s=sim_t_s,
                step_idx=self.step_idx,
                intent=self.state.intent(),
                round_progress=self.state.round_progress(),
                area_members=self.area_members,
            )
        except Exception as exc:
            self.state.publish_event({
                "event": "ui_bridge_warning",
                "summary": f"snapshot unavailable: {exc}",
            })
            return
        self.state.publish_snapshot(snapshot)

    def _agent_detail(self, agent_id: str) -> Optional[Dict[str, Any]]:
        builder = getattr(self.module, "build_agent_dashboard_snapshot", None)
        if not callable(builder):
            return None
        return builder(agent_id)


def _install_patches(runtime: BridgeRuntime) -> None:
    """Wrap the TraCI entry points the simulation script calls.

    ``traci.start`` is wrapped only to report the network load as progress.
    ``traci.simulationStep`` is wrapped to give the bridge a foothold on the
    simulation thread. Neither wrapper alters the arguments passed through.

    Raises:
        ImportError: If this interpreter cannot import TraCI, meaning it could
            not have run the simulator either.
    """
    import traci

    original_start = traci.start
    original_step = traci.simulationStep

    def start(*args: Any, **kwargs: Any) -> Any:
        runtime.state.set_phase(PHASE_LOADING, "loading the road network into SUMO")
        result = original_start(*args, **kwargs)
        runtime.state.set_phase(PHASE_LOADING, "network loaded, preparing households")
        return result

    def simulation_step(*args: Any, **kwargs: Any) -> Any:
        runtime.before_step()
        result = original_step(*args, **kwargs)
        runtime.after_step()
        return result

    traci.start = start
    traci.simulationStep = simulation_step


def main(argv: Optional[List[str]] = None) -> int:
    args, sim_argv = _parse_args(list(sys.argv[1:] if argv is None else argv))

    state = BridgeState(run_meta={
        "label": args.ui_label,
        "anchor_clock": args.ui_anchor_clock,
        "argv": sim_argv,
        "pid": os.getpid(),
    })
    state.apply_control("set_speed", args.ui_speed)
    if args.ui_start_paused:
        state.apply_control("pause")

    server = BridgeServer(state, host=args.ui_bridge_host, port=args.ui_bridge_port)
    port = server.start()
    if args.ui_bridge_port_file:
        with open(args.ui_bridge_port_file, "w", encoding="utf-8") as handle:
            json.dump({"host": args.ui_bridge_host, "port": port, "pid": os.getpid()}, handle)
    _say(f"[UI_BRIDGE] listening on http://{args.ui_bridge_host}:{port}")

    runtime = BridgeRuntime(state, args)
    try:
        _install_patches(runtime)
    except ImportError as exc:
        # This interpreter could not have run the simulator, so the run is
        # refused here with something an operator can act on. The bridge stays up
        # briefly so the console reads the reason instead of an exit code.
        message = (
            f"this interpreter cannot import TraCI, so it cannot start a simulation "
            f"({sys.executable}). Use one that has traci, sumolib, and agentevac, "
            f"which in this repository is venv/bin/python."
        )
        state.set_phase(PHASE_FAILED, message, error=f"{message} Underlying error: {exc}")
        _say(f"[UI_BRIDGE] failed: {message}")
        _linger(state, min(30.0, max(0.0, args.ui_linger_s)))
        server.close()
        return 1

    # The simulator reads its configuration from the process command line, so the
    # arguments meant for it are put back where it expects to find them.
    sys.argv = [SIM_MODULE] + sim_argv

    exit_code = 0
    try:
        __import__(SIM_MODULE)
        state.set_phase(PHASE_COMPLETE, "run reached its horizon")
    except EndRunRequested:
        state.set_phase(PHASE_COMPLETE, "run ended by the operator")
    except SystemExit as exc:
        code = int(exc.code or 0)
        if code == 0:
            state.set_phase(PHASE_COMPLETE, "run complete")
        else:
            state.set_phase(PHASE_FAILED, f"simulation exited with code {code}",
                            error=f"simulation exited with code {code}")
            exit_code = code
    except BaseException as exc:  # report the failure through the bridge, then re-raise the summary
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        traceback.print_exc()
        state.set_phase(PHASE_FAILED, "simulation stopped with an error", error=detail)
        exit_code = 1

    runtime.publish_snapshot()
    module = sys.modules.get(SIM_MODULE)
    if module is not None:
        state.update_run_meta(
            metrics_path=getattr(getattr(module, "metrics", None), "path", None),
            events_path=getattr(getattr(module, "events", None), "path", None),
            timeline_path=getattr(getattr(module, "timeline", None), "path", None),
            replay_path=getattr(getattr(module, "replay", None), "path", None),
        )
    state.publish_event({"event": "ui_run_finished", "summary": state.status()["phase_detail"]})
    _say(f"[UI_BRIDGE] {state.status()['phase']}: {state.status()['phase_detail']}")

    # Stay up briefly so the console can collect the closing state before the
    # process disappears. The backend shortens this by ending the linger itself.
    _linger(state, max(0.0, args.ui_linger_s))
    server.close()
    return exit_code


def _linger(state: BridgeState, seconds: float) -> None:
    """Keep serving until the console has read the closing state, or time runs out."""
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            if state.intent().get("shutdown_requested"):
                return
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
