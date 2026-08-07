"""Run lifecycle. One simulation process at a time, watched and broadcast.

The session owns the state machine the console mirrors::

    idle -> preparing -> running <-> paused -> finishing -> complete
                     \\-> failed          \\-> finishing -> ended_by_operator

It builds the command line from a validated run configuration, spawns
``ui.bridge.launcher``, follows the process until it exits, and merges the
bridge's snapshots and events into one stream for every connected browser.

Isolation
---------
Every artifact a console run produces is written under ``outputs/ui_runs/<run_id>/``,
including the decision log, which otherwise defaults into the shared outputs
directory. A console run therefore cannot overwrite or interleave with the
research campaign folders.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from ui.backend import history, packages
from ui.backend.sim_client import BridgeUnavailable, SimClient

REPO_ROOT = Path(__file__).resolve().parents[2]

PHASE_IDLE = "idle"
PHASE_PREPARING = "preparing"
PHASE_RUNNING = "running"
PHASE_PAUSED = "paused"
PHASE_FINISHING = "finishing"
PHASE_COMPLETE = "complete"
PHASE_ENDED = "ended_by_operator"
PHASE_FAILED = "failed"

ACTIVE_PHASES = frozenset({PHASE_PREPARING, PHASE_RUNNING, PHASE_PAUSED, PHASE_FINISHING})
TERMINAL_PHASES = frozenset({PHASE_COMPLETE, PHASE_ENDED, PHASE_FAILED})

#: How often the backend asks the bridge for continuous state.
SNAPSHOT_POLL_S = 0.5
#: How long the bridge may stay silent during startup before the launch is declared failed.
BRIDGE_STARTUP_TIMEOUT_S = 300.0
#: Lines of standard error kept for the failure panel.
STDERR_TAIL_LINES = 50


class Broadcaster:
    """Fans one merged stream out to every connected browser."""

    def __init__(self, backlog: int = 400):
        self._lock = threading.Lock()
        self._subscribers: List[queue.Queue] = []
        self._events: Deque[Dict[str, Any]] = deque(maxlen=backlog)
        self._latest: Dict[str, Dict[str, Any]] = {}

    def publish(self, message: Dict[str, Any]) -> None:
        kind = str(message.get("type", ""))
        with self._lock:
            if kind in ("session", "snapshot", "preview"):
                self._latest[kind] = message
            else:
                self._events.append(message)
            dead = []
            for sub in self._subscribers:
                try:
                    sub.put_nowait(message)
                except queue.Full:
                    dead.append(sub)
            for sub in dead:
                self._subscribers.remove(sub)

    def subscribe(self) -> "queue.Queue":
        """Register a browser and prime it with the state it needs to render at once."""
        sub: queue.Queue = queue.Queue(maxsize=4000)
        with self._lock:
            primer = [self._latest[k] for k in ("session", "preview", "snapshot") if k in self._latest]
            primer += list(self._events)[-120:]
            self._subscribers.append(sub)
        for message in primer:
            try:
                sub.put_nowait(message)
            except queue.Full:
                break
        return sub

    def unsubscribe(self, sub: "queue.Queue") -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)

    def latest(self, kind: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest.get(kind)

    def clear_run_state(self) -> None:
        with self._lock:
            self._events.clear()
            self._latest.pop("snapshot", None)
            self._latest.pop("preview", None)


class RunSession:
    """Holds the one run the console is allowed to have in flight."""

    def __init__(self, broadcaster: Broadcaster):
        self.broadcaster = broadcaster
        self._lock = threading.RLock()
        self.phase: str = PHASE_IDLE
        self.detail: str = "no run started"
        self.run_id: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self.label: str = ""
        self.error: Optional[str] = None
        self.started_wall: Optional[float] = None
        self.ended_wall: Optional[float] = None
        self.bridge: Optional[SimClient] = None
        self.bridge_port: Optional[int] = None
        self.run_dir: Optional[Path] = None
        self.artifacts: Dict[str, Any] = {}
        self.anchor_clock: Optional[str] = None
        self.end_requested: bool = False
        self.stderr_tail: Deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._process: Optional[subprocess.Popen] = None
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # State reporting
    # ------------------------------------------------------------------

    def session_json(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "type": "session",
                "phase": self.phase,
                "detail": self.detail,
                "run_id": self.run_id,
                "label": self.label,
                "config": dict(self.config),
                "error": self.error,
                "anchor_clock": self.anchor_clock,
                "elapsed_wall_s": round(time.time() - self.started_wall, 1) if self.started_wall else None,
                "artifacts": dict(self.artifacts),
                "stderr_tail": list(self.stderr_tail) if self.phase == PHASE_FAILED else [],
                "active": self.phase in ACTIVE_PHASES,
            }

    def _set_phase(self, phase: str, detail: str = "", error: Optional[str] = None) -> None:
        with self._lock:
            changed = phase != self.phase or (detail and detail != self.detail)
            self.phase = phase
            if detail:
                self.detail = detail
            if error is not None:
                self.error = error
            if phase in TERMINAL_PHASES and self.ended_wall is None:
                self.ended_wall = time.time()
        if changed:
            self.broadcaster.publish(self.session_json())

    # ------------------------------------------------------------------
    # Launching
    # ------------------------------------------------------------------

    def launch(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start one run. Raises RuntimeError if a run is already in flight."""
        with self._lock:
            if self.phase in ACTIVE_PHASES:
                raise RuntimeError(f"a run is already {self.phase}")
            self._stop.set()
            for thread in self._threads:
                thread.join(timeout=1.0)
            self._threads = []
            self._stop = threading.Event()

            run_id = time.strftime("%Y%m%d_%H%M%S")
            run_dir = history.UI_RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            package = packages.load_package(str(config.get("package", "")))
            self.run_id = run_id
            self.config = dict(config)
            self.label = str(config.get("label") or f"{package.id} · {config.get('scenario')}")
            self.error = None
            self.started_wall = time.time()
            self.ended_wall = None
            self.run_dir = run_dir
            self.anchor_clock = package.anchor_clock
            self.artifacts = {}
            self.end_requested = False
            self.stderr_tail.clear()
            self.broadcaster.clear_run_state()

            # A note beside the artifacts so the history list can name this run
            # the way the operator named it.
            with open(run_dir / "console_run.json", "w", encoding="utf-8") as handle:
                json.dump({"run_id": run_id, "label": self.label, "config": config,
                           "launched_wall": self.started_wall}, handle, indent=2)

            argv, env = self._build_command(config, run_dir)
            self._set_phase(PHASE_PREPARING, "starting the simulation process")
            process = subprocess.Popen(
                argv,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._process = process
            self.bridge_port = None
            self.bridge = None

            for target, name in (
                (self._pump_process_output, "console-run-output"),
                (self._watch_run, "console-run-watch"),
            ):
                thread = threading.Thread(target=target, daemon=True, name=name)
                thread.start()
                self._threads.append(thread)

            return self.session_json()

    def adopt(self, orphan: Dict[str, Any]) -> Dict[str, Any]:
        """Take over a simulation this backend started before it restarted.

        Restarting the console must not leave a live SUMO process running
        invisibly, holding its port and burning the demo clock. The bridge keeps
        answering across a backend restart, so the session reattaches to it and
        the operator regains pause, speed, and end.
        """
        with self._lock:
            if self.phase in ACTIVE_PHASES:
                raise RuntimeError(f"a run is already {self.phase}")
            self._stop = threading.Event()
            self._process = None
            self.run_id = str(orphan["run_id"])
            self.run_dir = Path(orphan["run_dir"])
            self.bridge_port = int(orphan["port"])
            self.bridge = SimClient("127.0.0.1", self.bridge_port)
            self.config = dict(orphan.get("config") or {})
            self.label = str(orphan.get("label") or f"adopted run {self.run_id}")
            self.anchor_clock = orphan.get("anchor_clock")
            self.error = None
            self.started_wall = orphan.get("launched_wall") or time.time()
            self.ended_wall = None
            self.artifacts = {}
            self.end_requested = False
            self.stderr_tail.clear()
            self.broadcaster.clear_run_state()
            self._set_phase(PHASE_RUNNING, "reattached to a run that was already in flight")

            self._start_follow_threads()
            thread = threading.Thread(target=lambda: self._poll_loop(None), daemon=True,
                                      name="console-adopted-watch")
            thread.start()
            self._threads.append(thread)
            return self.session_json()

    def _build_command(self, config: Dict[str, Any], run_dir: Path) -> tuple[List[str], Dict[str, str]]:
        """Translate a validated console configuration into the simulator's own flags."""
        engine = str(config.get("engine", "rule_based"))
        agent_type = "llm" if engine in ("llm", "replay") else "rule_based"

        sim_args: List[str] = [
            "--map", str(config["package"]),
            "--scenario", str(config["scenario"]),
            "--agent-type", agent_type,
            "--seed", str(int(config["seed"])),
            "--sim-end-time", str(float(config["sim_end_time_s"])),
            "--sumo-binary", "sumo",
            "--messaging", "on" if config.get("messaging") else "off",
            "--events", "on",
            "--events-stdout", "off",
            "--overlays", "off",
            "--metrics", "on",
            "--timeline", "on",
            "--metrics-log-path", str(run_dir / "run_metrics.json"),
            "--events-log-path", str(run_dir / "events.jsonl"),
            "--timeline-log-path", str(run_dir / "run_timeline.jsonl"),
            "--params-log-path", str(run_dir / "run_params.json"),
        ]
        if engine == "replay":
            recording = next(
                (rec for rec in history.list_recordings(limit=10000)
                 if rec["run_id"] == str(config.get("replay_run_id"))),
                None,
            )
            if recording is None:
                raise RuntimeError(f"recording {config.get('replay_run_id')} is no longer on disk")
            sim_args += ["--run-mode", "replay", "--replay-log-path", str(REPO_ROOT / recording["path"])]

        launcher_args: List[str] = [
            "--ui-bridge-host", "127.0.0.1",
            "--ui-bridge-port", "0",
            "--ui-bridge-port-file", str(run_dir / "bridge.json"),
            "--ui-speed", str(float(config.get("initial_speed", 16))),
            "--ui-label", self.label,
            "--ui-snapshot-interval-s", "0.35",
            # Long enough that a backend restart can still adopt the closing
            # state, short enough that an abandoned process does not sit around.
            "--ui-linger-s", "180",
        ]
        if self.anchor_clock:
            launcher_args += ["--ui-anchor-clock", self.anchor_clock]

        python = simulator_python()
        if python is None:
            raise RuntimeError(f"no interpreter on this machine can start a simulation. {NO_INTERPRETER_HINT}")
        argv = [python, "-m", "ui.bridge.launcher"] + launcher_args + ["--"] + sim_args

        env = dict(os.environ)
        env.setdefault("SUMO_HOME", os.getenv("SUMO_HOME", "/usr/share/sumo"))
        env["PYTHONUNBUFFERED"] = "1"
        # The decision log has no command-line flag, so it is redirected here to
        # keep console runs out of the shared outputs directory.
        env["REPLAY_LOG_PATH"] = str(run_dir / "llm_routes.jsonl")
        env["ALERT_TIME_OFFSET_S"] = str(-60.0 * float(config.get("alert_minutes_earlier", 0.0)))
        env["DECISION_PERIOD_S"] = str(float(config.get("decision_period_s", 240.0)))
        return argv, env

    # ------------------------------------------------------------------
    # Watching
    # ------------------------------------------------------------------

    def _pump_process_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            self.stderr_tail.append(line)
            if line.startswith("[SUMO]") or line.startswith("[MAP]") or line.startswith("[UI_BRIDGE]"):
                self.broadcaster.publish({"type": "sim_log", "line": line, "wall": time.time()})

    def _read_bridge_port(self) -> Optional[int]:
        if self.run_dir is None:
            return None
        path = self.run_dir / "bridge.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return int(json.load(handle)["port"])
        except (OSError, ValueError, KeyError):
            return None

    def _watch_run(self) -> None:
        """Follow one run from spawn to terminal phase."""
        process = self._process
        if process is None:
            return

        deadline = time.monotonic() + BRIDGE_STARTUP_TIMEOUT_S
        while not self._stop.is_set():
            port = self._read_bridge_port()
            if port is not None:
                self.bridge_port = port
                self.bridge = SimClient("127.0.0.1", port)
                break
            if process.poll() is not None:
                self._fail(f"the simulation process exited before it started serving "
                           f"(exit code {process.returncode})")
                return
            if time.monotonic() > deadline:
                self._fail("the simulation process never opened its bridge")
                return
            time.sleep(0.2)

        self._start_follow_threads()
        self._poll_loop(process)

    def _start_follow_threads(self) -> None:
        for target, name in (
            (self._pump_bridge_events, "console-bridge-events"),
            (self._fetch_preview, "console-bridge-preview"),
        ):
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()
            self._threads.append(thread)

    def _poll_loop(self, process: Optional[subprocess.Popen]) -> None:
        """Follow the bridge until the run reaches a terminal state.

        An adopted run has no process handle, so its end is detected by the
        bridge going quiet rather than by an exit code.
        """
        bridge = self.bridge
        assert bridge is not None
        misses = 0
        while not self._stop.is_set():
            exited = process is not None and process.poll() is not None
            try:
                status = bridge.status()
                misses = 0
            except BridgeUnavailable:
                misses += 1
                if exited:
                    break
                if process is None and misses > 6:
                    # Nothing is answering and there is no process to wait on, so
                    # what the run left on disk decides how it ended.
                    self._settle_from_disk()
                    return
                if misses > 20:
                    self._set_phase(PHASE_FINISHING, "the run is exporting its results")
                time.sleep(SNAPSHOT_POLL_S)
                continue

            self._apply_bridge_status(status)
            if str(status.get("phase")) in ("complete", "failed"):
                break

            try:
                snapshot = bridge.snapshot()
            except BridgeUnavailable:
                snapshot = None
            if snapshot is not None:
                snapshot["type"] = "snapshot"
                snapshot["anchor_clock"] = self.anchor_clock
                self.broadcaster.publish(snapshot)
            if exited:
                break
            time.sleep(SNAPSHOT_POLL_S)

        self._finalise(process)

    def _apply_bridge_status(self, status: Dict[str, Any]) -> None:
        bridge_phase = str(status.get("phase", ""))
        detail = str(status.get("phase_detail", ""))
        if bridge_phase in ("starting", "loading"):
            elapsed = status.get("elapsed_wall_s")
            suffix = f" ({elapsed:.0f} s so far)" if isinstance(elapsed, (int, float)) else ""
            self._set_phase(PHASE_PREPARING, detail + suffix)
        elif bridge_phase == "running":
            self._set_phase(PHASE_PAUSED if status.get("paused") else PHASE_RUNNING, detail)
        elif bridge_phase == "finishing":
            self._set_phase(PHASE_FINISHING, detail)
        run_meta = status.get("run") or {}
        if run_meta:
            with self._lock:
                self.artifacts.update({
                    key: run_meta.get(key)
                    for key in ("metrics_path", "events_path", "timeline_path", "replay_path")
                    if run_meta.get(key)
                })
                self.artifacts["projection"] = run_meta.get("projection")
                self.artifacts["total_households"] = run_meta.get("total_households")
                self.artifacts["sim_end_time_s"] = run_meta.get("sim_end_time_s")
                self.artifacts["decision_period_s"] = run_meta.get("decision_period_s")

    def _fetch_preview(self) -> None:
        bridge = self.bridge
        if bridge is None:
            return
        deadline = time.monotonic() + 600.0
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                preview = bridge.preview()
            except BridgeUnavailable:
                preview = None
            if preview is not None:
                preview["type"] = "preview"
                self.broadcaster.publish(preview)
                return
            time.sleep(1.0)

    def _pump_bridge_events(self) -> None:
        bridge = self.bridge
        if bridge is None:
            return
        while not self._stop.is_set():
            try:
                for record in bridge.events(backlog=200):
                    if self._stop.is_set():
                        return
                    self.broadcaster.publish({"type": "sim_event", **record})
            except BridgeUnavailable:
                pass
            if self.phase in TERMINAL_PHASES:
                return
            time.sleep(1.0)

    def _finalise(self, process: Optional[subprocess.Popen]) -> None:
        """Settle the terminal phase and collect what the run left on disk."""
        self._stop.set()
        code: Optional[int] = None
        if process is None:
            # An adopted run belongs to no process this backend owns, so it is
            # released rather than waited on.
            if self.bridge is not None:
                try:
                    self.bridge.control("shutdown")
                except BridgeUnavailable:
                    pass
        else:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # By design the bridge stays up after the run so the console can
                # read the closing state. That has happened by now.
                if self.bridge is not None:
                    try:
                        self.bridge.control("shutdown")
                    except BridgeUnavailable:
                        pass
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    self._terminate_process()
            code = process.returncode

        self._collect_artifacts()
        if self.phase == PHASE_FAILED:
            return
        if code not in (0, None):
            self._fail(f"the simulation process exited with code {code}")
            return
        if self.end_requested:
            self._set_phase(PHASE_ENDED, "run ended by the operator, results exported")
        else:
            self._set_phase(PHASE_COMPLETE, "run complete, results exported")

    def _collect_artifacts(self) -> None:
        if self.run_dir is None:
            return
        found: Dict[str, str] = {}
        for key, pattern in (
            ("metrics", "run_metrics_*.json"),
            ("params", "run_params*.json"),
            ("timeline", "run_timeline*.jsonl"),
            ("events", "events_*.jsonl"),
            ("profiles", "agent_profiles_*.json"),
            ("decisions", "llm_routes_*.jsonl"),
        ):
            matches = sorted(self.run_dir.glob(pattern))
            if matches:
                found[key] = str(matches[-1].relative_to(REPO_ROOT))
        with self._lock:
            self.artifacts.update(found)
            match = sorted(self.run_dir.glob("run_metrics_*.json"))
            if match:
                # The simulator stamps its own run identifier onto the artifacts,
                # and that is the identifier the history list keys on.
                stem = match[-1].stem
                if "_" in stem:
                    self.artifacts["sim_run_id"] = stem.split("run_metrics_", 1)[-1]

    def _settle_from_disk(self) -> None:
        """Decide how a run ended when the bridge can no longer be asked.

        A metrics summary on disk means the simulator's export path ran, so the
        run finished even though the console lost contact with it. Only an
        absent summary is a failure.
        """
        self._stop.set()
        self._collect_artifacts()
        if not self.artifacts.get("metrics"):
            self._fail("contact with the run was lost before it exported any results")
            return
        detail = "the run finished and exported its results, though the console lost contact with it"
        self._set_phase(PHASE_ENDED if self.end_requested else PHASE_COMPLETE, detail)

    def _fail(self, message: str) -> None:
        self._collect_artifacts()
        self._set_phase(PHASE_FAILED, message, error=message)

    def _terminate_process(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                process.kill()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def control(self, action: str, value: Any = None) -> Dict[str, Any]:
        if self.bridge is None:
            raise RuntimeError("the run is still starting, so it cannot be controlled")
        if action == "end":
            with self._lock:
                self.end_requested = True
        result = self.bridge.control(action, value)
        if action in ("pause", "resume", "toggle_pause"):
            intent = result.get("intent") or {}
            self._set_phase(PHASE_PAUSED if intent.get("paused") else PHASE_RUNNING,
                            "paused by the operator" if intent.get("paused") else "simulation running")
        elif action == "end":
            self._set_phase(PHASE_FINISHING, "stopping the run and exporting results")
        self.broadcaster.publish({"type": "control_ack", "action": action, "result": result,
                                  "wall": time.time()})
        return result

    def agent_detail(self, agent_id: str) -> Optional[Dict[str, Any]]:
        if self.bridge is None:
            return None
        try:
            return self.bridge.agent(agent_id)
        except BridgeUnavailable:
            return None

    def shutdown(self) -> None:
        self._stop.set()
        self._terminate_process()


#: Resolved once a working interpreter is found, since the environment does not
#: change while the console is up.
_simulator_python: Optional[str] = None
#: A failed search is remembered only briefly, so installing the dependencies is
#: picked up without restarting the backend, and a health check on a machine with
#: nothing usable does not re-probe every candidate on every request.
_probe_failed_until: float = 0.0
_FAILED_PROBE_TTL_S = 30.0
_PROBE_TIMEOUT_S = 20.0


def _interpreter_candidates() -> List[str]:
    """Interpreters to try, best first.

    The console is often started with whichever Python is on the path, which is
    not necessarily the one carrying ``traci``. In this repository that is the
    interpreter in ``venv/``.
    """
    candidates = [os.getenv("AGENTEVAC_PYTHON")]
    for folder in ("venv", ".venv"):
        candidates += [
            str(REPO_ROOT / folder / "bin" / "python"),
            str(REPO_ROOT / folder / "Scripts" / "python.exe"),
        ]
    candidates.append(sys.executable)
    seen: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen and Path(candidate).exists():
            seen.append(candidate)
    return seen


def simulator_python() -> Optional[str]:
    """The interpreter that can start a simulation, or None if there is none.

    Each candidate is asked to import what a run needs. Guessing from the path
    alone would let a launch fail deep inside the simulator with a traceback the
    operator has no way to act on.
    """
    global _simulator_python, _probe_failed_until
    if _simulator_python is not None:
        return _simulator_python
    if time.monotonic() < _probe_failed_until:
        return None
    env = dict(os.environ)
    env.setdefault("SUMO_HOME", os.getenv("SUMO_HOME", "/usr/share/sumo"))
    for candidate in _interpreter_candidates():
        try:
            probe = subprocess.run(
                [candidate, "-c", "import traci, sumolib, agentevac"],
                cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                timeout=_PROBE_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            _simulator_python = candidate
            return candidate
    _probe_failed_until = time.monotonic() + _FAILED_PROBE_TTL_S
    return None


#: Shown when no interpreter on this machine can start a run.
NO_INTERPRETER_HINT = (
    "Start the console with an interpreter that has traci, sumolib, and agentevac, "
    "which in this repository is venv/bin/python, or point AGENTEVAC_PYTHON at one."
)


def find_orphan_run(max_folders: int = 20) -> Optional[Dict[str, Any]]:
    """Look for a console run whose bridge is still answering.

    The launcher records its port beside the run's artifacts, so a backend that
    restarts can find a simulation it started earlier and reattach instead of
    leaving it running unattended.
    """
    if not history.UI_RUNS_DIR.is_dir():
        return None
    folders = sorted((p for p in history.UI_RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
    for run_dir in folders[:max_folders]:
        bridge_file = run_dir / "bridge.json"
        if not bridge_file.exists():
            continue
        try:
            with open(bridge_file, encoding="utf-8") as handle:
                info = json.load(handle)
            port = int(info["port"])
        except (OSError, ValueError, KeyError):
            continue
        try:
            status = SimClient("127.0.0.1", port, timeout_s=1.5).status()
        except BridgeUnavailable:
            continue
        if str(status.get("phase")) in ("complete", "failed"):
            continue
        note: Dict[str, Any] = {}
        note_path = run_dir / "console_run.json"
        if note_path.exists():
            try:
                with open(note_path, encoding="utf-8") as handle:
                    note = json.load(handle)
            except (OSError, ValueError):
                note = {}
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "port": port,
            "label": note.get("label"),
            "config": note.get("config"),
            "launched_wall": note.get("launched_wall"),
            "anchor_clock": (status.get("run") or {}).get("anchor_clock"),
            "phase": status.get("phase"),
            "sim_t_s": status.get("sim_t_s"),
        }
    return None


def sumo_available() -> Dict[str, Any]:
    """Report whether the pieces a run needs are present on this machine."""
    sumo_home = os.getenv("SUMO_HOME") or "/usr/share/sumo"
    binary = shutil.which("sumo")
    python = simulator_python()
    return {
        "sumo_home": sumo_home,
        "sumo_home_exists": Path(sumo_home).is_dir(),
        "sumo_binary": binary,
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "python": sys.executable,
        "simulator_python": python,
        "simulator_python_hint": None if python else NO_INTERPRETER_HINT,
    }
