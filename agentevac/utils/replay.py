"""Record and replay LLM-driven actions for deterministic re-runs.

This module provides ``RouteReplay``, a class that operates in one of two modes:

**record** — During a live simulation run, every replay-relevant action is logged
    to a JSONL file (one JSON record per line) along with agent cognition snapshots
    and LLM dialog transcripts.  Replay currently consumes:
        - ``departure_release`` events for vehicle release timing
        - ``route_change`` events for route application
    Cognition and dialog events are write-only metadata for research/debugging.

    Three output files are created:
        - ``routes_<run_id>.jsonl``         — Replayable route-change schedule.
        - ``routes_<run_id>.dialogs.log``   — Human-readable LLM dialog transcript.
        - ``routes_<run_id>.dialogs.csv``   — Machine-readable LLM dialog table.

**replay** — Loads a previously recorded JSONL file and, on each simulation step,
    releases vehicles according to the recorded ``departure_release`` schedule and
    applies the scheduled ``route_change`` events to the matching vehicle via
    ``traci.vehicle.setRoute()``.  This allows exact behavioural reproduction without
    making any OpenAI API calls.

Important constraint: ``traci.vehicle.setRoute()`` requires that the first edge in the
supplied route is the vehicle's *current* edge.  Both record and replay logic enforce
this by slicing the stored route to start from the current edge when needed.
"""

import traci
import json
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class RouteReplay:
    """
    Record/replay LLM-applied routes.
    We log *explicit edge lists* and replay by calling traci.vehicle.setRoute().
    Note: setRoute requires that the first edge is the vehicle's current edge. :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, mode: str, path: str):
        self.mode = mode
        self.path = path
        self.dialog_path: Optional[str] = None
        self.dialog_csv_path: Optional[str] = None
        self._fh = None
        self._dialog_fh = None
        self._dialog_csv_fh = None
        self._dialog_csv_writer = None
        self._schedule = {}  # step_idx -> veh_id -> route_change record
        self._departure_schedule = {}  # step_idx -> veh_id -> departure_release record

        if self.mode == "record":
            self.path = self._build_record_path(path)
            self.dialog_path = self._build_dialog_path(self.path)
            self.dialog_csv_path = self._build_dialog_csv_path(self.path)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            # Use exclusive create to avoid any accidental overwrite.
            self._fh = open(self.path, "x", encoding="utf-8")
            self._dialog_fh = open(self.dialog_path, "x", encoding="utf-8")
            self._dialog_csv_fh = open(self.dialog_csv_path, "x", encoding="utf-8", newline="")
            self._dialog_csv_writer = csv.DictWriter(
                self._dialog_csv_fh,
                fieldnames=[
                    "step",
                    "time_s",
                    "veh_id",
                    "control_mode",
                    "model",
                    "system_prompt",
                    "user_prompt",
                    "response_text",
                    "parsed_json",
                    "error",
                ],
            )
            self._dialog_csv_writer.writeheader()
            self._dialog_csv_fh.flush()
        elif self.mode == "replay":
            self._schedule, self._departure_schedule = self._load_schedule(self.path)
        else:
            raise ValueError(f"Unknown RUN_MODE={mode}. Use 'record' or 'replay'.")

    def _write_jsonl(self, rec: Dict[str, Any]):
        """Append a single JSON record to the JSONL log file.

        Args:
            rec: Dict to serialize and append; ignored if not in record mode.
        """
        if self.mode != "record" or self._fh is None:
            return
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        """Flush and close all open file handles.  Safe to call more than once."""
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        if self._dialog_fh:
            self._dialog_fh.flush()
            self._dialog_fh.close()
            self._dialog_fh = None
        if self._dialog_csv_fh:
            self._dialog_csv_fh.flush()
            self._dialog_csv_fh.close()
            self._dialog_csv_fh = None
            self._dialog_csv_writer = None

    @staticmethod
    def _load_schedule(path: str):
        """Load replayable events from a JSONL file by step index.

        Replay currently consumes ``route_change`` and ``departure_release`` events.
        All other events (cognition, metrics snapshots, dialogs) are silently ignored.

        Args:
            path: Path to the recorded JSONL file.

        Returns:
            Tuple of dicts:
                - ``route_schedule``: ``step_idx`` → {``veh_id`` → route-change record}
                - ``departure_schedule``: ``step_idx`` → {``veh_id`` → departure record}
        """
        route_schedule = {}
        departure_schedule = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                event = rec.get("event", "route_change")
                step = int(rec["step"])
                vid = rec["veh_id"]
                if event == "route_change":
                    route_schedule.setdefault(step, {})[vid] = rec
                elif event == "departure_release":
                    departure_schedule.setdefault(step, {})[vid] = rec
        return route_schedule, departure_schedule

    @staticmethod
    def _build_record_path(base_path: str) -> str:
        """Build a unique timestamped output path for the main JSONL log.

        Args:
            base_path: Base file path template.

        Returns:
            A unique path string with a timestamp suffix.
        """
        base = Path(base_path)
        ext = base.suffix or ".jsonl"
        stem = base.stem if base.suffix else base.name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        candidate = base.with_name(f"{stem}_{ts}{ext}")
        idx = 1
        while candidate.exists():
            candidate = base.with_name(f"{stem}_{ts}_{idx:02d}{ext}")
            idx += 1
        return str(candidate)

    @staticmethod
    def _build_dialog_path(route_log_path: str) -> str:
        """Derive the human-readable dialog log path from the route log path.

        Args:
            route_log_path: Path of the main JSONL route log.

        Returns:
            Path string for the ``.dialogs.log`` file.
        """
        route_path = Path(route_log_path)
        return str(route_path.with_name(f"{route_path.stem}.dialogs.log"))

    @staticmethod
    def _build_dialog_csv_path(route_log_path: str) -> str:
        """Derive the CSV dialog log path from the route log path.

        Args:
            route_log_path: Path of the main JSONL route log.

        Returns:
            Path string for the ``.dialogs.csv`` file.
        """
        route_path = Path(route_log_path)
        return str(route_path.with_name(f"{route_path.stem}.dialogs.csv"))

    def record_route_change(
        self,
        step: int,
        sim_t_s: float,
        veh_id: str,
        control_mode: str,
        choice_idx: int,
        chosen_name: str,
        chosen_edge: Optional[str],
        current_edge_before: str,
        applied_route_edges: List[str],
        reason: Optional[str] = None,
    ):
        """Log one LLM-applied route change to the JSONL file.

        Trims ``applied_route_edges`` to start from ``current_edge_before`` so the
        stored route is immediately compatible with ``traci.vehicle.setRoute()`` during
        replay without additional adjustment.

        Args:
            step: SUMO simulation step index.
            sim_t_s: Simulation time in seconds.
            veh_id: Vehicle ID.
            control_mode: ``"destination"`` or ``"route"``.
            choice_idx: Index of the chosen option in the menu.
            chosen_name: Name of the chosen destination/route.
            chosen_edge: Target destination edge (destination mode) or ``None``.
            current_edge_before: Edge the vehicle was on when the decision was applied.
            applied_route_edges: Full edge list passed to SUMO.
            reason: Optional human-readable reason string from the LLM.
        """
        if self.mode != "record" or self._fh is None:
            return

        # Ensure we store a route that can be replayed via setRoute:
        # setRoute assumes first edge is the current edge. :contentReference[oaicite:4]{index=4}
        route_for_replay = list(applied_route_edges)
        if current_edge_before in route_for_replay:
            k = route_for_replay.index(current_edge_before)
            route_for_replay = route_for_replay[k:]

        rec = {
            "event": "route_change",
            "step": int(step),
            "time_s": float(sim_t_s),
            "veh_id": str(veh_id),
            "control_mode": str(control_mode),
            "choice_idx": int(choice_idx),
            "chosen_name": str(chosen_name),
            "chosen_edge": chosen_edge,
            "current_edge": str(current_edge_before),
            "route_edges": route_for_replay,
            "reason": reason,
        }
        self._write_jsonl(rec)

    def record_departure_release(
        self,
        step: int,
        sim_t_s: float,
        veh_id: str,
        from_edge: str,
        to_edge: str,
        reason: Optional[str] = None,
    ):
        """Log one vehicle release event to the JSONL file.

        This is replayable metadata used to reproduce the actual departure timing of
        each vehicle in replay mode.

        Args:
            step: SUMO simulation step index.
            sim_t_s: Simulation time in seconds.
            veh_id: Vehicle ID.
            from_edge: Spawn edge used to initialize the route.
            to_edge: Initial destination edge used to initialize the route.
            reason: Optional departure reason.
        """
        rec = {
            "event": "departure_release",
            "step": int(step),
            "time_s": float(sim_t_s),
            "veh_id": str(veh_id),
            "from_edge": str(from_edge),
            "to_edge": str(to_edge),
            "reason": reason,
        }
        self._write_jsonl(rec)

    def record_agent_cognition(
        self,
        step: int,
        sim_t_s: float,
        veh_id: str,
        control_mode: str,
        phase: str,
        belief: Dict[str, Any],
        psychology: Dict[str, Any],
        env_signal: Optional[Dict[str, Any]] = None,
        social_signal: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Record the agent's subjective state for research/debugging.
        This is write-only metadata and is ignored by route replay.
        """
        rec = {
            "event": "agent_cognition",
            "step": int(step),
            "time_s": float(sim_t_s),
            "veh_id": str(veh_id),
            "control_mode": str(control_mode),
            "phase": str(phase),
            "belief_state": dict(belief or {}),
            "psychology": dict(psychology or {}),
            "environment_signal": dict(env_signal or {}),
            "social_signal": dict(social_signal or {}),
            "context": dict(context or {}),
        }
        self._write_jsonl(rec)

    def departure_record_for_step(self, step: int, veh_id: str) -> Optional[Dict[str, Any]]:
        """Return the recorded departure-release event for one vehicle at one step."""
        if self.mode != "replay":
            return None
        return self._departure_schedule.get(int(step), {}).get(str(veh_id))

    def has_departure_schedule(self) -> bool:
        """Whether the loaded replay log contains explicit departure-release events."""
        return bool(self._departure_schedule)

    def record_metric_snapshot(
        self,
        step: int,
        sim_t_s: float,
        snapshot_type: str,
        metrics_row: Dict[str, Any],
    ):
        """
        Record aggregate metrics snapshots over the course of a run.
        This is write-only metadata and is ignored by route replay.
        """
        rec = {
            "event": "metrics_snapshot",
            "step": int(step),
            "time_s": float(sim_t_s),
            "snapshot_type": str(snapshot_type),
            "metrics": dict(metrics_row or {}),
        }
        self._write_jsonl(rec)

    def record_llm_dialog(
        self,
        step: int,
        sim_t_s: float,
        veh_id: str,
        control_mode: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_text: Optional[str] = None,
        parsed: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """
        Record one LLM interaction for audit/debugging.
        This is write-only metadata and is not used for replay routing actions.
        """
        if self.mode != "record" or self._dialog_fh is None:
            return

        self._dialog_fh.write("=" * 80 + "\n")
        self._dialog_fh.write(
            f"step={int(step)} time_s={float(sim_t_s):.2f} veh_id={veh_id} "
            f"mode={control_mode} model={model}\n"
        )
        self._dialog_fh.write("-" * 80 + "\n")
        self._dialog_fh.write("SYSTEM PROMPT:\n")
        self._dialog_fh.write(system_prompt.strip() + "\n\n")
        self._dialog_fh.write("USER PROMPT:\n")
        self._dialog_fh.write(user_prompt.strip() + "\n\n")
        self._dialog_fh.write("MODEL RESPONSE:\n")
        if response_text:
            self._dialog_fh.write(response_text.strip() + "\n")
        else:
            self._dialog_fh.write("<none>\n")
        self._dialog_fh.write("\nPARSED OUTPUT:\n")
        if parsed is None:
            self._dialog_fh.write("<none>\n")
        else:
            self._dialog_fh.write(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n")
        if error:
            self._dialog_fh.write("\nERROR:\n")
            self._dialog_fh.write(str(error).strip() + "\n")
        self._dialog_fh.write("\n")
        self._dialog_fh.flush()

        if self._dialog_csv_writer is not None and self._dialog_csv_fh is not None:
            self._dialog_csv_writer.writerow({
                "step": int(step),
                "time_s": float(sim_t_s),
                "veh_id": str(veh_id),
                "control_mode": str(control_mode),
                "model": str(model),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_text": response_text if response_text is not None else "",
                "parsed_json": json.dumps(parsed, ensure_ascii=False) if parsed is not None else "",
                "error": str(error) if error is not None else "",
            })
            self._dialog_csv_fh.flush()

    def apply_step(self, step: int, controlled_vehicle_ids: List[str]):
        """
        Apply recorded actions scheduled for this step (if any), for vehicles currently in the sim.
        """
        if self.mode != "replay":
            return

        actions = self._schedule.get(int(step))
        if not actions:
            return

        alive = set(traci.vehicle.getIDList())
        for vid in controlled_vehicle_ids:
            if vid not in alive:
                continue
            rec = actions.get(vid)
            if not rec:
                continue

            try:
                cur = traci.vehicle.getRoadID(vid)
                if not cur or cur.startswith(":"):
                    continue

                route_edges = list(rec.get("route_edges") or [])
                if not route_edges:
                    continue

                # Make route compatible with setRoute requirement (first edge == current). :contentReference[oaicite:5]{index=5}
                if route_edges[0] != cur:
                    if cur in route_edges:
                        route_edges = route_edges[route_edges.index(cur):]
                    else:
                        # If we cannot align, skip rather than corrupting replay.
                        continue

                traci.vehicle.setRoute(vid, route_edges)
            except traci.TraCIException:
                continue
