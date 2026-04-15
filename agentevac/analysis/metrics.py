"""Run-level metrics collection and aggregation for evacuation simulations.

``RunMetricsCollector`` accumulates agent-level events over the course of a simulation
run and computes five aggregate KPIs plus one destination-share summary used for
calibration and cross-scenario comparison:

    1. **Departure-time variability** — Population variance of departure timestamps.
       High variability suggests agents are making nuanced, heterogeneous decisions.
       Low variability (everyone leaves at once) can indicate herding or over-reaction.

    2. **Route-choice entropy** — Shannon entropy over the distribution of chosen
       destinations/routes.  High entropy means agents spread across many options;
       low entropy means most converge on a single choice.

    3. **Decision instability** — Average number of times agents changed their chosen
       option across decision rounds.  Frequent changes suggest the LLM or belief model
       is producing erratic decisions; low instability suggests confident, stable routing.

    4. **Average hazard exposure** — Mean edge-level risk score sampled across all
       active vehicles each decision tick.  Proxies the cumulative fire danger
       experienced during the evacuation.

    5. **Average travel time** — Mean time from departure to arrival for agents that
       completed their evacuation during the simulation window.

    6. **Destination choice share** — Final per-agent destination commitments
       aggregated into counts and fractions for each designated evacuation point.

The collector writes a JSON summary to disk when ``export_run_metrics`` or ``close``
is called.  The file path is auto-timestamped to avoid overwrites across runs.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class RunMetricsCollector:
    """Stateful collector for run-level simulation metrics.

    Designed to be instantiated once per simulation run.  Event-recording methods
    (``record_departure``, ``record_arrival``, ``observe_active_vehicles``, etc.) are
    called by the main simulation loop during each step.  Aggregation methods
    (``compute_*``) are cheap and may be called at any time, including mid-run for
    live monitoring.

    Args:
        enabled: If ``False``, all recording and export methods are no-ops.
        base_path: Base file path for the output JSON (timestamp is appended).
        run_mode: Run mode string ("record" or "replay") stored in the summary.
    """

    def __init__(self, enabled: bool, base_path: str, run_mode: str):
        self.enabled = bool(enabled)
        self.run_mode = str(run_mode)
        self.path: Optional[str] = None
        if self.enabled:
            self.path = self._timestamped_path(base_path)
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self.total_agents: int = 0
        self._depart_times: Dict[str, float] = {}
        self._arrival_times: Dict[str, float] = {}
        self._last_seen_active: Set[str] = set()
        self._last_seen_time: Dict[str, float] = {}

        self._choice_counts: Dict[str, int] = {}
        self._decision_snapshot_count = 0
        self._decision_changes: Dict[str, int] = {}
        self._last_decision_state: Dict[str, str] = {}
        self._final_destination_by_agent: Dict[str, str] = {}
        self.token_usage: Optional[Dict[str, int]] = None

        self._exposure_sum = 0.0
        self._exposure_count = 0
        self._exposure_by_agent_sum: Dict[str, float] = {}
        self._exposure_by_agent_count: Dict[str, int] = {}

        self._conflict_sum = 0.0
        self._conflict_count = 0
        self._conflict_by_agent_sum: Dict[str, float] = {}
        self._conflict_by_agent_count: Dict[str, int] = {}

        self._agent_profiles: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _timestamped_path(base_path: str) -> str:
        """Generate a unique timestamped output path by appending ``YYYYMMDD_HHMMSS``.

        Args:
            base_path: Base file path (with or without extension).

        Returns:
            A unique file path string.
        """
        base = Path(base_path)
        ext = base.suffix or ".json"
        stem = base.stem if base.suffix else base.name
        ts = time.strftime("%Y%m%d_%H%M%S")
        candidate = base.with_name(f"{stem}_{ts}{ext}")
        idx = 1
        while candidate.exists():
            candidate = base.with_name(f"{stem}_{ts}_{idx:02d}{ext}")
            idx += 1
        return str(candidate)

    def record_departure(self, agent_id: str, sim_t_s: float, reason: Optional[str] = None) -> None:
        """Record the first departure event for an agent.

        Subsequent calls for the same ``agent_id`` are ignored so the original
        departure timestamp is preserved.

        Args:
            agent_id: Vehicle ID.
            sim_t_s: Simulation time of departure in seconds.
            reason: Optional departure trigger label (e.g., "risk_threshold").
        """
        if not self.enabled:
            return
        if agent_id not in self._depart_times:
            self._depart_times[agent_id] = float(sim_t_s)
        self._last_seen_time[agent_id] = float(sim_t_s)

    def record_arrival(self, agent_id: str, sim_t_s: float) -> None:
        """Record the first explicit arrival event for an agent.

        Arrival timestamps are only accepted for agents that have already
        departed.  Subsequent arrival records for the same agent are ignored so
        the original completion timestamp is preserved.

        Args:
            agent_id: Vehicle ID.
            sim_t_s: Simulation time of arrival in seconds.
        """
        if not self.enabled:
            return
        if agent_id not in self._depart_times or agent_id in self._arrival_times:
            return
        self._arrival_times[agent_id] = float(sim_t_s)
        self._last_seen_time[agent_id] = float(sim_t_s)

    def arrived_count(self) -> int:
        """Return the number of agents that have arrived at their destination."""
        return len(self._arrival_times)

    def observe_active_vehicles(self, active_vehicle_ids: List[str], sim_t_s: float) -> None:
        """Update the active-vehicle set for live bookkeeping only.

        Arrival timing is intentionally not inferred from disappearances because a
        transient omission from ``traci.vehicle.getIDList()`` can otherwise
        produce false travel-time completions.  True arrivals should be recorded
        through :meth:`record_arrival` using explicit SUMO arrival events.

        Args:
            active_vehicle_ids: List of vehicle IDs currently in the simulation.
            sim_t_s: Current simulation time in seconds.
        """
        if not self.enabled:
            return

        now = float(sim_t_s)
        current = set(active_vehicle_ids)
        for vid in current:
            self._last_seen_time[vid] = now

        self._last_seen_active = current

    def record_decision_snapshot(
        self,
        agent_id: str,
        sim_t_s: float,
        decision_round: int,
        state: Dict[str, Any],
        choice_idx: Optional[int],
        action_status: str,
    ) -> None:
        """Record a decision-round snapshot for entropy, instability, and final destination tracking.

        Detects decision-state changes by comparing ``control_mode::choice_idx``
        against the previous round's string for the same agent.  In destination
        mode, the latest selected destination name is also retained as the
        agent's current final destination commitment.

        Args:
            agent_id: Vehicle ID.
            sim_t_s: Simulation time of the decision in seconds.
            decision_round: Global decision-round counter.
            state: Full decision-state dict (supplies choice name and control mode).
            choice_idx: Index of the chosen option, or ``None`` if no decision was made.
            action_status: Action status string (e.g., "depart_now", "wait_predeparture").
        """
        if not self.enabled:
            return

        self._decision_snapshot_count += 1
        if choice_idx is None:
            return

        decision_state = f"{state.get('control_mode', 'unknown')}::{int(choice_idx)}"
        prev_state = self._last_decision_state.get(agent_id)
        if prev_state is not None and prev_state != decision_state:
            self._decision_changes[agent_id] = self._decision_changes.get(agent_id, 0) + 1
        self._last_decision_state[agent_id] = decision_state

        if int(choice_idx) < 0:
            return

        selected = state.get("selected_option") or {}
        choice_name = selected.get("name")
        if not choice_name:
            choice_name = f"choice_{int(choice_idx)}"
        label = f"{state.get('control_mode', 'unknown')}::{choice_name}"
        self._choice_counts[label] = self._choice_counts.get(label, 0) + 1
        if state.get("control_mode") == "destination":
            self._final_destination_by_agent[agent_id] = str(choice_name)

    def record_agent_profile(self, agent_id: str, profile: Dict[str, float]) -> None:
        """Record the sampled profile parameters for an agent.

        Only the first call per agent_id is stored (profiles are immutable).

        Args:
            agent_id: Vehicle ID.
            profile: Dict of psychological parameter values (theta_trust, theta_r, etc.).
        """
        if not self.enabled:
            return
        if agent_id not in self._agent_profiles:
            self._agent_profiles[agent_id] = dict(profile)

    def export_agent_profiles(self) -> Optional[str]:
        """Write per-agent profiles to a JSON file alongside the metrics file.

        Returns:
            The path of the written file, or ``None`` if metrics are disabled or no path is set.
        """
        if not self.enabled or not self.path:
            return None
        if not self._agent_profiles:
            return None
        base = Path(self.path)
        profiles_path = base.with_name(base.stem.replace("run_metrics", "agent_profiles", 1) + base.suffix)
        if str(profiles_path) == self.path:
            profiles_path = base.with_name(base.stem + "_profiles" + base.suffix)
        with open(profiles_path, "w", encoding="utf-8") as fh:
            json.dump(self._agent_profiles, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return str(profiles_path)

    def record_exposure_sample(
        self,
        agent_id: str,
        sim_t_s: float,
        current_edge: str,
        current_margin_m: Optional[float],
        risk_score: Optional[float] = None,
    ) -> None:
        """Record one hazard-exposure sample for an active vehicle.

        Called once per active vehicle per decision tick.

        Args:
            agent_id: Vehicle ID.
            sim_t_s: Current simulation time in seconds.
            current_edge: SUMO edge ID where the vehicle is located.
            current_margin_m: Fire margin (metres) on the current edge (retained for
                future per-edge analysis; not aggregated currently).
            risk_score: Edge-level fire risk score ∈ [0, 1]; treated as 0 if ``None``.
        """
        if not self.enabled:
            return
        exposure = float(risk_score if risk_score is not None else 0.0)
        self._exposure_sum += exposure
        self._exposure_count += 1
        self._exposure_by_agent_sum[agent_id] = self._exposure_by_agent_sum.get(agent_id, 0.0) + exposure
        self._exposure_by_agent_count[agent_id] = self._exposure_by_agent_count.get(agent_id, 0) + 1
        self._last_seen_time[agent_id] = float(sim_t_s)

    def record_conflict_sample(
        self,
        agent_id: str,
        signal_conflict: float,
    ) -> None:
        """Record one signal-conflict sample for an active vehicle.

        Called once per agent per decision round from the belief update.
        The conflict score (JSD between env and social beliefs, [0, 1]) enables
        post-hoc RQ1 analysis of the mediation pathway:
        σ_info → signal_conflict → behavioral DVs.

        Args:
            agent_id: Vehicle ID.
            signal_conflict: JSD-based conflict score ∈ [0, 1].
        """
        if not self.enabled:
            return
        val = float(signal_conflict)
        self._conflict_sum += val
        self._conflict_count += 1
        self._conflict_by_agent_sum[agent_id] = self._conflict_by_agent_sum.get(agent_id, 0.0) + val
        self._conflict_by_agent_count[agent_id] = self._conflict_by_agent_count.get(agent_id, 0) + 1

    def compute_average_signal_conflict(self) -> Dict[str, Any]:
        """Compute global and per-agent average signal conflict.

        Returns:
            Dict with ``global_average``, ``sample_count``, and ``per_agent_average``.
        """
        global_avg = (self._conflict_sum / float(self._conflict_count)) if self._conflict_count > 0 else 0.0
        per_agent: Dict[str, float] = {}
        for agent_id, total in self._conflict_by_agent_sum.items():
            cnt = self._conflict_by_agent_count.get(agent_id, 0)
            per_agent[agent_id] = (total / float(cnt)) if cnt > 0 else 0.0
        return {
            "global_average": round(global_avg, 6),
            "sample_count": self._conflict_count,
            "per_agent_average": per_agent,
        }

    def compute_departure_time_variability(self) -> float:
        """Compute the population variance of agent departure times (seconds²).

        Returns:
            Variance of departure timestamps, or 0.0 if fewer than two agents departed.
        """
        times = list(self._depart_times.values())
        n = len(times)
        if n <= 1:
            return 0.0
        mean = sum(times) / float(n)
        return sum((t - mean) ** 2 for t in times) / float(n)

    def compute_route_choice_entropy(self) -> float:
        """Compute Shannon entropy of the aggregated route/destination choice distribution.

        Returns:
            Entropy in nats (≥ 0); 0 if no choices have been recorded.
        """
        total = sum(self._choice_counts.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in self._choice_counts.values():
            p = float(count) / float(total)
            if p > 0.0:
                entropy -= p * math.log(p)
        return entropy

    def compute_decision_instability(self) -> Dict[str, Any]:
        """Compute per-agent and aggregate decision instability (number of choice changes).

        Returns:
            Dict with ``average_changes``, ``max_changes``, and ``per_agent_changes``.
        """
        if not self._last_decision_state:
            return {
                "average_changes": 0.0,
                "max_changes": 0,
                "per_agent_changes": {},
            }
        per_agent = {
            agent_id: int(self._decision_changes.get(agent_id, 0))
            for agent_id in self._last_decision_state.keys()
        }
        counts = list(per_agent.values())
        return {
            "average_changes": (sum(counts) / float(len(counts))) if counts else 0.0,
            "max_changes": max(counts) if counts else 0,
            "per_agent_changes": per_agent,
        }

    def compute_average_hazard_exposure(self) -> Dict[str, Any]:
        """Compute global and per-agent average hazard-exposure risk scores.

        Returns:
            Dict with ``global_average``, ``sample_count``, and ``per_agent_average``.
        """
        global_avg = (self._exposure_sum / float(self._exposure_count)) if self._exposure_count > 0 else 0.0
        per_agent = {}
        for agent_id, total in self._exposure_by_agent_sum.items():
            cnt = self._exposure_by_agent_count.get(agent_id, 0)
            per_agent[agent_id] = (total / float(cnt)) if cnt > 0 else 0.0
        return {
            "global_average": global_avg,
            "sample_count": self._exposure_count,
            "per_agent_average": per_agent,
        }

    def compute_average_travel_time(self) -> Dict[str, Any]:
        """Compute average travel time for agents that completed their evacuation.

        Agents still en route at simulation end are excluded.

        Returns:
            Dict with ``average`` (seconds), ``completed_agents`` count, and
            ``per_agent`` dict mapping agent ID to travel time.
        """
        durations = []
        per_agent = {}
        for agent_id, depart_t in self._depart_times.items():
            arrive_t = self._arrival_times.get(agent_id)
            if arrive_t is None:
                continue
            duration = max(0.0, float(arrive_t) - float(depart_t))
            per_agent[agent_id] = duration
            durations.append(duration)
        average = (sum(durations) / float(len(durations))) if durations else 0.0
        return {
            "average": average,
            "completed_agents": len(durations),
            "per_agent": per_agent,
        }

    def compute_destination_choice_share(self) -> Dict[str, Any]:
        """Compute counts and fractions of agents' latest destination commitments.

        Returns:
            Dict with ``counts``, ``fractions``, and
            ``total_agents_with_destination``.
        """
        counts: Dict[str, int] = {}
        for choice_name in self._final_destination_by_agent.values():
            counts[choice_name] = counts.get(choice_name, 0) + 1

        total = sum(counts.values())
        fractions = {
            choice_name: (float(count) / float(total)) if total > 0 else 0.0
            for choice_name, count in counts.items()
        }
        return {
            "counts": counts,
            "fractions": fractions,
            "total_agents_with_destination": total,
        }

    def summary(self) -> Dict[str, Any]:
        """Assemble the full run-metrics summary dict.

        Returns:
            A JSON-serializable dict containing all KPIs, destination-share
            summary, and bookkeeping fields.
        """
        return {
            "run_mode": self.run_mode,
            "departed_agents": len(self._depart_times),
            "arrived_agents": len(self._arrival_times),
            "total_agents": self.total_agents,
            "decision_snapshot_count": self._decision_snapshot_count,
            "departure_time_variability": round(self.compute_departure_time_variability(), 6),
            "route_choice_entropy": round(self.compute_route_choice_entropy(), 6),
            "decision_instability": self.compute_decision_instability(),
            "average_hazard_exposure": self.compute_average_hazard_exposure(),
            "average_travel_time": self.compute_average_travel_time(),
            "average_signal_conflict": self.compute_average_signal_conflict(),
            "destination_choice_share": self.compute_destination_choice_share(),
            **({"token_usage": self.token_usage} if self.token_usage else {}),
        }

    def export_run_metrics(self, path: Optional[str] = None) -> Optional[str]:
        """Write the metrics summary to a JSON file.

        Args:
            path: Override output path; falls back to the auto-timestamped path.

        Returns:
            The path of the written file, or ``None`` if metrics are disabled.
        """
        if not self.enabled:
            return None
        target = path or self.path
        if not target:
            return None
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return target

    def close(self) -> Optional[str]:
        """Flush and export metrics and agent profiles; typically called at simulation end.

        Returns:
            The path of the written metrics file, or ``None``.
        """
        if not self.enabled:
            return None
        self.export_agent_profiles()
        return self.export_run_metrics()
