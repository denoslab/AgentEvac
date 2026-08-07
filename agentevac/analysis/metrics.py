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
from typing import Any, Dict, List, Optional, Set, Tuple


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

    def __init__(
        self,
        enabled: bool,
        base_path: str,
        run_mode: str,
        *,
        ordered_areas: Optional[Dict[str, Dict[str, Any]]] = None,
        corridor_edges: Optional[Dict[str, List[str]]] = None,
        spawn_edge_by_agent: Optional[Dict[str, str]] = None,
        time_margin_warn_s: float = 600.0,
        home_key_by_agent: Optional[Dict[str, str]] = None,
        geometry_basis: str = "edge",
    ):
        """Create a collector.

        Args:
            enabled: If ``False``, all recording and export methods are no-ops.
            base_path: Base file path for the output JSON (timestamp is appended).
            run_mode: Run mode string ("record" or "replay") stored in the summary.
            ordered_areas: Map area -> ``{"order_t_s", "channel", "edges"}`` for the areas
                that received an evacuate order, from ``AlertSchedule.ordered_areas()``.
                Drives the order-compliance and area-clearance metrics.  Empty means no
                order was scheduled, so those metrics report empty.
            corridor_edges: Map corridor name -> list of edge IDs for the egress-flow
                split.  Empty until the corridor edge sets are defined against the network
                geometry during E0 validation, in which case the flow split reports empty.
            spawn_edge_by_agent: Map agent ID -> spawn edge, used to decide which agents
                belong to an ordered area and to count non-evacuees the fire reaches.
            time_margin_warn_s: Margin threshold in seconds below which a household counts
                as a close call in :meth:`compute_time_margin`.
            home_key_by_agent: Map agent ID -> the key its home fire-margin samples arrive
                under.  Defaults to ``spawn_edge_by_agent``, meaning one shared sample per
                spawn edge.  A config carrying building centroids passes one key per
                household instead, so time margin resolves per household.
            geometry_basis: What the home margin is measured from, ``edge`` for the spawn
                edge polyline or ``building_centroid`` for the household's own point.
                Reported in the summary so the two experiment batches stay distinguishable.
        """
        self.enabled = bool(enabled)
        self.run_mode = str(run_mode)
        self.path: Optional[str] = None
        if self.enabled:
            self.path = self._timestamped_path(base_path)
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self.total_agents: int = 0
        self._depart_times: Dict[str, float] = {}
        self._depart_reasons: Dict[str, str] = {}
        self._arrival_times: Dict[str, float] = {}
        self._last_seen_active: Set[str] = set()
        self._last_seen_time: Dict[str, float] = {}

        # --- Part J: awareness, order compliance, corridor flow, non-evacuee fire reach ---
        self._spawn_edge_by_agent: Dict[str, str] = {
            str(a): str(e) for a, e in (spawn_edge_by_agent or {}).items()
        }
        self._ordered_areas: Dict[str, Dict[str, Any]] = {
            str(area): dict(spec) for area, spec in (ordered_areas or {}).items()
        }
        # Which agents were ordered, and by which area and channel, decided by spawn edge.
        self._agent_order: Dict[str, Tuple[str, str]] = {}
        _ordered_edge_index: Dict[str, Tuple[str, str]] = {}
        for _area, _spec in self._ordered_areas.items():
            _channel = str(_spec.get("channel", "broadcast"))
            for _edge in _spec.get("edges", []) or []:
                _ordered_edge_index.setdefault(str(_edge), (_area, _channel))
        for _agent, _edge in self._spawn_edge_by_agent.items():
            if _edge in _ordered_edge_index:
                self._agent_order[_agent] = _ordered_edge_index[_edge]
        # First-awareness instant and source per agent (promoted from M2).
        self._awareness: Dict[str, Dict[str, Any]] = {}
        # Spawn edges the fire has crossed, and when, for the non-evacuee companion count.
        self._fire_reached_edges: Dict[str, float] = {}
        # Time margin: per home edge, the interpolated fire-arrival instant, the closest the
        # fire ever came, and the previous sample used to interpolate the zero crossing.
        self._time_margin_warn_s = float(time_margin_warn_s)
        self._geometry_basis = str(geometry_basis)
        self._home_key_by_agent: Dict[str, str] = (
            {str(a): str(k) for a, k in home_key_by_agent.items()}
            if home_key_by_agent
            else dict(self._spawn_edge_by_agent)
        )
        self._home_edge_arrival_t: Dict[str, float] = {}
        self._home_edge_closest: Dict[str, Dict[str, float]] = {}
        self._home_edge_prev_sample: Dict[str, Tuple[float, float]] = {}
        # Corridor edge reverse index and the distinct agents seen on each corridor.
        self._corridor_edge_index: Dict[str, Set[str]] = {}
        self._corridor_agents: Dict[str, Set[str]] = {}
        for _corridor, _edges in (corridor_edges or {}).items():
            self._corridor_agents[str(_corridor)] = set()
            for _edge in _edges or []:
                self._corridor_edge_index.setdefault(str(_edge), set()).add(str(_corridor))

        self._choice_counts: Dict[str, int] = {}
        self._decision_snapshot_count = 0
        self._decision_changes: Dict[str, int] = {}
        self._last_decision_state: Dict[str, str] = {}
        self._final_destination_by_agent: Dict[str, str] = {}
        self.token_usage: Optional[Dict[str, int]] = None

        self._fire_contact_agents: Set[str] = set()

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
            if reason is not None:
                self._depart_reasons[agent_id] = str(reason)
        self._last_seen_time[agent_id] = float(sim_t_s)

    def departure_reason(self, agent_id: str) -> Optional[str]:
        """Return the stored departure reason for an agent, or ``None`` if unrecorded."""
        return self._depart_reasons.get(agent_id)

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
        if current_margin_m is not None and current_margin_m <= 0.0:
            self._fire_contact_agents.add(agent_id)
        # Egress-flow split: count each distinct evacuating agent seen on a corridor edge.
        for corridor in self._corridor_edge_index.get(str(current_edge), ()):
            self._corridor_agents[corridor].add(agent_id)
        self._last_seen_time[agent_id] = float(sim_t_s)

    def record_awareness(self, agent_id: str, sim_t_s: float, source: str) -> None:
        """Record an agent's first-awareness instant and source (promoted from M2).

        Only the first call per ``agent_id`` is stored, so the earliest awareness stands.

        Args:
            agent_id: Vehicle ID.
            sim_t_s: Simulation time the agent first became aware, in seconds.
            source: First-warning channel, one of ``alert``, ``door_knock``,
                ``perception``, or ``peer``.
        """
        if not self.enabled:
            return
        if agent_id in self._awareness:
            return
        self._awareness[agent_id] = {"t_s": float(sim_t_s), "source": str(source)}

    def record_fire_reached_edge(self, edge_id: str, sim_t_s: float) -> None:
        """Record the first time the fire crosses ``edge_id`` (fire margin <= 0).

        Used to count non-evacuated households the fire reaches, which is the companion
        to the evacuating-only exposure average.  Only the earliest crossing is kept.

        Args:
            edge_id: SUMO edge ID the fire has reached.
            sim_t_s: Simulation time of the first crossing, in seconds.
        """
        if not self.enabled:
            return
        edge = str(edge_id)
        if edge not in self._fire_reached_edges:
            self._fire_reached_edges[edge] = float(sim_t_s)

    def record_home_edge_margin(
        self,
        home_key: str,
        sim_t_s: float,
        margin_m: Optional[float],
    ) -> None:
        """Record one fire-margin sample at a household's home location.

        ``home_key`` is whatever the run measures homes by, meaning the spawn edge under
        the edge basis and a per-household key under the building-centroid basis.  The
        collector does no geometry of its own, so the caller decides the basis and the
        matching ``home_key_by_agent`` map.

        Called once per home key per decision round with the same margin the hazard
        model computes for any other location.  Two quantities accumulate.  The fire-arrival
        instant is the first time the margin goes non-positive, linearly interpolated
        against the previous positive sample so arrival is not quantised to the sampling
        cadence.  The closest approach is the smallest margin ever seen at that home and
        the time it occurred, which is what the metric falls back on for the homes the
        fire never reaches.

        Non-finite margins are ignored, which is what the hazard model returns before any
        fire is active or when the location has no usable geometry.

        Args:
            home_key: Key the household's home is measured under.
            sim_t_s: Current simulation time in seconds.
            margin_m: Fire margin in metres at that home, negative once overtaken.
        """
        if not self.enabled or margin_m is None:
            return
        margin = float(margin_m)
        if not math.isfinite(margin):
            return
        edge = str(home_key)
        t_s = float(sim_t_s)

        closest = self._home_edge_closest.get(edge)
        if closest is None or margin < closest["margin_m"]:
            self._home_edge_closest[edge] = {"margin_m": margin, "t_s": t_s}

        if margin <= 0.0 and edge not in self._home_edge_arrival_t:
            arrival_t = t_s
            prev = self._home_edge_prev_sample.get(edge)
            if prev is not None:
                prev_t, prev_margin = prev
                if prev_margin > 0.0 and t_s > prev_t:
                    # prev_margin > 0 >= margin, so the denominator is strictly positive.
                    frac = prev_margin / (prev_margin - margin)
                    arrival_t = prev_t + frac * (t_s - prev_t)
            self._home_edge_arrival_t[edge] = arrival_t

        self._home_edge_prev_sample[edge] = (t_s, margin)

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

        Samples are taken from active vehicles only, so this average is conditional on
        evacuating.  A household that stays put contributes nothing here and is instead
        counted by :meth:`compute_non_evacuated_reached_by_fire` when the fire reaches it.

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

    def compute_mobilization_delay(self) -> Dict[str, Any]:
        """Compute the delay from first awareness to departure, per agent and aggregate.

        Only agents with both a recorded awareness instant and a departure are counted,
        so a household that is aware but never leaves is excluded.  This is the
        pre-evacuation delay the old simulation suppressed.

        Returns:
            Dict with ``average`` (seconds), ``count``, ``min``, ``max``, and
            ``per_agent`` mapping agent ID to its delay.
        """
        per_agent: Dict[str, float] = {}
        for agent_id, depart_t in self._depart_times.items():
            aware = self._awareness.get(agent_id)
            if aware is None:
                continue
            per_agent[agent_id] = float(depart_t) - float(aware["t_s"])
        delays = list(per_agent.values())
        return {
            "average": (sum(delays) / float(len(delays))) if delays else 0.0,
            "count": len(delays),
            "min": min(delays) if delays else 0.0,
            "max": max(delays) if delays else 0.0,
            "per_agent": per_agent,
        }

    def compute_order_compliance(self) -> Dict[str, Any]:
        """Compute the share of ordered households that evacuated, overall and split.

        A household is ordered when its spawn edge is in an area that received an evacuate
        order.  It counts as evacuated if it departed at any point.  The split is by the
        area's first-warning channel and by area.  The channel label reflects the warning
        schedule, not a claim that the channel drove the departure, so it is a per-area
        comparison anchored to the record, not a ranking of the channels.

        Returns:
            Dict with ``overall``, ``by_channel``, and ``by_area``, each a bucket with
            ``ordered``, ``evacuated``, and ``rate``.
        """
        def _bucket() -> Dict[str, Any]:
            return {"ordered": 0, "evacuated": 0}

        overall = _bucket()
        by_channel: Dict[str, Dict[str, Any]] = {}
        by_area: Dict[str, Dict[str, Any]] = {}
        for agent_id, (area, channel) in self._agent_order.items():
            evacuated = agent_id in self._depart_times
            for bucket in (overall,
                           by_channel.setdefault(channel, _bucket()),
                           by_area.setdefault(area, _bucket())):
                bucket["ordered"] += 1
                if evacuated:
                    bucket["evacuated"] += 1

        def _rate(bucket: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(bucket)
            out["rate"] = (bucket["evacuated"] / float(bucket["ordered"])) if bucket["ordered"] > 0 else 0.0
            return out

        return {
            "overall": _rate(overall),
            "by_channel": {k: _rate(v) for k, v in by_channel.items()},
            "by_area": {k: _rate(v) for k, v in by_area.items()},
        }

    def compute_area_clearance(self) -> Dict[str, Any]:
        """Compute per-area order clearance, the latest departure among ordered households.

        For each ordered area the clearance time is the latest departure among its ordered
        households.  ``fully_cleared`` is true when every ordered household in the area
        departed.  This is the primary E0 validation quantity and the source of the
        timeline's area-clearance rows.

        Returns:
            Dict mapping area to ``{ordered, departed, fully_cleared, clearance_t_s,
            channel, order_t_s}``.  ``clearance_t_s`` is ``None`` when no ordered household
            in the area departed.
        """
        members: Dict[str, List[str]] = {}
        for agent_id, (area, _channel) in self._agent_order.items():
            members.setdefault(area, []).append(agent_id)

        out: Dict[str, Any] = {}
        for area, spec in self._ordered_areas.items():
            area_members = members.get(area, [])
            departed_times = [self._depart_times[a] for a in area_members if a in self._depart_times]
            n_ordered = len(area_members)
            n_departed = len(departed_times)
            out[area] = {
                "ordered": n_ordered,
                "departed": n_departed,
                "fully_cleared": bool(n_ordered > 0 and n_departed == n_ordered),
                "clearance_t_s": (round(max(departed_times), 2) if departed_times else None),
                "channel": spec.get("channel"),
                "order_t_s": spec.get("order_t_s"),
            }
        return out

    def compute_awareness_source_share(self) -> Dict[str, Any]:
        """Compute the share of first awareness by channel and the aware count.

        Returns:
            Dict with ``counts`` per source, ``share`` per source, and ``n_aware``.
        """
        counts: Dict[str, int] = {}
        for rec in self._awareness.values():
            source = str(rec.get("source", "none"))
            counts[source] = counts.get(source, 0) + 1
        total = sum(counts.values())
        share = {s: (c / float(total)) for s, c in counts.items()} if total > 0 else {}
        return {"counts": counts, "share": share, "n_aware": total}

    def compute_corridor_flow(self) -> Dict[str, Any]:
        """Compute the distinct evacuating agents seen on each corridor's edges.

        Returns:
            Dict mapping corridor name to ``{agents, agent_ids}``.  Empty when no corridor
            edge sets are configured.
        """
        return {
            corridor: {"agents": len(agents), "agent_ids": sorted(agents)}
            for corridor, agents in self._corridor_agents.items()
        }

    def compute_non_evacuated_reached_by_fire(self) -> Dict[str, Any]:
        """Count households that never departed and whose spawn edge the fire crossed.

        This is the companion to the evacuating-only exposure average.  Without it a
        household that stays put contributes zero exposure, so a run where many stay would
        look safer even though staying near the fire is dangerous.

        Returns:
            Dict with ``count`` and ``agent_ids``.
        """
        reached: List[str] = []
        for agent_id, edge in self._spawn_edge_by_agent.items():
            if agent_id in self._depart_times:
                continue
            if str(edge) in self._fire_reached_edges:
                reached.append(agent_id)
        return {"count": len(reached), "agent_ids": sorted(reached)}

    @staticmethod
    def _quantile(sorted_vals: List[float], q: float) -> float:
        """Return the ``q`` quantile of an already-sorted list by linear interpolation."""
        n = len(sorted_vals)
        if n == 1:
            return float(sorted_vals[0])
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac

    @classmethod
    def _distribution(cls, values: List[float]) -> Dict[str, Any]:
        """Summarise a sample as count, min, p10, median, mean, and max."""
        if not values:
            return {"count": 0, "min": None, "p10": None, "median": None, "mean": None, "max": None}
        ordered = sorted(float(v) for v in values)
        return {
            "count": len(ordered),
            "min": round(ordered[0], 2),
            "p10": round(cls._quantile(ordered, 0.10), 2),
            "median": round(cls._quantile(ordered, 0.50), 2),
            "mean": round(sum(ordered) / float(len(ordered)), 2),
            "max": round(ordered[-1], 2),
        }

    def compute_time_margin(self, warn_s: Optional[float] = None) -> Dict[str, Any]:
        """Compute how much time each household had between leaving and the fire arriving.

        For a household whose home edge the fire reached, the margin is
        ``fire_arrival_t_s - depart_t_s``, so a non-positive value means the fire arrived
        while the household was still home.  For a household the fire never reached, the
        margin is undefined and the closest the fire ever came is reported instead, which
        keeps the metric informative on scenarios where the fire footprint stays clear of
        the populated edges.

        Each household falls into one status.  ``cleared`` left before the fire arrived,
        ``caught_at_home`` was still home when it arrived or never departed at all,
        ``never_threatened`` departed from an edge the fire never reached, and
        ``never_departed_safe`` stayed on an edge the fire never reached.

        Args:
            warn_s: Close-call threshold in seconds.  Defaults to the collector's
                ``time_margin_warn_s``.

        Returns:
            Dict with ``warn_s``, ``geometry_basis``, ``status_counts``, the headline
            counts ``n_threatened``, ``n_censored``, ``n_caught_at_home``,
            ``n_under_warn``, and ``tightest_margin_s``, the ``margin_s`` and
            ``closest_approach_m`` distributions, and a ``per_agent`` row per household.
            ``geometry_basis`` says whether homes were measured at the spawn edge or at
            the building centroid, so runs from the two batches are never pooled.
        """
        warn = float(self._time_margin_warn_s if warn_s is None else warn_s)
        per_agent: Dict[str, Any] = {}
        margins: List[float] = []
        closest_approaches: List[float] = []
        status_counts = {
            "cleared": 0,
            "caught_at_home": 0,
            "never_threatened": 0,
            "never_departed_safe": 0,
        }

        for agent_id, home_edge in self._spawn_edge_by_agent.items():
            edge = str(home_edge)
            key = self._home_key_by_agent.get(agent_id, edge)
            depart_t = self._depart_times.get(agent_id)
            arrival_t = self._home_edge_arrival_t.get(key)
            closest = self._home_edge_closest.get(key)

            margin: Optional[float] = None
            if arrival_t is not None and depart_t is not None:
                margin = float(arrival_t) - float(depart_t)

            if arrival_t is None:
                status = "never_threatened" if depart_t is not None else "never_departed_safe"
                if closest is not None:
                    closest_approaches.append(float(closest["margin_m"]))
            elif margin is None or margin <= 0.0:
                status = "caught_at_home"
            else:
                status = "cleared"
            status_counts[status] += 1

            if margin is not None:
                margins.append(margin)

            per_agent[agent_id] = {
                "home_edge": edge,
                "status": status,
                "depart_t_s": (round(float(depart_t), 2) if depart_t is not None else None),
                "fire_arrival_t_s": (round(float(arrival_t), 2) if arrival_t is not None else None),
                "margin_s": (round(margin, 2) if margin is not None else None),
                "closest_approach_m": (round(float(closest["margin_m"]), 2) if closest else None),
                "closest_approach_t_s": (round(float(closest["t_s"]), 2) if closest else None),
            }

        return {
            "warn_s": warn,
            "geometry_basis": self._geometry_basis,
            "status_counts": status_counts,
            "n_threatened": status_counts["cleared"] + status_counts["caught_at_home"],
            "n_censored": status_counts["never_threatened"] + status_counts["never_departed_safe"],
            "n_caught_at_home": status_counts["caught_at_home"],
            "n_under_warn": sum(1 for m in margins if m <= warn),
            "tightest_margin_s": (round(min(margins), 2) if margins else None),
            "margin_s": self._distribution(margins),
            "closest_approach_m": self._distribution(closest_approaches),
            "per_agent": per_agent,
        }

    def compute_departure_reasons(self) -> Dict[str, int]:
        """Return a histogram of stored departure reasons."""
        counts: Dict[str, int] = {}
        for reason in self._depart_reasons.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts

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
            "fire_contact": {
                "agents_ever_in_contact": len(self._fire_contact_agents),
                "fraction_of_total": len(self._fire_contact_agents) / max(1, self.total_agents),
                "agent_ids": sorted(self._fire_contact_agents),
            },
            # --- Part J additions ---
            # average_hazard_exposure above is conditional on evacuating, since only active
            # vehicles are sampled; non_evacuated_reached_by_fire is its companion count.
            "mobilization_delay": self.compute_mobilization_delay(),
            "compliance": self.compute_order_compliance(),
            "awareness_source_share": self.compute_awareness_source_share(),
            "n_aware": len(self._awareness),
            "n_never_aware": max(0, self.total_agents - len(self._awareness)),
            "area_clearance": self.compute_area_clearance(),
            "corridor_flow": self.compute_corridor_flow(),
            "non_evacuated_reached_by_fire": self.compute_non_evacuated_reached_by_fire(),
            # Time margin turns the exposure index into minutes of headroom per household,
            # and falls back to the closest approach where the fire never arrives.
            "time_margin": self.compute_time_margin(),
            "departure_reasons": self.compute_departure_reasons(),
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
