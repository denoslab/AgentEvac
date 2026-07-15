"""Per-run consolidated timeline export (M3 Part 2).

Each run emits one ``run_timeline_<run_id>.jsonl`` file that merges the scripted layers,
meaning fire, alert, and door-to-door, with the emergent layers, meaning awareness,
departure, arrival, reroute, and area clearance, on the ignition-anchored clock.  This is
the artifact the E0-versus-record validation reads.  It is sufficient on its own to
compute clearance sequencing, the departure curve, and the awareness-source distribution.

Every row shares one schema::

    {"t_s": float, "layer": str, "type": str,
     "agent_id"?: str, "area"?: str|list, "source"?: str, "detail"?: any}

Rows are written in the order they are emitted, so the scripted rows preloaded at startup
come first and the emergent rows follow as they fire, with the run-end area-clearance rows
last.  A consumer sorts by ``t_s`` to read the merged timeline chronologically.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


class RunTimeline:
    """JSONL writer for the consolidated per-run timeline.

    Args:
        enabled: If ``False``, all methods are no-ops and no file is opened.
        path: Full output path (``run_timeline_<run_id>.jsonl``), or ``None`` when disabled.
    """

    def __init__(self, enabled: bool, path: Optional[str]):
        self.enabled = bool(enabled) and bool(path)
        self.path: Optional[str] = path if self.enabled else None
        self._fh = None
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "x", encoding="utf-8")

    def close(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def emit(
        self,
        t_s: float,
        layer: str,
        event_type: str,
        *,
        agent_id: Optional[str] = None,
        area: Optional[Any] = None,
        source: Optional[str] = None,
        detail: Optional[Any] = None,
    ) -> None:
        """Write one timeline row.

        Args:
            t_s: Event time in seconds on the ignition-anchored clock.
            layer: One of ``fire``, ``alert``, ``door``, ``awareness``, ``departure``,
                ``arrival``, ``reroute``, ``clearance``.
            event_type: Row type within the layer, for example ``ignition``,
                ``evacuate_now``, ``aware``, ``depart``, ``arrive``, ``area_cleared``.
            agent_id: Vehicle ID for emergent per-agent rows.
            area: Area name or list of area names the row concerns.
            source: Channel or trigger, for example the alert channel or the awareness
                source.
            detail: Any JSON-serializable extra payload.
        """
        if not self.enabled or self._fh is None:
            return
        row: Dict[str, Any] = {
            "t_s": round(float(t_s), 2),
            "layer": str(layer),
            "type": str(event_type),
        }
        if agent_id is not None:
            row["agent_id"] = str(agent_id)
        if area is not None:
            row["area"] = area
        if source is not None:
            row["source"] = str(source)
        if detail is not None:
            row["detail"] = detail
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def emit_scripted(
        self,
        *,
        fire_sources: Optional[Iterable[Dict[str, Any]]] = None,
        alert_events: Optional[Iterable[Any]] = None,
        door_sweeps: Optional[Iterable[Tuple[str, float, float]]] = None,
    ) -> None:
        """Preload the scripted rows from the fire, alert, and door schedules.

        These come from static config, so preloading them at startup keeps the timeline
        complete even when a run ends early.  The alert events are read from the loaded
        ``AlertSchedule`` so any E1 time offset is already applied.

        Args:
            fire_sources: Fire-source dicts with ``id``, ``t0``, ``x``, ``y``, ``r0``,
                ``growth_m_per_s``, and optionally ``max_r_m``.
            alert_events: ``AlertEvent`` objects from ``AlertSchedule.scheduled_events()``.
            door_sweeps: ``(area, begin_s, cleared_by_s)`` tuples from
                ``AlertSchedule.door_sweeps()``.
        """
        for src in (fire_sources or []):
            self.emit(
                float(src.get("t0", 0.0)),
                "fire",
                "ignition",
                detail={
                    "id": src.get("id"),
                    "x": src.get("x"),
                    "y": src.get("y"),
                    "r0": src.get("r0"),
                    "growth_m_per_s": src.get("growth_m_per_s"),
                    "max_r_m": src.get("max_r_m"),
                },
            )
        for ev in (alert_events or []):
            self.emit(
                float(getattr(ev, "issue_time_s", 0.0)),
                "alert",
                str(getattr(ev, "instruction", "none")),
                area=list(getattr(ev, "areas", ()) or ()),
                source=getattr(ev, "channel", None),
                detail={
                    "id": getattr(ev, "id", None),
                    "comfort_centre": getattr(ev, "comfort_centre", None),
                    "routing_text": getattr(ev, "routing_text", None),
                    "hazard_text": getattr(ev, "hazard_text", None),
                },
            )
        for area, begin_s, cleared_by_s in (door_sweeps or []):
            self.emit(
                float(begin_s),
                "door",
                "sweep_begin",
                area=area,
                source="door",
                detail={"cleared_by_s": float(cleared_by_s)},
            )
            self.emit(
                float(cleared_by_s),
                "door",
                "sweep_clear",
                area=area,
                source="door",
                detail={"begin_s": float(begin_s)},
            )
