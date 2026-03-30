"""Information-regime configuration and signal filtering for evacuation scenarios.

AgentEvac models three empirically motivated information regimes that differ in how
much official hazard information agents receive each decision round:

    **no_notice** — No official warning exists yet.
        Agents rely solely on their own noisy margin observations and natural-language
        messages from neighbours.  Menu items include route identity, travel time, and
        an observation-based utility score (local road knowledge), but no fire-specific
        risk metrics.  This represents the typical onset of a rapidly spreading wildfire
        before emergency services have issued formal guidance.

    **alert_guided** — Official alerts broadcast general hazard information.
        Agents receive a full fire forecast summary and per-edge risk data for their
        current location, but do *not* receive route-specific advisories or expected
        utility scores.  They must synthesize hazard information themselves.

    **advice_guided** — Official guidance provides route-oriented recommendations.
        Agents receive the full forecast, per-route-head forecasts, advisory labels
        (Recommended / Use with caution / Avoid for now), and expected utility scores.
        This is the highest-information regime and models scenarios with active
        emergency operations-centre support.

The key functions ``apply_scenario_to_signals`` and ``filter_menu_for_scenario`` strip
fields from signals and menus based on the active regime before the data is embedded in
the LLM prompt.  This ensures information asymmetries are faithfully reproduced.
"""

from typing import Any, Dict, List, Tuple


# All valid scenario identifiers.
SCENARIO_CHOICES: Tuple[str, ...] = (
    "no_notice",
    "alert_guided",
    "advice_guided",
)


def load_scenario_config(mode: str) -> Dict[str, Any]:
    """Return the configuration dict for a given information regime.

    The config controls which data fields are surfaced to agents in the LLM prompt.

    Args:
        mode: One of ``"no_notice"``, ``"alert_guided"``, or ``"advice_guided"``.
            Any unrecognised value is treated as ``"advice_guided"``.

    Returns:
        A dict with keys:
            - ``mode``                          : Normalised mode string.
            - ``title``                         : Human-readable scenario name.
            - ``description``                   : One-sentence scenario description.
            - ``forecast_visible``              : Whether fire forecast summary is shown.
            - ``route_head_forecast_visible``   : Whether per-route-head risk is shown.
            - ``official_route_guidance_visible``: Whether advisory labels are shown.
            - ``expected_utility_visible``      : Whether computed utility scores are shown.
            - ``neighborhood_observation_visible``: Whether local system-authored
              neighborhood departure observations are shown.
    """
    name = str(mode).strip().lower()
    if name == "no_notice":
        return {
            "mode": name,
            "title": "No-Notice Wildfire",
            "description": (
                "No official warning is available yet. Agents rely on self-observation and neighbor messages."
            ),
            "forecast_visible": False,
            "route_head_forecast_visible": False,
            "official_route_guidance_visible": False,
            "expected_utility_visible": True,
            "neighborhood_observation_visible": True,
        }
    if name == "alert_guided":
        return {
            "mode": name,
            "title": "Alert-Guided Evacuation",
            "description": (
                "Official alerts expose hazard location and projected spread, but do not prescribe a route."
            ),
            "forecast_visible": True,
            "route_head_forecast_visible": False,
            "official_route_guidance_visible": False,
            "expected_utility_visible": True,
            "neighborhood_observation_visible": True,
        }
    return {
        "mode": "advice_guided",
        "title": "Advice-Guided Evacuation",
        "description": (
            "Official alerts include both hazard information and route-oriented guidance."
        ),
        "forecast_visible": True,
        "route_head_forecast_visible": True,
        "official_route_guidance_visible": True,
        "expected_utility_visible": True,
        "neighborhood_observation_visible": True,
    }


def apply_scenario_to_signals(
    mode: str,
    env_signal: Dict[str, Any],
    forecast: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter environment and forecast signals to match the active information regime.

    For ``no_notice``: strips the environment signal to subjective observations only
    (observed_state, delay flags) and replaces the forecast with a placeholder indicating
    no official forecast exists.

    For ``alert_guided``: preserves the forecast summary and current-edge risk, but
    removes route-head forecast data (agents see *what* is burning but not *which route*
    is safer).

    For ``advice_guided``: passes both signals through unmodified.

    Args:
        mode: Active scenario mode string.
        env_signal: Full environment signal dict from the information model.
        forecast: Full forecast dict from ``forecast_layer.render_forecast_briefing``.

    Returns:
        A ``(env_prompt, forecast_prompt)`` tuple filtered for the given regime.
    """
    cfg = load_scenario_config(mode)
    env_prompt = dict(env_signal or {})
    forecast_prompt = dict(forecast or {})

    if cfg["mode"] == "no_notice":
        # Strip everything except the raw perceptual observation and delay metadata.
        env_prompt = {
            "observed_state": env_prompt.get("observed_state"),
            "is_delayed": bool(env_prompt.get("is_delayed", False)),
            "delay_rounds_applied": int(env_prompt.get("delay_rounds_applied", 0) or 0),
            "source": "self_observation_and_neighbors_only",
            "note": "No official warning bulletin is available in this scenario.",
        }
        forecast_prompt = {
            "available": False,
            "briefing": "No official forecast is available yet.",
        }
        return env_prompt, forecast_prompt

    if cfg["mode"] == "alert_guided":
        # Keep the global fire forecast but suppress route-specific advice.
        route_head = dict(forecast_prompt.get("route_head") or {})
        forecast_prompt = {
            "available": True,
            "summary": dict(forecast_prompt.get("summary") or {}),
            "current_edge": dict(forecast_prompt.get("current_edge") or {}),
            "route_head": {
                "available": False,
                "note": "Alert-only mode: no official route-specific advice.",
                "head_edges_evaluated": route_head.get("head_edges_evaluated"),
            },
            "briefing": str(forecast_prompt.get("briefing") or ""),
        }
        return env_prompt, forecast_prompt

    # advice_guided: pass everything through unchanged.
    return env_prompt, forecast_prompt


def filter_menu_for_scenario(
    mode: str,
    menu: List[Dict[str, Any]],
    *,
    control_mode: str,
) -> List[Dict[str, Any]]:
    """Strip advisory and utility fields from each menu option based on the active regime.

    In ``no_notice`` mode, menu items are aggressively reduced to the minimal set
    the agent could plausibly know without official guidance (name, reachability for
    destinations; name and edge count for routes).

    In ``alert_guided`` mode, advisory labels and utility scores are removed but other
    risk metrics (risk_sum, blocked_edges, min_margin_m) remain visible.

    In ``advice_guided`` mode, menu items are returned unmodified.

    Args:
        mode: Active scenario mode string.
        menu: List of destination or route dicts (already annotated with utility scores).
        control_mode: ``"destination"`` or ``"route"``; determines which keys to retain
            in the ``no_notice`` minimum-information filter.

    Returns:
        A new list of dicts with scenario-inappropriate fields removed.
    """
    cfg = load_scenario_config(mode)
    prompt_menu: List[Dict[str, Any]] = []

    for item in menu:
        out = dict(item)
        # Strip internal fields that should never reach the LLM prompt.
        out.pop("_fastest_path_edges", None)
        if not cfg["official_route_guidance_visible"]:
            # Remove advisory labels and authority source produced by the operator briefing logic.
            out.pop("advisory", None)
            out.pop("briefing", None)
            out.pop("reasons", None)
            out.pop("guidance_source", None)

        if cfg["mode"] == "no_notice":
            # Keep fields an agent could plausibly know from local familiarity:
            # route identity, reachability, travel time/length (local knowledge),
            # and observation-based utility scores.
            if control_mode == "destination":
                keep_keys = {
                    "idx", "name", "dest_edge", "reachable", "note",
                    "travel_time_s_fastest_path", "len_edges_fastest_path",
                    "expected_utility", "utility_components",
                    # Visual fire observation fields (agent can see fire on
                    # the first few edges of their current route).
                    "visual_blocked_edges", "visual_min_margin_m",
                    # Proximity fire perception fields (agent is close enough
                    # to a fire to assess its impact on each candidate route).
                    "proximity_blocked_edges", "proximity_min_margin_m",
                }
            else:
                keep_keys = {
                    "idx", "name", "len_edges",
                    "expected_utility", "utility_components",
                }
            out = {k: v for k, v in out.items() if k in keep_keys}

        prompt_menu.append(out)

    return prompt_menu


def filter_history_for_scenario(
    mode: str,
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Strip scenario-inappropriate fields from agent self-history records.

    History records are stored unfiltered for post-hoc analysis, but this
    function creates sanitised copies that respect the active information
    regime before embedding them in the LLM prompt.

    Leaked fields prevented:

    * ``no_notice``: forecast data, advisory/briefing labels, fire-specific
      risk metrics (blocked_edges, risk_sum, min_margin_m), and raw margin
      numbers in the environment signal.
    * ``alert_guided``: advisory/briefing labels and route-head forecast
      detail.
    * ``advice_guided``: no filtering (all data visible).
    """
    cfg = load_scenario_config(mode)

    if cfg["mode"] == "advice_guided":
        return history

    filtered: List[Dict[str, Any]] = []
    for rec in history:
        out = dict(rec)  # shallow copy — only mutated keys are replaced below

        # --- Forecast ---
        if not cfg["forecast_visible"]:
            out["forecast"] = {"available": False}
        elif not cfg["route_head_forecast_visible"]:
            fc = dict(out.get("forecast") or {})
            fc.pop("route_head", None)
            out["forecast"] = fc

        # --- Selected option ---
        sel = out.get("selected_option")
        if sel:
            sel = dict(sel)
            if not cfg["official_route_guidance_visible"]:
                sel.pop("advisory", None)
                sel.pop("briefing", None)
            if cfg["mode"] == "no_notice":
                # Keep only fields plausible from local knowledge.
                sel = {k: v for k, v in sel.items()
                       if k in {"name", "dest_edge", "expected_utility", "travel_time_s"}}
            out["selected_option"] = sel

        # --- Environment signal (no_notice: strip raw margin numbers) ---
        if cfg["mode"] == "no_notice":
            env_sig = (out.get("signals") or {}).get("environment")
            if env_sig:
                out["signals"] = dict(out.get("signals", {}))
                out["signals"]["environment"] = {
                    "observed_state": env_sig.get("observed_state"),
                    "is_delayed": env_sig.get("is_delayed", False),
                }

        filtered.append(out)

    return filtered


def scenario_prompt_suffix(mode: str) -> str:
    """Return an LLM instruction suffix that contextualises the active information regime.

    Injected at the end of the LLM policy string so the model understands what
    information it legitimately has access to and how to frame its decision.

    Args:
        mode: Active scenario mode string.

    Returns:
        A one-to-two sentence instruction string for the LLM.
    """
    cfg = load_scenario_config(mode)
    if cfg["mode"] == "no_notice":
        return (
            "This is a no-notice wildfire scenario: no official warnings or route instructions exist yet. "
            "Do NOT invent official instructions. "
            "Base decisions on your_observation.environment_signal, inbox messages, "
            "and neighborhood_observation. Choose conservative actions if uncertain."
        )
    if cfg["mode"] == "alert_guided":
        return (
            "This is an alert-guided scenario: official alerts describe the fire but do not prescribe a specific route. "
            "Do NOT invent route guidance. Use the provided official alert content, "
            "hazard and forecast cues, and local road conditions to decide when, where, and how to evacuate."
        )
    return (
        "This is an advice-guided evacuation: the Emergency Operations Center has issued official route guidance for your area. "
        "Follow routes marked advisory='Recommended' unless they are physically blocked or impassable. "
        "If you must deviate from official guidance, state why and choose the safest feasible alternative. "
        "Delayed departure or ignoring recommended routes increases your exposure to dangerous fire conditions. "
        "Stay responsive to updated guidance as conditions change."
    )
