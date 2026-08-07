# P1a. M1 alert-event engine

Design source `docs/jason_0611_response_assessment.md` Part C. Co-developed with [C.7 belief-only compliance](02_c7_belief_compliance.md). Line numbers are current at time of writing and will drift as code lands, so anchor on the function names.

## Goal

Alerts stop being a run-global scenario flag and become a schedule of timed, area-scoped events. The scenario filters become per-agent and per-round functions of that schedule. The existing noise and delay machinery stays as sensitivity knobs.

## Why this is needed

The code entangles three bits and only ever sets them together through one run-global mode. H is hazard visible, D is a directive order present, R is routing visible. The legacy regimes occupy only three of eight corners, `no_notice` is (0,0,0), `alert_guided` is (1,0,0), `advice_guided` is (1,1,1). The real 2023 order was (1,1,0), meaning hazard plus an evacuate directive with no route guidance, which the triad cannot represent. M1 makes all the cells reachable from a schedule.

## Current state

- The scenario filters key only on `mode`, verified in `agentevac/agents/scenarios.py` at `apply_scenario_to_signals`, `filter_menu_for_scenario`, and `filter_history_for_scenario`. No time or position argument exists.
- The order block is injected run-global, guarded by `if SCENARIO_CONFIG["mode"] == "advice_guided"` at `main.py:2613`, present from round 1 for every agent.
- `config_loader.load_map_config` (`agentevac/config_loader.py:70`) loads a `(name, required)` list at lines 94 to 97, meaning spawns, fires, destinations, routes. It does not load `alerts.json`.
- The schedule data already exists and is populated, `configs/halifax_3town/alerts.json` and `configs/halifax_3town_e0/alerts.json`, with three bundled orders EA-1, EA-2, EA-3 at 6300, 9660, 15180 s, the door-to-door block, area-to-community-to-edge maps, comfort centres, and `routing_text` null for E0.

## Data model

The `alerts.json` schema is already authored, so no schema design is needed. Its shape, confirmed in the config, is an `areas` map from area name to `{model_community, edges, agents}`, a `schedule` list of events with `issue_time_s`, `areas`, `instruction`, `hazard_text`, `comfort_centre`, `routing_text`, `channel`, and a `door_to_door` block. Orders are cumulative, so once an area is ordered it stays ordered. Membership is an edge test, an agent belongs to an area if its spawn edge is in that area's `edges` list, which is exact for the pre-departure phase where every agent sits on its spawn edge.

## Changes

### New module `agentevac/agents/alert_schedule.py`

A pure resolver with no `traci` and no RNG, so it is fully unit-testable. Mirrors design Part C.2.

```python
@dataclass(frozen=True)
class AlertEvent:
    id: str
    issue_time_s: float
    areas: tuple[str, ...]        # newly ordered areas, cumulative across events
    instruction: str              # none | evacuate_now | shelter_in_place
    hazard_text: str
    routing_text: str | None
    comfort_centre: str | None
    channel: str

@dataclass(frozen=True)
class AlertState:
    received: bool                # any covering order issued by sim_t
    received_t_s: float | None    # earliest covering issue time, the alert-channel awareness instant
    instruction: str
    hazard_visible: bool
    routing_visible: bool
    order_text: dict | None       # bundled order block for C.7, or None

NO_ALERT = AlertState(False, None, "none", False, False, None)

class AlertSchedule:
    @classmethod
    def from_config(cls, cfg: dict) -> "AlertSchedule": ...
    @classmethod
    def empty(cls) -> "AlertSchedule": ...
    def areas_for_edge(self, edge_id: str) -> set[str]: ...
    def active_for_edge(self, sim_t_s: float, edge_id: str) -> AlertState: ...
    def door_knock_time(self, edge_id: str) -> float | None: ...   # M3 awareness trigger
```

`from_config` builds a reverse index edge to areas once for O(1) lookups. `active_for_edge` collects every event with `issue_time_s <= sim_t_s` whose `areas` intersect the agent's areas, takes the earliest as `received_t_s`, the strongest `instruction`, and ORs `routing_visible`, so an earlier order stays in effect when a later one extends the area. `empty()` returns `NO_ALERT` for every resolve, which backs `no_notice` and the E3-degenerate arm on the same code path.

### The adapter, no rewrite of scenarios.py

A thin adapter maps the resolved state to the mode the existing filters already implement.

```python
def effective_mode(a: AlertState) -> str:
    if not a.received:     return "no_notice"      # (0,0,0)
    if a.routing_visible:  return "advice_guided"  # (1,1,1)
    return "alert_guided"                          # (1,0,0) filter view
```

The directive bit D is set separately through the order block, so the historical (1,1,0) cell is the `alert_guided` filter view plus the order text. The internals of `scenarios.py` stay untouched because they already strip the right fields per mode.

### `config_loader.py`

Add `("alerts", False)` to the load list at `config_loader.py:94-97`, so `cfg["alerts"]` holds the parsed schedule or an empty default when absent. Absent means `AlertSchedule.empty()`.

### `main.py` wiring

- Build the schedule once at startup near the fire-source load, `sched = AlertSchedule.from_config(_MAP_CFG.get("alerts") or {})` or `AlertSchedule.empty()`.
- Add `ALERT_TIME_OFFSET_S` from the environment, applied at load to every `issue_time_s` for E1, capped at -3600 s.
- Swap the run-global `SCENARIO_MODE` for a per-agent `effective_mode` at the resolved call sites. Each site needs the agent's spawn edge and `sim_t` to resolve `AlertState`. Current sites are
  - `apply_scenario_to_signals` at 2507, 3101, 3732, 4050, 4361
  - `filter_menu_for_scenario` at 3098, 4063, 4373
  - `filter_history_for_scenario` at 3625
  - `scenario_prompt_suffix` at 2610, 3246, 4217, 4525
  - `scenario_system_prompt` at 2622, 3249, 4220, 4528
- Replace the order-block injection at `main.py:2613`. Under belief-only compliance this block is deleted from the predeparture prompt, see the C.7 file. The resolved `alert_state` still supplies the active-order flag and `order_text` to the belief channel and to metrics.
- Record the resolved `effective_mode` and `received_t_s` per agent-round in metrics and replay, since the lived-experience coding needs source and time of first official receipt.

## Determinism and replay

The schedule is static config, so M1 introduces no new RNG draws and replay stays bit-identical. `INFO_SIGMA` and `INFO_DELAY_S` are retained as sensitivity knobs that now modulate around the schedule.

## Tests

Unit tests in a new `tests/test_alert_schedule.py`.

- `areas_for_edge` reverse-index correctness.
- `active_for_edge` returns `NO_ALERT` before EA-1, Westwood after EA-1, Westwood plus Highland Park after EA-2 cumulatively, and flips `routing_visible` when `routing_text` is set.
- `effective_mode` mapping across the three corners.
- `empty()` always returns `NO_ALERT`.
- `door_knock_time` returns the expected sweep time for a Westwood edge and None elsewhere.

## Open sub-decisions, resolve before wiring

1. Institutional delay under a schedule. The natural reading is a per-agent receipt lag after `issue_time_s`, a refinement of `apply_institutional_delay` in `information_model.py`. Default for the first pass is to leave `apply_institutional_delay` as-is and note the refinement, so delay stays a global knob. Confirm before touching delay.
2. The (1,1,0) cell. Produced through the `alert_guided` filter plus the D-bit via the adapter, with no `scenarios.py` change. The alternative is a small refactor letting the filters accept a resolved config dict directly. Default is the adapter.

## Acceptance criteria

- With `alerts.json` present, an agent in Westwood sees `no_notice` filtering before 6300 s and the (1,1,0) view plus an active order after, and an agent in the outer extension stays `no_notice` until 15180 s.
- With no `alerts.json`, every resolve is `NO_ALERT`, so the run reproduces the legacy no_notice path.
- Replay of an M1 run is bit-identical to its record.
