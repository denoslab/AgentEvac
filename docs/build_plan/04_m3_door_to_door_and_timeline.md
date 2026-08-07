# P3. M3 door-to-door and the per-run timeline export

Design source `docs/jason_0611_response_assessment.md` Part E for M3 and Part D.7 for the timeline export. Depends on [M1](01_m1_alert_engine.md), [C.7](02_c7_belief_compliance.md), and [M2](03_m2_staggered_awareness.md).

## Part 1. M3 door-to-door

### Goal

A second official channel that sweeps edges inside an ordered area delivering an in-person evacuate instruction at a report-parameterised rate. It is awareness trigger 2 for M2 and a strong but overridable departure driver through C.7.

### Why it matters to the record

The record anchors it. RCMP began door-to-door at 15:42, about 14 min after ignition and about 90 min before the first broadcast alert. It was the primary notification channel for the first hour and a half, so a reconstruction without it mistimes first awareness for the origin community.

### Current state

- The parameters already exist in the `door_to_door` block of `configs/halifax_3town/alerts.json`, with `start_time_s` 840, `initial_areas` westwood_hills, and a `sweep` list carrying `begin_s` and `cleared_by_s` per area.
- No sweep or door-knock code exists. A repo-wide grep for `door_knock` returns nothing.
- The delivery path to reuse is the messaging layer, which already reaches pending agents through `messaging.begin_round` at `main.py:3596`.

### Changes

- `alert_schedule.py` gains `door_knock_time(edge_id)` from the `door_to_door` block, spreading per-edge knock times across each area's `begin_s` to `cleared_by_s` window by edge order. This is the pure function M2 trigger 2 calls.
- A small door-state object tracks which edges have been knocked by `sim_t`, updated each decision tick.
- The door-knock order is delivered through the C.7 belief channel with the in-person `channel_factor` of about 1.0, so it raises the household's danger belief strongly but does not force departure. Tag its departure `reason` as `order`, with the channel recorded so the compliance metric can split door-knock from broadcast.
- Expose the per-area sweep duration as an M3 sensitivity knob, since one subdivision took roughly 46 min of door-knocking before the broadcast order.

### Tests

- `door_knock_time` returns times inside the area window for Westwood edges, ordered by edge order, and None for edges outside any swept area.
- A Westwood agent with no other signal becomes aware via `door_knock` at its edge's knock time.
- Compliance split records door-knock and broadcast channels separately.

### Open sub-decisions

1. The sweep-rate shape, meaning how per-edge knock times distribute across the window. Default is linear by edge order across `begin_s` to `cleared_by_s`. Confirm before implementing if you want a different profile.

## Part 2. Per-run timeline export

### Goal

Each run emits one consolidated timeline that merges the scripted layers, meaning fire, alert, and door-to-door, with the emergent layer, meaning awareness, departure, arrival, reroute, and area clearance, on the ignition-anchored clock. This is the artifact the E0-versus-record validation reads.

### Current state

- The pieces exist scattered across the event stream and the metrics JSON. The event stream is `LiveEventStream` at `main.py:655` with `emit` at `main.py:699`, instantiated at `main.py:1546`.
- There is no single consolidated timeline file. The one `run_timeline` reference in the repo is an unrelated plot filename.

### Changes

- Emit `run_timeline_<run_id>.jsonl`, one row per event with a common schema, for example `{t_s, layer, type, agent_id?, area?, source?, detail}`, through the existing event log.
- Scripted rows come from the fire schedule, the alert schedule, and the door sweep at load or as they fire. Emergent rows come from the new M2 awareness events, the departure and arrival records, reroutes, and a new area-clearance event when the last ordered household in an area departs.
- The complete scripted layer for E0 is already enumerated in `docs/halifax_e0_event_timeline.md`, so that file is the fixture to check the scripted rows against.

### Tests

- A short toy run emits a timeline whose scripted rows match the fire and alert schedules, and whose emergent rows include at least one awareness and one departure with correct `t_s` ordering.
- Area clearance fires once per area at the last ordered departure.

### Acceptance criteria

- For an E0 run, the timeline reproduces the three alert waves at 6300, 9660, 15180 s and the door-to-door start at 840 s, and carries an awareness row per aware agent with its source.
- The timeline is sufficient on its own to compute clearance sequencing, the departure curve, and the awareness-source distribution for validation.
