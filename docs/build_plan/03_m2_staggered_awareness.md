# P2. M2 staggered awareness

Design source `docs/jason_0611_response_assessment.md` Part D. Depends on [M1](01_m1_alert_engine.md) for its primary trigger. This is decision 2, locked to staggered-only, with the awareness-mode flag deferred.

## Goal

The first policy call fires on first signal arrival instead of the universal opening round. This is the prerequisite for every timing result and retires the simultaneous-first-round limitation.

## Current state

- Every agent is eligible from t=0. The unconditional `effective_t0 = 0.0` sits in the `SPAWN_EVENTS` loop at `main.py:2401`, with the eligibility check on the next line.
- `ensure_agent_state` is called eagerly for all agents, which stamps `created_sim_t_s` at the opening round for everyone, so the urgency clock in `departure_model.py` runs from t=0 for all.
- The two ingredients for perception and peer triggers already exist. `_visible_fires` is at `main.py:2033` and is used at 3635, and `SPAWN_EDGE_MIDPOINT` is precomputed at `main.py:1706`. Peer messages already reach pending agents, since `messaging.begin_round` includes `pending_agent_ids` at `main.py:3596`.

## Decision 2 consequence

Staggered awareness changes the opening round, so the completed 774-run campaign no longer reproduces bit-for-bit on head once M2 lands. Per decision 2 we do not build the dual-mode `AWARENESS_MODE` flag now.

- Implement staggered as the single path.
- Before M2 lands, tag the current post-M0 commit, for example `pre-m2-baseline`, so the old campaign and the E3 bit-exact regression can be reproduced by checkout against that tag.
- The E3 ablation as a published result runs on the staggered path, which is the new baseline. The bit-exact reproduction of the old numbers is a development-time regression against the tag, not an on-head test.
- The awareness-mode flag stays on the deferred list as the future feature that would restore on-head reproduction.

## Changes

### The gate, `main.py` around 2401

Replace the unconditional `effective_t0 = 0.0` eligibility with a per-agent awareness gate. The agent state, and therefore its urgency clock, is created only when the agent first receives a signal.

```python
# inside the SPAWN_EVENTS loop, non-replay branch, replacing the
# unconditional effective_t0 = 0.0 eligibility block
if vid not in AGENT_STATES:                      # dormant, not yet aware
    pos = SPAWN_EDGE_MIDPOINT[vid]
    trig, source = check_awareness(
        vid, pos, sim_t, fire_geom, alert_sched, doorknock_state, inbox_of(vid)
    )
    if not trig:
        continue                                 # no LLM call, no urgency accrual
    agent_state = ensure_agent_state(vid, sim_t, ...)   # created_sim_t_s = awareness time
    agent_state.has_departed = False
    record_awareness(vid, sim_t, source)
else:
    agent_state = AGENT_STATES[vid]
    if not evaluate_departures:
        continue
```

The single most important consequence is that `created_sim_t_s` now equals the awareness time, so the urgency-decay clause in `departure_model.py` clocks from awareness rather than from t=0. A late-aware household gets a fresh urgency clock and does not instantly feel compelled to act. `departure_model.py` itself is unchanged, the field it reads now carries the right meaning.

### New helper `check_awareness`

Either in `main.py` or a small `agentevac/agents/awareness.py`. Checked each decision tick for not-yet-aware pending agents, returning the first trigger that fires and its source.

1. Alert receipt, needs M1. `alert_state.received` is true for the agent's spawn area at `sim_t`.
2. Door knock, needs M3. The RCMP sweep has reached the agent's spawn edge, via `alert_sched.door_knock_time(edge) <= sim_t`.
3. Perception crossing. A fire is within `FIRE_PERCEPTION_RANGE_M` of the spawn-edge midpoint, via the existing `_visible_fires`.
4. Peer message. The inbox is non-empty.

Record `awareness_t_s` and `awareness_source` in `{alert, door_knock, perception, peer}`. The source maps directly onto Jason's source-of-first-warning coding category.

### Why deferring state creation is safe

- Belief is created fresh as a uniform prior at awareness and updated from that round forward, no warm-up needed.
- Pre-aware agents still receive peer messages, so trigger 4 works, they simply do not send or decide until aware.
- The neighbor-departure clause only evaluates for aware agents, which is correct.
- `total_agents` counts spawn events, unaffected, and departure-time metrics key off actual release.
- The `MAX_VEHICLES_PER_DECISION` round-robin at `main.py:3568` still bounds LLM calls per round, awareness only adds an agent to the eligible pool.

## Replay

Awareness is deterministic given the schedule, fire timeline, message log, and positions. Replay releases on recorded departure events, so the gate is bypassed and there is no divergence. Record the awareness fields in record mode for fidelity and analysis.

## Tests

- A toy map with one early-order agent, one extension-area agent, and one never-ordered agent that perceives fire. Assert awareness times, sources, and that the urgency clock starts at awareness.
- Each `check_awareness` trigger in isolation.
- Determinism, same seed yields identical `awareness_t_s` and decisions across two runs.

## Open sub-decisions

1. Never-aware agents. A household whose area is never ordered and never perceives fire and never hears a peer stays dormant and never evacuates, which is realistic for far-field households. Add an `n_never_aware` run-end metric so a mis-scoped order polygon does not silently drop agents. For E0, verify the order areas plus perception cover the spawn population before trusting a run.

## Acceptance criteria

- A Westwood agent becomes aware at or before its area's order time, an outer-extension agent not before its order unless perception or a peer reaches it first, and the awareness source is recorded for each.
- The urgency clock for a late-aware agent starts at its awareness time, verified by a decision-history check.
- `n_never_aware` is reported and is zero for a correctly scoped E0 run.
