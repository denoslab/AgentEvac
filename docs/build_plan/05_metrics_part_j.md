# P4. Metrics, Part J

Design source `docs/jason_0611_response_assessment.md` Part J. Depends on awareness times from [M2](03_m2_staggered_awareness.md) and emergent compliance from [C.7](02_c7_belief_compliance.md). All anchors are in `agentevac/analysis/metrics.py`.

## Goal

The static-triad metric set needs additions and two reframes so the reconstruction can be validated against the record and so E4 can report emergent compliance.

## Current state

- `record_departure` at `metrics.py:111` takes a `reason` argument and discards it, storing only the timestamp.
- `record_exposure_sample` at `metrics.py:254` fires once per active vehicle per tick, so a household that never becomes aware or refuses the order is never an active vehicle and contributes zero exposure.
- `summary` at `metrics.py:442` builds the KPI JSON.
- There is no awareness time, no compliance, no area clearance, and no corridor flow.

## Add

1. **Mobilization delay**, awareness to departure. Record `awareness_t_s` from M2 alongside departure, report per-agent and as a distribution. SCOPING NOTE, 2026-07-17, a pre-evacuation preparation lag is deliberately not modeled, to avoid introducing another free variable, so departure equals the decision. In the usable msgoff baseline awareness arrives only through strong signals, perception or an active order, that compel departure the same round, so this reduces to awareness stagger and is about 0. It is stated as a limitation, and E1 is therefore a timing-shift study, earlier order gives earlier clearance and lower exposure with no behavioural delay dynamics.
2. **Order compliance rate**. With C.7 this is emergent and is the E4 headline. The fraction of ordered households that evacuated, split by channel, meaning door-knock versus broadcast, and by area, reported against `theta_auth`. The backbone is cheap, store the departure `reason` that `record_departure` currently discards and add the `order` reason from C.7, and a departure-reason histogram falls out for free.
3. **Order-area clearance time and sequencing**. Per area, the time the last ordered household departed. Primary E0 validation quantity. Reuse the alert-area edge sets for membership.
4. **Awareness-source distribution and never-aware count**. Share of first awareness by channel, meaning door, alert, perception, peer, plus `n_never_aware`. Validates the record's source of first warning. These are promoted from M2 fields to first-class metrics.
5. **Corridor egress-flow split** (OPTIONAL). Flow on the key corridors, the NS101 versus NS103 analog, about 2 to 1 in the record. Dropped as a calibration target, since the three-community, 182-agent scope cannot reproduce a regional highway split, so the metric adds nothing to the timing, compliance, and awareness core of E0, E1, and E4. Kept only as a descriptive output for the E2 congestion story, never fit to the ratio. Define corridor edge sets like the alert areas and count throughput.

The departure curve itself comes from the timeline export in [P3](04_m3_door_to_door_and_timeline.md), so it needs no separate metric.

## Reframe

1. **Exposure semantics for non-evacuees**, important. Because `record_exposure_sample` at `metrics.py:254` only samples active vehicles, a stayer contributes zero exposure, so under emergent non-compliance a low-trust run where many stay would show lower mean exposure even though staying is dangerous. Report exposure conditional on evacuating, and add a companion count of non-evacuated households the fire reaches, meaning the fire crossing a stayer's spawn edge. Without this, E1 and E4 exposure numbers mislead.
2. **`departure_time_variability` meaning**. It used to reflect decision heterogeneity. Under staggered awareness it is dominated by awareness stagger. Keep it but reinterpret it in the writeup, and lean on mobilization delay for the behavioural story.

## Keep

`average_travel_time`, `route_choice_entropy`, `decision_instability` with emphasis moved to the E3 ablation, `destination_choice_share` mapped to the real comfort centres for validation, `fire_contact`, and `average_signal_conflict`, optionally extended to three channels once C.7 adds the institutional belief.

## Which arm needs what

| Arm | Metrics |
|---|---|
| E0 validation | clearance sequencing, mobilization delay, awareness-source distribution, corridor split, comfort-centre share |
| E1 timing | exposure and travel time versus order time, mobilization delay |
| E2 content | corridor concentration, per-area clearance |
| E3 ablation | route entropy, decision instability, reproducing the old triad |
| E4 trust | compliance rate versus theta_auth |

## Touch points

- `metrics.py:111` store the departure `reason`, add the `order` reason via C.7.
- `metrics.py:254` exposure sampling, add a non-evacuee fire-reach count and a conditional-exposure split.
- New recorders, `record_awareness(agent_id, t_s, source)`, area tagging for clearance, and corridor edge-set throughput.
- `metrics.py:442` summary additions, `mobilization_delay`, `compliance` with overall, by_channel, and by_area, `awareness_source_share`, `n_never_aware`, `area_clearance`, `corridor_flow`, `non_evacuated_reached_by_fire`.

## Open sub-decisions

1. The exposure reframe reporting form, meaning whether the headline exposure number is conditional-on-evacuating with the stayer count as a companion, or a single combined index. Default is conditional plus companion count.
2. The corridor edge sets for the NS101 versus NS103 split, which need defining against the network geometry. This is a small data task, do it during E0 validation.

## Tests

- `record_departure` stores and returns the reason, and the reason histogram counts `order`.
- Compliance splits correctly by channel and area on a toy run.
- Conditional exposure excludes stayers and the companion count includes a stayer whose spawn edge the fire reaches.
- Clearance fires at the last ordered departure per area.
