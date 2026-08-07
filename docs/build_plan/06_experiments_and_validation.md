# P5. Experiments E0 to E4 and validation

Design source `docs/jason_0611_response_assessment.md` Parts G, I, and the validation gate in I.5. This runs only after M1, C.7, M2, M3, the timeline export, and the Part J metrics are in place.

## The arms as one engine

Every arm is the same engine with a different config plus a few flags, so no arm needs a code fork.

| Arm | alerts.json | flags | tone |
|---|---|---|---|
| E0 reconstruction | historical, `routing_text` null | none | directive |
| E1 timing | historical | `ALERT_TIME_OFFSET_S` in {-3600, -1800, -900, +900, +1800, +3600} s, capped at -60 min | directive |
| E2 content | historical plus `routing_text` populated | none | neutral, reuse `advice_guided_neutral` |
| E3 ablation | degenerate, empty for no_notice and instruction none for hazard-only | none | directive |
| E4 trust | any of the above | `theta_auth` sweep, 5 levels | per arm |

E1 is a single scalar applied at load, so earlier alerting at EOC-decision time is a specific negative offset near -57 min. Per decision 3, the buffer and deadline mechanism is deferred until E0 produces the per-community clearance times that size the buffers. E3 degenerate schedules reproduce the old static triad on the staggered path, which is the new baseline per decision 2, with the bit-exact old numbers checked against the pre-M2 tag rather than on head.

## Validation gate, run E0 first

This is the most important sequencing rule. Run E0 and validate the substrate against the record before spending runs on E1 to E4. If E0 does not reproduce the record's coarse pattern, the counterfactuals are built on sand.

Pre-register the acceptance criteria in writing before the E0 runs, so the comparison is not post-hoc fitting. The criteria, read from the per-run timeline export and the Part J metrics, are

- order-area clearance sequencing, Westwood then Highland Park then the outer ring,
- departure-curve shape against the documented order times,
- comfort-centre share direction, with Black Point named in EA-1,
- lived-experience coding per `docs/halifax_reconstruction_inputs.md`, meaning source and timeliness of first warning, guidance ambiguity, and destination support.

If E0 clears the gate, proceed to E1 through E4. If it does not, fix the substrate first.

OPTIONAL, dropped from the gate. The congestion-corridor rank, where Sen GPS shows NS101 about twice NS103, is not an acceptance criterion. The three-community, 182-agent scope cannot reproduce a regional highway split, so the metric adds nothing to the timing, compliance, and awareness core of E0, E1, and E4. If kept at all it is a descriptive output for the E2 congestion story only, never calibrated to the ratio, using the real Hwy 103 edges 70756039#1 and 697504315#1 with the Beaver Bank approach 1247430237#0 as the NS-101 proxy.

## E0 config

Built already, `configs/halifax_3town_e0/`. Same network, spawns, destinations, and routes as `halifax_3town`. The differences are the record-calibrated fire timing with `max_r_m`, which the cap now honors, and the alert schedule. See `configs/halifax_3town_e0/README.md`.

## Second layout, deferred

Per decision 4, author `configs/halifax_tantallon_e0/` only after the 3-town E0 validates. When built, it re-times its `fires.json`, adds `max_r_m`, and carries an `alerts.json` scoped to the subdivisions that layout actually spawns, verified against its spawn edges. Until then the first pass is single-layout, so the grid below is halved on the layout axis for the first pass.

## Experiment grid

About 104 LLM runs at full two-layout scope, plus the rule-based mirror, well under a fifth of the old campaign. First-pass single-layout numbers are half the LLM column.

| Arm | Cells, two-layout | LLM runs, two-layout |
|---|---|---|
| E0 reconstruction baseline | 2 layouts x 10 seeds | 20 |
| E1 timing, 6 offsets | 6 x 2 layouts x 3 seeds | 36 |
| E2 content | 1 x 2 layouts x 3 seeds | 6 |
| E3 ablation | 2 x 2 layouts x 3 seeds | 12 |
| E4 trust, 5 levels | 5 x 2 layouts x 3 seeds | 30 |

Every arm is mirrored with the rule-based policy, which is free of API cost and faster.

## Run settings

- `DECISION_PERIOD_S=240`, `SIM_STEP_LENGTH_S=0.2`, so 1200 steps per round, both in place from M0.
- `sim_end_time_s=28800`, the 8-hour horizon, so EA-3 at 15180 s is inside the run.
- `--map halifax_3town_e0`, and later `halifax_tantallon_e0`.
- Seeds via the seed flag, LLM versus rule-based via the agent-type flag.
- Arm flags, E1 `ALERT_TIME_OFFSET_S`, E2 the routing alerts variant, E3 the degenerate alerts variant, E4 the `theta_auth` sweep.

## Open sub-decisions

1. The E2 `routing_text` content, meaning the recommended shelter plus egress corridor per area, authored from the municipal plan in `docs/halifax_reconstruction_inputs.md`. Draft it when E2 comes up.
2. The E4 `theta_auth` sweep levels, tied to the C.7 calibration.
3. The pre-registered E0 acceptance thresholds, meaning how close the clearance sequencing and corridor rank must be to count as reproducing the record. Settle before the E0 runs.

## E4 trust gradient, resolved as a structural null (2026-07-22)

E4 is flat on the outcome at every offset and every `theta_auth`, while the trust channel is demonstrably wired, the departure-reason histogram shifts from `order` to a backstop clause as `theta_auth` falls. The masked-by-urgency hypothesis was tested on the free rule_based mirror at the early offset with `DEFAULT_GAMMA=0.999` and `DEFAULT_THETA_U=0.15`, set through the environment. It did not unmask the gradient. Urgency departures fell to zero as intended, but `low_confidence_precaution` took over as the backstop and released the same households at the same 6240 s clearance, so clearance and exposure stayed flat. E4early is now confirmed flat at both `0.995/0.30` and `0.999/0.15`.

The conclusion is structural, not a one-parameter calibration. The departure model has several self-protective clauses, meaning risk, urgency, low-confidence precaution, and neighbour pressure, so an aware household that ignores a low-trust order is released by whichever precaution fires first, and there is no lingering or stayer state for trust to gate. Authority trust therefore sets the attributed reason for departure, not the evacuation outcome, which is the reportable E4 result. A trust-gated compliance gradient would require an added stayer mechanism, out of scope for the same reason mobilization delay is.
