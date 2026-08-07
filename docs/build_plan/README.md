# AgentEvac reconstruction build plan

High-level index for the Halifax reconstruction engineering effort. This turns the domain-expert exchange with Jason into implementable modules. Each work item has its own file in this folder. Read this file first, then the item file for whatever you are about to build.

## What this is

The exchange in `docs/response-0617.txt` and `docs/QA-0618.txt` asks us to rebuild the experiment around the real 28 May 2023 Upper Tantallon wildfire, with bundled evacuation alerts on the documented timeline, an alert-timing counterfactual, an official-routing counterfactual, and calibration against evacuee movement. The engineering design that decomposes this into modules is `docs/jason_0611_response_assessment.md`, which labels them M0 to M4 and experiment arms E0 to E4. This build plan follows that design, corrected to the current code and to the four decisions locked below.

Source material to keep open while building.

| Document | Role |
|---|---|
| `docs/jason_0611_response_assessment.md` | Master design, module and arm definitions, code touch points |
| `docs/halifax_reconstruction_inputs.md` | Real-event provenance, alert times, door-to-door, page citations |
| `docs/halifax_e0_event_timeline.md` | Complete scripted E0 timeline on the ignition-anchored clock |
| `configs/halifax_3town_e0/` | Built E0 config, re-timed fires with `max_r_m`, alert schedule |
| `configs/halifax_3town/alerts.json` | Populated 28 May 2023 alert and door-to-door schedule |

## Status

| Item | Status | Plan file |
|---|---|---|
| M0 clock fix | Done, this session | recorded below |
| Fire radius cap | Done, this session | recorded below |
| M1 alert engine | Done | [01_m1_alert_engine.md](01_m1_alert_engine.md) |
| C.7 belief-only compliance | Done | [02_c7_belief_compliance.md](02_c7_belief_compliance.md) |
| M2 staggered awareness | Done | [03_m2_staggered_awareness.md](03_m2_staggered_awareness.md) |
| M3 door-to-door and timeline export | Planned | [04_m3_door_to_door_and_timeline.md](04_m3_door_to_door_and_timeline.md) |
| Metrics, Part J | Planned | [05_metrics_part_j.md](05_metrics_part_j.md) |
| Experiments E0 to E4 and validation | Planned | [06_experiments_and_validation.md](06_experiments_and_validation.md) |
| Config authoring in the operator console | Stage 0 in progress | [07_config_authoring_ui.md](07_config_authoring_ui.md) |

Item 07 sits outside the reconstruction sequence below. It depends on the operator
console in `ui/` and produces new packages without touching the ones already run.

### M0 and radius cap, completed

Done in this session and covered by the passing suite plus a new `tests/test_clock_mapping.py`.

- `DECISION_PERIOD_S` default moved from 60.0 to 240.0 and a `SIM_STEP_LENGTH_S` constant extracted, both env-overridable (`main.py:144-145`).
- A module-load guard exits unless `DECISION_PERIOD_S` is an integer multiple of `SIM_STEP_LENGTH_S`, float-tolerant (`main.py:505`).
- A `[CLOCK]` startup log prints both clocks and the resolved `delay_rounds`, and the run-params snapshot records `sim_step_length_s` and `decision_period_steps`.
- The delay-rounds mapping is centralized in `delay_rounds_for` in `agentevac/utils/run_parameters.py` and used at both former inline sites, so {0, 240, 480, 720} maps to {0, 1, 2, 3} at the 240 s round.
- `active_fires` honors an optional per-source `max_r_m`, so E0 sources bound at 1500 m while legacy configs without the key stay unbounded and bit-identical (`main.py:626`).

## Locked decisions

| # | Decision | Locked choice |
|---|---|---|
| 1 | Compliance model | Belief-only. A `theta_auth` belief channel, delete the mandatory order rule, and no order text at the departure step. Factual-hybrid stays a stated fallback if the lived-experience coding later needs visible order text |
| 2 | Legacy reproduction under M2 | Staggered-only now. The awareness-mode flag is a deferred future feature. Reproduce the old campaign from a pre-M2 tag in the interim |
| 3 | Alert timing | Global offset first, `ALERT_TIME_OFFSET_S` capped at -60 min. Buffer and deadline alerting deferred until E0 yields per-community clearance times |
| 4 | Second layout | Defer Tantallon E0 until the 3-town E0 reproduces the record |

## Build sequence and dependencies

```
M0 clock fix ─┐
              ├─> done, unblocks everything below
radius cap ───┘

P1  M1 alert engine  ──co-developed──  C.7 belief-only compliance
      │                                     │
      │  (alert_schedule.py resolver first, then wiring, then belief channel)
      v
P2  M2 staggered awareness   (alert receipt is its primary trigger, needs M1)
      v
P3  M3 door-to-door  +  per-run timeline export   (door is M2 trigger 2; timeline consumes awareness + clearance)
      v
P4  Metrics, Part J   (needs awareness times from M2 and compliance from C.7)
      v
P5  E0 validation on 3-town, then E1 global offset, E2 routing, E3 ablation, E4 theta_auth
```

Start P1 with the pure `agentevac/agents/alert_schedule.py` resolver and its unit tests, because it is SUMO-free and nothing imports it, so it lands with zero behavioural risk.

## Experiment grid

From the design Part I.3, about 104 LLM runs plus the rule-based mirror. Each arm derives from the E0 config with a few flags.

| Arm | Cells | LLM runs |
|---|---|---|
| E0 reconstruction baseline | 2 layouts x 10 seeds | 20 |
| E1 timing, 6 offsets capped at -60 min | 6 x 2 layouts x 3 seeds | 36 |
| E2 content, routing added, neutral tone | 1 x 2 layouts x 3 seeds | 6 |
| E3 ablation, hazard-only and no-notice | 2 x 2 layouts x 3 seeds | 12 |
| E4 trust sweep on theta_auth, 5 levels | 5 x 2 layouts x 3 seeds | 30 |

The second layout enters only after 3-town E0 validates, per decision 4, so the first pass is single-layout.

## Run settings

After M0 and the cap, which are in place.

- `DECISION_PERIOD_S=240`, `SIM_STEP_LENGTH_S=0.2`, so 1200 steps per round.
- `sim_end_time_s=28800`, the 8-hour horizon, so EA-3 at 15180 s sits inside the run.
- `--map halifax_3town_e0`.
- Seeds via the seed flag, LLM versus rule-based via the agent-type flag.
- Arm flags, E1 `ALERT_TIME_OFFSET_S`, E2 the routing alerts variant, E3 the degenerate alerts variant, E4 the `theta_auth` sweep.

## Deferred, in order of likely need

1. Tantallon E0 config, after 3-town validation.
2. Buffer and deadline alerting arm, after E0 produces per-community clearance times that size the buffers.
3. Awareness-mode flag, a future feature that restores on-head reproduction of the pre-M2 campaign.
4. M4 shelter-in-place, cut for the MVP.
5. Large-scale demand above 10000 vehicles, stated as a limitation, and the 50-call-per-round throttle it needs already exists at `main.py:639`.
6. GPS calibration against Sen 2025, an analysis and data task.
