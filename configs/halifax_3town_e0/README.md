# halifax_3town_e0

E0 reconstruction config for the 28 May 2023 Upper Tantallon wildfire, derived from `halifax_3town`. Same network, spawns, destinations, and routes. The differences are the record-calibrated fire timing and the alert schedule. Do not edit `halifax_3town`, it stays as the original campaign config.

References and provenance: `docs/halifax_e0_event_timeline.md` (calibration section) and `docs/halifax_reconstruction_inputs.md`. Design: `docs/jason_0611_response_assessment.md`.

## What changed vs halifax_3town

| File | Change |
|---|---|
| `fires.json` | Sources 1 to 10 unchanged, already within about 2 min of the record. The three eastward anchors New_Spread_1, New_Spread_3 and New_Spread_4 re-timed to the record. The four interpolated sources New_Spread_2, New_Morningsong_Ln, New_Carmel_Cr and New_Shelby_Dr removed as unreferenced. A 700 m `max_r_m` cap on every source. See below. |
| `alerts.json` | Copied unchanged. The M1 schedule (EA-1 6300 s, EA-2 9660 s, EA-3 15180 s) and the door-to-door block. |
| `map.json`, `spawns.json`, `destinations.json`, `routes.json` | Copied unchanged. |

## Fire re-timing and pruning of the eastward sources

The record's eastward spread runs 107 to 205 min (HRM after-action report, physical PDF p.10). Only the three eastward points that carry a record timestamp are kept, re-timed to those anchors. The four interpolated points between them are removed, because they have no record basis and, sitting close to Highland Park, they made agents perceive fire before the order.

| Source | t0 (s) | +min | Status | Record anchor (HRM p.10) |
|---|---|---|---|---|
| New_Spread_1 | 6420 | 107 | kept | "Fire moved into woods beyond Westwood sub-division", 17:15 |
| New_Spread_2 | removed | removed | removed | none, interpolated |
| New_Morningsong_Ln | removed | removed | removed | none, interpolated |
| New_Carmel_Cr | removed | removed | removed | none, interpolated |
| New_Shelby_Dr | removed | removed | removed | none, interpolated |
| New_Spread_3 | 9840 | 164 | kept | "Fire jump the road hwy103", 18:12 |
| New_Spread_4 | 12300 | 205 | kept | "Crews responding to North/East Pockwock", 18:53 |

Sources 1 to 10, the Westwood origin area from 0 to 56 min, are unchanged because they already match the record within about two minutes. Ignition times, locations, `r0` and `growth_m_per_s` for those ten are identical to `halifax_3town`.

## Radius cap `max_r_m`, honored by the simulator

Every source carries `max_r_m: 700`. `active_fires` in `main.py` honors it, `r = min(r0 + growth * dt, max_r_m)`.

Why it is needed. The fire growth model is linear with no burnout, so over the multi-hour E0 horizon the origin Westwood fires would otherwise grow without bound. The cap also sets how far each fire reaches by radius alone, which through the 1200 m `FIRE_PERCEPTION_RANGE_M` decides when a distant community first sees fire. At the earlier 1500 m cap the origin fires reached Highland Park 1908 m away and made it self-evacuate about 100 min before its order, an artefact of the growth model rather than the record.

A cap of 700 m keeps each origin fire inside Westwood, so Highland Park first perceives fire from New_Spread_1 at +107 min, the record instant the fire moved beyond Westwood, and the outer ring from New_Spread_4 at +205 min. The value is calibrated against the perception geometry and is open to further tuning. The residual, each community seeing its fire before its order, is faithful to the record, in which the fire reached each area ahead of the evacuation order.

## Run settings for E0

- `DECISION_PERIOD_S=240` and the 0.2 s SUMO step (after the M0 fix).
- `sim_end_time_s=28800` (8 h, 120 decision rounds) so EA-3 at 15180 s is inside the horizon.
- `--map halifax_3town_e0`.
