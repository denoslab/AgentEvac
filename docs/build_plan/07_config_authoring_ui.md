# P7. Config authoring in the operator console

Authoring a scenario package by drawing on the map instead of hand-editing JSON. A
selection box picks buildings by centroid, each selected building carries an editable
agent count defaulting to one, and a second box picks the buildings an alert order
covers. Fire sources are placed by click with their growth parameters beside them.

This item sits outside the Halifax reconstruction sequence in
[README.md](README.md). It depends on the operator console in `ui/`, not on M1 to M3,
and it produces new packages without touching the ones the campaigns already ran.

## Why buildings and not edges

Every package today is authored as edges with counts, which is how `spawns.json` and the
`alerts.json` area lists are written. An operator does not think in SUMO edge IDs, and
the record describes which subdivisions were ordered out, not which road segments. The
building is the object both an operator and the source material name, so it is the right
handle for selection. The edge stays underneath, because SUMO inserts vehicles on edges,
and the mapping from one to the other is computed once at authoring time.

The centroid work in `configs/halifax_3town_e0_centroid/` already put `home_xy` and
`building_id` into `spawns.json`, so the household-to-building link exists. The authoring
UI produces the same format from the other direction.

## Status

| Stage | What it delivers | Status |
|---|---|---|
| 0 | Nearest drivable edge per building in the preview bundle | Done |
| 1 | A write path from the console into `configs/` | Done |
| 2 | Spawn authoring, selection box and per-building counts | Done |
| 3 | Alert areas selected as buildings | Done |
| 4 | Fire source placement and growth parameters | Done |
| 5 | Round trip, validation, and the bad-snap guard | Done |

Composition, validation, and the write path live in `ui/backend/authoring.py`, reachable
at `POST /api/packages/validate` and `POST /api/packages`. The interface is an Author tab
beside Setup, built from `ui/frontend/src/authoring/`, where `selection.ts` holds the
selection and draft rules as plain functions, `AuthorMap.tsx` draws and hit-tests, and
`AuthorView.tsx` carries the three tools and the validation panel.

Stage 3 resolved without touching `agentevac/`. An area's buildings are translated into
its edges when the package is written, so `alerts.json` keeps the edge format the alert
engine already reads and no research code changed.

Stage 0 and Stage 1 are prerequisites for everything below them. Stage 4 is independent
of Stages 2 and 3 and is the smallest, so it is the cheapest way to prove the write path
on something low risk.

## Stage 0. Nearest edge in the bundle

The building layer landed in `ui/tools/build_map_assets.py` and carries `id`, centroid
`lon` and `lat`, and a `poly` footprint, scoped to the incident bbox padded by 0.02
degrees. What it lacks is the edge each building would spawn onto.

The browser cannot compute that. Snapping needs the road network, which is 239 MB and
takes about fifteen seconds to read, so it cannot happen per interaction and it cannot
happen client side at all. The builder already holds the network, so the mapping is
computed there once and travels in the bundle. With it the authoring UI can emit a
complete `spawns.json` with no backend geometry call.

Two accuracy notes. `scripts/generate_spawns_from_buildings.py` snaps to the nearest edge
midpoint through a KD-tree, so a building near the end of a long edge can be assigned to
a different street whose midpoint happens to be closer. Stage 0 measures to the edge
polyline with the same `polygonOffsetAndDistanceToPoint` call the hazard model uses, so
authored packages and hazard distance agree. And `rtree` is not installed, so
`sumolib.net.getNeighboringEdges` falls back to a scan of every edge per query, which is
why Stage 0 builds its own uniform grid index.

Output per building is `edge` and `edge_dist_m`, both null when nothing drivable is
within range.

## Stage 1. The write path

The console writes only into `ui/assets/` and `outputs/ui_runs/`. That separation is
deliberate and it is the same boundary that left ad-hoc alert injection unbuilt. Authoring
requires writing into `configs/`, so the boundary moves and it should move on a stated
rule.

The rule to adopt is that the console may create a new package directory and may never
write into one that exists. A `POST /api/packages` endpoint validates a proposed package
and refuses any id already present under `configs/`. Nothing an operator does in the
console can alter a config a campaign has run against.

Validation returns a field, a message, and a fix per problem, which is the shape the
Setup view already consumes for run configs.

## Stage 2. Spawn authoring

Draw a box, take every building whose centroid falls inside, default one agent each, edit
the count on any building, click to add or remove one.

Emission produces compact groups keyed by the snapped edge, with `home_xy` and
`building_id` lists in count order, which is the format `expand_spawn_groups` reads
today. Buildings sharing an edge merge into one group and their centroids concatenate in
a stable order, so agent IDs stay predictable.

Frontend work is a Setup sub-view plus a buildings layer with box draw and hit testing.
Backend work is the emit and the validation pass.

## Stage 3. Alert areas as buildings

The same interaction over a second box, producing the buildings an order covers.

This is where the format decision lands. `alerts.json` areas are edge lists, and a
building selection produces buildings. The recommendation is to let an area carry either
form, so an existing edge list keeps working untouched and a building list becomes the
new optional shape, resolved to edges at load through the Stage 0 mapping. Areas then
follow the spawn set by construction, which is the adaptive behaviour the alert coverage
problem calls for.

That coverage problem is real and measured. A selection box over the three communities
maps to 327 edges while the current schedule names 45, so 1108 of 1355 households would
sit outside every ordered area and run silently as no-notice. Order compliance, area
clearance, and awareness share would all describe a different population than intended.

The change lands in `agentevac/agents/alert_schedule.py`, so it carries the same
requirement the centroid work did, meaning existing configs stay bit-identical.

## Stage 4. Fire authoring

Click to place a source, then set `t0`, `r0`, `growth_m_per_s`, and `max_r_m` beside it,
with a time scrubber drawing the footprint at a chosen instant.

Cheaper than Stages 2 and 3. Fire sources are points in simulation coordinates with no
snapping, `fires.json` is a flat list, and the scrubber reuses the growth model the
simulator already applies in `active_fires`.

## Stage 5. Round trip and validation

Author a package, build its preview bundle, launch it, and confirm the run matches what
the map showed.

One guard belongs here specifically. A household whose snapped edge is missing or not
drivable fails at insertion, and `main.py` catches the exception, prints a warning, and
drops the household before `metrics.record_departure` runs, so it appears as
never-departed with no recorded cause. Validation should reject an authored package that
would hit this, since at authoring scale it would otherwise go unnoticed.

## Constraints carried from earlier work

- Existing packages and existing results are never modified. Authored packages are new
  directories and form their own experiment batches.
- The research suite under `tests/` stays untouched. Console tests live in `ui/tests/`.
- The console adds no package to the environment the simulator runs in, so the backend
  stays standard library only.
