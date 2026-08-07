# AgentEvac Operator Console

A browser console for configuring, running, and reviewing a wildfire evacuation
simulation. It implements [docs/UI_PLAN.md](../docs/UI_PLAN.md).

The console wraps the simulator. It does not modify it, extend it, or
reimplement any part of its behaviour.

---

## Running it

Three commands, the first two only once.

```bash
# 1. Build the map bundles the Setup view draws before a run exists.
#    About 15 seconds per package, since it reads the road network.
python -m ui.tools.build_map_assets

# 2. Build the interface.
cd ui/frontend && npm install && npm run build && cd ../..

# 3. Serve it.
export SUMO_HOME=/path/to/sumo
python -m ui.backend --port 8000
```

Then open `http://127.0.0.1:8000`.

The backend does not have to run on the interpreter that runs simulations. It
finds one that can, preferring `AGENTEVAC_PYTHON`, then `venv/bin/python` in this
repository, then whichever Python is running the console, and it probes each for
`traci`, `sumolib`, and `agentevac` before using it. The interpreter it settled
on is printed at startup and reported by `GET /api/health`. When none of them
work the Setup view says so and holds the launch button, instead of letting the
run die inside the simulator.

Pass `--host 0.0.0.0` to reach the console from a second machine, such as a
projector beside an operator laptop.

For frontend work, `npm run dev` serves the interface on port 5173 with instant
reload and proxies the API to the backend on port 8000.

### What it needs

Nothing beyond the repository. The backend is standard library only, so running
the console adds no package to the environment the simulator runs in. The
interface has its own `node_modules`, contained entirely within `ui/frontend/`.

Once built, the console runs with networking switched off. The basemap is drawn
from the simulator's own road network, the fonts and stylesheets are bundled, and
no layer fetches a tile.

---

## The three views

**Setup.** Choose a scenario package, an information regime, and who makes the
decisions. Shift the alert timing to run the counterfactual, meaning orders
issued up to sixty minutes earlier than the record. The map beside the form shows
the households, fire origins, shelters, and alert areas of the selected package,
with the incident schedule beneath it. Everything else stays in the JSON packages
where the research team maintains it.

**Operations.** The map fills the view. Households move continuously at any
speed, fire fronts grow, and alert areas fill in as their orders issue. The right
rail carries population accounting, per-community clearance, the evacuation curve,
and fire status. The bottom strip is the filterable event log, including the
messages households send each other. Clicking a household opens its belief and
decision history.

**Debrief.** Outcome tiles, the clearance table, the evacuation curve with an
optional overlay of a second run, and the secondary metrics in collapsed groups.
Every artifact the run produced is downloadable, individually or as one archive.

Keyboard: space pauses and resumes, `1` to `5` select speed presets, `L` jumps
the event log to the latest row.

---

## How it stays out of the simulator's way

The console never edits `agentevac/`. It runs the simulation script as it is,
through a launcher that attaches a bridge at runtime.

```
Browser ──SSE /api/stream──  UI backend  ──HTTP──  Simulation process
        ──REST commands───▶  (:8000)     ◀───────  ui.bridge.launcher
                                                   + agentevac.simulation.main
```

`agentevac/simulation/main.py` is a script whose simulation loop runs at import
time and drives SUMO through `traci.simulationStep()`. `ui.bridge.launcher`
replaces that one TraCI entry point with a wrapper before importing the script.
The wrapper runs on the simulation thread between steps, where it publishes
snapshots, paces wall time, and honours pause and end.

Three properties follow, and each is covered by a test.

1. **The trajectory is unchanged.** The wrapper only reads state and sleeps. It
   issues no TraCI command and changes no simulation variable.
   `ui/tests/test_run_parity.py` runs the same configuration twice, once plain
   and once through the console with pacing, a mid-run pause, and a speed change
   applied, then compares the metrics summary, the timeline, and every recorded
   decision key by key. They are identical.

2. **The results stay separated.** Every artifact a console run produces lands in
   `outputs/ui_runs/<run_id>/`, including the decision log, which has no
   command-line flag and is redirected by environment variable. A console run
   cannot land beside a research campaign.

3. **TraCI stays single-threaded.** HTTP handler threads read a cached snapshot
   and never call TraCI. Requests that need live agent state are parked and
   answered by the simulation thread on its next step.

Ending a run raises inside the patched step call, which hands control to the
simulator's own `finally` block. Metrics, the timeline, and the event log are
exported exactly as they are on a natural end.

---

## Layout

```
ui/
  bridge/                 runs inside the simulation process
    launcher.py           patches traci, starts the bridge, imports the simulator
    control.py            thread-safe state, control intent, wall-clock pacing
    collector.py          builds snapshots and previews from simulator state
    projection.py         simulation coordinates to longitude and latitude
    server.py             /status /snapshot /preview /events /agent /control
  backend/                the local service the browser talks to
    server.py             routing, the merged event stream, static files
    session.py            run lifecycle, subprocess management, orphan adoption
    packages.py           scenario packages and run validation
    history.py            past runs and replay recordings from outputs/
    sim_client.py         HTTP client for the bridge
  frontend/               React, TypeScript, Vite, Tailwind, MapLibre, uPlot
  tools/
    build_map_assets.py   offline map bundles for the Setup view
  tests/                  console tests, separate from the research suite
```

### Endpoints

| Method and path | Purpose |
|---|---|
| `GET /api/health` | SUMO presence, the resolved simulator interpreter, API key presence, an orphaned run if one exists |
| `GET /api/packages` | Scenario packages with households, fires, alert waves, horizon |
| `GET /api/packages/{id}/preview` | Map bundle for the Setup view |
| `GET /api/packages/{id}/schedule` | Ignitions and alert waves on the incident clock |
| `GET /api/recordings` | Language-model runs that replay can play back |
| `POST /api/runs/validate` | Check a configuration without launching it |
| `POST /api/runs` | Validate and launch, 409 while a run is active |
| `GET /api/runs/current` | Session state |
| `POST /api/runs/current/control` | pause, resume, toggle_pause, set_speed, end |
| `GET /api/runs/current/agents/{id}` | One household's belief and history |
| `POST /api/runs/adopt` | Take over a run that outlived a backend restart |
| `POST /api/runs/discard-orphan` | Stop that run instead, exporting what it has |
| `GET /api/stream` | Merged event stream, primed with current state on connect |
| `GET /api/history` | Finished runs found under `outputs/` |
| `GET /api/history/{run_id}/metrics` | Metrics, evacuation curve, artifact paths |
| `GET /api/history/{run_id}/export` | Every artifact of that run as one archive |
| `GET /api/files/{path}` | One artifact, restricted to paths under `outputs/` |

---

## Tests

```bash
python -m pytest ui/tests/                    # console backend and bridge
cd ui/frontend && npm test                    # formatting, map style, state store

# The parity guarantee. Starts SUMO twice, so it is opt-in.
AGENTEVAC_UI_SLOW_TESTS=1 python -m pytest ui/tests/test_run_parity.py -v
```

The research suite is untouched. `python -m pytest tests/` runs exactly what it
ran before this directory existed, and a bare `pytest` still collects only
`tests/`, because `pyproject.toml` pins `testpaths`.

---

## Choices that differ from the plan

**The backend uses the standard library, not FastAPI.** The plan chose FastAPI
for typed request models and async fan-out. Installing it would add packages to
the environment the simulator runs in, which the decoupling requirement rules
out, and the plan's own argument for a web app was that it installs nothing on
the demo machine beyond the repository. `ThreadingHTTPServer` is also what the
simulator already uses for its dashboard. Validation is explicit and returns a
field, a message, and a fix for every problem, which is what the Setup view needs
regardless of how it is produced.

**The basemap is drawn from the road network, not a PMTiles archive.** The plan
proposed a self-hosted tile bundle to avoid preprocessing the 240 MB network.
Selecting edges in simulation coordinates before projecting them turns out to
cost about fifteen seconds per package and yields a 2 MB bundle, so the console
draws the actual network the vehicles drive on. Local streets are drawn across
the incident area and arterials across the corridors out to the shelters. This
needs no tile server and no internet at any point. A PMTiles layer could still be
added underneath if place labels are wanted for the stage.

**Coordinates are converted without `pyproj`.** `sumolib` can invert the
network's UTM projection, but only when `pyproj` is installed, and it is not
present here. `ui/bridge/projection.py` implements the inverse transverse
Mercator series directly. It is checked against SUMO's own `convertGeo` on sample
points before the console draws anything, and the run refuses to show geography
if the two disagree by more than five metres. On the bundled networks they agree
to within a millimetre.

**Not built.** Ad-hoc alert injection during a run, which the plan left open
pending sign-off from the simulator's owner, since it would write into the alert
schedule inside research code at runtime. The alert timing field in Setup carries
the counterfactual instead.
