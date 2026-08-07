# AgentEvac Operator Console, UI build plan

This plan answers `docs/optimized-ui-plan-prompt.md`. It specifies a demonstration user interface that lets practicing emergency operators configure, launch, and observe an AgentEvac wildfire evacuation run, then review the outcome. The plan is grounded in the code as it exists today. The simulator remains the single source of behavior. The UI wraps it through a small bridge and never reimplements simulation logic.

Repo facts the plan relies on. The simulation entry point `agentevac/simulation/main.py` already runs an embedded HTTP server with a Server-Sent Events (SSE) stream of decision, message, and movement events on port 8765. Scenario packages live under `configs/<map>/` as JSON files for the network, spawns, fires, destinations, routes, and the alert schedule. The flagship demo package `halifax_3town_e0` reconstructs the 28 May 2023 Upper Tantallon wildfire with 182 households in 45 spawn groups, 13 timed fire sources, 3 alert waves at 6300 s, 9660 s, and 15180 s after ignition, and a recommended 28800 s horizon anchored at 15:28 local time. Both SUMO networks carry a UTM projection, so the sim process can convert vehicle positions to longitude and latitude through the `sumolib` net object it already loads. The Halifax network file is 240 MB, which rules out shipping network geometry to the browser. Runs are launched entirely through CLI flags and environment variables, agents can run in `llm`, `rule_based`, or deterministic `replay` mode, and the metrics collector exports a rich KPI summary JSON at run end.

---

## 1. Overview and goals

We are building a browser-based operator console served from a local Python backend. An operator selects a scenario package, reviews it on a map, adjusts a small set of operationally meaningful parameters, launches the run, and watches it unfold on a live map of the real Halifax road network showing fire perimeters, evacuating households, alert waves, and shelter destinations, with population accounting and clearance measures alongside. Afterward the operator reviews outcome measures and can compare two runs, which directly supports the counterfactual story the research is built on, such as issuing the first alert 30 minutes earlier. The console has three views that mirror the operator workflow, Setup, Operations, and Debrief.

Success criteria for the demo, all checkable.

- A first-time operator completes configure and launch in under 3 minutes using a single settings page with at most 7 visible decisions.
- At any moment during a run, a viewer 3 meters from the screen can answer four questions within 2 seconds. Is it running or paused. What time is it in the incident. How many households remain unaccounted. Where is the fire.
- A full 8 hour scenario plays end to end in roughly 8 minutes at 60x speed with continuous motion on the map, and every stall is labeled as an AI decision round with visible progress.
- The stage demo runs with no internet and no OpenAI access, using replay of a recorded LLM run for authentic decisions with zero API latency, with `rule_based` as the interactive fallback for audience-driven what-if runs.
- Every reachable failure state, such as a missing API key, a crashed sim process, or a dropped stream, shows a labeled status and one recovery action. No blank panels, no frozen screens without explanation.

Out of scope for this MVP. Editing spawn, fire, or destination JSON beyond the whitelisted parameters. Multiple concurrent runs. Authentication and remote deployment. Mobile layouts. Replacing SUMO GUI for research work.

---

## 2. Recommended tech stack and tooling

Platform tradeoff. A desktop app through Electron or Tauri adds packaging, signing, and update machinery that a local demonstration never exercises, and it would duplicate an HTTP layer the simulator already has. A browser app served by a local Python backend installs nothing on the demo machine beyond the repo itself, lets a projector and an operator laptop show the same session simultaneously, and matches the team's Python-first skill set. The choice is a web app served locally.

Transport tradeoff. The telemetry direction is server to client, and commands are discrete request and response. WebSockets would give one bidirectional channel but require new server dependencies and a reconnect state machine. SSE is already the pattern inside `main.py`, the browser `EventSource` reconnects automatically, and every message is inspectable with curl. The choice is SSE for telemetry plus REST POST for commands.

Map tradeoff. Rendering the SUMO network in the browser would require preprocessing the 240 MB Halifax net into tiled geometry. A real basemap makes that unnecessary because the roads are already drawn, and it reads as far more credible to operators since they see their own place names. Fires and agents become small live GeoJSON overlays in geographic coordinates, converted inside the sim process where the net object already exists. For offline safety the basemap ships as a PMTiles archive, meaning a single-file tile bundle served by the backend over HTTP range requests. The choice is MapLibre GL JS with a self-hosted PMTiles basemap and live GeoJSON sources.

Charting tradeoff. The evacuation curve accumulates one point per snapshot over thousands of sim seconds. SVG chart libraries degrade there, and we need exactly one streaming line chart plus static tiles. uPlot renders to canvas, handles hundreds of thousands of points, and weighs about 50 KB. The choice is uPlot for time series and plain markup for stat tiles.

| Area | Choice | Rationale |
|---|---|---|
| Platform | Web app, local backend serves the built bundle | Nothing to install, projector plus laptop views, reuses the existing HTTP pattern |
| Backend service | FastAPI with uvicorn, Python 3.11 | Typed request models with the pydantic already in the repo, async SSE fan-out, the team writes Python |
| Frontend framework | React 18 with TypeScript, built by Vite | Component model fits a panel console, mainstream maintainability, instant dev reload |
| State management | Zustand | One small store per domain, subscription without boilerplate, telemetry writes stay out of React render churn |
| Styling | Tailwind CSS with design tokens as CSS custom properties | Fast consistent styling, tokens keep the operational look in one place |
| Map | MapLibre GL JS, PMTiles basemap, GeoJSON live layers | Real road network with zero net-file preprocessing, fully offline |
| Charts | uPlot | Canvas performance for streaming series at trivial size |
| Realtime transport | SSE from backend to browser, REST POST for commands | Matches existing sim SSE, native auto-reconnect, curl-debuggable |
| Sim to backend bridge | HTTP on the sim's embedded server, extended with snapshot and control endpoints | The server already exists in `main.py`, one mechanism for both directions |
| Testing | pytest for backend, Vitest plus React Testing Library, one Playwright happy path | Covers the launch flow and the stream contract where regressions hurt the demo |
| Lint and format | ruff for Python as already configured, ESLint plus Prettier for TypeScript | Keeps both halves consistent with zero debate |

New code lives outside the research package.

```
ui/
  backend/
    app.py          FastAPI app, routes, SSE fan-out to browsers
    session.py      run lifecycle state machine, subprocess management
    sim_client.py   HTTP client polling the sim bridge
    packages.py     scenario package listing, validation, run history
    assets.py       serves PMTiles basemap, preview GeoJSON, fonts
  frontend/
    src/
      app/          shell, view switching, top bar
      state/        Zustand stores, SSE client, types
      setup/        Setup view
      operations/   Operations view
      debrief/      Debrief view
      map/          MapLibre wrapper, layers, interpolation
      ui/           shared primitives, tiles, badges, modal, toast
scripts/build_ui_map_assets.py   offline preview-asset generator
```

---

## 3. High-level architecture

Three processes. The browser talks only to the UI backend. The backend owns session continuity across runs and spawns one simulation process per run, passing the exact CLI flags and environment variables the simulator already accepts. The sim process stays disposable, and everything the research team does today keeps working untouched.

```
+-------------------+        SSE /api/stream         +----------------------+
|  Browser (React)  | <----------------------------- |  UI backend          |
|  Setup/Ops/Debrief| -----------------------------> |  (FastAPI, :8000)    |
+-------------------+   REST: config, launch, control|  session state       |
                                                     |  machine, fan-out    |
                                                     +----------+-----------+
                                                                |
                                        spawn subprocess        | HTTP poll GET /snapshot (2 Hz)
                                        python -m agentevac...  | SSE client on /events
                                        --ui-bridge on          | POST /control  (pause, speed, end)
                                                                v
                                                     +----------------------+
                                                     |  Simulation process  |
                                                     |  main.py + TraCI     |
                                                     |  embedded server     |
                                                     |  (:free port)        |
                                                     +----------+-----------+
                                                                |
                                                     reads configs/<map>/*.json
                                                     writes outputs/*.json[l]
```

Data flow for one run. The Setup view submits a run configuration. The backend validates it, allocates a free bridge port, spawns the sim process, and reports the `preparing` phase while SUMO loads the network, which takes tens of seconds for Halifax and must be visible progress. Once the bridge answers, the backend polls `GET /snapshot` at 2 Hz for continuous state and subscribes to the sim's existing SSE `/events` for discrete events, then re-broadcasts both to every browser on one merged stream. Control actions go browser to backend to bridge as POSTs, each acknowledged. At run end the backend reads the metrics summary, timeline, and events files from `outputs/` and serves them to the Debrief view. Continuous state travels as full snapshots, so a dropped and reconnected stream self-heals with no gap bookkeeping.

Session state machine held by the backend and mirrored by every client.

```
idle -> preparing -> running <-> paused -> finishing -> complete
                 \-> failed        \-> ending -> finishing -> ended_by_operator
```

Run configuration submitted by the Setup view. The backend maps it onto existing flags such as `--map`, `--scenario`, `--agent-type`, `--seed`, `--sim-end-time`, `--run-mode`, `--run-id`, and env vars such as `ALERT_TIME_OFFSET_S`.

```json
{
  "package": "halifax_3town_e0",
  "scenario": "advice_guided",
  "agent_engine": "replay",
  "replay_run_id": "20260716_155238",
  "seed": 1024,
  "sim_end_time_s": 28800,
  "alert_time_offset_s": 0,
  "messaging": true,
  "initial_speed": 16,
  "label": "Tantallon reconstruction, historical alert timing"
}
```

Merged stream envelope from backend to browser, one SSE channel, discriminated by `type`.

```json
{ "type": "session",  "phase": "running", "run_id": "20260805_141210", "label": "..." }

{ "type": "snapshot", "sim_t_s": 1842.4, "anchor_clock": "15:58:42",
  "speed_target": 16, "speed_actual": 15.8,
  "round": { "in_progress": true, "index": 8, "completed": 23, "total": 41 },
  "counts": { "total": 182, "waiting": 118, "evacuating": 41, "arrived": 21, "fire_contact": 2 },
  "agents": [ { "id": "70646940_2", "lon": -63.912, "lat": 44.703,
                "status": "evacuating", "aware": true, "speed_mps": 12.2 } ],
  "fires":  [ { "id": "Juneberry_Lane", "lon": -63.921, "lat": 44.711, "r_m": 583 } ],
  "alerts": { "issued": ["EA-1"], "next_id": "EA-2", "next_at_s": 9660 } }

{ "type": "sim_event", "event": "llm_decision", "sim_t_s": 1680.0,
  "summary": "70646940_2 chose Black_Point", "veh_id": "70646940_2" }
```

---

## 4. Screen and layout design

One window, a persistent top bar, and three views the operator moves through in order. The view switcher enforces the workflow, so Operations activates only once a run is launched and Debrief activates once a run finishes or is ended. This is the progressive disclosure the brief asks for, each stage shows only its own controls.

Top bar, identical in every view. Left side holds the product mark from `static/agent-evac-logo.png` and the run label. Center holds the run state pill, the incident clock showing both sim seconds and the anchored local time such as 16:12, and the speed readout. Right side holds the run controls, pause or resume, speed selector, end run, plus the connection badge. State and the controls that change state sit together, which is the single most important layout rule for clarity under pressure.

Setup view. A two-column layout. The left column, about 420 px, is the settings form in strict priority order, scenario package cards, information regime cards with one plain-language sentence each, alert timing offset presented as orders issued N minutes earlier with a 0 to 60 slider, decision engine, seed, run label, then an Advanced accordion collapsed by default holding messaging, horizon, decision cadence, and replay recording picker. The launch button and a validation summary anchor the bottom of the column. The right column is a map preview of the selected package showing spawn household dots, fire origin markers with ignition times, destination markers with names, alert area outlines, and beneath the map a horizontal incident schedule listing ignitions and alert waves on the sim clock. The operator sees what they are about to run before they run it.

Operations view. The map owns the center at roughly 70 percent width because the map is the demo. The right rail, about 340 px, stacks the population accounting tiles, per-community clearance bars, the evacuation curve, and the fire status tile. The bottom strip, about 180 px and collapsible, is the event log with filter tabs. A decision round banner slides down under the top bar whenever the sim clock is stalled on LLM calls, showing processing decisions 23 of 41, which converts the single most confusing system state into visible work. Clicking an agent opens a right-side drawer with its status, belief, and decision history from the existing per-agent snapshot endpoint, on top of the rail.

Debrief view. A headline row of outcome tiles, cleared households, mean mobilization delay, households reached by fire, total travel time. Below it the evacuation curve for the full run with an optional overlay of a second run chosen from run history, then the area clearance table with order and clearance times per community, then collapsed groups for the remaining metrics, compliance by channel, awareness sources, destination shares, route entropy, decision instability, token usage. A metadata footer shows the exact configuration and seed with export buttons for the metrics, timeline, events, and parameter files. A prominent button starts a new run, returning to Setup with the previous settings prefilled.

---

## 5. Component inventory

| Component | Module | Purpose | Key inputs |
|---|---|---|---|
| TopBar | Shell | Frame holding identity, state, clock, controls | session phase, run label |
| RunStatePill | Shell | Single glanceable state, color plus word | session phase |
| IncidentClock | Shell | Sim seconds plus anchored local time, large numerals | sim_t_s, anchor_clock |
| SpeedSelector | Run control | Target speed 1x, 4x, 16x, 60x, max | speed_target, speed_actual |
| RunControls | Run control | Pause, resume, end run with confirm | session phase |
| ConnectionBadge | Shell | Stream health, reconnect countdown | SSE client state |
| ViewSwitcher | Shell | Setup, Operations, Debrief with gating | session phase |
| PackageCardList | Setup | Choose scenario package with facts, households, fires, waves, horizon | packages from backend |
| RegimeSelector | Setup | Information regime cards, no_notice, alert_guided, advice_guided | scenario value |
| AlertTimingField | Setup | Orders issued N minutes earlier slider | alert_time_offset_s |
| EngineSelector | Setup | llm, rule_based, replay with recording picker | agent_engine, recordings list |
| SeedField | Setup | Seed number with randomize | seed |
| AdvancedSettings | Setup | Collapsed accordion of secondary knobs | messaging, horizon, cadence |
| ValidationSummary | Setup | Blocking problems with fix hints | backend validation response |
| LaunchButton | Setup | Submit run, disabled with reason | validation state |
| SetupMapPreview | Setup | Spawns, fire origins, destinations, areas before launch | package preview GeoJSON |
| IncidentSchedule | Setup | Ignitions and alert waves on the sim clock | package preview data |
| LiveMap | Live map | Basemap plus fires, agents, destinations, areas | snapshots, preview GeoJSON |
| MapLegend | Live map | Status color and shape key, always visible | static tokens |
| LayerToggles | Live map | Show or hide overlays, zoom presets | layer visibility state |
| AgentDrawer | Live map | One agent's status, belief, history | agent snapshot endpoint |
| RoundBanner | Run status | LLM round in progress with counts | round progress |
| AccountingTiles | KPI panel | Total, waiting, evacuating, arrived, fire contact | counts |
| ClearanceBars | KPI panel | Per-community ordered and departed progress | area clearance live data |
| EvacuationCurve | KPI panel | Cumulative departures and arrivals over sim time | snapshot counts series |
| FireStatusTile | KPI panel | Active fires, newest ignition, next scheduled | fires, package schedule |
| EventLog | Event log | Filterable scrolling feed | merged sim events |
| EventFilterTabs | Event log | All, alerts, fire, movement, decisions, messages | filter state |
| CommunicationsFeed | Event log | Household message texts as a chat-style tab | message events |
| OutcomeTiles | Debrief | Headline results | metrics summary |
| ClearanceTable | Debrief | Per-area order time, clearance time, stragglers | metrics summary |
| ComparisonPicker | Debrief | Overlay a second run on the curve | run history |
| MetricGroups | Debrief | Collapsed secondary metric sections | metrics summary |
| ExportButtons | Debrief | Download run artifacts | backend file endpoints |
| ConfirmModal | Shared | Destructive action confirmation | title, consequence text |
| Toast | Shared | Command acknowledgements and notices | message, severity |
| EmptyState | Shared | Labeled empty and loading placeholders | context text, action |
| StatTile | Shared | Number plus label plus trend, tabular numerals | value, label, tone |
| StatusDot | Shared | Color plus shape encoded status | status value |

---

## 6. Module-by-module breakdown

### 6.1 Simulation bridge, inside the sim process

**Responsibility.** Expose continuous state and accept control from the backend, entirely behind a new `--ui-bridge on|off` flag defaulting to off. With the flag off the simulation behaves exactly as today, byte for byte, which preserves the reproducibility discipline the research campaigns depend on.

**Work items, each small and anchored in existing code.**

1. Add the `--ui-bridge` flag in `_parse_cli_args` and force the embedded dashboard server on when set.
2. Snapshot cache. TraCI is single-threaded, so HTTP handler threads must never call it. Inside the step loop near `main.py:5461`, every 5 steps, meaning one sim second, build a snapshot dict from `traci.vehicle.getPosition`, the existing `agent_live_status`, `active_fires`, and alert schedule state, convert coordinates with `net.convertXY2LonLat` using the net object loaded at `main.py:1866`, and store it in a lock-guarded module variable. Add `GET /snapshot` to the embedded handler serving the cached copy.
3. Pacing and pause. A small control object with a target multiplier and a paused flag, updated by a new `POST /control` route accepting pause, resume, set_speed, and end. The loop checks it each iteration, sleeps the remainder of `step_length / multiplier` for pacing, sleeps in 50 ms slices while paused, and breaks on end so the existing `finally` block flushes metrics and the timeline as it already does.
4. Round progress. Set a shared counter when a decision round dispatches its futures and increment it as results are consumed in the existing phase 2 loops near `main.py:2988` and `main.py:5156`, then emit a `decision_round_end` event. The snapshot reports the counter, so the UI can label the stall.
5. Status. `GET /status` returning phase, sim time, config echo, and run identifiers, available as soon as the server starts so the backend can detect readiness during the long SUMO network load.

Keep the new logic in a separate `agentevac/simulation/ui_bridge.py` with pure functions where possible, imported by `main.py`, so it is unit-testable without SUMO.

**Inputs and outputs.** Inputs are control POSTs. Outputs are the snapshot, status, the existing SSE events, and unchanged run artifacts in `outputs/`.

**Edge cases.** Snapshot requested before the first step returns phase `preparing` with zero agents. Pause during an LLM round takes effect at the next step boundary while in-flight futures resolve, and the clock is frozen either way. At max speed the one-sim-second cache refresh is a trivial cost. Port collisions are avoided because the backend passes a probed free port through the existing `--web-dashboard-port` flag.

**Dependencies.** None new. Standard library HTTP as today.

### 6.2 UI backend service

**Responsibility.** Serve the frontend bundle and map assets, list scenario packages and run history, validate and launch runs, own the session state machine, aggregate sim telemetry, and fan it out to browsers on one SSE stream.

**Endpoints.**

| Method and path | Purpose |
|---|---|
| GET /api/health | SUMO_HOME presence, API key presence, version |
| GET /api/packages | Scenario packages with household count, fires, alert waves, horizon |
| GET /api/packages/{id}/preview | Preview GeoJSON bundle and incident schedule |
| GET /api/recordings | Replay candidates paired with their run_params companions |
| POST /api/runs | Validate and launch, 409 while a run is active |
| GET /api/runs/current | Session state |
| POST /api/runs/current/control | pause, resume, set_speed, end |
| GET /api/stream | Merged SSE, session then latest snapshot replayed on connect |
| GET /api/history | Past runs from outputs with labels and headline numbers |
| GET /api/history/{run_id}/metrics | Metrics summary JSON |
| GET /api/history/{run_id}/export | Zip of metrics, timeline, events, params |

**States.** The session machine from section 3. `preparing` covers subprocess spawn plus SUMO network load and reports elapsed seconds so the Halifax load reads as progress. `finishing` covers metrics export after the loop ends. `failed` captures a nonzero exit or an unreachable bridge, retaining the last 50 stderr lines.

**Edge cases.** Backend restart while a sim process lives, on boot it scans for an orphan by probing the recorded bridge port and offers adopt or end. Sim crash mid-run, the `finally` block in `main.py` still writes partial metrics, so Debrief opens with a partial-results notice. Two browser tabs are safe because snapshots are idempotent and control actions are acknowledged per request. Launch with a missing OpenAI key fails validation before spawn when the engine is `llm`, with the fix hint to choose `rule_based` or `replay`.

**Dependencies.** FastAPI, uvicorn. Reads `configs/`, `outputs/`, spawns `python -m agentevac.simulation.main`.

### 6.3 Shell and session state, frontend

**Responsibility.** Boot, connect the SSE client, hold the Zustand stores, render the top bar and view gating, surface connection health and toasts.

**Inputs and outputs.** Consumes the merged stream and health endpoint. Produces control POSTs from the top bar.

**States.** Disconnected with auto-reconnect countdown, connected idle, connected with each session phase. On reconnect the store resets from the replayed session and snapshot, so no divergence accumulates.

**Edge cases.** Backend down at load shows a full-screen labeled state with a retry action. Clock display holds the last sim time with a stalled marker if snapshots stop while events continue.

### 6.4 Scenario setup

**Responsibility.** Let the operator assemble the run configuration from whitelisted parameters, see the package on the preview map, and launch with confidence. Whitelist for the MVP, package, scenario regime, alert timing offset, engine, replay recording, seed, messaging, horizon, label. Everything else stays in the JSON packages where the research team maintains it.

**Inputs and outputs.** Reads packages, previews, recordings, health. Produces the run configuration POST.

**States.** Pristine with defaults from the flagship package, editing, validating, blocked with reasons, launching, then hand-off to Operations at `preparing`.

**Edge cases.** Replay engine selected locks scenario, seed, and package to the recording's companion parameters and shows why. Alert offset is only meaningful when the package ships an alert schedule, so the field disables with a note for packages without one. A package whose files fail validation lists the failing file by name. If the API key is absent the llm option carries an inline warning before launch is attempted.

**Dependencies.** Backend package endpoints. Preview GeoJSON from the asset build script.

### 6.5 Run control and status

**Responsibility.** Start, pause, resume, change speed, end. Reflect true state within 200 ms of acknowledgement, and never lie about state while a command is in flight.

**Inputs and outputs.** Session phase and snapshot pacing fields in, control POSTs out.

**States.** Buttons render from session phase, with an in-flight spinner between click and acknowledgement. End run always passes through the ConfirmModal stating that the run will finish and export partial results.

**Edge cases.** Speed change during an LLM round applies at the next step, and the selector shows target next to actual, so the operator sees 60x target with 0x actual during a round instead of suspecting a hang. Repeated clicks are idempotent server-side. Keyboard space toggles pause when no input is focused.

### 6.6 Live map

**Responsibility.** The centerpiece. Basemap of the real region, fire perimeters as filled circles with a subtle ignition pulse, households as status-colored dots with client-side linear interpolation between snapshots so motion is continuous at any speed, destination markers with names, alert area outlines that fill amber when their wave issues, and the legend always visible.

**Inputs and outputs.** Snapshots and preview GeoJSON in, agent selection out to the drawer, zoom preset actions internal.

**States.** Empty before launch showing the package preview, live during a run, frozen at the final snapshot after completion with a completed watermark, and a stream-lost overlay chip when disconnected.

**Edge cases.** An ignition outside the viewport raises a toast with a zoom-to action. A clicked agent that has arrived keeps a drawer with its final record. Interpolation clamps when an agent teleports between network edges after a reroute, snapping when displacement exceeds a threshold. At 60x a household covers up to a kilometer between one-second snapshots, which interpolation renders as smooth motion.

**Dependencies.** MapLibre, PMTiles archive, snapshot stream, preview assets, the per-agent endpoint for the drawer.

### 6.7 KPI and status panels

**Responsibility.** The right rail. Population accounting tiles, total, waiting, evacuating, arrived, fire contact with fire contact styled as the alarm figure. Clearance bars per community showing ordered versus departed from the live area data. The evacuation curve accumulating departures and arrivals per snapshot. Fire status naming active sources and the next scheduled ignition. The round banner documented in 6.5 renders from the same data.

**Inputs and outputs.** Snapshots in, nothing out.

**States.** Zeroed with muted styling before launch, live during the run, final values after completion.

**Edge cases.** Fire contact above zero flips the tile to the alarm treatment exactly once per change with no flashing. The curve downsamples beyond 5000 points, which uPlot handles without visible cost. Community bars appear only for packages with an alert schedule.

### 6.8 Event log and communications

**Responsibility.** A time-ordered feed of the run's discrete events with tabs for all, alerts, fire, movement, AI decisions, and household messages. Each row carries the sim clock, a severity chip, and the existing event summary text. The messages tab shows the natural-language texts households exchange, which demonstrates the LLM agents in their own words.

**Inputs and outputs.** Merged sim events in, a click on a row with an agent focuses that agent on the map.

**States.** Empty with an explanation before the first event, streaming with auto-scroll, paused auto-scroll on hover with a jump-to-latest button, capped at 2000 rows in memory with the full record always on disk in the events JSONL.

**Edge cases.** Bursts at alert waves render batched per animation frame. `llm_error` events render with the warning treatment and the fallback the sim applied, which turns an ugly failure into visible resilience.

### 6.9 Debrief and comparison

**Responsibility.** Present the run outcome from the metrics summary, timeline, and history files, support one-run comparison overlays, and export artifacts.

**Inputs and outputs.** History and metrics endpoints in, export downloads out.

**States.** Loading, loaded, partial-results notice when the run ended early or failed, comparison active.

**Edge cases.** Comparing runs from different packages or horizons is allowed with a visible caveat line naming the differing settings, since mismatched comparisons are how a demo audience explores. Missing timeline file degrades to metrics-only. Token usage renders only for llm runs.

**Dependencies.** Metrics summary structure as exported today, including area_clearance, mobilization_delay, compliance, awareness_source_share, destination_choice_share, fire_contact, non_evacuated_reached_by_fire, and token_usage.

---

## 7. Design system essentials

**Typography.** Inter for UI text, self-hosted. JetBrains Mono for identifiers, coordinates, and seeds. Scale in px, 12 for dense table cells, 13 for secondary text, 14 body default, 16 panel titles, 20 view titles, 28 for the incident clock and headline tiles, with `font-variant-numeric: tabular-nums` on every figure so counters never jitter. Minimum 14 px for anything an operator must read during a run.

**Color.** Dark neutral chrome around a light basemap, the standard emergency operations center look, which also makes the map the brightest object on the projector. Chrome surfaces from a slate ramp, roughly #12161C background, #1A2028 panels, #232B35 raised elements, #E6EAF0 primary text, #9AA5B1 secondary text. Status semantics keep the red, amber, green convention the brief requires, tuned to colorblind-safe hues from the Okabe-Ito palette. Green #009E73 for nominal, running, arrived, cleared. Amber #E69F00 for caution, paused, alerted, degraded stream. Red #D55E00 reserved exclusively for hazard and loss, fire perimeters, fire contact, failed. Blue #0072B2 for in-progress movement, evacuating households, so motion never competes with hazard. Neutral #8D99A6 for waiting and idle. Every status pairs color with a second channel, shape or icon or text, dots are circles for waiting, triangles for evacuating, squares for arrived, a flame glyph for contact, so no meaning rides on hue alone.

**Spacing and layout.** 4 px base unit, 8 px grid for component internals, 16 px panel padding, 12 px gaps in the rail stack. Panels are flat with 1 px #2A333E borders and 6 px radii, no shadows or gradients, which keeps the restrained operational register.

**Interaction patterns.** Every command shows an in-flight state and lands a toast or visible state change within 200 ms of the acknowledgement. Destructive actions, end run only, require a typed-out confirm modal naming the consequence. Disabled controls always carry a reason in a tooltip and in the validation summary. Keyboard, space for pause and resume, digits 1 to 5 for speed presets, L to jump the event log to latest. Motion is limited to dot interpolation, the ignition pulse, and 150 ms panel transitions, honoring `prefers-reduced-motion` by disabling all three.

**Accessibility.** WCAG 2.1 AA contrast on all text and status pairings, verified for the amber-on-dark cases. Full keyboard reachability with visible focus rings. Live-region announcements for phase changes and alert waves. Hit targets at 32 px minimum. The legend and status shapes carry the colorblind redundancy described above.

**Projector legibility.** The top bar clock, state pill, and accounting tiles are sized for 3 meter reading on a 1080p projector, meaning 28 px numerals and 600-weight labels, checked during rehearsal from the back of the room.

---

## 8. Build phases

Fifteen working days, two workstreams that parallelize after phase 0, one engineer on the Python side, one on the frontend, either can be the same person sequentially at reduced scope. Each phase ends with a demoable exit criterion, so the demo exists from day 5 onward and only gains fidelity.

| Phase | Days | Deliverables | Exit criterion |
|---|---|---|---|
| 0, prove the pipe | 1 to 2 | Sim bridge, snapshot, control, pacing, round counter, `--ui-bridge` flag. Backend skeleton, launch, session machine, merged stream | A `rule_based` Halifax run launched by curl, paused by curl, snapshot JSON shows moving lon/lat positions |
| 1, live map | 3 to 5 | Vite app shell, SSE client, stores, top bar with state pill and clock. LiveMap with basemap, agent dots, fire circles, interpolation. Asset script for PMTiles and preview GeoJSON | The Tantallon fire and 182 households visibly evacuate on the projector from a run launched by curl |
| 2, operate the run | 6 to 8 | RunControls, SpeedSelector, RoundBanner, AccountingTiles, ClearanceBars, EvacuationCurve, FireStatusTile, EventLog with tabs | First full stakeholder demo, an 8 hour scenario driven entirely from the UI at 60x in about 8 minutes |
| 3, configure the run | 9 to 11 | Setup view, package cards, regime and timing and engine fields, advanced accordion, validation, preview map, incident schedule, launch flow | A non-developer configures and launches an earlier-alert counterfactual in under 3 minutes unassisted |
| 4, review the run | 12 to 13 | Debrief view, outcome tiles, clearance table, metric groups, comparison overlay, exports, run history, replay recording picker | Reconstruction versus 30-minutes-earlier comparison told entirely inside the UI, replay of a recorded LLM run plays with zero API calls |
| 5, harden and rehearse | 14 to 15 | Failure-state pass over every module, orphan adoption, offline check with networking disabled, projector legibility pass, demo script with a recorded fallback for every live segment | Full rehearsal offline, someone force-kills the sim process mid-demo and the console recovers with partial results in Debrief |

Dependency note. Phase 1 needs only the snapshot from phase 0. Phases 2 through 4 touch the backend only through endpoints defined in section 6.2, so the Python engineer spends phases 2 to 4 on history, exports, recordings, validation, and hardening. Record the demo LLM run during phase 4 and pin its run id in the demo script.

---

## 9. Risks and open questions

**Risks.**

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Live LLM latency and failures on stage. 120 rounds at seconds each of API latency dominate wall time at high speed | High | High | Stage demo defaults to replay of a recorded LLM run, authentic decisions with zero API latency. `rule_based` for audience-driven runs. Live llm reserved for one short segment framed by the round banner |
| Halifax SUMO load, a 240 MB net read twice at startup, tens of seconds of dead air | High | Medium | `preparing` phase with elapsed time and stage text. Pre-warm a run before the audience arrives. Measure in phase 0, if above 90 s, evaluate a cropped demo network as a fallback package |
| Sim clock stalls at LLM rounds read as hangs | High | Medium | RoundBanner with live progress counts, speed selector showing target next to actual |
| Venue internet absent or captive | Medium | High | Everything self-hosted, PMTiles basemap, bundled fonts and glyphs, phase 5 rehearsal runs with networking disabled |
| TraCI thread-safety violated by bridge handlers | Medium | High | Snapshot cache written only by the sim thread, handlers read the cache, enforced by keeping all TraCI calls out of `ui_bridge.py` request paths and covered by a unit test |
| Bridge changes perturb research runs | Low | High | All bridge behavior behind `--ui-bridge`, default off, plus one replay determinism test comparing artifacts with the flag off |
| Scope creep into config editing | Medium | Medium | The whitelist in 6.4 is the contract, JSON packages stay read-only in the UI |
| Two operators issue conflicting controls | Low | Low | Last-write-wins with every acknowledgement broadcast on the stream, both screens converge |

**Open questions, each with a recommendation so the build starts without waiting.**

1. Ad-hoc alert injection, an operator pressing issue evacuation order for a community mid-run. Highest-value interactive feature for this audience, and it writes into the alert schedule inside research code at runtime. Recommendation, decide by end of phase 3, build only if the sim owner signs off on a bridge-only injection path, otherwise the alert timing slider in Setup carries the counterfactual story.
2. Household communications tab visibility during the stage demo. The texts are compelling and imperfect. Recommendation, keep the tab, review the recorded demo run's messages during phase 4, and hide the tab by a config switch if any text undermines credibility.
3. Second scenario package for the demo, Lytton as a contrast or Tantallon-only depth. Recommendation, demo depth on `halifax_3town_e0` and its variants, keep Lytton listed to show the tool is general.
4. Demo hardware, projector resolution, and room lighting. Determines the final legibility pass. Needed before phase 5.
5. Basemap attribution placement for OpenStreetMap data, required text in the map corner. Confirm wording with the basemap extract chosen in phase 1.
6. Whether the recorded demo run comes from the existing E0 campaign outputs or a fresh recording after phase 3, which would include any bridge-era event additions. Recommendation, fresh recording in phase 4.
