# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentEvac is an agent-based simulator for wildfire evacuations. It couples a SUMO traffic simulation with LLM-driven agents (GPT-4o-mini) that make real-time evacuation decisions under different information regimes. See [README.md](README.md) for project background, objectives, and quickstart.

## Setup & Running

**Requirements:** SUMO must be installed and `SUMO_HOME` set. Python dependencies: `openai`, `pydantic`, `sumolib`, `traci`.

```bash
export SUMO_HOME=/path/to/sumo

# Run simulation (interactive with SUMO GUI)
python Traci_GPT2.py --sumo-binary sumo-gui --scenario advice_guided

# Run headless
python Traci_GPT2.py --sumo-binary sumo --scenario no_notice --messaging on --metrics on

# Record LLM decisions to replay later
python Traci_GPT2.py --run-mode record --scenario alert_guided

# Replay a previous run deterministically (uses logged LLM responses)
python Traci_GPT2.py --run-mode replay --run-id 20260209_012156

# Parameter sweep study (calibration)
python study_runner.py --reference metrics.json \
  --sigma-values "20,40,60" --delay-values "0,5" \
  --trust-values "0.3,0.5,0.7" --scenario-values "advice_guided"
```

**Key CLI flags for `Traci_GPT2.py`:** `--scenario` (no_notice|alert_guided|advice_guided), `--messaging` (on|off), `--events` (on|off), `--web-dashboard` (on|off), `--metrics` (on|off), `--overlays` (on|off).

**Key environment variables:** `OPENAI_MODEL` (default: `gpt-4o-mini`), `DECISION_PERIOD_S` (default: `5.0`), `RUN_MODE`, `REPLAY_LOG_PATH`, `EVENTS_LOG_PATH`, `METRICS_LOG_PATH`.

There is no test suite or linter configuration.

## Architecture

`Traci_GPT2.py` is the main simulation loop (~2,800 lines). It manages the SUMO lifecycle and orchestrates the agent pipeline each tick. All other modules are domain libraries it imports.

**Agent decision pipeline (each `DECISION_PERIOD_S` seconds):**
1. `information_model.py` — sample edge margins (with Gaussian noise + delay), build social signals from inbox messages
2. `belief_model.py` — Bayesian update: categorize hazard → fuse env+social beliefs → compute entropy
3. `departure_model.py` — check if `p_danger > theta_r` or urgency decayed below `theta_u`
4. `routing_utility.py` — score each destination/route by exposure + travel cost, weighted by agent belief
5. `scenarios.py` — filter what information the agent sees based on information regime
6. **OpenAI API call** — GPT-4o-mini with Pydantic-validated structured output chooses destination/route
7. `metrics.py` — log departure time, route entropy, hazard exposure, decision instability

**Information regimes** (`scenarios.py`):
- `no_notice` — agent sees only own observations and neighbor messages
- `alert_guided` — adds fire forecast (`forecast_layer.py`)
- `advice_guided` — adds forecast + route guidance + expected utility scores

**Agent state** (`agent_state.py`): Each agent carries a profile of psychological parameters (`theta_trust`, `theta_r`, `theta_u`, `gamma`, `lambda_e`, `lambda_t`) and runtime state (belief distribution, signal/decision histories). All agents stored in the global `AGENT_STATES` dict.

**Record/replay** (`replay.py`): All LLM prompts and responses logged to JSONL. Replay mode substitutes logged responses instead of calling the API, enabling deterministic re-runs.

**Calibration** (`calibration.py` + `experiments.py` + `study_runner.py`): `study_runner.py` drives a parameter sweep by spawning `Traci_GPT2.py` subprocesses across a grid of `(info_sigma, info_delay_s, theta_trust, scenario)` values, collects metrics JSON from each run, and fits against a reference dataset via weighted loss.

## Key Config in `Traci_GPT2.py`

At the top of the file (labeled `USER CONFIG`):
- `CONTROL_MODE` — `"destination"` (default) or `"route"`
- `NET_FILE` — path to SUMO `.net.xml` file
- `DESTINATION_LIBRARY` / `ROUTE_LIBRARY` — hardcoded choice menus for agents
- `OPENAI_MODEL` / `DECISION_PERIOD_S` — overridable via env vars

Vehicle spawns are defined in `Spawn_Events.py` as a list of `(veh_id, spawn_edge, dest_edge, depart_time, lane, pos, speed, color)` tuples.
