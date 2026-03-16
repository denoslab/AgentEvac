# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentEvac is an agent-based simulator for wildfire evacuations. It couples a SUMO traffic simulation with LLM-driven agents (GPT-4o-mini) that make real-time evacuation decisions under different information regimes. See [README.md](README.md) for project background, objectives, and quickstart.

## Project Layout

```
agentevac/
├── agents/      # Per-agent decision pipeline modules
├── analysis/    # Calibration, experiment sweep, metrics
├── utils/       # Fire forecast & record/replay
└── simulation/  # Main SUMO/TraCI loop (main.py) + spawn config
sumo/            # SUMO network/route/config files
tests/           # pytest test suite
docs/            # Project documentation
```

## Setup & Running

**Requirements:** SUMO must be installed and `SUMO_HOME` set. Install the package in development mode:

```bash
export SUMO_HOME=/path/to/sumo
pip install -e .

# Run simulation (interactive with SUMO GUI)
python -m agentevac.simulation.main --sumo-binary sumo-gui --scenario advice_guided

# Run headless
python -m agentevac.simulation.main --sumo-binary sumo --scenario no_notice --messaging on --metrics on

# Record LLM decisions to replay later
python -m agentevac.simulation.main --run-mode record --scenario alert_guided

# Replay a previous run deterministically (uses logged LLM responses)
python -m agentevac.simulation.main --run-mode replay --run-id 20260209_012156

# Parameter sweep study (calibration)
agentevac-study --reference metrics.json \
  --sigma-values "20,40,60" --delay-values "0,5" \
  --trust-values "0.3,0.5,0.7" --scenario-values "advice_guided"

# Run tests
python -m pytest tests/
```

**Key CLI flags for the simulation:** `--scenario` (no_notice|alert_guided|advice_guided), `--messaging` (on|off), `--events` (on|off), `--web-dashboard` (on|off), `--metrics` (on|off), `--overlays` (on|off).

**Key environment variables:** `OPENAI_MODEL` (default: `gpt-4o-mini`), `DECISION_PERIOD_S` (default: `5.0`), `NET_FILE` (default: `sumo/Repaired.net.xml`), `SUMO_CFG` (default: `sumo/Repaired.sumocfg`), `RUN_MODE`, `REPLAY_LOG_PATH`, `EVENTS_LOG_PATH`, `METRICS_LOG_PATH`.

## Architecture

`agentevac/simulation/main.py` is the main simulation loop (~3,400 lines). It manages the SUMO lifecycle and orchestrates the agent pipeline each tick. All other modules are domain libraries it imports.

**Agent decision pipeline (each `DECISION_PERIOD_S` seconds):**
1. `agentevac/agents/information_model.py` — sample edge margins (with Gaussian noise + delay), build social signals from inbox messages
2. `agentevac/agents/belief_model.py` — Bayesian update: categorize hazard → fuse env+social beliefs → compute entropy
3. `agentevac/agents/departure_model.py` — check if `p_danger > theta_r` or urgency decayed below `theta_u`
4. `agentevac/agents/routing_utility.py` — score each destination/route by exposure + travel cost, weighted by agent belief
5. `agentevac/agents/scenarios.py` — filter what information the agent sees based on information regime
6. **OpenAI API call** — GPT-4o-mini with Pydantic-validated structured output chooses destination/route
7. `agentevac/analysis/metrics.py` — log departure time, route entropy, hazard exposure, decision instability

**Information regimes** (`agentevac/agents/scenarios.py`):
- `no_notice` — agent sees only own observations and neighbor messages
- `alert_guided` — adds fire forecast (`agentevac/utils/forecast_layer.py`)
- `advice_guided` — adds forecast + route guidance + expected utility scores

**Agent state** (`agentevac/agents/agent_state.py`): Each agent carries a profile of psychological parameters (`theta_trust`, `theta_r`, `theta_u`, `gamma`, `lambda_e`, `lambda_t`) and runtime state (belief distribution, signal/decision histories). All agents stored in the global `AGENT_STATES` dict.

**Record/replay** (`agentevac/utils/replay.py`): All LLM prompts and responses logged to JSONL. Replay mode substitutes logged responses instead of calling the API, enabling deterministic re-runs.

**Calibration** (`agentevac/analysis/`): `study_runner.py` drives a parameter sweep by spawning simulation subprocesses across a grid of `(info_sigma, info_delay_s, theta_trust, scenario)` values, collects metrics JSON from each run, and fits against a reference dataset via weighted loss.

## Key Config in `agentevac/simulation/main.py`

At the top of the file (labeled `USER CONFIG`):
- `CONTROL_MODE` — `"destination"` (default) or `"route"`
- `NET_FILE` — path to SUMO route/network file (overridable via `NET_FILE` env var; default: `sumo/Repaired.net.xml`)
- `DESTINATION_LIBRARY` / `ROUTE_LIBRARY` — hardcoded choice menus for agents
- `OPENAI_MODEL` / `DECISION_PERIOD_S` — overridable via env vars

Vehicle spawns are defined in `agentevac/simulation/spawn_events.py` as a list of `(veh_id, spawn_edge, dest_edge, depart_time, lane, pos, speed, color)` tuples.
