# AgentEvac Documentation

AgentEvac is an agent-based wildfire evacuation simulator that couples a
[SUMO](https://eclipse.dev/sumo/) traffic simulation with LLM-driven agents
(GPT-4o-mini) that make real-time evacuation decisions under different
information regimes.

## Package layout

```
agentevac/
├── agents/      # Per-agent decision pipeline
├── analysis/    # Calibration, experiment sweep, metrics
├── utils/       # Fire forecast & record/replay utilities
└── simulation/  # Main SUMO/TraCI simulation loop
```

## Quick-start

```bash
# Install in development mode (from repo root)
pip install -e .

# Run simulation headless
python -m agentevac.simulation.main --sumo-binary sumo --scenario advice_guided

# Run interactive with SUMO GUI
python -m agentevac.simulation.main --sumo-binary sumo-gui --scenario no_notice

# Parameter sweep study
agentevac-study --reference metrics.json \
  --sigma-values "20,40,60" --trust-values "0.3,0.5,0.7"
```

## Modules

### `agentevac.agents`

| Module | Purpose |
|---|---|
| `agent_state` | Per-agent runtime state and psychological profile |
| `information_model` | Noisy/delayed edge margin sampling and social signal parsing |
| `belief_model` | Bayesian hazard belief update (env + social fusion) |
| `departure_model` | Three-clause departure decision rule |
| `routing_utility` | Expected-utility scoring for destination/route menus |
| `scenarios` | Information-regime filtering (no_notice / alert_guided / advice_guided) |

### `agentevac.analysis`

| Module | Purpose |
|---|---|
| `metrics` | Per-run KPI collection (departure time, entropy, exposure, travel time) |
| `calibration` | Weighted relative-loss fit against a reference scenario |
| `experiments` | Cartesian-product parameter sweep driver |
| `study_runner` | End-to-end orchestrator: sweep → fit → report |

### `agentevac.utils`

| Module | Purpose |
|---|---|
| `forecast_layer` | Fire growth model, edge risk scoring, route briefings |
| `replay` | Record/replay LLM route decisions for deterministic re-runs |

## SUMO configuration

All SUMO network and scenario files live under `sumo/`:

```
sumo/
├── Repaired.sumocfg   # SUMO simulation config
├── Repaired.rou.xml   # Route/vehicle definitions
└── Repaired.netecfg   # SUMO netedit config
```

The external Lytton road-network files (`lytton.net.xml`, `polygons.xml`) are
**not committed** and must be placed at `../Lytton/` relative to the repo root,
or their paths overridden via the `SUMO_CFG` and `NET_FILE` environment
variables.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SUMO_HOME` | *(required)* | Path to SUMO installation |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model ID |
| `DECISION_PERIOD_S` | `5.0` | Seconds between LLM decision rounds |
| `NET_FILE` | `sumo/Repaired.rou.xml` | Path to the SUMO route/network file |
| `SUMO_CFG` | `sumo/Repaired.sumocfg` | Path to the SUMO config file |
| `INFO_SIGMA` | `40.0` | Gaussian noise std-dev on margin observations (m) |
| `INFO_DELAY_S` | `0.0` | Information delay in seconds |
| `DEFAULT_THETA_TRUST` | `0.5` | Default social-signal trust weight |
