"""AgentEvac — agent-based wildfire evacuation simulator.

Couples a SUMO traffic simulation with LLM-driven agents (GPT-4o-mini) that make
real-time evacuation decisions under different information regimes.

Sub-packages:
    agents      — per-agent decision pipeline (belief, departure, routing, etc.)
    analysis    — calibration, experiment sweep, and metrics collection.
    utils       — shared utilities: record/replay and fire forecast.
    simulation  — main simulation loop and vehicle spawn configuration.
"""

__version__ = "0.1.0"
