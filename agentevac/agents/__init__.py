"""Agent decision pipeline modules.

Contains the per-agent state management and the sequential pipeline stages
that run every ``DECISION_PERIOD_S`` seconds:

    agent_state       — runtime state store and profile parameters.
    information_model — sample noisy/delayed environment and social signals.
    belief_model      — Bayesian hazard belief update.
    departure_model   — three-clause departure decision logic.
    routing_utility   — expected-utility scoring for destination/route menus.
    scenarios         — information-regime filtering (no_notice / alert / advice).
"""
