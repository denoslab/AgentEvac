"""Allow ``python -m agentevac.simulation`` as an alias for the main script."""

from agentevac.simulation.main import *  # noqa: F401,F403 — re-export script namespace
