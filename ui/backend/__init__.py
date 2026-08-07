"""Local service behind the AgentEvac operator console.

Serves the console, lists scenario packages and past runs, launches one
simulation process at a time through :mod:`ui.bridge.launcher`, and merges that
process's telemetry into a single stream for the browser.

Built on the standard library alone, so running the console adds no package to
the environment the simulator runs in.
"""
