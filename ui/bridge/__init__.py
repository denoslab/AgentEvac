"""In-process bridge that exposes a running simulation to the operator console.

The bridge lives in the simulation process and is installed by
``ui.bridge.launcher`` before the simulation script is imported. It adds an HTTP
endpoint for continuous state and control, and it never edits the simulator.
"""
