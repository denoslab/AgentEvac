#!/bin/sh
# Copy external scenario files (lytton.net.xml, polygons.xml) from a mounted
# volume into the expected location beside sumo/ if provided.
if [ -d /app/scenario_src ] && [ "$(ls -A /app/scenario_src 2>/dev/null)" ]; then
    cp -r /app/scenario_src/* /app/
fi

exec python -m agentevac.simulation.main "$@"
