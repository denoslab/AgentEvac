#!/bin/sh
# Copy scenario files from the mounted volume into the working directory so
# they match the hardcoded paths in Traci_GPT2.py (Repaired.sumocfg, etc.).
if [ -d /app/scenario_src ] && [ "$(ls -A /app/scenario_src 2>/dev/null)" ]; then
    cp /app/scenario_src/* /app/
fi

exec python Traci_GPT2.py "$@"
