FROM python:3.11-slim

# Install SUMO (headless)
RUN apt-get update \
    && apt-get install -y --no-install-recommends sumo \
    && rm -rf /var/lib/apt/lists/*

ENV SUMO_HOME=/usr/share/sumo

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Scenario files (Repaired.sumocfg, *.net.xml, *.rou.xml) are not committed
# to the repo and must be mounted at runtime — see docker-compose.yml.
# outputs/ is also expected to be a host-mounted volume.
RUN mkdir -p outputs

EXPOSE 8765

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--sumo-binary", "sumo", "--scenario", "advice_guided", "--metrics", "on"]
