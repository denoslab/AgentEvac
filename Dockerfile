FROM python:3.11-slim

# Install SUMO (headless)
RUN apt-get update \
    && apt-get install -y --no-install-recommends sumo \
    && rm -rf /var/lib/apt/lists/*

ENV SUMO_HOME=/usr/share/sumo

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY agentevac/ ./agentevac/
COPY sumo/ ./sumo/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# External scenario files (lytton.net.xml, polygons.xml) are not committed
# and must be mounted at runtime — see docker-compose.yml.
# outputs/ is also expected to be a host-mounted volume.
RUN mkdir -p outputs

EXPOSE 8765

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--sumo-binary", "sumo", "--scenario", "advice_guided", "--metrics", "on"]
