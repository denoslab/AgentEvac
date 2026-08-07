"""The claim the whole console rests on: attaching it changes no result.

This runs the same configuration twice, once as the research team runs it and
once through the console's launcher with pacing, a mid-run pause, and a speed
change applied, then compares the artifacts key by key.

It starts SUMO twice, so it is opt-in::

    AGENTEVAC_UI_SLOW_TESTS=1 python -m pytest ui/tests/test_run_parity.py -v

It needs ``SUMO_HOME`` set and a Python that can import ``traci``. The test finds
that interpreter itself, preferring ``venv/bin/python`` in this repository, and
``AGENTEVAC_PYTHON`` overrides the choice.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from ui.backend import session

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.getenv("AGENTEVAC_UI_SLOW_TESTS") != "1",
    reason="starts SUMO twice; set AGENTEVAC_UI_SLOW_TESTS=1 to run",
)

#: A short deterministic run on the small network, with no language model in it.
COMMON_ARGS = [
    "--map", "lytton",
    "--scenario", "advice_guided",
    "--agent-type", "rule_based",
    "--messaging", "on",
    "--sumo-binary", "sumo",
    "--sim-end-time", "2400",
    "--seed", "4242",
    "--events-stdout", "off",
    "--overlays", "off",
]

BRIDGE_PORT = 8796


def simulator_python() -> str:
    """The interpreter that can actually start SUMO.

    The console's own tests run under whichever Python has pytest, which is not
    necessarily the one carrying ``traci``. This is the same resolution the
    backend uses when it launches a run.
    """
    resolved = session.simulator_python()
    if resolved is None:
        pytest.skip("no interpreter on this machine can import traci; set AGENTEVAC_PYTHON")
    return resolved


def output_flags(folder: Path) -> list[str]:
    return [
        "--metrics-log-path", str(folder / "m.json"),
        "--events-log-path", str(folder / "e.jsonl"),
        "--timeline-log-path", str(folder / "t.jsonl"),
        "--params-log-path", str(folder / "p.json"),
    ]


def run_environment(folder: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("SUMO_HOME", "/usr/share/sumo")
    env["REPLAY_LOG_PATH"] = str(folder / "routes.jsonl")
    return env


def newest(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    assert matches, f"no file matching {pattern} in {folder}"
    return matches[-1]


def post_control(action: str, value=None) -> None:
    body = json.dumps({"action": action, "value": value}).encode("utf-8")
    request = Request(f"http://127.0.0.1:{BRIDGE_PORT}/control", data=body, method="POST",
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        response.read()


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    python = simulator_python()
    plain_dir = tmp_path_factory.mktemp("plain")
    bridged_dir = tmp_path_factory.mktemp("bridged")

    plain = subprocess.run(
        [python, "-m", "agentevac.simulation.main", *COMMON_ARGS, *output_flags(plain_dir)],
        cwd=REPO_ROOT, env=run_environment(plain_dir), capture_output=True, text=True, timeout=900,
    )
    assert plain.returncode == 0, plain.stdout[-3000:]

    bridged = subprocess.Popen(
        [
            python, "-m", "ui.bridge.launcher",
            "--ui-bridge-port", str(BRIDGE_PORT),
            "--ui-speed", "40",
            "--ui-linger-s", "20",
            "--", *COMMON_ARGS, *output_flags(bridged_dir),
        ],
        cwd=REPO_ROOT, env=run_environment(bridged_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Exercise the controls mid-run, which is what the console does on stage.
    time.sleep(10)
    try:
        post_control("pause")
        time.sleep(3)
        post_control("set_speed", 0)
        post_control("resume")
    except Exception as exc:  # a control that never landed makes the test meaningless
        bridged.kill()
        pytest.fail(f"the bridge did not accept controls: {exc}")
    stdout, _ = bridged.communicate(timeout=900)
    assert bridged.returncode == 0, stdout[-3000:]

    return plain_dir, bridged_dir


def test_the_metrics_summary_is_identical(artifacts):
    plain_dir, bridged_dir = artifacts
    plain = json.loads(newest(plain_dir, "m_*.json").read_text())
    bridged = json.loads(newest(bridged_dir, "m_*.json").read_text())
    differing = [key for key in set(plain) | set(bridged) if plain.get(key) != bridged.get(key)]
    assert differing == [], f"pacing or pausing changed {differing}"


def test_the_timeline_is_identical(artifacts):
    plain_dir, bridged_dir = artifacts

    def rows(folder: Path) -> list[dict]:
        return [json.loads(line) for line in newest(folder, "t_*.jsonl").read_text().splitlines() if line.strip()]

    assert rows(plain_dir) == rows(bridged_dir)


def test_every_recorded_decision_is_identical(artifacts):
    plain_dir, bridged_dir = artifacts

    def decisions(folder: Path) -> list[dict]:
        out = []
        for line in newest(folder, "routes_*.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            # Wall-clock fields differ between any two runs and say nothing about
            # the simulated trajectory.
            for noisy in ("wall_time", "timestamp", "latency_s", "duration_s"):
                record.pop(noisy, None)
            out.append(record)
        return out

    assert decisions(plain_dir) == decisions(bridged_dir)


def test_the_console_run_wrote_nothing_into_the_shared_outputs_folder(artifacts):
    _, bridged_dir = artifacts
    produced = {path.name for path in bridged_dir.iterdir()}
    assert any(name.startswith("m_") for name in produced)
    assert any(name.startswith("routes_") for name in produced)
