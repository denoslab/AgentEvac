"""Guards on the boundary between the console and the simulator.

The console is a demonstration wrapper. It must not change the simulator's
source, its behaviour, or the artifacts of the research campaigns. These tests
fail loudly if a future change crosses that line.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

from ui.backend import session
from ui.bridge import launcher

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_PACKAGE = REPO_ROOT / "agentevac"


def test_the_simulator_never_imports_the_console():
    """Nothing under agentevac/ may depend on ui/, in either direction of use."""
    offenders = []
    for path in SIM_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "ui" or name.startswith("ui.") for name in names):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"the simulator imports the console at {offenders}"


def test_the_launcher_patches_only_the_two_traci_entry_points():
    """The bridge takes a foothold on the simulation thread and nothing more.

    Every extra assignment onto an imported module is a behaviour change the
    research runs would inherit, so the set is pinned here.
    """
    tree = ast.parse(inspect.getsource(launcher._install_patches))
    patched = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                patched.add(f"{target.value.id}.{target.attr}")
    assert patched == {"traci.start", "traci.simulationStep"}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Identify docstring constants so prose is not mistaken for a code path."""
    marked = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    marked.add(id(body[0].value))
    return marked


def test_no_console_module_names_a_file_inside_the_simulator_package():
    """The console reaches the simulator by import, never by filesystem path.

    A literal path under ``agentevac/`` in console code would be the first step
    toward reading or rewriting the simulator's own files, so none is allowed.
    Prose in docstrings is exempt, since it describes the boundary rather than
    crossing it.
    """
    offenders = []
    for path in (REPO_ROOT / "ui").rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or id(node) in docstrings:
                continue
            if isinstance(node.value, str) and "agentevac/" in node.value:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.value!r}")
    assert offenders == [], f"console code names a simulator file at {offenders}"


def test_the_console_opens_no_file_for_writing_outside_its_own_areas():
    """Writes are confined to the run folder and the generated map assets."""
    allowed_roots = ("outputs", "ui/assets", "dist")
    offenders = []
    for path in (REPO_ROOT / "ui").rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open"):
                continue
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = str(keyword.value.value)
            if not any(flag in mode for flag in ("w", "a", "x", "+")):
                continue
            target = node.args[0] if node.args else None
            if isinstance(target, ast.Constant) and not str(target.value).startswith(allowed_roots):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"a console module writes to a literal path at {offenders}"


class _StubSession(session.RunSession):
    """A session that can build a command line without spawning anything."""

    def __init__(self):
        super().__init__(session.Broadcaster())
        self.label = "test run"
        self.anchor_clock = "15:28:00"


@pytest.fixture
def command(tmp_path):
    stub = _StubSession()
    config = {
        "package": "halifax_3town_e0",
        "scenario": "advice_guided",
        "engine": "rule_based",
        "seed": 1024,
        "sim_end_time_s": 28800,
        "alert_minutes_earlier": 30,
        "messaging": True,
        "decision_period_s": 240.0,
        "initial_speed": 16,
    }
    return stub._build_command(config, tmp_path)


def test_every_output_path_is_redirected_out_of_the_shared_outputs_folder(command, tmp_path):
    """A console run must not be able to land beside a research campaign."""
    argv, env = command
    flags = {
        "--metrics-log-path",
        "--events-log-path",
        "--timeline-log-path",
        "--params-log-path",
    }
    seen = set()
    for flag in flags:
        assert flag in argv, f"{flag} is not passed, so that artifact would use the shared default"
        value = argv[argv.index(flag) + 1]
        assert str(tmp_path) in value, f"{flag} points outside the run folder: {value}"
        seen.add(flag)
    assert seen == flags
    # The decision log has no flag of its own, so it is redirected by environment.
    assert str(tmp_path) in env["REPLAY_LOG_PATH"]


def test_the_alert_offset_is_negative_for_orders_issued_earlier(command):
    _, env = command
    assert float(env["ALERT_TIME_OFFSET_S"]) == -1800.0


def test_a_zero_offset_reproduces_the_simulator_default(tmp_path):
    stub = _StubSession()
    _, env = stub._build_command(
        {
            "package": "halifax_3town_e0",
            "scenario": "advice_guided",
            "engine": "rule_based",
            "seed": 1,
            "sim_end_time_s": 100,
            "alert_minutes_earlier": 0,
            "messaging": False,
            "decision_period_s": 240.0,
            "initial_speed": 16,
        },
        tmp_path,
    )
    assert float(env["ALERT_TIME_OFFSET_S"]) == 0.0
    assert float(env["DECISION_PERIOD_S"]) == 240.0


def test_the_console_does_not_assume_its_own_interpreter_can_run_a_simulation():
    """The console is often started with a Python that has no TraCI.

    Picking ``sys.executable`` blindly makes a launch die deep inside the
    simulator with a traceback the operator cannot act on, so the interpreter is
    resolved and probed first.
    """
    source = inspect.getsource(session.RunSession._build_command)
    assert "sys.executable" not in source
    assert "simulator_python()" in source

    candidates = session._interpreter_candidates()
    assert candidates, "no interpreter candidate exists on this machine"
    # The repository's own environment is preferred over whatever is running the
    # console, and an explicit override wins over both.
    venv = str(REPO_ROOT / "venv" / "bin" / "python")
    if venv in candidates and sys.executable in candidates:
        assert candidates.index(venv) < candidates.index(sys.executable)


def test_an_explicit_interpreter_override_is_tried_first(monkeypatch):
    monkeypatch.setenv("AGENTEVAC_PYTHON", sys.executable)
    assert session._interpreter_candidates()[0] == sys.executable


def test_a_launch_is_refused_when_no_interpreter_can_run_it(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "simulator_python", lambda: None)
    stub = _StubSession()
    with pytest.raises(RuntimeError, match="no interpreter"):
        stub._build_command({
            "package": "halifax_3town_e0",
            "scenario": "advice_guided",
            "engine": "rule_based",
            "seed": 1,
            "sim_end_time_s": 100,
            "alert_minutes_earlier": 0,
            "messaging": False,
            "decision_period_s": 240.0,
            "initial_speed": 16,
        }, tmp_path)


def test_the_console_launches_the_simulator_through_the_launcher(command):
    argv, _ = command
    assert argv[1:3] == ["-m", "ui.bridge.launcher"]
    separator = argv.index("--")
    # Everything after the separator is the simulator's own command line, typed
    # exactly as it would be by hand.
    simulator_args = argv[separator + 1 :]
    assert "--map" in simulator_args and "--scenario" in simulator_args
    assert not any(arg.startswith("--ui-") for arg in simulator_args)


def test_run_folders_live_under_the_console_directory():
    assert session.history.UI_RUNS_DIR.name == "ui_runs"
    assert session.history.UI_RUNS_DIR.parent.name == "outputs"
