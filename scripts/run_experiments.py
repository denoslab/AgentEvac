#!/usr/bin/env python3
"""Driver for the E0, E1 and E4 experiment grid on the ``halifax_3town_e0`` config.

Runs every cell of the grid as a separate ``agentevac.simulation.main`` subprocess,
using the interpreter that runs this script (``sys.executable``), so launching it from
a PyCharm Python configuration with the project venv reuses that venv automatically.

PyCharm setup
    Script path        scripts/run_experiments.py
    Working directory   the repo root
    Interpreter         the project venv (has openai + traci)
    Environment         OPENAI_API_KEY=sk-...   (only for the llm agent)
                        SUMO_HOME defaults to /usr/share/sumo if unset

Examples
    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py --arms e0 --agents rule_based      # free half first
    python scripts/run_experiments.py --arms e1,e4 --agents llm --skip-existing

Fixed knobs, matching docs/build_plan and the calibration
    map=halifax_3town_e0, sim-end-time=28800, scenario=no_notice,
    FIRE_PERCEPTION_RANGE_M=1200. E1 sweeps ALERT_TIME_OFFSET_S, E4 sweeps
    DEFAULT_THETA_AUTH. The counterfactual arms run messaging off, where the alert
    channel is visible. E0 runs messaging on and off.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUMO_HOME = "/usr/share/sumo"
MAP_NAME = "halifax_3town_e0"

# --- grid axes ---
SEEDS_E0 = [47, 1024, 7, 13] # [47, 1024, 7, 13, 91, 128, 256, 512, 777, 2024]  # 10 seeds
SEEDS_CF = SEEDS_E0[:3]                                       # 3 seeds for E1/E4
ALL_AGENTS = ["llm", "rule_based"]
E1_OFFSETS = [-3600, -1800, -900, 900, 1800, 3600]           # capped at -3600 in code
E4_AUTH = [0.1, 0.3, 0.5, 0.7, 0.9]

FIXED_FLAGS = [
    "--map", MAP_NAME, "--sim-end-time", "28800",
    "--scenario", "no_notice", "--metrics", "on", "--events", "on",
]
PERCEPTION_RANGE = "1200"

MANIFEST_HDR = ["arm", "subdir", "agent", "seed", "messaging", "offset_s", "theta_auth",
                "outdir", "status", "elapsed_s", "departed", "arrived", "total", "usable"]

# stdout lines worth keeping in each cell's run.log; the rest is per-step spam
KEEP_PREFIXES = (
    "[ALERTS]", "[CLOCK]", "[SCENARIO]", "[MESSAGING]", "[AGENT_TYPE]",
    "[M2]", "[METRICS]", "[SUMO]", "[SEED", "[CLI_FLAGS]",
)
ERR_RE = re.compile(r"Traceback|Error|Exception|Failed|CRITICAL")


@dataclass
class Cell:
    arm: str            # e0 | e1 | e4
    subdir: str         # e.g. e0_msgoff, e1_off-3600, e4_auth0.1
    agent: str          # llm | rule_based
    seed: int
    messaging: str      # on | off
    offset_s: int       # ALERT_TIME_OFFSET_S
    theta_auth: float   # DEFAULT_THETA_AUTH

    @property
    def outdir(self) -> Path:
        # Grouped under an E0/E1/E4 family folder so outputs/ stays tidy.
        return REPO / "outputs" / self.arm.upper() / self.subdir / f"{self.agent}_seed{self.seed}"

    @property
    def name(self) -> str:
        return f"{self.arm.upper()}/{self.subdir}/{self.agent}_seed{self.seed}"

    @property
    def usable(self) -> str:
        # E0 messaging-on is the unbounded-broadcast flood, marked unusable in the folder name.
        return "FALSE" if "UNUSABLE" in self.subdir else "TRUE"


def build_cells(arms, agents) -> list[Cell]:
    cells: list[Cell] = []
    if "e0" in arms:
        # Messaging only affects LLM agents. rule_based decides heuristically and never
        # composes outbox text, so messaging on is a byte-identical no-op there, verified
        # empirically. So LLM runs both on and off, rule_based runs once (off).
        for agent in agents:
            msgs = ("on", "off") if agent == "llm" else ("off",)
            for msg in msgs:
                # msg-on floods the map via unbounded broadcast, so it is marked unusable.
                sub = "e0_msgon__UNUSABLE" if msg == "on" else "e0_msgoff"
                for seed in SEEDS_E0:
                    cells.append(Cell("e0", sub, agent, seed, msg, 0, 0.5))
    if "e1" in arms:
        for off in E1_OFFSETS:
            for agent in agents:
                for seed in SEEDS_CF:
                    cells.append(Cell("e1", f"e1_off{off:+d}", agent, seed, "off", off, 0.5))
    if "e4" in arms:
        for auth in E4_AUTH:
            for agent in agents:
                for seed in SEEDS_CF:
                    cells.append(Cell("e4", f"e4_auth{auth}", agent, seed, "off", 0, auth))
    return cells


def cell_cmd(cell: Cell, sumo_binary: str) -> list[str]:
    o = cell.outdir
    return [
        sys.executable, "-m", "agentevac.simulation.main",
        "--sumo-binary", sumo_binary, *FIXED_FLAGS,
        "--agent-type", cell.agent, "--messaging", cell.messaging, "--seed", str(cell.seed),
        "--metrics-log-path", str(o / "run_metrics.json"),
        "--events-log-path", str(o / "events.jsonl"),
        "--params-log-path", str(o / "run_params.json"),
        "--replay-log-path", str(o / "llm_routes.jsonl"),
    ]


def cell_env(cell: Cell) -> dict:
    env = os.environ.copy()
    env["SUMO_HOME"] = os.environ.get("SUMO_HOME", DEFAULT_SUMO_HOME)
    env["FIRE_PERCEPTION_RANGE_M"] = PERCEPTION_RANGE
    env["ALERT_TIME_OFFSET_S"] = str(cell.offset_s)
    env["DEFAULT_THETA_AUTH"] = str(cell.theta_auth)
    return env


def has_result(cell: Cell) -> bool:
    return bool(glob.glob(str(cell.outdir / "run_metrics_*.json")))


def read_result(cell: Cell):
    files = sorted(glob.glob(str(cell.outdir / "run_metrics_*.json")))
    if not files:
        return None
    try:
        d = json.load(open(files[-1]))
        return d.get("departed_agents"), d.get("arrived_agents"), d.get("total_agents")
    except Exception:
        return None


def merge_manifest(manifest: Path, rows) -> None:
    """Upsert this run's rows into the manifest keyed by outdir, keeping other cells' rows.

    The driver used to overwrite the manifest each invocation, dropping cells from other
    runs. Merging by outdir means an llm batch and a rule_based batch accumulate into one
    complete record, and a re-run updates its own row in place.
    """
    existing: dict = {}
    if manifest.exists():
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f):
                existing[r.get("outdir", "")] = r
    for c, status, dt, res in rows:
        dep, arr, tot = res if res else (None, None, None)
        key = str(c.outdir.relative_to(REPO))
        existing[key] = {
            "arm": c.arm, "subdir": c.subdir, "agent": c.agent, "seed": c.seed,
            "messaging": c.messaging, "offset_s": c.offset_s, "theta_auth": c.theta_auth,
            "outdir": key, "status": status, "elapsed_s": f"{dt:.0f}",
            "departed": dep, "arrived": arr, "total": tot, "usable": c.usable,
        }
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_HDR)
        w.writeheader()
        for key in sorted(existing):
            w.writerow({h: existing[key].get(h, "") for h in MANIFEST_HDR})


def run_cell(cell: Cell, sumo_binary: str):
    """Run one cell, streaming a filtered log. Returns (returncode, seconds, m2_line, tail)."""
    cell.outdir.mkdir(parents=True, exist_ok=True)
    cmd, env = cell_cmd(cell, sumo_binary), cell_env(cell)
    tail: deque[str] = deque(maxlen=60)
    m2 = None
    t0 = time.time()
    with open(cell.outdir / "run.log", "w") as log, subprocess.Popen(
        cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    ) as p:
        for line in p.stdout:
            tail.append(line.rstrip("\n"))
            if line.startswith(KEEP_PREFIXES) or ERR_RE.search(line):
                log.write(line)
                if line.startswith("[M2]"):
                    m2 = line.strip()
        rc = p.wait()
    return rc, time.time() - t0, m2, tail


def preflight(cells, dry_run) -> bool:
    ok = True
    map_dir = REPO / "configs" / MAP_NAME
    if not map_dir.is_dir():
        print(f"  MISSING map config {map_dir}")
        ok = False
    fires = map_dir / "fires.json"
    if fires.is_file():
        d = json.load(open(fires))
        caps = sorted({s.get("max_r_m") for s in d.get("sources", [])})
        print(f"  config {MAP_NAME}: {len(d.get('sources', []))} fire sources, max_r_m={caps}")
    if any(c.agent == "llm" for c in cells) and not os.environ.get("OPENAI_API_KEY"):
        print("  OPENAI_API_KEY not set, the llm cells will fail. Set it or use --agents rule_based.")
        ok = ok and dry_run
    if not os.environ.get("SUMO_HOME"):
        print(f"  SUMO_HOME not set, defaulting to {DEFAULT_SUMO_HOME}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the E0/E1/E4 experiment grid.")
    ap.add_argument("--arms", default="e0,e1,e4", help="Comma list of arms to run (default all).")
    ap.add_argument("--agents", default="llm,rule_based", help="Comma list, llm and/or rule_based.")
    ap.add_argument("--sumo-binary", default="sumo", help="sumo or sumo-gui (default sumo).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip cells that already have metrics.")
    ap.add_argument("--continue-on-error", action="store_true", help="Keep going past a failed cell.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit without running.")
    ap.add_argument("--limit", type=int, default=0, help="Run at most N cells (0 = all). Useful for a test.")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    cells = build_cells(arms, agents)
    if args.limit:
        cells = cells[:args.limit]
    if not cells:
        print("No cells selected.")
        return 1

    n_llm = sum(1 for c in cells if c.agent == "llm")
    print(f"Grid: {len(cells)} cells  (arms={arms} agents={agents})")
    for arm in ("e0", "e1", "e4"):
        k = sum(1 for c in cells if c.arm == arm)
        if k:
            print(f"  {arm}: {k} cells")
    print(f"  llm cells {n_llm} (budget API time), rule_based cells {len(cells) - n_llm} (free)")
    if not preflight(cells, args.dry_run):
        print("Preflight failed.")
        return 2

    if args.dry_run:
        print("\n-- dry run, sample cell --")
        c = cells[0]
        print("  env:", {k: cell_env(c)[k] for k in ("SUMO_HOME", "FIRE_PERCEPTION_RANGE_M",
                                                       "ALERT_TIME_OFFSET_S", "DEFAULT_THETA_AUTH")})
        print("  cmd:", " ".join(cell_cmd(c, args.sumo_binary)))
        print("\n-- all cells --")
        for c in cells:
            print(f"  {c.name:44} msg={c.messaging} off={c.offset_s:+d} auth={c.theta_auth}")
        return 0

    manifest = REPO / "outputs" / "experiments_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    for i, c in enumerate(cells, 1):
        if args.skip_existing and has_result(c):
            print(f"[{i}/{len(cells)}] skip (exists) {c.name}")
            res = read_result(c)
            rows.append((c, "skipped", 0.0, res))
            continue
        print(f"[{i}/{len(cells)}] run  {c.name}  msg={c.messaging} off={c.offset_s:+d} auth={c.theta_auth}")
        rc, dt, m2, tail = run_cell(c, args.sumo_binary)
        res = read_result(c)
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        extra = f"  {m2}" if m2 else ""
        depinfo = f"  departed/arrived={res[0]}/{res[1]}" if res else ""
        print(f"      {status}  {dt/60:.1f} min{depinfo}{extra}")
        rows.append((c, status, dt, res))
        if rc != 0:
            failed += 1
            print("      --- last output lines ---")
            for ln in list(tail)[-15:]:
                print("      | " + ln)
            if not args.continue_on_error:
                print("      stopping (use --continue-on-error to keep going)")
                break

    merge_manifest(manifest, rows)

    done = sum(1 for _, s, _, _ in rows if s in ("ok", "skipped"))
    print(f"\nDone. {done}/{len(cells)} cells accounted for, {failed} failed. Manifest: {manifest.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
