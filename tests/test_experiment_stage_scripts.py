"""Basic validation for staged experiment shell scripts."""

from pathlib import Path
import subprocess


SCRIPT_NAMES = [
    "run_stage0_pilot.sh",
    "run_stage1_scenarios.sh",
    "run_stage2_uncertainty.sh",
    "run_stage3_trust_messaging.sh",
    "run_stage4_calibration.sh",
    "run_stage5_refine_calibration.sh",
]


def test_stage_scripts_exist():
    for name in SCRIPT_NAMES:
        assert Path("scripts", name).exists()


def test_stage_scripts_are_valid_bash():
    for name in SCRIPT_NAMES:
        path = Path("scripts", name)
        subprocess.run(["bash", "-n", str(path)], check=True)
