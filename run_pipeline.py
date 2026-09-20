#!/usr/bin/env python3
"""Run the frozen Version 1.0 analysis pipeline in its documented order."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "active_comparator_sedation_study.py",
    "refresh_mimic_index_weight.py",
    "harmonized_hourly_outcome.py",
    "create_harmonized_diagnostics.py",
    "additional_harmonized_sensitivity.py",
    "recompute_secondary_harmonized.py",
]


def main() -> None:
    db_root = os.environ.get("SEDATION_DB_ROOT")
    if not db_root:
        raise SystemExit("Set SEDATION_DB_ROOT to the parent directory containing eICU-CRD and MIMIC-IV.")
    env = os.environ.copy()
    env["SEDATION_PROJECT_ROOT"] = str(ROOT)
    (ROOT / "outputs" / "active_comparator_sedation").mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        print(f"Running {step}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "work" / step)], check=True, env=env)


if __name__ == "__main__":
    main()
