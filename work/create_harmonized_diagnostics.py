#!/usr/bin/env python3
"""Create diagnostics for the final harmonized-hourly analysis."""

from __future__ import annotations

import os

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUT = ROOT / "outputs" / "active_comparator_sedation"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_frame(base, harmonized, database, prefix):
    frame = base.enforce_current_definitions(
        pd.read_csv(OUT / f"{prefix}_active_comparator_patient_level_v2.csv")
    )
    if database == "eICU-CRD":
        frame = base.refresh_eicu_baseline_ventilation(frame)
    else:
        frame = base.refresh_mimic_baseline_ventilation(frame)
    hourly = pd.read_csv(OUT / f"{prefix}_harmonized_hourly_outcomes_v3.csv")
    return harmonized.prepare_harmonized_frame(
        frame.merge(hourly, on="id", how="left", validate="one_to_one")
    )


def main():
    base = load_module("base", ROOT / "work" / "active_comparator_sedation_study.py")
    harmonized = load_module("harmonized", ROOT / "work" / "harmonized_hourly_outcome.py")

    balance, ps_diagnostics, missingness, hospital_rows, baseline = [], [], [], [], []
    covariates = [
        "age", "female", "weight", "baseline_map", "baseline_hr", "index_hour",
        "vent", "baseline_bp_n", "baseline_hr_n", "baseline_midazolam",
        "baseline_opioid", "baseline_ketamine",
    ]

    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
        frame = load_frame(base, harmonized, database, prefix)
        evaluable = frame[frame.evaluable == 1].copy()

        if database == "eICU-CRD":
            treatment_count = evaluable.groupby("hospital_id")["dex"].nunique()
            both_hospitals = set(treatment_count[treatment_count == 2].index)
            for label, ids in (
                ("both_treatments", both_hospitals),
                ("single_treatment", set(treatment_count.index) - both_hospitals),
            ):
                subset = evaluable[evaluable.hospital_id.isin(ids)]
                hospital_rows.append({
                    "database": database,
                    "hospital_group": label,
                    "hospitals": len(ids),
                    "patients": len(subset),
                    "propofol": int((subset.dex == 0).sum()),
                    "dexmedetomidine": int((subset.dex == 1).sum()),
                })
            pre_complete_case = evaluable[evaluable.hospital_id.isin(both_hospitals)].copy()
        else:
            pre_complete_case = evaluable

        for dex, treatment in ((0, "propofol"), (1, "dexmedetomidine")):
            group = pre_complete_case[pre_complete_case.dex == dex]
            for variable in covariates:
                missingness.append({
                    "database": database,
                    "treatment": treatment,
                    "variable": variable,
                    "eligible_n": len(group),
                    "missing_n": int(group[variable].isna().sum()),
                    "missing_percent": group[variable].isna().mean() * 100,
                })

        work, cols = base.encode_analysis(frame, database)
        work.to_csv(
            OUT / f"{prefix}_harmonized_hourly_weighted_analysis_cohort.csv",
            index=False,
        )
        balance.extend(base.smd_rows(work, cols, database))
        ps_diagnostics.extend(base.diagnostic_rows(work, database))
        baseline.extend(base.baseline_table(work, database))
        if database == "MIMIC-IV":
            ps_diagnostics.append({
                "database": database,
                "treatment": "all",
                "diagnostic": "pre_index_RASS_observed_fraction",
                "n": len(work),
                "median": work.baseline_rass.notna().mean(),
            })

    pd.DataFrame(balance).to_csv(
        OUT / "table13_harmonized_balance.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(ps_diagnostics).to_csv(
        OUT / "table14_harmonized_ps_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(missingness).to_csv(
        OUT / "table15_harmonized_missingness.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(hospital_rows).to_csv(
        OUT / "table16_hospital_restriction.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(baseline).to_csv(
        OUT / "table17_harmonized_baseline.csv", index=False, encoding="utf-8-sig"
    )

    b = pd.DataFrame(balance)
    print("Maximum absolute SMD after weighting")
    print(b.assign(abs_smd=b.SMD_after.abs()).groupby("database").abs_smd.max())
    print("\nHospital restriction")
    print(pd.DataFrame(hospital_rows).to_string(index=False))
    print("\nMissing values")
    print(pd.DataFrame(missingness).query("missing_n > 0").to_string(index=False))


if __name__ == "__main__":
    main()
