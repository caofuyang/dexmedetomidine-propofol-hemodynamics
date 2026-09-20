#!/usr/bin/env python3
"""Recompute secondary outcome-definition and time-zero sensitivities."""

from __future__ import annotations

import os

import importlib.util
import math
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


def clean_saved_cohort(frame):
    exact = {
        "age10", "weight10", "map10", "hr10", "index6", "bp_obs_log",
        "hr_obs_log", "ps", "overlap_weight",
    }
    prefixes = ("unit_type_", "admission_source_", "anchor_year_group_")
    hospital_dummy = [
        col for col in frame.columns
        if col.startswith("hospital_id_") and col != "hospital_id"
    ]
    generated = [
        col for col in frame.columns
        if col in exact or col.startswith(prefixes)
    ] + hospital_dummy
    return frame.drop(columns=sorted(set(generated)), errors="ignore")


def difference_row(effects, draws):
    e = effects["eICU-CRD"]
    m = effects["MIMIC-IV"]
    e_draw = np.asarray(draws["eICU-CRD"]["rd"])
    m_draw = np.asarray(draws["MIMIC-IV"]["rd"])
    size = min(len(e_draw), len(m_draw))
    difference_draws = e_draw[:size] - m_draw[:size]
    observed = e["risk_difference_percent"] - m["risk_difference_percent"]
    se = np.std(difference_draws, ddof=1) * 100
    p = math.erfc(abs(observed / se) / math.sqrt(2)) if se > 0 else np.nan
    ci = np.percentile(difference_draws * 100, [2.5, 97.5])
    return {
        "outcome": "hemodynamic_instability_24h",
        "eICU_RD_percent": e["risk_difference_percent"],
        "MIMIC_RD_percent": m["risk_difference_percent"],
        "RD_difference_percent": observed,
        "CI95_low": ci[0],
        "CI95_high": ci[1],
        "heterogeneity_p": p,
        "bootstrap_replicates": size,
    }


def main():
    base = load_module("active_comparator", ROOT / "work" / "active_comparator_sedation_study.py")
    hourly = load_module("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")

    event_rows = []
    event_effects = {}
    event_draws = {}
    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
        frame = clean_saved_cohort(pd.read_csv(
            OUT / f"{prefix}_harmonized_hourly_weighted_analysis_cohort.csv"
        ))
        _, rows, draws = hourly.analyze_harmonized(
            base, frame, database,
            analysis_label="unharmonized_event_level_on_harmonized_cohort",
            n_boot=1000,
            outcomes=["hemodynamic_instability_24h"],
        )
        event_rows.extend(rows)
        event_effects[database] = rows[0]
        event_draws[database] = draws["hemodynamic_instability_24h"]

    pd.DataFrame(event_rows).to_csv(
        OUT / "table19_unharmonized_event_level_same_cohort.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([difference_row(event_effects, event_draws)]).to_csv(
        OUT / "table20_unharmonized_event_level_heterogeneity.csv",
        index=False, encoding="utf-8-sig",
    )

    timezero_rows = []
    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
        current = base.enforce_current_definitions(pd.read_csv(
            OUT / f"{prefix}_active_comparator_patient_level_v2.csv"
        ))
        old = pd.read_csv(
            OUT / f"{prefix}_active_comparator_patient_level_v2.pre_opioid_vaso_fix.csv",
            usecols=["id", "baseline_vasopressor"],
        ).rename(columns={"baseline_vasopressor": "pre_fix_baseline_vasopressor"})
        hourly_outcome = pd.read_csv(OUT / f"{prefix}_harmonized_hourly_outcomes_v3.csv")
        frame = current.merge(old, on="id", how="left", validate="one_to_one")
        frame["baseline_vasopressor"] = frame.pre_fix_baseline_vasopressor.fillna(
            frame.baseline_vasopressor
        )
        frame = frame.drop(columns="pre_fix_baseline_vasopressor")
        frame = frame.merge(hourly_outcome, on="id", how="left", validate="one_to_one")
        frame = (
            base.refresh_eicu_baseline_ventilation(frame)
            if database == "eICU-CRD"
            else base.refresh_mimic_baseline_ventilation(frame)
        )
        _, rows, _ = hourly.analyze_harmonized(
            base, frame, database,
            analysis_label="timezero_vasopressor_not_excluded",
            n_boot=1000,
        )
        timezero_rows.extend(rows)

    pd.DataFrame(timezero_rows).to_csv(
        OUT / "table12_timezero_vasopressor_sensitivity.csv",
        index=False, encoding="utf-8-sig",
    )
    print(pd.DataFrame(event_rows).to_string(index=False))
    print(pd.DataFrame([difference_row(event_effects, event_draws)]).to_string(index=False))
    print(pd.DataFrame(timezero_rows).query(
        "outcome == 'hourly_repeated_instability_24h'"
    ).to_string(index=False))


if __name__ == "__main__":
    main()
