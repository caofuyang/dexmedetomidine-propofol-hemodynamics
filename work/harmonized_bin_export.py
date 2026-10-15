#!/usr/bin/env python3
"""v2.0: scan vital signs ONCE and export per-hour-bin detail.

Why this exists
---------------
v1.0 collapses every hourly bin into a handful of patient-level summaries, so
any alternative outcome definition (only-MAP hypotension, only-SBP
hypotension, k-adjacent-window grids, monitoring-density covariates) would
force a fresh scan of the multi-gigabyte raw tables (MIMIC chartevents ~3.5 GB,
eICU vitalPeriodic/vitalAperiodic).  This script performs that expensive scan a
single time and persists one row per (stay, relative hour bin) carrying the mean
and record count of each signal component.

It also writes the patient-level summary that must reproduce the frozen v1.0
outputs, and regression-checks that summary against the cached
``*_harmonized_hourly_outcomes_v3.csv`` files.  If the check fails the script
aborts, so a silent divergence cannot slip into v2.0.

Outputs (under outputs/active_comparator_sedation/):
  eicu_harmonized_bins_v4.csv / mimic_harmonized_bins_v4.csv
  eicu_harmonized_hourly_outcomes_v4.csv / mimic_harmonized_hourly_outcomes_v4.csv
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
DB = Path(os.environ.get("SEDATION_DB_ROOT", "/path/to/deidentified_databases")).expanduser().resolve()
EICU = DB / "eICU-CRD" / "data" / "eicu-collaborative-research-database-2.0"
MIMIC = DB / "MIMIC-IV" / "data" / "mimic-iv-3.1"
OUT = ROOT / "outputs" / "active_comparator_sedation"

# signal components carried into the per-bin table
BIN_COMPONENTS = ("hr", "inv_sbp", "nibp_sbp", "inv_map", "nibp_map")

# patient-level columns that must match the frozen v1.0 scan exactly
CHECK_COLS = (
    "hourly_repeated_hypotension_24h",
    "hourly_repeated_bradycardia_24h",
    "hourly_repeated_instability_24h",
    "hourly_any_instability_24h",
    "hourly_repeated_instability_nibp_priority_24h",
    "hourly_repeated_instability_any_source_24h",
    "hourly_repeated_instability_concordant_source_24h",
    "baseline_bp_hour_bins",
    "baseline_hr_hour_bins",
    "baseline_low_hour_bins",
    "baseline_brady_hour_bins",
    "post_bp_hour_bins",
    "post_hr_hour_bins",
    "post_low_hour_bins",
    "post_brady_hour_bins",
    "post_bp_both_source_hour_bins",
    "post_bp_source_discordant_hour_bins",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hourly = load_module("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")


def bin_records(stay, state):
    """One row per observed relative hour bin with component means and counts."""
    records = []
    for hour_bin in sorted(state):
        values = state[hour_bin]
        rec = {"id": int(stay), "hour_bin": int(hour_bin)}
        for comp in BIN_COMPONENTS:
            rec[f"{comp}_mean"] = hourly.mean_value(values, comp)
            rec[f"{comp}_n"] = values[f"{comp}_n"]
        records.append(rec)
    return records


def summarize(ids, states):
    rows, bin_rows = [], []
    for stay in ids:
        rows.append({"id": int(stay), **hourly.finalize_state(states[stay])})
        bin_rows.extend(bin_records(stay, states[stay]))
    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def build_eicu(cohort):
    ids = set(cohort["id"].astype("int64").astype(str))
    index_minutes = dict(zip(cohort["id"].astype("int64").astype(str), cohort["index_hour"] * 60))
    states = defaultdict(lambda: defaultdict(hourly.blank_bin))

    periodic_cols = ["patientunitstayid", "observationoffset", "heartrate", "systemicsystolic", "systemicmean"]
    reader = pd.read_csv(
        EICU / "vitalPeriodic.csv.gz", compression="gzip", usecols=periodic_cols,
        dtype={"patientunitstayid": "string"}, chunksize=1_000_000,
    )
    for chunk_no, chunk in enumerate(reader, 1):
        chunk = chunk[chunk.patientunitstayid.isin(ids)].copy()
        if chunk.empty:
            continue
        rel_min = pd.to_numeric(chunk.observationoffset, errors="coerce") - chunk.patientunitstayid.map(index_minutes)
        chunk = chunk.assign(rel_min=rel_min)
        chunk = chunk[(chunk.rel_min >= -360) & (chunk.rel_min <= 1440) & chunk.rel_min.ne(0)]
        for row in chunk.itertuples(index=False):
            hour_bin = hourly.relative_hour_bin(row.rel_min)
            if hour_bin is None:
                continue
            target = states[row.patientunitstayid][hour_bin]
            hourly.add_value(target, "hr", row.heartrate, 20, 250)
            hourly.add_value(target, "inv_sbp", row.systemicsystolic, 30, 300)
            hourly.add_value(target, "inv_map", row.systemicmean, 20, 200)
        if chunk_no % 25 == 0:
            print(f"[eICU] periodic chunks scanned: {chunk_no}", flush=True)

    aperiodic_cols = ["patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivemean"]
    reader = pd.read_csv(
        EICU / "vitalAperiodic.csv.gz", compression="gzip", usecols=aperiodic_cols,
        dtype={"patientunitstayid": "string"}, chunksize=1_000_000,
    )
    for chunk in reader:
        chunk = chunk[chunk.patientunitstayid.isin(ids)].copy()
        if chunk.empty:
            continue
        rel_min = pd.to_numeric(chunk.observationoffset, errors="coerce") - chunk.patientunitstayid.map(index_minutes)
        chunk = chunk.assign(rel_min=rel_min)
        chunk = chunk[(chunk.rel_min >= -360) & (chunk.rel_min <= 1440) & chunk.rel_min.ne(0)]
        for row in chunk.itertuples(index=False):
            hour_bin = hourly.relative_hour_bin(row.rel_min)
            if hour_bin is None:
                continue
            target = states[row.patientunitstayid][hour_bin]
            hourly.add_value(target, "nibp_sbp", row.noninvasivesystolic, 30, 300)
            hourly.add_value(target, "nibp_map", row.noninvasivemean, 20, 200)

    return summarize(ids, states)


def build_mimic(cohort):
    ids = set(cohort["id"].astype("int64").astype(str))
    index_seconds = dict(zip(
        cohort["id"].astype("int64").astype(str),
        pd.to_datetime(cohort["intime"]).astype("int64") / 1e9 + cohort["index_hour"] * 3600,
    ))
    states = defaultdict(lambda: defaultdict(hourly.blank_bin))
    wanted = {
        hourly.MIMIC_HR, hourly.MIMIC_INV_SBP, hourly.MIMIC_NIBP_SBP,
        hourly.MIMIC_INV_MAP, hourly.MIMIC_NIBP_MAP,
    }
    reader = pd.read_csv(
        MIMIC / "icu" / "chartevents.csv.gz", compression="gzip",
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        dtype={"stay_id": "string", "itemid": "string", "valuenum": "float64"},
        chunksize=2_000_000,
    )
    for chunk_no, chunk in enumerate(reader, 1):
        chunk = chunk[chunk.itemid.isin(wanted) & chunk.stay_id.isin(ids)].copy()
        if chunk.empty:
            continue
        chart_seconds = pd.to_datetime(chunk.charttime, errors="coerce").astype("int64") / 1e9
        rel_seconds = chart_seconds - chunk.stay_id.map(index_seconds)
        chunk = chunk.assign(rel_seconds=rel_seconds)
        chunk = chunk[
            (chunk.rel_seconds >= -21600) & (chunk.rel_seconds <= 86400)
            & chunk.rel_seconds.ne(0)
        ]
        for row in chunk.itertuples(index=False):
            hour_bin = hourly.relative_hour_bin(row.rel_seconds / 60)
            if hour_bin is None:
                continue
            target = states[row.stay_id][hour_bin]
            if row.itemid == hourly.MIMIC_HR:
                hourly.add_value(target, "hr", row.valuenum, 20, 250)
            elif row.itemid == hourly.MIMIC_INV_SBP:
                hourly.add_value(target, "inv_sbp", row.valuenum, 30, 300)
            elif row.itemid == hourly.MIMIC_NIBP_SBP:
                hourly.add_value(target, "nibp_sbp", row.valuenum, 30, 300)
            elif row.itemid == hourly.MIMIC_INV_MAP:
                hourly.add_value(target, "inv_map", row.valuenum, 20, 200)
            elif row.itemid == hourly.MIMIC_NIBP_MAP:
                hourly.add_value(target, "nibp_map", row.valuenum, 20, 200)
        if chunk_no % 10 == 0:
            print(f"[MIMIC] chartevents chunks scanned: {chunk_no}", flush=True)

    return summarize(ids, states)


def check_regression(old, new, database):
    """Recomputed v4 summary must equal the frozen v3 summary, row for row."""
    merged = old.merge(new, on="id", suffixes=("_old", "_new"))
    problems = []
    for col in CHECK_COLS:
        if f"{col}_old" not in merged or f"{col}_new" not in merged:
            problems.append((col, "missing"))
            continue
        a = merged[f"{col}_old"].astype(float)
        b = merged[f"{col}_new"].astype(float)
        mismatch = ((a - b).abs() > 1e-9) & ~(a.isna() & b.isna())
        if mismatch.any():
            problems.append((col, int(mismatch.sum())))
    if len(merged) != len(new):
        problems.append(("row_count", f"{len(merged)} vs {len(new)}"))
    if problems:
        raise SystemExit(f"[{database}] regression check FAILED: {problems}")
    print(f"[{database}] regression check PASSED on {len(merged)} stays", flush=True)


def main():
    base = hourly.load_base_module()
    jobs = (
        ("eICU-CRD", "eicu", build_eicu, OUT / "eicu_active_comparator_patient_level_v2.csv"),
        ("MIMIC-IV", "mimic", build_mimic, OUT / "mimic_active_comparator_patient_level_v2.csv"),
    )
    for database, prefix, builder, cohort_path in jobs:
        cohort = base.enforce_current_definitions(pd.read_csv(cohort_path))
        summary, bins = builder(cohort)
        old_path = OUT / f"{prefix}_harmonized_hourly_outcomes_v3.csv"
        if old_path.exists():
            check_regression(pd.read_csv(old_path), summary, database)
        else:
            print(f"[{database}] WARNING: frozen v3 summary not found at {old_path}; "
                  f"cannot regression-check the scan.", flush=True)
        bins.to_csv(OUT / f"{prefix}_harmonized_bins_v4.csv", index=False)
        summary.to_csv(OUT / f"{prefix}_harmonized_hourly_outcomes_v4.csv", index=False)
        print(f"[{database}] wrote {len(bins)} bin rows for {len(summary)} stays", flush=True)


if __name__ == "__main__":
    main()
