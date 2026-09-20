#!/usr/bin/env python3
"""Harmonize post-index vital-sign sampling across eICU-CRD and MIMIC-IV.

The original event-level outcome is sensitive to database-specific sampling
frequency.  This script maps both databases to 24 one-hour bins after the
index infusion, averages valid observations within each bin, prioritizes
invasive over non-invasive blood pressure when both are available, and defines
repeat abnormality as abnormal hourly summaries in two adjacent bins.
"""

from __future__ import annotations

import os

import importlib.util
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
DB = Path(os.environ.get("SEDATION_DB_ROOT", "/path/to/deidentified_databases")).expanduser().resolve()
EICU = DB / "eICU-CRD" / "data" / "eicu-collaborative-research-database-2.0"
MIMIC = DB / "MIMIC-IV" / "data" / "mimic-iv-3.1"
OUT = ROOT / "outputs" / "active_comparator_sedation"
BASE_SCRIPT = ROOT / "work" / "active_comparator_sedation_study.py"

MIMIC_HR = "220045"
MIMIC_INV_SBP = "220050"
MIMIC_NIBP_SBP = "220179"
MIMIC_INV_MAP = "220052"
MIMIC_NIBP_MAP = "220181"

FIELDS = (
    "hr_sum", "hr_n",
    "inv_sbp_sum", "inv_sbp_n", "inv_map_sum", "inv_map_n",
    "nibp_sbp_sum", "nibp_sbp_n", "nibp_map_sum", "nibp_map_n",
)


def load_base_module():
    spec = importlib.util.spec_from_file_location("active_comparator", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def blank_bin():
    return {field: 0.0 for field in FIELDS}


def add_value(bin_state, field, value, low, high):
    value = pd.to_numeric(value, errors="coerce")
    if pd.notna(value) and low <= float(value) <= high:
        bin_state[f"{field}_sum"] += float(value)
        bin_state[f"{field}_n"] += 1.0


def mean_value(bin_state, field):
    n = bin_state[f"{field}_n"]
    return bin_state[f"{field}_sum"] / n if n else np.nan


def has_adjacent(bins):
    ordered = sorted(set(bins))
    return any(second - first == 1 for first, second in zip(ordered, ordered[1:]))


def relative_hour_bin(relative_minutes):
    """Map -6 to 0 h and 0 to 24 h to fixed relative-hour bins."""
    if -360 <= relative_minutes < 0:
        return int(math.floor(relative_minutes / 60))
    if 0 < relative_minutes <= 1440:
        return min(int(math.floor((relative_minutes - 1e-9) / 60)), 23)
    return None


def finalize_state(state):
    baseline_bp_bins, baseline_hr_bins = set(), set()
    baseline_low_bins, baseline_brady_bins = set(), set()
    baseline_maps, baseline_hrs = [], []
    bp_bins, hr_bins, low_bins, brady_bins = set(), set(), set(), set()
    nibp_priority_low_bins, any_source_low_bins, concordant_low_bins = set(), set(), set()
    both_source_bins, discordant_source_bins = set(), set()
    for hour_bin, values in state.items():
        hr = mean_value(values, "hr")
        inv_sbp = mean_value(values, "inv_sbp")
        inv_map = mean_value(values, "inv_map")
        nibp_sbp = mean_value(values, "nibp_sbp")
        nibp_map = mean_value(values, "nibp_map")

        # Prefer invasive measurements for each BP component.  If an invasive
        # component is absent, retain the non-invasive component rather than
        # discarding an otherwise usable hourly bin.
        sbp = inv_sbp if np.isfinite(inv_sbp) else nibp_sbp
        map_value = inv_map if np.isfinite(inv_map) else nibp_map
        low = (np.isfinite(map_value) and map_value < 65) or (np.isfinite(sbp) and sbp < 90)
        if hour_bin < 0:
            if np.isfinite(sbp) or np.isfinite(map_value):
                baseline_bp_bins.add(hour_bin)
                if low:
                    baseline_low_bins.add(hour_bin)
            if np.isfinite(map_value):
                baseline_maps.append(map_value)
            if np.isfinite(hr):
                baseline_hr_bins.add(hour_bin)
                baseline_hrs.append(hr)
                if hr < 50:
                    baseline_brady_bins.add(hour_bin)
        else:
            if np.isfinite(sbp) or np.isfinite(map_value):
                bp_bins.add(hour_bin)
                if low:
                    low_bins.add(hour_bin)
            if np.isfinite(hr):
                hr_bins.add(hour_bin)
                if hr < 50:
                    brady_bins.add(hour_bin)

            invasive_observed = np.isfinite(inv_sbp) or np.isfinite(inv_map)
            noninvasive_observed = np.isfinite(nibp_sbp) or np.isfinite(nibp_map)
            invasive_low = (
                (np.isfinite(inv_map) and inv_map < 65)
                or (np.isfinite(inv_sbp) and inv_sbp < 90)
            )
            noninvasive_low = (
                (np.isfinite(nibp_map) and nibp_map < 65)
                or (np.isfinite(nibp_sbp) and nibp_sbp < 90)
            )
            if noninvasive_low or (not noninvasive_observed and invasive_low):
                nibp_priority_low_bins.add(hour_bin)
            if invasive_low or noninvasive_low:
                any_source_low_bins.add(hour_bin)
            if (
                (invasive_observed and noninvasive_observed and invasive_low and noninvasive_low)
                or (invasive_observed and not noninvasive_observed and invasive_low)
                or (noninvasive_observed and not invasive_observed and noninvasive_low)
            ):
                concordant_low_bins.add(hour_bin)
            if invasive_observed and noninvasive_observed:
                both_source_bins.add(hour_bin)
                if invasive_low != noninvasive_low:
                    discordant_source_bins.add(hour_bin)

    repeated_low = has_adjacent(low_bins)
    repeated_brady = has_adjacent(brady_bins)
    repeated_nibp_priority_low = has_adjacent(nibp_priority_low_bins)
    repeated_any_source_low = has_adjacent(any_source_low_bins)
    repeated_concordant_low = has_adjacent(concordant_low_bins)
    return {
        "baseline_bp_hour_bins": len(baseline_bp_bins),
        "baseline_hr_hour_bins": len(baseline_hr_bins),
        "baseline_low_hour_bins": len(baseline_low_bins),
        "baseline_brady_hour_bins": len(baseline_brady_bins),
        "hourly_baseline_map": np.mean(baseline_maps) if baseline_maps else np.nan,
        "hourly_baseline_hr": np.mean(baseline_hrs) if baseline_hrs else np.nan,
        "post_bp_hour_bins": len(bp_bins),
        "post_hr_hour_bins": len(hr_bins),
        "post_low_hour_bins": len(low_bins),
        "post_brady_hour_bins": len(brady_bins),
        "hourly_repeated_hypotension_24h": int(repeated_low),
        "hourly_repeated_bradycardia_24h": int(repeated_brady),
        "hourly_repeated_instability_24h": int(repeated_low or repeated_brady),
        "hourly_any_instability_24h": int(bool(low_bins or brady_bins)),
        "hourly_repeated_instability_nibp_priority_24h": int(
            repeated_nibp_priority_low or repeated_brady
        ),
        "hourly_repeated_instability_any_source_24h": int(
            repeated_any_source_low or repeated_brady
        ),
        "hourly_repeated_instability_concordant_source_24h": int(
            repeated_concordant_low or repeated_brady
        ),
        "post_bp_both_source_hour_bins": len(both_source_bins),
        "post_bp_source_discordant_hour_bins": len(discordant_source_bins),
    }


def build_eicu_hourly(cohort):
    ids = set(cohort["id"].astype("int64").astype(str))
    index_minutes = dict(zip(cohort["id"].astype("int64").astype(str), cohort["index_hour"] * 60))
    states = defaultdict(lambda: defaultdict(blank_bin))

    periodic_path = EICU / "vitalPeriodic.csv.gz"
    periodic_cols = ["patientunitstayid", "observationoffset", "heartrate", "systemicsystolic", "systemicmean"]
    reader = pd.read_csv(
        periodic_path, compression="gzip", usecols=periodic_cols,
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
            hour_bin = relative_hour_bin(row.rel_min)
            if hour_bin is None:
                continue
            target = states[row.patientunitstayid][hour_bin]
            add_value(target, "hr", row.heartrate, 20, 250)
            add_value(target, "inv_sbp", row.systemicsystolic, 30, 300)
            add_value(target, "inv_map", row.systemicmean, 20, 200)
        if chunk_no % 25 == 0:
            print(f"[eICU] periodic chunks scanned: {chunk_no}", flush=True)

    aperiodic_path = EICU / "vitalAperiodic.csv.gz"
    aperiodic_cols = ["patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivemean"]
    reader = pd.read_csv(
        aperiodic_path, compression="gzip", usecols=aperiodic_cols,
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
            hour_bin = relative_hour_bin(row.rel_min)
            if hour_bin is None:
                continue
            target = states[row.patientunitstayid][hour_bin]
            add_value(target, "nibp_sbp", row.noninvasivesystolic, 30, 300)
            add_value(target, "nibp_map", row.noninvasivemean, 20, 200)

    rows = []
    for stay in ids:
        rows.append({"id": int(stay), **finalize_state(states[stay])})
    return pd.DataFrame(rows)


def build_mimic_hourly(cohort):
    ids = set(cohort["id"].astype("int64").astype(str))
    index_seconds = dict(zip(
        cohort["id"].astype("int64").astype(str),
        pd.to_datetime(cohort["intime"]).astype("int64") / 1e9 + cohort["index_hour"] * 3600,
    ))
    states = defaultdict(lambda: defaultdict(blank_bin))
    wanted = {MIMIC_HR, MIMIC_INV_SBP, MIMIC_NIBP_SBP, MIMIC_INV_MAP, MIMIC_NIBP_MAP}
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
            hour_bin = relative_hour_bin(row.rel_seconds / 60)
            if hour_bin is None:
                continue
            target = states[row.stay_id][hour_bin]
            if row.itemid == MIMIC_HR:
                add_value(target, "hr", row.valuenum, 20, 250)
            elif row.itemid == MIMIC_INV_SBP:
                add_value(target, "inv_sbp", row.valuenum, 30, 300)
            elif row.itemid == MIMIC_NIBP_SBP:
                add_value(target, "nibp_sbp", row.valuenum, 30, 300)
            elif row.itemid == MIMIC_INV_MAP:
                add_value(target, "inv_map", row.valuenum, 20, 200)
            elif row.itemid == MIMIC_NIBP_MAP:
                add_value(target, "nibp_map", row.valuenum, 20, 200)
        if chunk_no % 10 == 0:
            print(f"[MIMIC] chartevents chunks scanned: {chunk_no}", flush=True)

    rows = []
    for stay in ids:
        rows.append({"id": int(stay), **finalize_state(states[stay])})
    return pd.DataFrame(rows)


def prepare_harmonized_frame(frame):
    frame = frame.copy()
    frame["stable_baseline"] = (
        frame.baseline_bp_hour_bins.ge(1)
        & frame.baseline_hr_hour_bins.ge(1)
        & frame.baseline_low_hour_bins.eq(0)
        & frame.baseline_brady_hour_bins.eq(0)
        & frame.baseline_vasopressor.eq(0)
    ).astype(int)
    frame["evaluable"] = (
        frame.stable_baseline.eq(1)
        & frame.post_bp_hour_bins.ge(2)
        & frame.post_hr_hour_bins.ge(2)
    ).astype(int)
    # Reuse the audited propensity-score implementation with harmonized
    # baseline summaries and observation-count covariates.
    frame["baseline_map"] = frame.hourly_baseline_map
    frame["baseline_hr"] = frame.hourly_baseline_hr
    frame["baseline_bp_n"] = frame.baseline_bp_hour_bins
    frame["baseline_hr_n"] = frame.baseline_hr_hour_bins
    return frame


def analyze_harmonized(
    base, frame, database, analysis_label="primary", n_boot=1000,
    restrict_eicu_hospitals=True, include_hospital_effects=True, outcomes=None,
    enriched_ps=False, extra_covariates=None,
):
    frame = prepare_harmonized_frame(frame)
    work, cols = base.encode_analysis(
        frame, database,
        restrict_eicu_hospitals=restrict_eicu_hospitals,
        include_hospital_effects=include_hospital_effects,
        enriched_ps=enriched_ps,
        extra_covariates=extra_covariates,
    )
    if outcomes is None:
        outcomes = [
            "hourly_repeated_instability_24h",
            "hourly_repeated_hypotension_24h",
            "hourly_repeated_bradycardia_24h",
            "hourly_any_instability_24h",
        ]
    intervals, draws = base.bootstrap_intervals(
        work, cols, outcomes,
        seed=20260919 if database == "eICU-CRD" else 20260920,
        n_boot=n_boot,
        cluster_col="hospital_id" if database == "eICU-CRD" else None,
    )
    effects = []
    for outcome in outcomes:
        groups, rd, _, rr, _ = base.weighted_risk(work, outcome)
        effects.append({
            "database": database,
            "analysis": analysis_label,
            "outcome": outcome,
            "analysis_n": len(work),
            "propofol_n": int((work.dex == 0).sum()),
            "dexmedetomidine_n": int((work.dex == 1).sum()),
            "propofol_weighted_percent": groups[0]["weighted_events_percent"],
            "dexmedetomidine_weighted_percent": groups[1]["weighted_events_percent"],
            "risk_difference_percent": rd * 100,
            "RD_CI95_low": intervals[outcome]["rd"][0] * 100,
            "RD_CI95_high": intervals[outcome]["rd"][1] * 100,
            "risk_ratio": rr,
            "RR_CI95_low": intervals[outcome]["rr"][0],
            "RR_CI95_high": intervals[outcome]["rr"][1],
        })
    return work, effects, draws


def main():
    base = load_base_module()
    eicu_path = OUT / "eicu_active_comparator_patient_level_v2.csv"
    mimic_path = OUT / "mimic_active_comparator_patient_level_v2.csv"
    eicu = base.enforce_current_definitions(pd.read_csv(eicu_path))
    mimic = base.enforce_current_definitions(pd.read_csv(mimic_path))
    eicu = base.refresh_eicu_baseline_ventilation(eicu)
    mimic = base.refresh_mimic_baseline_ventilation(mimic)

    eicu_hourly_path = OUT / "eicu_harmonized_hourly_outcomes_v3.csv"
    mimic_hourly_path = OUT / "mimic_harmonized_hourly_outcomes_v3.csv"
    if eicu_hourly_path.exists():
        eicu_hourly = pd.read_csv(eicu_hourly_path)
    else:
        eicu_hourly = build_eicu_hourly(eicu)
        eicu_hourly.to_csv(eicu_hourly_path, index=False)
    if mimic_hourly_path.exists():
        mimic_hourly = pd.read_csv(mimic_hourly_path)
    else:
        mimic_hourly = build_mimic_hourly(mimic)
        mimic_hourly.to_csv(mimic_hourly_path, index=False)

    eicu = eicu.merge(eicu_hourly, on="id", how="left", validate="one_to_one")
    mimic = mimic.merge(mimic_hourly, on="id", how="left", validate="one_to_one")

    effects, sensitivity, draws_by_db, diagnostics, flow = [], [], {}, [], []
    for database, frame in (("eICU-CRD", eicu), ("MIMIC-IV", mimic)):
        prepared = prepare_harmonized_frame(frame)
        for stage, mask in (
            ("initial_strategy", np.ones(len(prepared), dtype=bool)),
            ("harmonized_stable_baseline", prepared.stable_baseline.eq(1)),
            ("harmonized_stable_and_outcome_evaluable", prepared.evaluable.eq(1)),
        ):
            subset = prepared[mask]
            flow.append({
                "database": database,
                "stage": stage,
                "total": len(subset),
                "propofol": int((subset.dex == 0).sum()),
                "dexmedetomidine": int((subset.dex == 1).sum()),
            })
        work, rows, draws = analyze_harmonized(base, frame, database)
        flow.append({
            "database": database,
            "stage": "complete_case_weighted_analysis",
            "total": len(work),
            "propofol": int((work.dex == 0).sum()),
            "dexmedetomidine": int((work.dex == 1).sum()),
        })
        effects.extend(rows)
        draws_by_db[database] = draws
        _, source_rows, _ = analyze_harmonized(
            base, frame, database, analysis_label="alternative_bp_source_rule", n_boot=300,
            outcomes=[
                "hourly_repeated_instability_nibp_priority_24h",
                "hourly_repeated_instability_any_source_24h",
                "hourly_repeated_instability_concordant_source_24h",
            ],
        )
        sensitivity.extend(source_rows)
        for dex, treatment in ((0, "propofol"), (1, "dexmedetomidine")):
            group = work[work.dex == dex]
            diagnostics.append({
                "database": database,
                "treatment": treatment,
                "n": len(group),
                "bp_hour_bins_median": group.post_bp_hour_bins.median(),
                "bp_hour_bins_q1": group.post_bp_hour_bins.quantile(.25),
                "bp_hour_bins_q3": group.post_bp_hour_bins.quantile(.75),
                "hr_hour_bins_median": group.post_hr_hour_bins.median(),
                "hr_hour_bins_q1": group.post_hr_hour_bins.quantile(.25),
                "hr_hour_bins_q3": group.post_hr_hour_bins.quantile(.75),
                "bp_both_source_hour_bins_median": group.post_bp_both_source_hour_bins.median(),
                "bp_source_discordant_hour_bins_median": group.post_bp_source_discordant_hour_bins.median(),
                "fraction_with_any_both_source_hour": group.post_bp_both_source_hour_bins.gt(0).mean(),
                "fraction_with_any_source_discordance": group.post_bp_source_discordant_hour_bins.gt(0).mean(),
                "discordant_fraction_among_both_source_hours": (
                    group.post_bp_source_discordant_hour_bins.sum()
                    / group.post_bp_both_source_hour_bins.sum()
                    if group.post_bp_both_source_hour_bins.sum() else np.nan
                ),
                "remaining_icu_time_under_24h_fraction": (
                    (group.icu_los_days * 24 - group.index_hour) < 24
                ).mean(),
            })

        sensitivity_frames = {
            "exclude_24h_crossover": frame[frame.crossover_24h == 0].copy(),
            "at_least_20_observed_hour_bins": frame[
                (frame.post_bp_hour_bins >= 20) & (frame.post_hr_hour_bins >= 20)
            ].copy(),
            "at_least_24h_remaining_in_icu": frame[
                (frame.icu_los_days * 24 - frame.index_hour) >= 24
            ].copy(),
        }
        for label, sensitivity_frame in sensitivity_frames.items():
            _, sensitivity_rows, _ = analyze_harmonized(
                base, sensitivity_frame, database,
                analysis_label=label, n_boot=300,
            )
            sensitivity.extend(sensitivity_rows)
        _, sensitivity_rows, _ = analyze_harmonized(
            base, frame, database,
            analysis_label="enriched_propensity_score", n_boot=300,
            enriched_ps=True,
        )
        sensitivity.extend(sensitivity_rows)
        if database == "eICU-CRD":
            _, sensitivity_rows, _ = analyze_harmonized(
                base, frame, database,
                analysis_label="all_hospitals_no_hospital_FE", n_boot=300,
                restrict_eicu_hospitals=False, include_hospital_effects=False,
            )
            sensitivity.extend(sensitivity_rows)

    heterogeneity = base.heterogeneity_rows(effects, draws_by_db)
    pd.DataFrame(effects).to_csv(OUT / "table7_harmonized_hourly_effects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(heterogeneity).to_csv(OUT / "table8_harmonized_hourly_heterogeneity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(OUT / "table9_hourly_monitoring_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(OUT / "table10_harmonized_hourly_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow).to_csv(OUT / "table11_harmonized_flow.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(effects).to_string(index=False), flush=True)
    print(pd.DataFrame(heterogeneity).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
