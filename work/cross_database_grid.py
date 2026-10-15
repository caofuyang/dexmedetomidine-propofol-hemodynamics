#!/usr/bin/env python3
"""v2.0: agreement between the two databases across the whole definition grid.

The main pipeline reports how the *within-database* risk difference moves as the
outcome-construction rule changes, but reports a between-database difference only
for the pre-specified rule and for the single-hour rule.  This script closes that
gap: it recomputes the between-database difference for every rule in the grid,
using the same seeds and the same resampling as the rest of the pipeline.

Efficiency note: the propensity model is refitted once per bootstrap replicate and
every definition is scored inside that replicate.  Since the random draws depend
only on the resampling, this yields exactly the draws that
``base.bootstrap_intervals`` would produce if it were called once per definition,
at a fraction of the cost (one fit per replicate rather than one per definition).

Two gates: the observed-data estimate and the bootstrap interval for the
pre-specified rule must both reproduce the frozen v1.0 ``table7`` entry.

Output: ``outputs/active_comparator_sedation/table27_cross_database_grid.csv``
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUT = ROOT / "outputs" / "active_comparator_sedation"
N_BOOT = int(os.environ.get("SEDATION_GRID_BOOT", "1000"))
SENSITIVITY_BOOT = 300
CLIP = 1e-6
PRIMARY = "def_k2_adjacent_primary"

# outcome column -> (analysis label, human-readable rule)
DEFINITIONS = [
    ("v2_ip_composite_k1", "def_k1_any_hour", "k = 1 adjacent abnormal hour"),
    ("v2_ip_composite_k2", PRIMARY, "k = 2 adjacent abnormal hours (pre-specified)"),
    ("v2_ip_composite_k3", "def_k3_adjacent", "k = 3 consecutive abnormal hours"),
    ("v2_ip_map_k2", "def_hypot_map_only_k2", "Hypotension by mean arterial pressure <65 mmHg only"),
    ("v2_ip_sbp_k2", "def_hypot_sbp_only_k2", "Hypotension by systolic pressure <90 mmHg only"),
    ("v2_np_composite_k2", "def_source_nibp_priority_k2", "Non-invasive priority source rule"),
    ("v2_any_composite_k2", "def_source_any_k2", "Any available source rule"),
    ("v2_conc_composite_k2", "def_source_concordant_k2", "Concordant source rule"),
]
COLUMNS = [column for column, _, _ in DEFINITIONS]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hourly = load_module("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")
variants = load_module("variants", ROOT / "work" / "harmonized_definition_variants.py")


def build_work(base, database, prefix):
    cohort = base.enforce_current_definitions(pd.read_csv(OUT / f"{prefix}_active_comparator_patient_level_v2.csv"))
    cohort = (
        base.refresh_eicu_baseline_ventilation(cohort)
        if database == "eICU-CRD"
        else base.refresh_mimic_baseline_ventilation(cohort)
    )
    summary = pd.read_csv(OUT / f"{prefix}_harmonized_hourly_outcomes_v4.csv")
    bins = pd.read_csv(OUT / f"{prefix}_harmonized_bins_v4.csv")
    ids = cohort["id"]
    outcomes = variants.build_outcome_table(bins).set_index("id").reindex(ids).fillna(0).astype(int)
    outcomes = outcomes.rename_axis("id").reset_index()
    frame = (
        cohort.merge(summary, on="id", how="left", validate="one_to_one")
        .merge(outcomes, on="id", how="left", validate="one_to_one")
    )
    return base.encode_analysis(hourly.prepare_harmonized_frame(frame), database)


def bootstrap_draws(base, work, cols, database, n_boot):
    """Reproduce the pipeline's own bootstrap, scoring every definition per replicate."""
    rng = np.random.default_rng(20260919 if database == "eICU-CRD" else 20260920)
    cluster_col = "hospital_id" if database == "eICU-CRD" else None
    clusters = work[cluster_col].fillna("Unknown").astype(str).unique() if cluster_col else None
    roster = work[cluster_col].fillna("Unknown").astype(str) if cluster_col else None
    draws = {column: [] for column in COLUMNS}

    for replicate in range(n_boot):
        if cluster_col:
            selected = rng.choice(clusters, size=len(clusters), replace=True)
            sample = pd.concat([work[roster == cluster] for cluster in selected], ignore_index=True)
        else:
            sample = work.iloc[rng.integers(0, len(work), len(work))].copy()
        if len(sample) == 0:
            continue
        means = sample[cols].mean()
        sds = sample[cols].std().replace(0, 1)
        X = ((sample[cols] - means) / sds).to_numpy(float)
        _, ps = base.logistic_fit(X, sample.dex.to_numpy(float))
        ps = np.clip(ps, CLIP, 1 - CLIP)
        sample["overlap_weight"] = np.where(sample.dex == 1, 1 - ps, ps)
        for column in COLUMNS:
            draws[column].append(base.weighted_risk(sample, column)[1])
        if (replicate + 1) % 200 == 0:
            print(f"[{database}] bootstrap {replicate + 1}/{n_boot}", flush=True)
    return draws


def main():
    base = hourly.load_base_module()
    frozen = pd.read_csv(OUT / "table7_harmonized_hourly_effects.csv")
    effects, draws = {}, {}

    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
        work, cols = build_work(base, database, prefix)
        print(f"[{database}] analysis n={len(work)}", flush=True)

        row = frozen.query("database == @database and outcome == 'hourly_repeated_instability_24h'").iloc[0]
        observed = base.weighted_risk(work, "v2_ip_composite_k2")[1] * 100
        if abs(observed - row.risk_difference_percent) > 1e-9:
            raise SystemExit(f"[{database}] pre-specified RD {observed!r} != frozen table7 "
                             f"{row.risk_difference_percent!r} -- abort.")
        print(f"[{database}] pre-specified rule reproduces the frozen point estimate", flush=True)

        effects[database] = {column: base.weighted_risk(work, column)[1] for column in COLUMNS}
        draws[database] = bootstrap_draws(base, work, cols, database, N_BOOT)

        interval = np.percentile(draws[database]["v2_ip_composite_k2"], [2.5, 97.5]) * 100
        if abs(interval[0] - row.RD_CI95_low) > 1e-6 or abs(interval[1] - row.RD_CI95_high) > 1e-6:
            raise SystemExit(f"[{database}] pre-specified bootstrap interval "
                             f"({interval[0]!r}, {interval[1]!r}) != frozen "
                             f"({row.RD_CI95_low!r}, {row.RD_CI95_high!r}) -- abort.")
        print(f"[{database}] pre-specified rule reproduces the frozen bootstrap interval", flush=True)

    rows = []
    for column, label, text in DEFINITIONS:
        e = effects["eICU-CRD"][column] * 100
        m = effects["MIMIC-IV"][column] * 100
        count = N_BOOT if label == PRIMARY else SENSITIVITY_BOOT
        e_draw = np.asarray(draws["eICU-CRD"][column][:count], float)
        m_draw = np.asarray(draws["MIMIC-IV"][column][:count], float)
        size = min(len(e_draw), len(m_draw))
        difference_draws = (e_draw[:size] - m_draw[:size]) * 100
        difference = e - m
        se = np.std(difference_draws, ddof=1)
        p = math.erfc(abs(difference / se) / math.sqrt(2)) if se > 0 else np.nan
        ci = np.percentile(difference_draws, [2.5, 97.5])
        rows.append({
            "rule": text,
            "analysis": label,
            "eICU_RD_percent": e,
            "MIMIC_RD_percent": m,
            "difference_eICU_minus_MIMIC_percent": difference,
            "difference_CI95_low": ci[0],
            "difference_CI95_high": ci[1],
            "heterogeneity_p": p,
            "bootstrap_replicates": size,
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table27_cross_database_grid.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
