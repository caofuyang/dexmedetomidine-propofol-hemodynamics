#!/usr/bin/env python3
"""v2.0: patient-important anchors for the thresholded haemodynamic outcome.

Everything else in this package measures the *instability* outcome the literature
uses — repeated low blood pressure or heart rate within 24 hours.  That outcome is
a surrogate: it is defined by a threshold the analyst chooses and it is not, by
itself, something a patient experiences.  This script asks whether the surrogate
points the same way as outcomes patients and clinicians actually care about, in
the same cohort and under the same weighting.

Three outcomes are treated as anchors because their definitions involve no
analyst-chosen vital-sign threshold:

  * ``new_vasopressor_24h``   — a vasopressor started after the index time, in a
                                 cohort that required no vasopressor at baseline,
  * ``icu_mortality``,
  * ``hospital_mortality``.

The comparison is descriptive.  A mortality difference of the size seen here
(RR near 0.5) is far larger than any plausible sedative effect and should be read
as confounding by indication, not as a drug effect; what the comparison shows is
that the surrogate and the hard outcomes are not measuring the same thing.

Gate: the pre-specified instability definition must reproduce frozen ``table7``.

Output: ``outputs/active_comparator_sedation/table28_anchor_outcomes.csv``
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUT = ROOT / "outputs" / "active_comparator_sedation"
N_BOOT = int(os.environ.get("SEDATION_ANCHOR_BOOT", "1000"))
SEED_OFFSET = int(os.environ.get("SEDATION_ANCHOR_SEED_OFFSET", "0"))
CLIP = 1e-6
PRE_SPECIFIED = "hourly_repeated_instability_24h"

# (column, group, row label)
OUTCOMES = [
    ("v2_ip_composite_k1", "Thresholded surrogate", "k = 1 adjacent abnormal hour"),
    (PRE_SPECIFIED, "Thresholded surrogate", "k = 2 adjacent abnormal hours (pre-specified)"),
    ("v2_ip_composite_k3", "Thresholded surrogate", "k = 3 consecutive abnormal hours"),
    ("v2_ip_map_k2", "Thresholded surrogate", "Hypotension by MAP <65 mmHg only"),
    ("v2_ip_sbp_k2", "Thresholded surrogate", "Hypotension by SBP <90 mmHg only"),
    ("new_vasopressor_24h", "Patient-important anchor", "New vasopressor after the index time"),
    ("icu_mortality", "Patient-important anchor", "ICU mortality"),
    ("hospital_mortality", "Patient-important anchor", "Hospital mortality"),
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hourly = load_module("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")
variants = load_module("variants", ROOT / "work" / "harmonized_definition_variants.py")


def build_work(base, database, prefix):
    """Use the cached prepared frame when present (it carries the refreshed
    ventilation status); otherwise rebuild it, which needs the raw database."""
    cached = OUT / f"{prefix}_prepared_analysis_frame.csv"
    if cached.exists():
        frame = pd.read_csv(cached)
    else:
        cohort = base.enforce_current_definitions(
            pd.read_csv(OUT / f"{prefix}_active_comparator_patient_level_v2.csv")
        )
        cohort = (
            base.refresh_eicu_baseline_ventilation(cohort)
            if database == "eICU-CRD"
            else base.refresh_mimic_baseline_ventilation(cohort)
        )
        summary = pd.read_csv(OUT / f"{prefix}_harmonized_hourly_outcomes_v4.csv")
        bins = pd.read_csv(OUT / f"{prefix}_harmonized_bins_v4.csv")
        ids = cohort["id"]
        derived = variants.build_outcome_table(bins).set_index("id").reindex(ids).fillna(0).astype(int)
        derived = derived.rename_axis("id").reset_index()
        frame = (
            cohort.merge(summary, on="id", how="left", validate="one_to_one")
            .merge(derived, on="id", how="left", validate="one_to_one")
        )
        frame = hourly.prepare_harmonized_frame(frame)
    return base.encode_analysis(frame, database)


def bootstrap_draws(base, work, cols, database, n_boot, seed):
    """Shared-replicate bootstrap: one propensity fit per replicate, all outcomes scored."""
    rng = np.random.default_rng(seed)
    cluster_col = "hospital_id" if database == "eICU-CRD" else None
    clusters = work[cluster_col].fillna("Unknown").astype(str).unique() if cluster_col else None
    roster = work[cluster_col].fillna("Unknown").astype(str) if cluster_col else None
    draws = {column: [] for column, _, _ in OUTCOMES}

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
        for column in draws:
            draws[column].append(base.weighted_risk(sample, column)[1])
        if (replicate + 1) % 100 == 0:
            print(f"[{database}] bootstrap {replicate + 1}/{n_boot}", flush=True)
    return draws


def main():
    base = hourly.load_base_module()
    frozen = pd.read_csv(OUT / "table7_harmonized_hourly_effects.csv")
    rows = []
    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
        work, cols = build_work(base, database, prefix)
        print(f"[{database}] analysis n={len(work)}", flush=True)

        row = frozen.query("database == @database and outcome == @PRE_SPECIFIED").iloc[0]
        observed_primary = base.weighted_risk(work, PRE_SPECIFIED)[1] * 100
        if abs(observed_primary - row.risk_difference_percent) > 1e-9:
            raise SystemExit(f"[{database}] pre-specified RD {observed_primary!r} != frozen table7 "
                             f"{row.risk_difference_percent!r} -- abort.")
        print(f"[{database}] pre-specified rule reproduces the frozen primary result", flush=True)

        draws = bootstrap_draws(base, work, cols, database, N_BOOT,
                                20260919 + SEED_OFFSET if database == "eICU-CRD" else 20260920 + SEED_OFFSET)

        for column, group, label in OUTCOMES:
            groups, rd, _, rr, _ = base.weighted_risk(work, column)
            values = np.asarray(draws[column], float)
            rd_ci = np.percentile(values, [2.5, 97.5]) * 100
            rows.append({
                "database": database,
                "outcome_group": group,
                "outcome": label,
                "column": column,
                "analysis_n": len(work),
                "propofol_weighted_percent": groups[0]["weighted_events_percent"],
                "dexmedetomidine_weighted_percent": groups[1]["weighted_events_percent"],
                "risk_difference_percent": rd * 100,
                "RD_CI95_low": rd_ci[0],
                "RD_CI95_high": rd_ci[1],
                "risk_ratio": rr,
                "bootstrap_replicates": int(np.isfinite(values).sum()),
            })
            print(f"[{database}] {label}: RD={rd * 100:+.2f} ({rd_ci[0]:+.2f} to {rd_ci[1]:+.2f})", flush=True)

        pd.DataFrame(rows).to_csv(OUT / "table28_anchor_outcomes.csv", index=False, encoding="utf-8-sig")

    table = pd.DataFrame(rows)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
