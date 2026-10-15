#!/usr/bin/env python3
"""v2.0: effective sample size after overlap weighting.

Overlap weighting balances the included covariates exactly, but it does so by
down-weighting patients who could only plausibly have received one of the two
drugs.  The number of patients the analysis effectively rests on is therefore
much smaller than the nominal sample size, and it is that effective number, not
the nominal one, that governs the width of every interval reported in the paper.

This script writes the effective sample size per database and treatment arm, so
that the effective-sample-size table in the manuscript is backed by a file like
every other reported number.

Effective sample size is (sum w)^2 / sum(w^2), the standard Kish measure.

Output: ``outputs/active_comparator_sedation/table29_effective_sample_size.csv``
"""

from __future__ import annotations

import importlib.util
import os
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


hourly = load_module("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")
variants = load_module("variants", ROOT / "work" / "harmonized_definition_variants.py")


def build_work(base, database, prefix):
    """Cached prepared frame when available; otherwise rebuild from the raw data."""
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


def main():
    base = hourly.load_base_module()
    frozen = pd.read_csv(OUT / "table7_harmonized_hourly_effects.csv")
    rows = []
    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
        work, _ = build_work(base, database, prefix)

        # gate: the analysis frame must be the frozen one
        expected = int(frozen.query(
            "database == @database and outcome == 'hourly_repeated_instability_24h'"
        ).analysis_n.iloc[0])
        if len(work) != expected:
            raise SystemExit(f"[{database}] analysis n={len(work)}, expected {expected} -- abort.")
        print(f"[{database}] analysis n={len(work)} matches the frozen sample", flush=True)

        weights = work.overlap_weight.to_numpy(float)
        dex = work.dex.to_numpy()
        for arm, label in ((0, "propofol"), (1, "dexmedetomidine")):
            w = weights[dex == arm]
            nominal = len(w)
            ess = float(w.sum() ** 2 / (w ** 2).sum())
            rows.append({
                "database": database,
                "treatment": label,
                "nominal_n": nominal,
                "effective_n": ess,
                "effective_fraction": ess / nominal,
                "median_weight": float(np.median(w)),
                "max_weight": float(w.max()),
            })
            print(f"[{database}] {label}: nominal {nominal}, effective {ess:.0f} "
                  f"({100 * ess / nominal:.0f}%)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table29_effective_sample_size.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
