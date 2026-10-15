#!/usr/bin/env python3
"""v2.0 helper: cache the prepared analysis frames.

Most of the v2.0 scripts need one thing that cannot be reconstructed from the
cached files: the ventilation status refreshed from the raw database at the index
time (``respiratoryCare.csv.gz`` in eICU-CRD, ``icu/procedureevents.csv.gz`` in
MIMIC-IV).  Both are small single tables — this script reads them once, builds the
exact analysis frame each database uses, and writes it to disk so that later
analyses can be run without access to the raw databases again.

It changes no analysis; it only snapshots what the pipeline already computes.  Run
it once after ``harmonized_bin_export.py`` and ``harmonized_definition_variants.py``.

Output: ``{prefix}_prepared_analysis_frame.csv`` for each database.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

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


def main():
    base = hourly.load_base_module()
    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
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
        outcomes = variants.build_outcome_table(bins).set_index("id").reindex(ids).fillna(0).astype(int)
        outcomes = outcomes.rename_axis("id").reset_index()
        density = variants.build_density_table(bins).set_index("id").reindex(ids)
        for count_col in ("post_hr_bins", "hr_records_total", "post_bp_bins",
                          "bp_records_total", "nibp_map_records_total"):
            density[count_col] = density[count_col].fillna(0)
        density = density.rename_axis("id").reset_index()

        frame = (
            cohort.merge(summary, on="id", how="left", validate="one_to_one")
            .merge(outcomes, on="id", how="left", validate="one_to_one")
            .merge(density, on="id", how="left", validate="one_to_one")
        )
        prepared = hourly.prepare_harmonized_frame(frame)

        # gate: the cached frame must reproduce the frozen analysis sample
        work, _ = base.encode_analysis(prepared, database)
        frozen = pd.read_csv(OUT / "table7_harmonized_hourly_effects.csv")
        expected = int(frozen.query(
            "database == @database and outcome == 'hourly_repeated_instability_24h'"
        ).analysis_n.iloc[0])
        if len(work) != expected:
            raise SystemExit(f"[{database}] cached frame gives n={len(work)}, expected {expected} -- abort.")
        print(f"[{database}] cached frame reproduces the frozen analysis sample (n={len(work)})", flush=True)

        path = OUT / f"{prefix}_prepared_analysis_frame.csv"
        prepared.to_csv(path, index=False)
        print(f"[{database}] wrote {path.name} ({path.stat().st_size // 1024} KB)", flush=True)


if __name__ == "__main__":
    main()
