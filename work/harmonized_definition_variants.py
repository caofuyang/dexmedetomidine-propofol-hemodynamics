#!/usr/bin/env python3
"""v2.0: outcome-definition grid, single-component hypotension, monitoring density.

Reads the per-hour-bin table produced once by ``harmonized_bin_export.py`` and
derives, for every patient, a family of outcome definitions that v1.0 could not
report without re-scanning the raw databases:

  * repeated(k) for k = 1, 2, 3 adjacent hourly bins (k = 2 reproduces the v1.0
    primary; k = 1 reproduces the v1.0 "any abnormal hour" outcome),
  * hypotension defined by MAP < 65 only, or by SBP < 90 only (v1.0 only ever
    reported the union),
  * the three alternative blood-pressure source rules already used in v1.0,
  * monitoring density (records per observed hourly bin) as a description and as
    an effect-modifier stratum.

Everything runs through the frozen v1.0 estimator (``analyze_harmonized``), so
only the outcome column changes -- the model, weights and bootstrap are
untouched.  n_boot is 1000 for the definition that mirrors the primary outcome
and 300 elsewhere, matching the v1.0 convention that non-primary sensitivities
use 300 replicates.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUT = ROOT / "outputs" / "active_comparator_sedation"
GRID_BOOT = int(os.environ.get("SEDATION_GRID_BOOT", "300"))
PRIMARY_BOOT = 1000


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hourly = load_module("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")


# ---------------------------------------------------------------- pure logic

def has_run(hours, k):
    """True if ``hours`` (integer hour bins) contains k consecutive integers."""
    hours = np.sort(np.asarray(hours, dtype=int))
    if k <= 1:
        return bool(len(hours) >= 1)
    run = 1
    for previous, current in zip(hours, hours[1:]):
        if current - previous == 1:
            run += 1
            if run >= k:
                return True
        else:
            run = 1
    return False


def add_bin_flags(bins):
    """Attach one boolean low/high flag per bin for each definition family."""
    b = bins.copy()
    inv_obs = (b.inv_map_n > 0) | (b.inv_sbp_n > 0)
    nibp_obs = (b.nibp_map_n > 0) | (b.nibp_sbp_n > 0)
    inv_low = ((b.inv_map_n > 0) & (b.inv_map_mean < 65)) | ((b.inv_sbp_n > 0) & (b.inv_sbp_mean < 90))
    nibp_low = ((b.nibp_map_n > 0) & (b.nibp_map_mean < 65)) | ((b.nibp_sbp_n > 0) & (b.nibp_sbp_mean < 90))

    map_ip = np.where(b.inv_map_n > 0, b.inv_map_mean, b.nibp_map_mean)
    sbp_ip = np.where(b.inv_sbp_n > 0, b.inv_sbp_mean, b.nibp_sbp_mean)
    map_np = np.where(b.nibp_map_n > 0, b.nibp_map_mean, b.inv_map_mean)
    sbp_np = np.where(b.nibp_sbp_n > 0, b.nibp_sbp_mean, b.inv_sbp_mean)

    b["low_ip_composite"] = (map_ip < 65) | (sbp_ip < 90)
    b["low_ip_map"] = map_ip < 65
    b["low_ip_sbp"] = sbp_ip < 90
    b["low_np_composite"] = (map_np < 65) | (sbp_np < 90)
    b["low_any_composite"] = inv_low | nibp_low
    b["low_conc_composite"] = (
        (inv_obs & nibp_obs & inv_low & nibp_low)
        | (inv_obs & ~nibp_obs & inv_low)
        | (~inv_obs & nibp_obs & nibp_low)
    )
    b["brady"] = (b.hr_n > 0) & (b.hr_mean < 50)
    return b


# family -> (low column, add bradycardia)
FAMILIES = {
    "ip_composite": ("low_ip_composite", True),
    "ip_map": ("low_ip_map", False),
    "ip_sbp": ("low_ip_sbp", False),
    "np_composite": ("low_np_composite", True),
    "any_composite": ("low_any_composite", True),
    "conc_composite": ("low_conc_composite", True),
}


def build_outcome_table(bins, ks=(1, 2, 3)):
    post = add_bin_flags(bins[bins.hour_bin >= 0].copy()).sort_values(["id", "hour_bin"])
    records = []
    for stay, grp in post.groupby("id", sort=False):
        hours = grp.hour_bin.to_numpy()
        brady_hours = hours[grp.brady.to_numpy()]
        rec = {"id": int(stay)}
        for family, (lowcol, with_brady) in FAMILIES.items():
            low_hours = hours[grp[lowcol].to_numpy()]
            for k in ks:
                flag = has_run(low_hours, k)
                if with_brady:
                    flag = flag or has_run(brady_hours, k)
                rec[f"v2_{family}_k{k}"] = int(flag)
        records.append(rec)
    return pd.DataFrame(records)


def build_density_table(bins):
    post = bins[(bins.hour_bin >= 0) & (bins.hr_n > 0)]
    agg = post.groupby("id").agg(
        post_hr_bins=("hour_bin", "size"),
        hr_records_total=("hr_n", "sum"),
    )
    agg["hr_records_per_bin"] = agg.hr_records_total / agg.post_hr_bins
    post_bp = bins[(bins.hour_bin >= 0) & ((bins.inv_map_n > 0) | (bins.nibp_map_n > 0))]
    bp = post_bp.groupby("id").agg(
        post_bp_bins=("hour_bin", "size"),
        bp_records_total=("inv_map_n", "sum"),
    )
    bp["nibp_map_records_total"] = post_bp.groupby("id")["nibp_map_n"].sum()
    agg = agg.join(bp, how="left")
    return agg.reset_index()


# ------------------------------------------------------------------ driver

# definition label -> (outcome column, bootstrap replicates)
DEFINITIONS = {
    "def_k1_any_hour": ("v2_ip_composite_k1", GRID_BOOT),
    "def_k2_adjacent_primary": ("v2_ip_composite_k2", PRIMARY_BOOT),
    "def_k3_adjacent": ("v2_ip_composite_k3", GRID_BOOT),
    "def_hypot_map_only_k2": ("v2_ip_map_k2", PRIMARY_BOOT),
    "def_hypot_sbp_only_k2": ("v2_ip_sbp_k2", PRIMARY_BOOT),
    "def_source_nibp_priority_k2": ("v2_np_composite_k2", GRID_BOOT),
    "def_source_any_k2": ("v2_any_composite_k2", GRID_BOOT),
    "def_source_concordant_k2": ("v2_conc_composite_k2", GRID_BOOT),
}

DENSITY_TERM = "hr_records_per_bin"


def main():
    base = hourly.load_base_module()
    grid_rows, density_rows, stratified_rows = [], [], []

    jobs = (
        ("eICU-CRD", "eicu"),
        ("MIMIC-IV", "mimic"),
    )
    for database, prefix in jobs:
        cohort = base.enforce_current_definitions(pd.read_csv(OUT / f"{prefix}_active_comparator_patient_level_v2.csv"))
        cohort = (
            base.refresh_eicu_baseline_ventilation(cohort)
            if database == "eICU-CRD"
            else base.refresh_mimic_baseline_ventilation(cohort)
        )
        summary = pd.read_csv(OUT / f"{prefix}_harmonized_hourly_outcomes_v4.csv")
        bins = pd.read_csv(OUT / f"{prefix}_harmonized_bins_v4.csv")

        ids = cohort["id"]
        outcomes = build_outcome_table(bins)
        density = build_density_table(bins)
        # A stay with no post-index hourly bin (no records at all, or only the
        # pre-index baseline bins that are filtered out above) has no derived
        # row.  The frozen ``finalize_state`` scores such an empty state 0, so
        # reindex onto the full cohort and default the outcome flags to 0 rather
        # than letting the left-merge introduce NaN.
        outcomes = (
            outcomes.set_index("id").reindex(ids).fillna(0).astype(int)
            .rename_axis("id").reset_index()
        )
        density = density.set_index("id").reindex(ids)
        for count_col in (
            "post_hr_bins", "hr_records_total", "post_bp_bins",
            "bp_records_total", "nibp_map_records_total",
        ):
            density[count_col] = density[count_col].fillna(0)
        density = density.rename_axis("id").reset_index()
        frame = (
            cohort.merge(summary, on="id", how="left", validate="one_to_one")
            .merge(outcomes, on="id", how="left", validate="one_to_one")
            .merge(density, on="id", how="left", validate="one_to_one")
        )

        # ---- sanity: the k=2 invasive-priority composite must equal v1.0 primary
        observed = frame.hourly_repeated_instability_24h.astype(int)
        derived = frame.v2_ip_composite_k2.astype(int)
        mismatch = int((derived != observed).sum())
        if mismatch:
            bad = frame.loc[
                derived != observed,
                ["id", "v2_ip_composite_k2", "hourly_repeated_instability_24h"],
            ]
            raise SystemExit(
                f"[{database}] v2_ip_composite_k2 disagrees with the frozen primary outcome "
                f"in {mismatch} stays -- abort.\n" + bad.head(20).to_string(index=False)
            )
        print(f"[{database}] k=2 definition reproduces the frozen primary outcome", flush=True)

        # ---- definition grid
        for label, (col, n_boot) in DEFINITIONS.items():
            print(f"[{database}] definition {label} (n_boot={n_boot}) ...", flush=True)
            _, rows, _ = hourly.analyze_harmonized(
                base, frame, database, analysis_label=label, n_boot=n_boot, outcomes=[col],
            )
            grid_rows.extend(rows)
            print(f"[{database}] definition {label} done", flush=True)

        # persist the grid as soon as it is complete, so a crash while handling a
        # later database cannot discard the bootstrap work already done
        pd.DataFrame(grid_rows).to_csv(
            OUT / "table22_definition_grid.csv", index=False, encoding="utf-8-sig"
        )

        # ---- monitoring density description
        for dex, treatment in ((0, "propofol"), (1, "dexmedetomidine")):
            group = frame[frame.dex == dex]
            density_rows.append({
                "database": database,
                "treatment": treatment,
                "n": len(group),
                "hr_records_per_bin_median": group.hr_records_per_bin.median(),
                "hr_records_per_bin_q1": group.hr_records_per_bin.quantile(.25),
                "hr_records_per_bin_q3": group.hr_records_per_bin.quantile(.75),
                "post_hr_bins_median": group.post_hr_bins.median(),
                "hr_records_total_median": group.hr_records_total.median(),
            })

        pd.DataFrame(density_rows).to_csv(
            OUT / "table23_monitoring_density.csv", index=False, encoding="utf-8-sig"
        )

        # ---- effect by monitoring-density tertile (primary outcome)
        # hr records per observed bin is very heavily tied (half the eICU cohort
        # sits at exactly ~12, a quarter of MIMIC at ~1), so quantile cut points
        # taken on the raw values collapse into a degenerate stratum.  Rank the
        # variable first, which yields three equal-sized groups whatever the tie
        # structure, and record the realised value range of each stratum.
        valid = frame[frame.hr_records_per_bin.notna()].copy()
        if len(valid) >= 30:
            valid["density_tertile"] = pd.qcut(
                valid.hr_records_per_bin.rank(method="first"), 3,
                labels=["low", "mid", "high"],
            )
        else:
            valid["density_tertile"] = "all"
        for tertile, subset in valid.groupby("density_tertile", observed=True):
            if len(subset) < 20:
                print(f"[{database}] density stratum {tertile} skipped (n={len(subset)})", flush=True)
                continue
            print(f"[{database}] density stratum {tertile} (n={len(subset)}) ...", flush=True)
            try:
                _, rows, _ = hourly.analyze_harmonized(
                    base, subset.copy(), database,
                    analysis_label=f"density_{tertile}", n_boot=GRID_BOOT,
                    outcomes=["hourly_repeated_instability_24h"],
                )
            except ValueError as exc:
                # e.g. the single-treatment-hospital restriction empties a small
                # stratum; report it loudly rather than aborting the whole run.
                print(f"[{database}] density stratum {tertile} failed ({exc}); skipped", flush=True)
                continue
            for row in rows:
                row["density_tertile"] = str(tertile)
                row["tertile_density_min"] = float(subset.hr_records_per_bin.min())
                row["tertile_density_median"] = float(subset.hr_records_per_bin.median())
                row["tertile_density_max"] = float(subset.hr_records_per_bin.max())
            stratified_rows.extend(rows)
            print(f"[{database}] density stratum {tertile} done", flush=True)
        pd.DataFrame(stratified_rows).to_csv(
            OUT / "table24_density_stratified.csv", index=False, encoding="utf-8-sig"
        )

    pd.DataFrame(grid_rows).to_csv(OUT / "table22_definition_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(density_rows).to_csv(OUT / "table23_monitoring_density.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stratified_rows).to_csv(OUT / "table24_density_stratified.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(grid_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
