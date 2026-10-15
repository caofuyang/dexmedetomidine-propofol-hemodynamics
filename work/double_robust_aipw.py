#!/usr/bin/env python3
"""v2.0: doubly robust (AIPW) re-analysis of the overlap-weighted effect.

v1.0 reports the overlap-weighted (IPW) risk difference.  Because that estimator
relies entirely on the propensity-score model being right, this script adds the
two other members of the same family and reports all three from identical
bootstrap replicates:

  * ``overlap_weighted_IPW``   -- the v1.0 estimator (gate below reproduces it),
  * ``overlap_weighted_gcomp`` -- outcome-model standardisation over the overlap
                                  population (correct if the outcome model holds),
  * ``overlap_weighted_AIPW``  -- augmented IPW (correct if *either* model holds).

The estimand is unchanged: the average treatment effect in the overlap
population (ATO), i.e. the same target the overlap weights already define.

Gate: the observed-data IPW risk difference for the primary outcome must equal
the frozen v1.0 ``table7`` value; the script aborts otherwise, so a divergence in
the shared design matrix cannot slip through.

Output: ``outputs/active_comparator_sedation/table25_double_robust.csv``
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
N_BOOT = int(os.environ.get("SEDATION_AIPW_BOOT", "1000"))
CLIP = 1e-6

# outcome column -> (analysis label, frozen v1.0 RD for the gate or None)
OUTCOMES = (
    ("hourly_repeated_instability_24h", "primary_k2"),
    ("v2_ip_composite_k1", "definition_k1"),
    ("v2_ip_composite_k3", "definition_k3"),
)

ESTIMATORS = ("overlap_weighted_IPW", "overlap_weighted_gcomp", "overlap_weighted_AIPW")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hourly = load_module("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")
variants = load_module("variants", ROOT / "work" / "harmonized_definition_variants.py")


def predict_logistic(beta, X):
    design = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    return 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))


def arm_model(base, X, y, arm):
    """Return a callable X -> P(Y=1) fitted within one treatment arm.

    Falls back to the arm's event rate when the arm is too small or the fit does
    not converge, which happens only on tiny bootstrap resamples."""
    Xa, ya = X[arm], y[arm]
    rate = float(ya.mean()) if len(ya) else np.nan
    if len(ya) < 20 or np.unique(ya).size < 2:
        return lambda new, r=rate: np.full(len(new), r)
    beta, _ = base.logistic_fit(Xa, ya)
    if not np.all(np.isfinite(beta)):
        return lambda new, r=rate: np.full(len(new), r)
    return lambda new, b=beta: predict_logistic(b, new)


def estimator_block(ps, dex, y, X, model1, model0):
    """Return the three (treated_mean, control_mean) pairs for one replicate."""
    h = ps * (1.0 - ps)
    denom = h.sum()
    if not np.isfinite(denom) or denom <= 0:
        return None
    m1 = model1(X)
    m0 = model0(X)

    # IPW: identical arithmetic to base.weighted_risk on the same weights
    w = np.where(dex == 1, 1.0 - ps, ps)
    ipw1 = np.sum(w[dex == 1] * y[dex == 1]) / np.sum(w[dex == 1])
    ipw0 = np.sum(w[dex == 0] * y[dex == 0]) / np.sum(w[dex == 0])

    # g-computation: standardise the outcome models over the overlap population
    g1 = np.sum(h * m1) / denom
    g0 = np.sum(h * m0) / denom

    # AIPW: IPW plus the outcome-model residual correction
    aipw1 = np.sum(h * (dex * y / ps - (dex / ps - 1.0) * m1)) / denom
    aipw0 = np.sum(h * ((1 - dex) * y / (1 - ps) - ((1 - dex) / (1 - ps) - 1.0) * m0)) / denom

    return {"overlap_weighted_IPW": (ipw1, ipw0),
            "overlap_weighted_gcomp": (g1, g0),
            "overlap_weighted_AIPW": (aipw1, aipw0)}


def ratios(pair):
    treated, control = pair
    if not (np.isfinite(treated) and np.isfinite(control)):
        return np.nan, np.nan
    return treated - control, (treated / control if control else np.nan)


def build_frame(base, database, prefix, bins):
    cohort = base.enforce_current_definitions(pd.read_csv(OUT / f"{prefix}_active_comparator_patient_level_v2.csv"))
    cohort = (
        base.refresh_eicu_baseline_ventilation(cohort)
        if database == "eICU-CRD"
        else base.refresh_mimic_baseline_ventilation(cohort)
    )
    summary = pd.read_csv(OUT / f"{prefix}_harmonized_hourly_outcomes_v4.csv")
    ids = cohort["id"]
    outcomes = variants.build_outcome_table(bins).set_index("id").reindex(ids).fillna(0).astype(int)
    outcomes = outcomes.rename_axis("id").reset_index()
    return cohort.merge(summary, on="id", how="left", validate="one_to_one").merge(
        outcomes, on="id", how="left", validate="one_to_one"
    )


def main():
    base = hourly.load_base_module()
    rows = []
    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
        bins = pd.read_csv(OUT / f"{prefix}_harmonized_bins_v4.csv")
        frame = hourly.prepare_harmonized_frame(build_frame(base, database, prefix, bins))
        work, cols = base.encode_analysis(frame, database)
        n_propofol = int((work.dex == 0).sum())
        n_dex = int((work.dex == 1).sum())
        print(f"[{database}] analysis n={len(work)} (propofol {n_propofol} / dex {n_dex})", flush=True)

        dex = work.dex.to_numpy(float)
        X = ((work[cols] - work[cols].mean()) / work[cols].std().replace(0, 1)).to_numpy(float)
        ps = work.ps.to_numpy(float)

        # ---- observed-data estimates + gate against the frozen primary
        observed = {}
        for outcome, label in OUTCOMES:
            y = work[outcome].to_numpy(float)
            block = estimator_block(
                ps, dex, y, X,
                arm_model(base, X, y, dex == 1), arm_model(base, X, y, dex == 0),
            )
            if block is None:
                raise SystemExit(f"[{database}] observed-data estimator failed for {outcome}")
            observed[outcome] = block
            rd, rr = ratios(block["overlap_weighted_IPW"])
            if label == "primary_k2":
                frozen = pd.read_csv(OUT / "table7_harmonized_hourly_effects.csv")
                frozen_rd = frozen.query(
                    "database == @database and outcome == 'hourly_repeated_instability_24h'"
                ).risk_difference_percent.iloc[0]
                if abs(rd * 100 - frozen_rd) > 1e-9:
                    raise SystemExit(
                        f"[{database}] overlap-weighted IPW RD {rd * 100!r} does not reproduce the "
                        f"frozen table7 value {frozen_rd!r} -- abort."
                    )
            print(f"[{database}] {label}: IPW RD={rd*100:.3f} "
                  f"gcomp RD={ratios(block['overlap_weighted_gcomp'])[0]*100:.3f} "
                  f"AIPW RD={ratios(block['overlap_weighted_AIPW'])[0]*100:.3f}", flush=True)
        print(f"[{database}] primary outcome reproduces the frozen overlap-weighted estimate", flush=True)

        # ---- cluster (eICU) or patient (MIMIC) bootstrap, same seeds as v1.0
        rng = np.random.default_rng(20260919 if database == "eICU-CRD" else 20260920)
        cluster_col = "hospital_id" if database == "eICU-CRD" else None
        clusters = (
            work[cluster_col].fillna("Unknown").astype(str).unique() if cluster_col else None
        )
        roster = work[cluster_col].fillna("Unknown").astype(str) if cluster_col else None
        draws = {
            (outcome, estimator): {"rd": [], "rr": []}
            for outcome, _ in OUTCOMES for estimator in ESTIMATORS
        }
        for replicate in range(N_BOOT):
            if cluster_col:
                selected = rng.choice(clusters, size=len(clusters), replace=True)
                sample = pd.concat(
                    [work[roster == cluster] for cluster in selected], ignore_index=True
                )
            else:
                sample = work.iloc[rng.integers(0, len(work), len(work))].copy()
            if len(sample) == 0:
                continue
            means = sample[cols].mean()
            sds = sample[cols].std().replace(0, 1)
            Xs = ((sample[cols] - means) / sds).to_numpy(float)
            _, ps_sample = base.logistic_fit(Xs, sample.dex.to_numpy(float))
            ps_sample = np.clip(ps_sample, CLIP, 1 - CLIP)
            dexs = sample.dex.to_numpy(float)
            for outcome, _ in OUTCOMES:
                ys = sample[outcome].to_numpy(float)
                block = estimator_block(
                    ps_sample, dexs, ys, Xs,
                    arm_model(base, Xs, ys, dexs == 1), arm_model(base, Xs, ys, dexs == 0),
                )
                if block is None:
                    continue
                for estimator in ESTIMATORS:
                    rd, rr = ratios(block[estimator])
                    draws[(outcome, estimator)]["rd"].append(rd)
                    draws[(outcome, estimator)]["rr"].append(rr)
            if (replicate + 1) % 100 == 0:
                print(f"[{database}] bootstrap {replicate + 1}/{N_BOOT}", flush=True)

        for outcome, label in OUTCOMES:
            for estimator in ESTIMATORS:
                values = draws[(outcome, estimator)]
                rd_draws = np.asarray(values["rd"], float)
                rr_draws = np.asarray(values["rr"], float)
                rd, rr = ratios(observed[outcome][estimator])
                treated, control = observed[outcome][estimator]
                rd_ci = np.percentile(rd_draws, [2.5, 97.5]) * 100
                rr_ci = np.percentile(rr_draws, [2.5, 97.5])
                rows.append({
                    "database": database,
                    "analysis": label,
                    "outcome": outcome,
                    "estimator": estimator,
                    "analysis_n": len(work),
                    "propofol_n": n_propofol,
                    "dexmedetomidine_n": n_dex,
                    "propofol_weighted_percent": control * 100,
                    "dexmedetomidine_weighted_percent": treated * 100,
                    "risk_difference_percent": rd * 100,
                    "RD_CI95_low": rd_ci[0],
                    "RD_CI95_high": rd_ci[1],
                    "risk_ratio": rr,
                    "RR_CI95_low": rr_ci[0],
                    "RR_CI95_high": rr_ci[1],
                    "bootstrap_replicates": int(np.isfinite(rd_draws).sum()),
                })

        pd.DataFrame(rows).to_csv(OUT / "table25_double_robust.csv", index=False, encoding="utf-8-sig")
        print(f"[{database}] wrote {len(rows)} rows to table25_double_robust.csv", flush=True)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
