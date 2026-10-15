#!/usr/bin/env python3
"""v2.0: multiple-imputation sensitivity analysis.

Missingness in the analysis covariates is under 1% in both databases (weight
<=1.0%, MIMIC baseline MAP 0.28%, eICU age 0.02%), so the complete-case analysis
of v1.0 discards almost nothing.  This script nevertheless re-runs the frozen
estimator under multiple imputation so that missing-data handling is not an
unexamined assumption:

  * a compact MICE (chained equations) imputes the raw covariates with missing
    values from all other covariates, treatment and the outcome, drawing both
    the regression coefficients and the residual (proper imputation);
  * each completed dataset is analysed with the untouched frozen estimator;
  * point estimates and variances are combined with Rubin's rules.

Within-imputation variance comes from the robust sandwich estimator already
returned by ``weighted_risk`` (not from a fresh bootstrap), which keeps the run
to seconds; the complete-case bootstrap interval is repeated alongside so the
two are directly comparable.

Output: ``outputs/active_comparator_sedation/table26_multiple_imputation.csv``
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
M_IMPUTATIONS = int(os.environ.get("SEDATION_MI_COUNT", "20"))
OUTCOME = "hourly_repeated_instability_24h"

# raw covariates that can carry missing values (everything else is complete)
IMPUTE_CANDIDATES = ("age", "weight", "baseline_map", "baseline_hr")
PREDICTOR_CANDIDATES = (
    "age", "weight", "baseline_map", "baseline_hr", "index_hour", "vent", "female",
    "baseline_bp_n", "baseline_hr_n", "baseline_midazolam", "baseline_opioid",
    "baseline_ketamine", "dex",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hourly = load_module("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")
variants = load_module("variants", ROOT / "work" / "harmonized_definition_variants.py")


def ols_draw(X, y, rng, sigma_floor=1e-6):
    """Least-squares fit with a proper posterior draw of the coefficients."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma = max(math.sqrt(float(resid @ resid) / dof), sigma_floor)
    cov = sigma ** 2 * XtX_inv
    try:
        chol = np.linalg.cholesky(cov + np.eye(X.shape[1]) * 1e-12)
        beta = beta + chol @ rng.normal(size=X.shape[1])
    except np.linalg.LinAlgError:
        pass
    return beta, sigma


def impute_once(frame, targets, predictors, rng):
    completed = frame.copy()
    for target in targets:
        missing = completed[target].isna()
        if not missing.any():
            continue
        observed = ~missing
        X_obs = np.column_stack([
            np.ones(int(observed.sum())),
            completed.loc[observed, predictors].to_numpy(float),
        ])
        beta, sigma = ols_draw(X_obs, completed.loc[observed, target].to_numpy(float), rng)
        X_miss = np.column_stack([
            np.ones(int(missing.sum())),
            completed.loc[missing, predictors].to_numpy(float),
        ])
        completed.loc[missing, target] = X_miss @ beta + rng.normal(0.0, sigma, int(missing.sum()))
    return completed


def rubin(points, variances, m):
    """Rubin's rules for a point estimate and its sampling variance."""
    q_bar = float(np.mean(points))
    u_bar = float(np.mean(variances))
    b = float(np.var(points, ddof=1)) if m > 1 else 0.0
    total = u_bar + (1.0 + 1.0 / m) * b
    if b > 0:
        dof = (m - 1) * (1.0 + u_bar / ((1.0 + 1.0 / m) * b)) ** 2
    else:
        dof = math.inf
    half = 1.96 if not math.isfinite(dof) else _t_quantile(dof, 0.975)
    return q_bar, q_bar - half * math.sqrt(total), q_bar + half * math.sqrt(total)


def _t_quantile(dof, p):
    """Student-t quantile, by bisection on the t density.

    Uses log-gamma so that the normalising constant stays finite for the large
    degrees of freedom that Rubin's rules routinely produce."""
    log_const = (
        math.lgamma((dof + 1) / 2)
        - 0.5 * math.log(dof * math.pi)
        - math.lgamma(dof / 2)
    )
    const = math.exp(log_const)

    def cdf(x):
        grid = np.linspace(0.0, abs(x), 2001)
        density = (1.0 + grid ** 2 / dof) ** (-(dof + 1) / 2.0)
        area = np.trapezoid(density, grid)
        return 0.5 + const * area if x >= 0 else 0.5 - const * area
    low, high = 0.0, 100.0
    for _ in range(200):
        mid = (low + high) / 2
        if cdf(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def analyse(base, prepared, database):
    """Analyse an already-prepared frame (see hourly.prepare_harmonized_frame).

    The frame must arrive prepared because that step is what fills
    ``baseline_map``/``baseline_hr`` from the hourly summary; preparing again
    afterwards would overwrite any imputed value.
    """
    work, _ = base.encode_analysis(prepared, database)
    rows, rd, rd_ci, rr, rr_ci = base.weighted_risk(work, OUTCOME)
    se_rd = (rd_ci[1] - rd_ci[0]) / (2 * 1.96)
    se_log_rr = (
        (math.log(rr_ci[1]) - math.log(rr_ci[0])) / (2 * 1.96)
        if rr > 0 and rr_ci[0] > 0 else np.nan
    )
    return len(work), rd, se_rd, rr, se_log_rr, rows


def main():
    base = hourly.load_base_module()
    out_rows = []
    for database, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
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

        # prepare once, then restrict to the evaluable rows the analysis actually
        # uses: baseline_map/baseline_hr are only populated where a baseline
        # window exists, and those rows are dropped by ``evaluable`` anyway, so
        # judging missingness on the full frame would flag variables that are in
        # fact complete within the analysed population.
        prepared = hourly.prepare_harmonized_frame(frame)
        analysis_frame = prepared[prepared.evaluable == 1].copy()
        targets = [c for c in IMPUTE_CANDIDATES if analysis_frame[c].isna().any()]
        predictors = [c for c in PREDICTOR_CANDIDATES if c in analysis_frame.columns and c not in targets]
        missing_total = int(analysis_frame[targets].isna().sum().sum()) if targets else 0
        print(f"[{database}] imputing {targets} ({missing_total} missing cells of "
              f"{len(analysis_frame) * max(len(targets), 1)}) with {len(predictors)} predictors, "
              f"m={M_IMPUTATIONS}", flush=True)

        # ---- complete case, for a like-for-like comparison
        n_cc, rd_cc, se_cc, rr_cc, selogrr_cc, _ = analyse(
            base, analysis_frame.dropna(subset=targets or None), database
        )
        frozen = pd.read_csv(OUT / "table7_harmonized_hourly_effects.csv")
        boot = frozen.query("database == @database and outcome == @OUTCOME").iloc[0]
        if abs(rd_cc * 100 - boot.risk_difference_percent) > 1e-9:
            raise SystemExit(
                f"[{database}] complete-case RD {rd_cc * 100!r} does not reproduce frozen "
                f"table7 {boot.risk_difference_percent!r} -- abort."
            )
        print(f"[{database}] complete case reproduces the frozen estimate (n={n_cc})", flush=True)
        out_rows.append({
            "database": database, "outcome": OUTCOME, "method": "complete_case_bootstrap",
            "analysis_n": n_cc, "imputations": 0,
            "risk_difference_percent": rd_cc * 100,
            "RD_CI95_low": boot.RD_CI95_low, "RD_CI95_high": boot.RD_CI95_high,
            "risk_ratio": rr_cc, "RR_CI95_low": boot.RR_CI95_low, "RR_CI95_high": boot.RR_CI95_high,
            "RD_CI_basis": "cluster/patient bootstrap (v1.0)",
        })

        # ---- multiple imputation
        rng = np.random.default_rng(20261015)
        rds, ses, rrs, selog, sizes = [], [], [], [], []
        for i in range(M_IMPUTATIONS):
            completed = analysis_frame if not targets else impute_once(analysis_frame, targets, predictors, rng)
            n_i, rd_i, se_i, rr_i, selog_i, _ = analyse(base, completed, database)
            rds.append(rd_i); ses.append(se_i); rrs.append(rr_i); selog.append(selog_i); sizes.append(n_i)
            if (i + 1) % 5 == 0:
                print(f"[{database}] imputation {i + 1}/{M_IMPUTATIONS}", flush=True)
        rd_bar, rd_lo, rd_hi = rubin(rds, np.square(ses), M_IMPUTATIONS)
        log_rr = [math.log(v) if v > 0 else np.nan for v in rrs]
        rr_bar, rr_lo, rr_hi = rubin(log_rr, np.square(selog), M_IMPUTATIONS)
        out_rows.append({
            "database": database, "outcome": OUTCOME, "method": "multiple_imputation",
            "analysis_n": int(round(np.mean(sizes))), "imputations": M_IMPUTATIONS,
            "risk_difference_percent": rd_bar * 100,
            "RD_CI95_low": rd_lo * 100, "RD_CI95_high": rd_hi * 100,
            "risk_ratio": math.exp(rr_bar),
            "RR_CI95_low": math.exp(rr_lo), "RR_CI95_high": math.exp(rr_hi),
            "RD_CI_basis": "Rubin's rules (sandwich within-imputation variance)",
        })
        print(f"[{database}] MI RD={rd_bar * 100:.3f} ({rd_lo * 100:.3f}, {rd_hi * 100:.3f}) "
              f"vs complete-case bootstrap {boot.risk_difference_percent:.3f} "
              f"({boot.RD_CI95_low:.3f}, {boot.RD_CI95_high:.3f})", flush=True)

    table = pd.DataFrame(out_rows)
    table.to_csv(OUT / "table26_multiple_imputation.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
