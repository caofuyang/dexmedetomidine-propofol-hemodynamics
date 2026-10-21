#!/usr/bin/env python3
"""Criterion (prognostic) validity of the hypotension-definition grid.

For each definition used in the manuscript, compute within the same analysis
cohort the discrimination (AUC) for three patient-important outcomes and
compare each definition with the pre-specified one by DeLong's test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
ROOT = DATA.resolve().parents[1]


def has_run(hours, k):
    hours = np.sort(np.asarray(hours, dtype=int))
    if k <= 1:
        return bool(len(hours) >= 1)
    run = 1
    for prev, cur in zip(hours, hours[1:]):
        if cur - prev == 1:
            run += 1
            if run >= k:
                return True
        else:
            run = 1
    return False


FAMILIES = {
    "ip_composite": ("low_ip_composite", True),
    "ip_map": ("low_ip_map", False),
    "ip_sbp": ("low_ip_sbp", False),
    "np_composite": ("low_np_composite", True),
    "any_composite": ("low_any_composite", True),
    "conc_composite": ("low_conc_composite", True),
}
KEEP = [
    "ip_composite_k1", "ip_composite_k2", "ip_composite_k3",
    "ip_map_k2", "ip_sbp_k2", "np_composite_k2",
    "any_composite_k2", "conc_composite_k2",
]


def add_flags(b):
    inv_obs = (b.inv_map_n > 0) | (b.inv_sbp_n > 0)
    nibp_obs = (b.nibp_map_n > 0) | (b.nibp_sbp_n > 0)
    inv_low = ((b.inv_map_n > 0) & (b.inv_map_mean < 65)) | ((b.inv_sbp_n > 0) & (b.inv_sbp_mean < 90))
    nibp_low = ((b.nibp_map_n > 0) & (b.nibp_map_mean < 65)) | ((b.nibp_sbp_n > 0) & (b.nibp_sbp_mean < 90))
    map_ip = np.where(b.inv_map_n > 0, b.inv_map_mean, b.nibp_map_mean)
    sbp_ip = np.where(b.inv_sbp_n > 0, b.inv_sbp_mean, b.nibp_sbp_mean)
    map_np = np.where(b.nibp_map_n > 0, b.nibp_map_mean, b.inv_map_mean)
    sbp_np = np.where(b.nibp_sbp_n > 0, b.nibp_sbp_mean, b.inv_sbp_mean)
    b = b.copy()
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


def outcome_table(bins):
    post = add_flags(bins[bins.hour_bin >= 0]).sort_values(["id", "hour_bin"])
    rows = []
    for stay, g in post.groupby("id", sort=False):
        hours = g.hour_bin.to_numpy()
        brady = hours[g.brady.to_numpy()]
        rec = {"id": int(stay)}
        for fam, (col, with_brady) in FAMILIES.items():
            low = hours[g[col].to_numpy()]
            for k in (1, 2, 3):
                flag = has_run(low, k)
                if with_brady:
                    flag = flag or has_run(brady, k)
                rec[f"{fam}_k{k}"] = int(flag)
        rows.append(rec)
    return pd.DataFrame(rows)


def auc(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def delong(y, x1, x2):
    """Paired DeLong test for AUC difference (fast version for 0/1 tests)."""
    y = np.asarray(y, dtype=int)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def midrank(a):
        return pd.Series(a).rank(method="average").to_numpy()

    def comp(z):
        tx = midrank(z)
        tz = midrank(z[pos])
        ty = midrank(z[neg])
        return (tx[pos] - tz) / n, 1.0 - (tx[neg] - ty) / m

    v1, v2 = comp(x1)
    w1, w2 = comp(x2)
    auc1, auc2 = v1.mean(), w1.mean()
    s = np.cov(np.vstack([v1, w1])) / m + np.cov(np.vstack([v2, w2])) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return auc1, auc2, np.nan, np.nan
    z = (auc1 - auc2) / np.sqrt(var)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return auc1, auc2, z, p


def auc_ci(y, x):
    """DeLong variance for a single AUC."""
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan
    tx = pd.Series(x).rank(method="average").to_numpy()
    tz = pd.Series(x[pos]).rank(method="average").to_numpy()
    ty = pd.Series(x[neg]).rank(method="average").to_numpy()
    v1 = (tx[pos] - tz) / n
    v2 = 1.0 - (tx[neg] - ty) / m
    a = v1.mean()
    se = np.sqrt(np.var(v1, ddof=1) / m + np.var(v2, ddof=1) / n)
    return a, a - 1.96 * se, a + 1.96 * se


def main():
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    hourly = load("hourly", ROOT / "work" / "harmonized_hourly_outcome.py")
    base = hourly.load_base_module()

    rows = []
    for db, prefix in (("eICU-CRD", "eicu"), ("MIMIC-IV", "mimic")):
        frame = pd.read_csv(DATA / f"{prefix}_prepared_analysis_frame.csv")
        bins = pd.read_csv(DATA / f"{prefix}_harmonized_bins_v4.csv")
        # identical cohort to the published analysis: encode_analysis drops stays
        # with missing propensity covariates and applies the eICU hospital rule
        work, _ = base.encode_analysis(
            frame.copy(), db, restrict_eicu_hospitals=True, include_hospital_effects=True
        )
        cohort = frame[frame.id.isin(work.id)].copy()
        ids = cohort["id"]
        out = outcome_table(bins).set_index("id").reindex(ids).fillna(0).astype(int)
        df = pd.concat([cohort.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        print(f"{db}: n={len(df)}, dex={(df.treatment=='dexmedetomidine').sum()}")

        for outcome in ("icu_mortality", "hospital_mortality", "new_vasopressor_24h"):
            y = df[outcome].to_numpy()
            print(f"  -- {outcome} (events={int(y.sum())})")
            base_x = df["ip_composite_k2"].to_numpy()
            for d in KEEP:
                a, lo, hi = auc_ci(y, df[d].to_numpy())
                if d == "ip_composite_k2":
                    print(f"     {d:20s} AUC={a:.4f} ({lo:.4f} to {hi:.4f}) (reference)")
                    rows.append(dict(database=db, outcome=outcome, definition=d, auc=a, lo=lo, hi=hi, p_vs_primary=1.0))
                else:
                    _, _, z, p = delong(y, df[d].to_numpy(), base_x)
                    print(f"     {d:20s} AUC={a:.4f} ({lo:.4f} to {hi:.4f})  vs k=2: z={z:+.2f} p={p:.3f}")
                    rows.append(dict(database=db, outcome=outcome, definition=d, auc=a, lo=lo, hi=hi, p_vs_primary=p))
        # prevalence by arm
        print("  -- prevalence by arm (%)")
        for d in KEEP:
            pr = df.groupby("treatment")[d].mean().mul(100).round(1).to_dict()
            print(f"     {d:20s} {pr}")
            for arm, v in pr.items():
                rows.append(dict(database=db, outcome=f"prevalence_{arm}", definition=d, auc=v))
    pd.DataFrame(rows).to_csv(OUT / "criterion_validity.csv", index=False)


if __name__ == "__main__":
    main()
