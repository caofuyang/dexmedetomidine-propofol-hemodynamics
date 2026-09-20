#!/usr/bin/env python3
"""Refresh MIMIC-IV weight from the initial study-drug order at time zero."""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUT = ROOT / "outputs" / "active_comparator_sedation"
DB = Path(os.environ.get("SEDATION_DB_ROOT", "/path/to/deidentified_databases")).expanduser().resolve()
INPUTEVENTS = DB / "MIMIC-IV" / "data" / "mimic-iv-3.1" / "icu" / "inputevents.csv.gz"
PROP = {"222168"}
DEX = {"225150", "229420"}


def main():
    cache_path = OUT / "mimic_active_comparator_patient_level_v2.csv"
    cohort = pd.read_csv(cache_path)
    cohort["stay_key"] = cohort.id.astype("int64").astype(str)
    cohort["index_time"] = (
        pd.to_datetime(cohort.intime)
        + pd.to_timedelta(cohort.index_hour, unit="h")
    )
    index_time = dict(zip(cohort.stay_key, cohort.index_time))
    treatment = dict(zip(cohort.stay_key, cohort.treatment))
    weights = {}

    usecols = [
        "stay_id", "starttime", "endtime", "itemid", "patientweight",
        "rate", "ordercategorydescription", "statusdescription",
    ]
    reader = pd.read_csv(
        INPUTEVENTS, compression="gzip", usecols=usecols,
        dtype={"stay_id": "string", "itemid": "string"}, chunksize=500_000,
    )
    for chunk_no, chunk in enumerate(reader, 1):
        chunk = chunk[chunk.stay_id.isin(index_time)].copy()
        if chunk.empty:
            continue
        chunk = chunk[chunk.statusdescription.ne("Rewritten")]
        chunk["starttime"] = pd.to_datetime(chunk.starttime, errors="coerce")
        chunk["endtime"] = pd.to_datetime(chunk.endtime, errors="coerce")
        chunk["patientweight"] = pd.to_numeric(chunk.patientweight, errors="coerce")
        chunk["rate"] = pd.to_numeric(chunk.rate, errors="coerce")
        chunk["index_time"] = chunk.stay_id.map(index_time)
        chunk["treatment"] = chunk.stay_id.map(treatment)
        correct_item = (
            (chunk.treatment.eq("propofol") & chunk.itemid.isin(PROP))
            | (chunk.treatment.eq("dexmedetomidine") & chunk.itemid.isin(DEX))
        )
        valid = (
            correct_item
            & chunk.ordercategorydescription.eq("Continuous Med")
            & chunk.rate.gt(0)
            & chunk.endtime.gt(chunk.starttime)
            & chunk.patientweight.between(20, 400)
            & (chunk.starttime.sub(chunk.index_time).abs().dt.total_seconds() <= 1)
        )
        for row in chunk.loc[valid, ["stay_id", "patientweight"]].itertuples(index=False):
            weights.setdefault(row.stay_id, float(row.patientweight))
        if chunk_no % 10 == 0:
            print(f"inputevents chunks scanned: {chunk_no}", flush=True)

    old_weight = cohort.weight.copy()
    cohort["weight"] = cohort.stay_key.map(weights)
    changed = (
        old_weight.notna() & cohort.weight.notna()
        & old_weight.sub(cohort.weight).abs().gt(1e-9)
    )
    print(f"index-order weights available: {cohort.weight.notna().sum()} / {len(cohort)}")
    print(f"weights changed: {changed.sum()}")
    print(f"newly missing: {(old_weight.notna() & cohort.weight.isna()).sum()}")
    cohort.drop(columns=["stay_key", "index_time"]).to_csv(
        cache_path, index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
