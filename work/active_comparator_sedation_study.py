#!/usr/bin/env python3
"""Active-comparator study of dexmedetomidine versus propofol in eICU and MIMIC-IV."""

from __future__ import annotations

import os

import csv
import gzip
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
DB = Path(os.environ.get("SEDATION_DB_ROOT", "/path/to/deidentified_databases")).expanduser().resolve()
EICU = DB / "eICU-CRD" / "data" / "eicu-collaborative-research-database-2.0"
MIMIC = DB / "MIMIC-IV" / "data" / "mimic-iv-3.1"
OUT = ROOT / "outputs" / "active_comparator_sedation"

VASO_TERMS = ("norepinephrine", "levophed", "epinephrine", "phenylephrine", "vasopressin", "dopamine")
MIMIC_PROP = {"222168"}
MIMIC_DEX = {"225150", "229420"}
MIMIC_VASO = {"221289", "221749", "221906", "222315", "229617", "229630", "229631", "229632"}
MIMIC_HR = {"220045"}
MIMIC_SBP = {"220050", "220179"}
MIMIC_MAP = {"220052", "220181"}
MIMIC_RASS = {"228096"}
MIMIC_MIDAZOLAM = {"221668"}
MIMIC_OPIOID = {"221744", "221833", "225154", "225942", "225972"}
MIMIC_KETAMINE = {"221712", "227211"}


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    return open(path, "r", newline="", encoding="utf-8", errors="replace")


def fnum(value):
    try:
        value = float(str(value).strip())
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_dt(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def age_value(value):
    text = str(value or "").strip()
    if text.startswith(">"):
        return 90.0
    return fnum(text)


def vitals_blank():
    return {
        "baseline_bp_times": set(), "baseline_low_times": set(),
        "baseline_map_by_time": {}, "baseline_hr_by_time": {},
        "baseline_brady_times": set(), "baseline_rass_by_time": {},
        "post_bp_times": set(), "post_low_times": set(),
        "post_severe_low_times": set(), "post_hr_times": set(),
        "post_brady_times": set(), "min_map_24h": np.nan,
    }


def update_vitals(state, rel_h, hr=np.nan, sbp=np.nan, mapv=np.nan, rass=np.nan):
    if not np.isfinite(rel_h):
        return
    minute = int(round(rel_h * 60))
    valid_hr = np.isfinite(hr) and 20 <= hr <= 250
    valid_sbp = np.isfinite(sbp) and 30 <= sbp <= 300
    valid_map = np.isfinite(mapv) and 20 <= mapv <= 200
    valid_rass = np.isfinite(rass) and -5 <= rass <= 4
    low = (valid_map and mapv < 65) or (valid_sbp and sbp < 90)
    severe = valid_map and mapv < 55
    if -6 <= rel_h < 0:
        if valid_sbp or valid_map:
            state["baseline_bp_times"].add(minute)
            if low:
                state["baseline_low_times"].add(minute)
        if valid_map:
            state["baseline_map_by_time"][minute] = mapv
        if valid_hr:
            state["baseline_hr_by_time"][minute] = hr
            if hr < 50:
                state["baseline_brady_times"].add(minute)
        if valid_rass:
            state["baseline_rass_by_time"][minute] = rass
    elif 0 < rel_h <= 24:
        if valid_sbp or valid_map:
            state["post_bp_times"].add(minute)
            if low:
                state["post_low_times"].add(minute)
            if severe:
                state["post_severe_low_times"].add(minute)
        if valid_map:
            state["min_map_24h"] = mapv if not np.isfinite(state["min_map_24h"]) else min(state["min_map_24h"], mapv)
        if valid_hr:
            state["post_hr_times"].add(minute)
            if hr < 50:
                state["post_brady_times"].add(minute)


def repeated_within_window(times, minimum_separation_minutes=15, maximum_separation_minutes=60):
    """Return whether two abnormal observations recur within a clinically local window."""
    ordered = sorted(set(times))
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            gap = second - first
            if gap > maximum_separation_minutes:
                break
            if gap >= minimum_separation_minutes:
                return True
    return False


def repeated_over_time(times, minimum_separation_minutes=15):
    """Broad sensitivity definition: two events at least 15 minutes apart in 24 h."""
    return len(times) >= 2 and max(times) - min(times) >= minimum_separation_minutes


def finalize_vitals(row, state):
    row["baseline_bp_n"] = len(state["baseline_bp_times"])
    row["baseline_low_n"] = len(state["baseline_low_times"])
    row["baseline_map_n"] = len(state["baseline_map_by_time"])
    row["baseline_hr_n"] = len(state["baseline_hr_by_time"])
    row["baseline_brady_n"] = len(state["baseline_brady_times"])
    row["post_bp_n"] = len(state["post_bp_times"])
    row["post_low_n"] = len(state["post_low_times"])
    row["post_severe_low_n"] = len(state["post_severe_low_times"])
    row["post_hr_n"] = len(state["post_hr_times"])
    row["post_brady_n"] = len(state["post_brady_times"])
    row["min_map_24h"] = state["min_map_24h"]
    row["baseline_map"] = np.mean(list(state["baseline_map_by_time"].values())) if state["baseline_map_by_time"] else np.nan
    row["baseline_hr"] = np.mean(list(state["baseline_hr_by_time"].values())) if state["baseline_hr_by_time"] else np.nan
    row["baseline_rass"] = (
        state["baseline_rass_by_time"][max(state["baseline_rass_by_time"])]
        if state["baseline_rass_by_time"] else np.nan
    )
    row["sustained_hypotension_24h"] = int(repeated_within_window(state["post_low_times"]))
    row["severe_hypotension_24h"] = int(repeated_within_window(state["post_severe_low_times"]))
    row["bradycardia_24h"] = int(repeated_within_window(state["post_brady_times"]))
    row["recurrent_hypotension_24h"] = int(repeated_over_time(state["post_low_times"]))
    row["recurrent_bradycardia_24h"] = int(repeated_over_time(state["post_brady_times"]))
    row["recurrent_instability_24h"] = int(row["recurrent_hypotension_24h"] or row["recurrent_bradycardia_24h"])
    row["hypotension_burden"] = row["post_low_n"] / row["post_bp_n"] if row["post_bp_n"] else np.nan
    row["stable_baseline"] = int(
        row["baseline_bp_n"] >= 1 and row["baseline_hr_n"] >= 1
        and row["baseline_low_n"] == 0 and row["baseline_brady_n"] == 0
        and not row["baseline_vasopressor"]
    )
    # The primary composite includes both hypotension and bradycardia, so both
    # blood pressure and heart rate must be observable after treatment start.
    row["evaluable"] = int(
        row["stable_baseline"]
        and row["post_bp_n"] >= 2
        and row["post_hr_n"] >= 2
    )
    row["hemodynamic_instability_24h"] = int(row["sustained_hypotension_24h"] or row["bradycardia_24h"])


def choose_strategy(prop_time, dex_time, concurrent_window_hours=0.5):
    prop = fnum(prop_time)
    dex = fnum(dex_time)
    # Starts exactly at the 30-minute boundary are considered concurrent.
    tolerance = 1e-12
    if np.isfinite(prop) and (
        not np.isfinite(dex) or dex - prop > concurrent_window_hours + tolerance
    ):
        return "propofol", prop
    if np.isfinite(dex) and (
        not np.isfinite(prop) or prop - dex > concurrent_window_hours + tolerance
    ):
        return "dexmedetomidine", dex
    return None, np.nan


def enforce_current_definitions(df):
    """Reapply current eligibility rules when loading a patient-level cache."""
    prop = pd.to_numeric(df["prop_time"], errors="coerce")
    dex = pd.to_numeric(df["dex_time"], errors="coerce")
    concurrent = prop.notna() & dex.notna() & ((prop - dex).abs() <= 0.5 + 1e-12)
    current = df.loc[~concurrent].copy()
    current["stable_baseline"] = (
        current["baseline_bp_n"].ge(1)
        & current["baseline_hr_n"].ge(1)
        & current["baseline_low_n"].eq(0)
        & current.get("baseline_brady_n", pd.Series(np.nan, index=current.index)).eq(0)
        & current["baseline_vasopressor"].eq(0)
    ).astype(int)
    current["evaluable"] = (
        current["stable_baseline"].eq(1)
        & current["post_bp_n"].ge(2)
        & current["post_hr_n"].ge(2)
    ).astype(int)
    return current


def refresh_eicu_baseline_ventilation(df):
    """Mark invasive ventilation that is active at treatment initiation."""
    index_minutes = dict(zip(df["id"].astype("int64").astype(str), df["index_hour"] * 60))
    active = set()
    usecols = [
        "patientunitstayid", "ventstartoffset", "ventendoffset",
        "priorventstartoffset", "priorventendoffset",
    ]
    reader = pd.read_csv(
        EICU / "respiratoryCare.csv.gz", compression="gzip", usecols=usecols,
        dtype={"patientunitstayid": "string"}, chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk[chunk.patientunitstayid.isin(index_minutes)].copy()
        if chunk.empty:
            continue
        chunk["index_minute"] = chunk.patientunitstayid.map(index_minutes)
        current = (
            chunk.ventstartoffset.notna()
            & chunk.ventstartoffset.le(chunk.index_minute)
            & (
                chunk.ventendoffset.isna()
                | chunk.ventendoffset.le(chunk.ventstartoffset)
                | chunk.ventendoffset.ge(chunk.index_minute)
            )
        )
        prior = (
            chunk.priorventstartoffset.notna()
            & chunk.priorventstartoffset.le(chunk.index_minute)
            & (
                chunk.priorventendoffset.isna()
                | chunk.priorventendoffset.le(chunk.priorventstartoffset)
                | chunk.priorventendoffset.ge(chunk.index_minute)
            )
        )
        active.update(chunk.loc[current | prior, "patientunitstayid"].tolist())
    refreshed = df.copy()
    refreshed["vent"] = refreshed["id"].astype("int64").astype(str).isin(active).astype(int)
    return refreshed


def refresh_mimic_baseline_ventilation(df):
    """Mark invasive ventilation whose recorded interval contains time zero."""
    ids = df["id"].astype("int64").astype(str)
    index_times = dict(zip(
        ids,
        (pd.to_datetime(df["intime"]) + pd.to_timedelta(df["index_hour"], unit="h")).to_numpy(),
    ))
    active = set()
    reader = pd.read_csv(
        MIMIC / "icu" / "procedureevents.csv.gz", compression="gzip",
        usecols=["stay_id", "itemid", "starttime", "endtime"],
        dtype={"stay_id": "string", "itemid": "string"}, chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk[chunk.stay_id.isin(index_times) & chunk.itemid.eq("225792")].copy()
        if chunk.empty:
            continue
        chunk["index_time"] = chunk.stay_id.map(index_times)
        chunk["starttime"] = pd.to_datetime(chunk.starttime, errors="coerce")
        chunk["endtime"] = pd.to_datetime(chunk.endtime, errors="coerce")
        at_index = chunk.starttime.le(chunk.index_time) & chunk.endtime.gt(chunk.index_time)
        active.update(chunk.loc[at_index, "stay_id"].tolist())
    refreshed = df.copy()
    refreshed["vent"] = refreshed["id"].astype("int64").astype(str).isin(active).astype(int)
    return refreshed


def build_eicu():
    raw = {}
    first = {}
    with open_text(EICU / "patient.csv.gz") as f:
        for x in csv.DictReader(f):
            visit = int(fnum(x.get("unitvisitnumber"))) if np.isfinite(fnum(x.get("unitvisitnumber"))) else 999
            person = x.get("uniquepid", "")
            if person not in first or visit < first[person][0]:
                first[person] = (visit, x["patientunitstayid"])
            raw[x["patientunitstayid"]] = {
                "database": "eICU-CRD", "id": x["patientunitstayid"], "person_id": person,
                "hospital_id": x.get("hospitalid", ""), "age": age_value(x.get("age")),
                "female": int(str(x.get("gender", "")).lower().startswith("f")),
                "weight": fnum(x.get("admissionweight")), "unit_type": x.get("unittype", ""),
                "admission_source": x.get("unitadmitsource", ""),
                "icu_los_days": fnum(x.get("unitdischargeoffset")) / 1440,
                "icu_mortality": int(str(x.get("unitdischargestatus", "")).lower() == "expired"),
                "hospital_mortality": int(str(x.get("hospitaldischargestatus", "")).lower() == "expired"),
                "prop_time": np.nan, "dex_time": np.nan, "baseline_vasopressor": 0,
                "new_vasopressor_24h": 0, "vent": 0, "intubated": 0,
                "baseline_midazolam": 0, "baseline_opioid": 0, "baseline_ketamine": 0,
                "apache": np.nan, "aps": np.nan,
            }
    first_stays = {stay for _, stay in first.values()}
    with open_text(EICU / "apachePatientResult.csv.gz") as f:
        for x in csv.DictReader(f):
            row = raw.get(x["patientunitstayid"])
            if row is not None:
                score = fnum(x.get("apachescore")); aps = fnum(x.get("acutephysiologyscore"))
                if np.isfinite(score): row["apache"] = score
                if np.isfinite(aps): row["aps"] = aps
    with open_text(EICU / "apacheApsVar.csv.gz") as f:
        for x in csv.DictReader(f):
            row = raw.get(x["patientunitstayid"])
            if row is not None:
                row["vent"] = int(x.get("vent") == "1")
                row["intubated"] = int(x.get("intubated") == "1")
    infusion = defaultdict(list)
    other_sedatives = defaultdict(list)
    with open_text(EICU / "infusionDrug.csv.gz") as f:
        for x in csv.DictReader(f):
            stay = x["patientunitstayid"]
            if stay not in first_stays:
                continue
            name = (x.get("drugname") or "").lower()
            offset = fnum(x.get("infusionoffset")) / 60
            rate = fnum(x.get("drugrate"))
            if not np.isfinite(offset) or not np.isfinite(rate) or rate <= 0:
                continue
            if "propofol" in name and 0 <= offset <= 24:
                raw[stay]["prop_time"] = min(raw[stay]["prop_time"], offset) if np.isfinite(raw[stay]["prop_time"]) else offset
            elif ("dexmedetomidine" in name or "precedex" in name) and 0 <= offset <= 24:
                raw[stay]["dex_time"] = min(raw[stay]["dex_time"], offset) if np.isfinite(raw[stay]["dex_time"]) else offset
            if any(term in name for term in VASO_TERMS):
                infusion[stay].append(offset)
            if "midazolam" in name or "versed" in name:
                other_sedatives[stay].append(("baseline_midazolam", offset))
            if any(term in name for term in ("fentanyl", "morphine", "hydromorphone", "dilaudid")):
                other_sedatives[stay].append(("baseline_opioid", offset))
            if "ketamine" in name:
                other_sedatives[stay].append(("baseline_ketamine", offset))
    cohort = {}
    for stay in first_stays:
        row = raw[stay]
        strategy, index = choose_strategy(row["prop_time"], row["dex_time"])
        if strategy is None:
            continue
        row["treatment"] = strategy; row["dex"] = int(strategy == "dexmedetomidine"); row["index_hour"] = index
        for when in infusion.get(stay, []):
            # A vasopressor first charted exactly at sedation initiation is
            # treated as baseline instability rather than left unclassified.
            if -6 <= when - index <= 0: row["baseline_vasopressor"] = 1
            if 0 < when - index <= 24: row["new_vasopressor_24h"] = 1
        for field, when in other_sedatives.get(stay, []):
            if -6 <= when - index < 0:
                row[field] = 1
        row["crossover_6h"] = int(
            np.isfinite(row["prop_time"]) and np.isfinite(row["dex_time"])
            and max(row["prop_time"], row["dex_time"]) - index <= 6
        )
        row["crossover_24h"] = int(
            np.isfinite(row["prop_time"]) and np.isfinite(row["dex_time"])
            and max(row["prop_time"], row["dex_time"]) - index <= 24
        )
        cohort[stay] = row
    states = {stay: vitals_blank() for stay in cohort}
    with open_text(EICU / "vitalPeriodic.csv.gz") as f:
        for i, x in enumerate(csv.DictReader(f), 1):
            stay = x["patientunitstayid"]
            if stay not in states:
                continue
            rel = fnum(x.get("observationoffset")) / 60 - cohort[stay]["index_hour"]
            update_vitals(states[stay], rel, fnum(x.get("heartrate")), fnum(x.get("systemicsystolic")), fnum(x.get("systemicmean")))
            if i % 50_000_000 == 0: print(f"[eICU] periodic rows {i:,}")
    with open_text(EICU / "vitalAperiodic.csv.gz") as f:
        for x in csv.DictReader(f):
            stay = x["patientunitstayid"]
            if stay not in states:
                continue
            rel = fnum(x.get("observationoffset")) / 60 - cohort[stay]["index_hour"]
            update_vitals(states[stay], rel, sbp=fnum(x.get("noninvasivesystolic")), mapv=fnum(x.get("noninvasivemean")))
    rows = []
    for stay, row in cohort.items():
        finalize_vitals(row, states[stay]); rows.append(row)
    return pd.DataFrame(rows)


def build_mimic():
    stays = {}
    first = {}
    with open_text(MIMIC / "icu" / "icustays.csv.gz") as f:
        for x in csv.DictReader(f):
            intime = parse_dt(x.get("intime")); person = x["subject_id"]
            if person not in first or intime < first[person][0]: first[person] = (intime, x["stay_id"])
            stays[x["stay_id"]] = {
                "database": "MIMIC-IV", "id": x["stay_id"], "person_id": person,
                "hadm_id": x["hadm_id"], "intime": intime, "outtime": parse_dt(x.get("outtime")),
                "unit_type": x.get("first_careunit", ""), "icu_los_days": fnum(x.get("los")),
                "prop_time": np.nan, "dex_time": np.nan,
                "prop_weight": np.nan, "dex_weight": np.nan, "weight": np.nan,
                "baseline_vasopressor": 0, "new_vasopressor_24h": 0, "vent": 0,
                "baseline_midazolam": 0, "baseline_opioid": 0, "baseline_ketamine": 0,
            }
    first_stays = {stay for _, stay in first.values()}
    patient_info = {}
    with open_text(MIMIC / "hosp" / "patients.csv.gz") as f:
        for x in csv.DictReader(f): patient_info[x["subject_id"]] = x
    admission_info = {}
    with open_text(MIMIC / "hosp" / "admissions.csv.gz") as f:
        for x in csv.DictReader(f): admission_info[x["hadm_id"]] = x
    events = defaultdict(list)
    other_sedatives = defaultdict(list)
    with open_text(MIMIC / "icu" / "inputevents.csv.gz") as f:
        for x in csv.DictReader(f):
            stay = x["stay_id"]
            if stay not in first_stays or x.get("statusdescription") == "Rewritten": continue
            item = x.get("itemid"); start = parse_dt(x.get("starttime")); end = parse_dt(x.get("endtime"))
            if not start: continue
            offset = (start - stays[stay]["intime"]).total_seconds() / 3600
            weight = fnum(x.get("patientweight"))
            rate = fnum(x.get("rate"))
            continuous = (
                x.get("ordercategorydescription") == "Continuous Med"
                and np.isfinite(rate) and rate > 0
                and end is not None and end > start
            )
            if continuous and item in MIMIC_PROP and 0 <= offset <= 24:
                if (
                    not np.isfinite(stays[stay]["prop_time"])
                    or offset < stays[stay]["prop_time"]
                    or (
                        abs(offset - stays[stay]["prop_time"]) <= 1e-12
                        and not np.isfinite(stays[stay]["prop_weight"])
                        and np.isfinite(weight) and 20 <= weight <= 400
                    )
                ):
                    stays[stay]["prop_time"] = offset
                    stays[stay]["prop_weight"] = weight if np.isfinite(weight) and 20 <= weight <= 400 else np.nan
            elif continuous and item in MIMIC_DEX and 0 <= offset <= 24:
                if (
                    not np.isfinite(stays[stay]["dex_time"])
                    or offset < stays[stay]["dex_time"]
                    or (
                        abs(offset - stays[stay]["dex_time"]) <= 1e-12
                        and not np.isfinite(stays[stay]["dex_weight"])
                        and np.isfinite(weight) and 20 <= weight <= 400
                    )
                ):
                    stays[stay]["dex_time"] = offset
                    stays[stay]["dex_weight"] = weight if np.isfinite(weight) and 20 <= weight <= 400 else np.nan
            if continuous and item in MIMIC_VASO:
                end_offset = (end - stays[stay]["intime"]).total_seconds() / 3600
                events[stay].append((offset, end_offset))
            if continuous and item in (MIMIC_MIDAZOLAM | MIMIC_OPIOID | MIMIC_KETAMINE):
                end_offset = (end - stays[stay]["intime"]).total_seconds() / 3600
                field = (
                    "baseline_midazolam" if item in MIMIC_MIDAZOLAM else
                    "baseline_opioid" if item in MIMIC_OPIOID else "baseline_ketamine"
                )
                other_sedatives[stay].append((field, offset, end_offset))
    with open_text(MIMIC / "icu" / "procedureevents.csv.gz") as f:
        for x in csv.DictReader(f):
            stay = x["stay_id"]
            if stay not in first_stays or x.get("itemid") != "225792": continue
            start = parse_dt(x.get("starttime")); end = parse_dt(x.get("endtime"))
            if start and end:
                sh = (start - stays[stay]["intime"]).total_seconds() / 3600
                eh = (end - stays[stay]["intime"]).total_seconds() / 3600
                if sh <= 24 and eh >= 0: stays[stay]["vent"] = 1
    cohort = {}
    for stay in first_stays:
        row = stays[stay]; strategy, index = choose_strategy(row["prop_time"], row["dex_time"])
        if strategy is None: continue
        row["weight"] = row.pop("dex_weight" if strategy == "dexmedetomidine" else "prop_weight")
        row.pop("prop_weight", None)
        row.pop("dex_weight", None)
        p = patient_info.get(row["person_id"], {}); a = admission_info.get(row["hadm_id"], {})
        death_dt = parse_dt((p.get("dod") or "") + " 00:00:00")
        anchor_age = fnum(p.get("anchor_age"))
        anchor_year = fnum(p.get("anchor_year"))
        admit_time = parse_dt(a.get("admittime"))
        age = anchor_age
        if np.isfinite(anchor_age) and np.isfinite(anchor_year) and admit_time is not None:
            age = min(anchor_age + admit_time.year - anchor_year, 90)
        row.update({
            "treatment": strategy, "dex": int(strategy == "dexmedetomidine"), "index_hour": index,
            "age": age, "female": int(p.get("gender") == "F"),
            "anchor_year_group": p.get("anchor_year_group", "Unknown"),
            "admission_source": a.get("admission_location", ""),
            "hospital_mortality": int(a.get("hospital_expire_flag", "0") == "1"),
            "icu_mortality": int(death_dt is not None and row["outtime"] is not None and death_dt <= row["outtime"]),
        })
        for start_offset, end_offset in events.get(stay, []):
            if start_offset <= index and end_offset > index - 6:
                row["baseline_vasopressor"] = 1
            if 0 < start_offset - index <= 24:
                row["new_vasopressor_24h"] = 1
        for field, start_offset, end_offset in other_sedatives.get(stay, []):
            if start_offset < index and end_offset > index - 6:
                row[field] = 1
        row["crossover_6h"] = int(
            np.isfinite(row["prop_time"]) and np.isfinite(row["dex_time"])
            and max(row["prop_time"], row["dex_time"]) - index <= 6
        )
        row["crossover_24h"] = int(
            np.isfinite(row["prop_time"]) and np.isfinite(row["dex_time"])
            and max(row["prop_time"], row["dex_time"]) - index <= 24
        )
        cohort[stay] = row
    states = {stay: vitals_blank() for stay in cohort}
    wanted = MIMIC_HR | MIMIC_SBP | MIMIC_MAP | MIMIC_RASS
    # MIMIC timestamps are timezone-naive. datetime.timestamp() would apply the
    # host timezone and shift every chart event relative to treatment initiation.
    index_times = {
        stay: pd.Timestamp(row["intime"]).value / 1e9 + row["index_hour"] * 3600
        for stay, row in cohort.items()
    }
    path = MIMIC / "icu" / "chartevents.csv.gz"
    reader = pd.read_csv(
        path, compression="gzip", usecols=["stay_id", "charttime", "itemid", "valuenum"],
        dtype={"stay_id": "string", "itemid": "string", "valuenum": "float64"},
        chunksize=2_000_000,
    )
    for chunk_no, chunk in enumerate(reader, 1):
        chunk = chunk[chunk.itemid.isin(wanted) & chunk.stay_id.isin(states)]
        if chunk.empty:
            continue
        chart_seconds = pd.to_datetime(chunk.charttime, errors="coerce").astype("int64") / 1e9
        chunk = chunk.assign(rel_h=(chart_seconds - chunk.stay_id.map(index_times)) / 3600)
        for x in chunk.itertuples(index=False):
            update_vitals(
                states[x.stay_id], x.rel_h,
                hr=x.valuenum if x.itemid in MIMIC_HR else np.nan,
                sbp=x.valuenum if x.itemid in MIMIC_SBP else np.nan,
                mapv=x.valuenum if x.itemid in MIMIC_MAP else np.nan,
                rass=x.valuenum if x.itemid in MIMIC_RASS else np.nan,
            )
        if chunk_no % 10 == 0:
            print(f"[MIMIC] chartevents chunks scanned: {chunk_no}")
    rows = []
    for stay, row in cohort.items():
        finalize_vitals(row, states[stay]); rows.append(row)
    return pd.DataFrame(rows)


def logistic_fit(X, y, weights=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(X)), X]); beta = np.zeros(X.shape[1])
    base_w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = np.maximum(p * (1 - p) * base_w, 1e-9)
        h = X.T @ (X * w[:, None]) + np.eye(X.shape[1]) * 1e-7
        step = np.linalg.solve(h, X.T @ ((y - p) * base_w))
        beta += step
        if np.max(np.abs(step)) < 1e-8: break
    return beta, 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))


def encode_analysis(
    df, database, restrict_eicu_hospitals=True, include_hospital_effects=True,
    enriched_ps=False, extra_covariates=None,
):
    work = df[df.evaluable == 1].copy()
    work["age10"] = work.age / 10; work["weight10"] = work.weight / 10
    work["map10"] = work.baseline_map / 10; work["hr10"] = work.baseline_hr / 10
    work["index6"] = work.index_hour / 6
    work["bp_obs_log"] = np.log1p(work.baseline_bp_n)
    work["hr_obs_log"] = np.log1p(work.baseline_hr_n)
    cols = [
        "age10", "female", "weight10", "map10", "hr10", "index6", "vent",
        "bp_obs_log", "hr_obs_log", "baseline_midazolam", "baseline_opioid",
        "baseline_ketamine",
    ]
    if extra_covariates:
        cols.extend(extra_covariates)
    if enriched_ps:
        continuous = ["age10", "weight10", "map10", "hr10", "index6", "bp_obs_log", "hr_obs_log"]
        for col in continuous:
            squared = f"{col}_sq"
            work[squared] = work[col] ** 2
            cols.append(squared)
        interactions = {
            "map10_x_hr10": work.map10 * work.hr10,
            "vent_x_index6": work.vent * work.index6,
            "vent_x_bp_obs_log": work.vent * work.bp_obs_log,
            "vent_x_hr_obs_log": work.vent * work.hr_obs_log,
        }
        for name, values in interactions.items():
            work[name] = values
            cols.append(name)
    work = work.dropna(subset=["dex", *cols]).copy()
    if database == "eICU-CRD" and restrict_eicu_hospitals:
        treatment_count = work.groupby("hospital_id")["dex"].nunique()
        work = work[work.hospital_id.isin(treatment_count[treatment_count == 2].index)].copy()
    categorical_cols = []
    sources = [("unit_type", 50), ("admission_source", 50)]
    if database == "eICU-CRD" and include_hospital_effects:
        sources.append(("hospital_id", 1))
    if database == "MIMIC-IV":
        sources.append(("anchor_year_group", 1))
    for source, minimum_count in sources:
        values = work[source].fillna("Unknown").astype(str)
        counts = values.value_counts()
        values = values.where(values.map(counts) >= minimum_count, "Other")
        dummies = pd.get_dummies(values, prefix=source, drop_first=True, dtype=float)
        work = pd.concat([work, dummies], axis=1)
        categorical_cols.extend(dummies.columns.tolist())
    cols.extend(categorical_cols)
    means = work[cols].mean(); sds = work[cols].std().replace(0, 1)
    X = (work[cols] - means) / sds
    _, ps = logistic_fit(X.to_numpy(), work.dex.to_numpy())
    # Overlap weights are inherently bounded, so probability truncation is not
    # needed and would prevent the exact mean-balance property of the method.
    work["ps"] = np.clip(ps, 1e-6, 1 - 1e-6)
    work["overlap_weight"] = np.where(work.dex == 1, 1 - work.ps, work.ps)
    return work, cols


def weighted_risk(work, outcome):
    rows = []
    risks = {}
    variances = {}
    for dex, name in [(0, "propofol"), (1, "dexmedetomidine")]:
        g = work[work.dex == dex]; w = g.overlap_weight.to_numpy(); y = g[outcome].to_numpy(float)
        risk = np.sum(w * y) / np.sum(w); risks[dex] = risk
        variance = np.sum((w ** 2) * ((y - risk) ** 2)) / (np.sum(w) ** 2)
        variances[dex] = variance
        rows.append({"treatment": name, "weighted_events_percent": risk * 100, "effective_n": np.sum(w) ** 2 / np.sum(w ** 2)})
    rd = risks[1] - risks[0]; rr = risks[1] / risks[0] if risks[0] else np.nan
    rd_se = math.sqrt(variances[1] + variances[0])
    rd_ci = (rd - 1.96 * rd_se, rd + 1.96 * rd_se)
    if risks[0] > 0 and risks[1] > 0:
        log_rr_se = math.sqrt(variances[1] / risks[1] ** 2 + variances[0] / risks[0] ** 2)
        rr_ci = (math.exp(math.log(rr) - 1.96 * log_rr_se), math.exp(math.log(rr) + 1.96 * log_rr_se))
    else:
        rr_ci = (np.nan, np.nan)
    return rows, rd, rd_ci, rr, rr_ci


def smd_rows(work, cols, database):
    rows = []
    for col in cols:
        a = work[work.dex == 1]; b = work[work.dex == 0]
        pooled = math.sqrt((a[col].var() + b[col].var()) / 2)
        pre = (a[col].mean() - b[col].mean()) / pooled if pooled else 0
        def wm(g):
            return np.average(g[col], weights=g.overlap_weight)
        ma, mb = wm(a), wm(b)
        va = np.average((a[col] - ma) ** 2, weights=a.overlap_weight)
        vb = np.average((b[col] - mb) ** 2, weights=b.overlap_weight)
        post = (ma - mb) / math.sqrt((va + vb) / 2) if va + vb else 0
        rows.append({"database": database, "covariate": col, "SMD_before": pre, "SMD_after": post})
    return rows


def bootstrap_intervals(work, cols, outcomes, seed, n_boot=1000, cluster_col=None):
    rng = np.random.default_rng(seed)
    draws = {outcome: {"rd": [], "rr": []} for outcome in outcomes}
    clusters = work[cluster_col].fillna("Unknown").astype(str).unique() if cluster_col else None
    for _ in range(n_boot):
        if cluster_col:
            selected = rng.choice(clusters, size=len(clusters), replace=True)
            sample = pd.concat(
                [work[work[cluster_col].fillna("Unknown").astype(str) == cluster] for cluster in selected],
                ignore_index=True,
            )
        else:
            sample = work.iloc[rng.integers(0, len(work), len(work))].copy()
        means = sample[cols].mean()
        sds = sample[cols].std().replace(0, 1)
        X = (sample[cols] - means) / sds
        _, ps = logistic_fit(X.to_numpy(), sample.dex.to_numpy())
        ps = np.clip(ps, 1e-6, 1 - 1e-6)
        sample["overlap_weight"] = np.where(sample.dex == 1, 1 - ps, ps)
        for outcome in outcomes:
            _, rd, _, rr, _ = weighted_risk(sample, outcome)
            draws[outcome]["rd"].append(rd)
            draws[outcome]["rr"].append(rr)
    summary = {
        outcome: {
            "rd": np.percentile(values["rd"], [2.5, 97.5]),
            "rr": np.percentile(values["rr"], [2.5, 97.5]),
        }
        for outcome, values in draws.items()
    }
    return summary, draws


def analyze(
    df, database, analysis_label="primary", n_boot=1000,
    restrict_eicu_hospitals=True, include_hospital_effects=True,
):
    work, cols = encode_analysis(
        df, database,
        restrict_eicu_hospitals=restrict_eicu_hospitals,
        include_hospital_effects=include_hospital_effects,
    )
    balance = smd_rows(work, cols, database)
    effects = []
    outcomes = [
        "hemodynamic_instability_24h", "sustained_hypotension_24h",
        "severe_hypotension_24h", "bradycardia_24h",
        "new_vasopressor_24h", "hospital_mortality",
        "recurrent_instability_24h", "recurrent_hypotension_24h",
        "recurrent_bradycardia_24h",
    ]
    bootstrap, draws = bootstrap_intervals(
        work, cols, outcomes,
        seed=20260917 if database == "eICU-CRD" else 20260918,
        n_boot=n_boot,
        cluster_col="hospital_id" if database == "eICU-CRD" else None,
    )
    for outcome in outcomes:
        groups, rd, rd_ci, rr, rr_ci = weighted_risk(work, outcome)
        rd_ci = bootstrap[outcome]["rd"]
        rr_ci = bootstrap[outcome]["rr"]
        effects.append({
            "database": database, "analysis": analysis_label,
            "outcome": outcome, "analysis_n": len(work),
            "propofol_n": int((work.dex == 0).sum()), "dexmedetomidine_n": int((work.dex == 1).sum()),
            "propofol_weighted_percent": groups[0]["weighted_events_percent"],
            "dexmedetomidine_weighted_percent": groups[1]["weighted_events_percent"],
            "propofol_effective_n": groups[0]["effective_n"], "dexmedetomidine_effective_n": groups[1]["effective_n"],
            "risk_difference_percent": rd * 100, "RD_CI95_low": rd_ci[0] * 100, "RD_CI95_high": rd_ci[1] * 100,
            "risk_ratio": rr, "RR_CI95_low": rr_ci[0], "RR_CI95_high": rr_ci[1],
            "CI_method": (
                f"{n_boot}-replicate hospital-cluster bootstrap with propensity-score refitting"
                if database == "eICU-CRD"
                else f"{n_boot}-replicate patient-level bootstrap with propensity-score refitting"
            ),
        })
    return work, balance, effects, draws


def diagnostic_rows(work, database):
    rows = []
    for dex, treatment in [(0, "propofol"), (1, "dexmedetomidine")]:
        g = work[work.dex == dex]
        for variable in ["ps", "overlap_weight"]:
            x = g[variable]
            rows.append({
                "database": database, "treatment": treatment, "diagnostic": variable,
                "n": len(g), "minimum": x.min(), "p01": x.quantile(.01),
                "p05": x.quantile(.05), "median": x.median(), "p95": x.quantile(.95),
                "p99": x.quantile(.99), "maximum": x.max(),
            })
        w = g.overlap_weight.to_numpy()
        rows.append({
            "database": database, "treatment": treatment, "diagnostic": "effective_sample_size",
            "n": len(g), "minimum": np.nan, "p01": np.nan, "p05": np.nan,
            "median": (w.sum() ** 2 / np.square(w).sum()), "p95": np.nan,
            "p99": np.nan, "maximum": np.nan,
        })
    return rows


def baseline_table(work, database):
    rows = []
    for col in ["age", "weight", "baseline_map", "baseline_hr", "index_hour"]:
        if col not in work or work[col].notna().sum() == 0:
            continue
        row = {"database": database, "variable": col, "type": "continuous"}
        for dex, label in [(0, "propofol"), (1, "dexmedetomidine")]:
            x = work.loc[work.dex == dex, col].dropna()
            row[label] = f"{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]"
        rows.append(row)
    for col in ["female", "vent", "baseline_midazolam", "baseline_opioid", "baseline_ketamine"]:
        row = {"database": database, "variable": col, "type": "categorical"}
        for dex, label in [(0, "propofol"), (1, "dexmedetomidine")]:
            x = work.loc[work.dex == dex, col]
            row[label] = f"{int(x.sum())} ({x.mean()*100:.1f}%)"
        rows.append(row)
    return rows


def heterogeneity_rows(effects, draws_by_database):
    rows = []
    primary = pd.DataFrame(effects)
    primary = primary[primary.analysis.eq("primary")]
    for outcome in primary.outcome.unique():
        e = primary[(primary.database == "eICU-CRD") & (primary.outcome == outcome)]
        m = primary[(primary.database == "MIMIC-IV") & (primary.outcome == outcome)]
        if e.empty or m.empty:
            continue
        e_draw = np.asarray(draws_by_database["eICU-CRD"][outcome]["rd"])
        m_draw = np.asarray(draws_by_database["MIMIC-IV"][outcome]["rd"])
        size = min(len(e_draw), len(m_draw))
        difference_draws = e_draw[:size] - m_draw[:size]
        observed = e.iloc[0].risk_difference_percent - m.iloc[0].risk_difference_percent
        se = np.std(difference_draws, ddof=1) * 100
        z = observed / se if se > 0 else np.nan
        p = math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else np.nan
        ci = np.percentile(difference_draws * 100, [2.5, 97.5])
        rows.append({
            "outcome": outcome,
            "eICU_RD_percent": e.iloc[0].risk_difference_percent,
            "MIMIC_RD_percent": m.iloc[0].risk_difference_percent,
            "RD_difference_eICU_minus_MIMIC_percent": observed,
            "difference_CI95_low": ci[0], "difference_CI95_high": ci[1],
            "heterogeneity_p": p,
            "method": "difference between independent database-specific overlap-weighted risk differences",
        })
    return rows


def make_workbook(flow, baseline, balance, effects, diagnostics, sensitivity, heterogeneity):
    path = OUT / "可编辑表格与图形_右美托咪定_vs_丙泊酚.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        frames = {
            "Flow": pd.DataFrame(flow), "Baseline": pd.DataFrame(baseline),
            "Balance": pd.DataFrame(balance), "Effects": pd.DataFrame(effects),
            "Diagnostics": pd.DataFrame(diagnostics),
            "Sensitivity": pd.DataFrame(sensitivity),
            "Heterogeneity": pd.DataFrame(heterogeneity),
        }
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]; ws.freeze_panes(1, 0); ws.set_column(0, len(frame.columns) - 1, 20)
        wb = writer.book; fig = wb.add_worksheet("Editable_Figures")
        n = len(effects)
        chart = wb.add_chart({"type": "column"})
        chart.add_series({"name": "Propofol", "categories": ["Effects", 1, 1, n, 1], "values": ["Effects", 1, 5, n, 5]})
        chart.add_series({"name": "Dexmedetomidine", "categories": ["Effects", 1, 1, n, 1], "values": ["Effects", 1, 6, n, 6]})
        chart.set_title({"name": "Overlap-weighted outcomes"}); chart.set_y_axis({"name": "Weighted incidence (%)"})
        fig.insert_chart("A2", chart, {"x_scale": 1.7, "y_scale": 1.5})
        bchart = wb.add_chart({"type": "bar"})
        bn = len(balance)
        bchart.add_series({"name": "Before", "categories": ["Balance", 1, 1, bn, 1], "values": ["Balance", 1, 2, bn, 2]})
        bchart.add_series({"name": "After", "categories": ["Balance", 1, 1, bn, 1], "values": ["Balance", 1, 3, bn, 3]})
        bchart.set_title({"name": "Covariate balance"}); bchart.set_x_axis({"name": "Standardized mean difference"})
        fig.insert_chart("A28", bchart, {"x_scale": 1.7, "y_scale": 1.5})
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # v2 caches encode strict pre-index HR stability and the 15--60 minute
    # repeated-abnormality endpoint. Retaining v1 makes the definition change auditable.
    eicu_cache = OUT / "eicu_active_comparator_patient_level_v2.csv"
    mimic_cache = OUT / "mimic_active_comparator_patient_level_v2.csv"
    if eicu_cache.exists():
        print("[1/6] loading cached eICU cohort")
        eicu = enforce_current_definitions(pd.read_csv(eicu_cache))
    else:
        print("[1/6] eICU cohort")
        eicu = build_eicu(); eicu.to_csv(eicu_cache, index=False)
    if mimic_cache.exists():
        print("[2/6] loading cached MIMIC cohort")
        mimic = enforce_current_definitions(pd.read_csv(mimic_cache))
    else:
        print("[2/6] MIMIC cohort")
        mimic = build_mimic(); mimic.to_csv(mimic_cache, index=False)
    print("[3/6] refreshing ventilation status at treatment initiation")
    eicu = refresh_eicu_baseline_ventilation(eicu)
    mimic = refresh_mimic_baseline_ventilation(mimic)
    flow = []; baseline = []; balance = []; effects = []; diagnostics = []; sensitivity = []
    draws_by_database = {}
    for database, frame in [("eICU-CRD", eicu), ("MIMIC-IV", mimic)]:
        for stage, mask in [
            ("符合首次镇静策略", np.ones(len(frame), dtype=bool)),
            ("基线血流动力学稳定", frame.stable_baseline == 1),
            ("基线稳定且结局可评价", frame.evaluable == 1),
        ]:
            sub = frame[mask]
            flow.append({"database": database, "stage": stage, "total": len(sub), "propofol": int((sub.dex == 0).sum()), "dexmedetomidine": int((sub.dex == 1).sum())})
        work, brows, erows, draws = analyze(frame, database)
        draws_by_database[database] = draws
        flow.append({
            "database": database,
            "stage": "完整病例加权分析",
            "total": len(work),
            "propofol": int((work.dex == 0).sum()),
            "dexmedetomidine": int((work.dex == 1).sum()),
        })
        work.to_csv(OUT / f"{database}_weighted_analysis_cohort.csv", index=False)
        baseline.extend(baseline_table(work, database))
        balance.extend(brows); effects.extend(erows); diagnostics.extend(diagnostic_rows(work, database))
        for dex, treatment in [(0, "propofol"), (1, "dexmedetomidine")]:
            eligible = frame[(frame.stable_baseline == 1) & (frame.dex == dex)]
            diagnostics.append({
                "database": database, "treatment": treatment,
                "diagnostic": "complete_case_fraction_after_stable_baseline",
                "n": len(eligible), "median": eligible.evaluable.mean(),
            })
            diagnostics.append({
                "database": database, "treatment": treatment,
                "diagnostic": "crossover_24h_fraction_in_primary_cohort",
                "n": int((work.dex == dex).sum()),
                "median": work.loc[work.dex == dex, "crossover_24h"].mean(),
            })
        if database == "MIMIC-IV":
            diagnostics.append({
                "database": database, "treatment": "all",
                "diagnostic": "pre_index_RASS_observed_fraction",
                "n": len(work), "median": work.baseline_rass.notna().mean(),
            })

        no_cross = frame[frame.crossover_24h == 0].copy()
        _, _, no_cross_effects, _ = analyze(
            no_cross, database, analysis_label="exclude_24h_crossover", n_boot=300,
        )
        sensitivity.extend(no_cross_effects)
        if database == "eICU-CRD":
            _, _, all_hospital_effects, _ = analyze(
                frame, database, analysis_label="all_hospitals_no_hospital_FE", n_boot=300,
                restrict_eicu_hospitals=False, include_hospital_effects=False,
            )
            sensitivity.extend(all_hospital_effects)
    heterogeneity = heterogeneity_rows(effects, draws_by_database)
    pd.DataFrame(flow).to_csv(OUT / "table1_flow.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(baseline).to_csv(OUT / "table2_baseline.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(balance).to_csv(OUT / "table2_balance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(effects).to_csv(OUT / "table3_effects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(OUT / "table4_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(OUT / "table5_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(heterogeneity).to_csv(OUT / "table6_heterogeneity.csv", index=False, encoding="utf-8-sig")
    print("[5/6] editable workbook")
    make_workbook(flow, baseline, balance, effects, diagnostics, sensitivity, heterogeneity)
    print("[6/6] complete")


if __name__ == "__main__":
    main()
