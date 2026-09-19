# dexmedetomidine-propofol-hemodynamics

Analysis code for the manuscript:

> **Hemodynamic abnormalities within 24 hours of initial dexmedetomidine versus propofol sedation after vital-sign sampling harmonization: an active-comparator study in eICU-CRD and MIMIC-IV**

## Overview

This repository contains the de-identified analysis code used to build active-comparator cohorts of adults receiving a first continuous infusion of dexmedetomidine or propofol within 24 hours of their first ICU admission, harmonize vital-sign measurements onto a fixed hourly time grid, and estimate overlap-weighted associations between initial sedation strategy and repeated hemodynamic abnormalities within 24 hours.

The analysis was performed separately in two public critical-care databases:

- **eICU-CRD v2.0** (multi-center, USA)
- **MIMIC-IV v3.1** (Beth Israel Deaconess Medical Center, single center)

## Data availability

This repository intentionally contains **no patient-level data**. eICU-CRD and MIMIC-IV are available to credentialed researchers after completing required training and signing the respective data-use agreements (DUAs). The DUAs prohibit redistribution of the underlying data. Only analysis code, the variable dictionary, and documentation are provided here.

## Study design (brief)

- **Eligibility:** adults (age ≥ 18) with a first continuous infusion of dexmedetomidine or propofol within 24 h of first ICU admission.
- **Harmonization:** measurements during the 6-h baseline and 24-h follow-up were mapped to fixed 1-h bins relative to treatment initiation and averaged within each bin. Invasive measurements were prioritized separately for mean and systolic arterial pressure, with the corresponding noninvasive component used when unavailable.
- **Primary outcome:** the same abnormality — hypotension (MAP < 65 mmHg or SBP < 90 mmHg) or bradycardia (HR < 50 beats/min) — in two adjacent post-index hourly bins.
- **Estimation:** complete-case propensity-score overlap weighting fitted separately in each database; 1,000 bootstrap replicates for risk differences (RDs), risk ratios (RRs), and the between-database RD difference.

## Environment

- Python 3.12.14
- pandas 2.2.3
- NumPy 2.3.5

See `requirements.txt`.

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── VARIABLE_DICTIONARY.md
└── *.py          # analysis scripts (see below)
```

### Analysis scripts

> **TODO (to be finalized):** replace this section with the actual list of `.py` scripts and a one-line description of each, once the script filenames are confirmed.

The pipeline follows the manuscript's three stages: (1) cohort construction and data extraction, (2) vital-sign harmonization and outcome definition, and (3) propensity-score overlap weighting and bootstrap inference.

## Variable dictionary

`VARIABLE_DICTIONARY.md` defines every exposure, outcome, covariate, and analysis rule (cleaning ranges, time-window mapping, blood-pressure source priority, baseline-stability requirements).

## License

This code is released under the [MIT License](LICENSE).

## Citation

Please cite this work using the metadata in [`CITATION.cff`](CITATION.cff), or cite the associated manuscript.
