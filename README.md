# 分析代码 Version 2.0（「加厚设计」修订版）

本目录是 v1.0 的完整副本加上三项新增分析，用于把手稿从"药物比较"改造成以**测量/结局构建依赖**为主线的 SCI 稿。v1.0 保持冻结、可回退；本目录不改动任何既有脚本的统计定义。

## 相对 v1.0 的新增

| 新增 | 文件 | 作用 |
|---|---|---|
| 一次扫描、落盘小时窗表 | `work/harmonized_bin_export.py` | 把原始数据（MIMIC chartevents ~3.5 GB、eICU vitalPeriodic/vitalAperiodic）**只扫一次**，输出每患者每小时窗的分量血压/心率均值与记录数；同时对 v1.0 冻结的摘要做**逐行回归校验**，不一致即中止 |
| 结局定义网格 | `work/harmonized_definition_variants.py` | 从小时窗表派生"仅 MAP<65""仅 SBP<90"、相邻 k∈{1,2,3} 窗、三种血压来源规则、采样密度分层；全部走 v1.0 冻结估计器（仅换结局列） |

## 运行

```bat
cd 07_Analysis_code_v2.0
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set SEDATION_DB_ROOT=<含 eICU-CRD 与 MIMIC-IV 的父目录>
python run_pipeline.py
```

预期原始数据结构：

```
<SEDATION_DB_ROOT>/eICU-CRD/data/eicu-collaborative-research-database-2.0
<SEDATION_DB_ROOT>/MIMIC-IV/data/mimic-iv-3.1
```

## 新产出

- `table22_definition_grid.csv`：各结局定义下的加权 RD/95%CI/RR（含"仅 MAP""仅 SBP"、k 窗网格、来源规则）
- `table23_monitoring_density.csv`：两组"每小时窗内记录数"的描述统计
- `table24_density_stratified.csv`：主结局按监测强度三分位的分层效应
- `eicu_/mimic_harmonized_bins_v4.csv`：小时窗中间表（后续再改定义可直接复用，无需重扫）
- `eicu_/mimic_harmonized_hourly_outcomes_v4.csv`：与 v1.0 逐行一致的摘要（已回归校验）

## 两道内置校验（不通过即中止）

1. `harmonized_bin_export.py`：重算的摘要必须与 v1.0 缓存的 `*_harmonized_hourly_outcomes_v3.csv` 逐行相等 —— 防止 v2.0 悄悄偏离 v1.0。
2. `harmonized_definition_variants.py`：`v2_ip_composite_k2` 必须与冻结的主结局 `hourly_repeated_instability_24h` 完全一致 —— 证明网格的 k=2 就是原主结局。

## 待续（下一步补入本目录）

- 双稳健（增广重叠加权 / AIPW）平行估计
- 多重插补与完整病例对照

## Steps 9-10 (added 2026-10-15)

Both steps run on the cohort/summary/bin files already on disk, so neither touches the
raw databases again.

**`work/double_robust_aipw.py`** — re-analyses the pre-specified primary outcome plus the
k=1 and k=3 definitions with three estimators computed from *identical* bootstrap
replicates: the v1.0 overlap-weighted IPW, overlap-weighted g-computation, and
overlap-weighted AIPW. A gate aborts unless the observed-data IPW risk difference
reproduces the frozen `table7` value to machine precision, so the shared design matrix
cannot drift. Writes `table25_double_robust.csv`. Runtime is ~3x the v1.0 bootstrap (three
model fits per replicate); set `SEDATION_AIPW_BOOT=300` for a quicker run.

**`work/multiple_imputation_sensitivity.py`** — MICE (proper imputation; `SEDATION_MI_COUNT`,
default 20) for the covariates that actually carry missingness *within the evaluable
population* (eICU: age, weight; MIMIC: weight, baseline MAP — the last being stays with a
baseline blood pressure but no MAP value). Each completed dataset is analysed with the
untouched frozen estimator and combined by Rubin's rules; within-imputation variance comes
from the sandwich estimator, and the complete-case bootstrap interval is repeated alongside
for a like-for-like comparison. A gate aborts unless the complete-case estimate reproduces
`table7`. Writes `table26_multiple_imputation.csv`.

## Step 11 and auxiliary scripts (added 2026-10-15)

**`work/cross_database_grid.py`** — recomputes the between-database difference for
*every* rule in the definition grid, not just the pre-specified one, with the same
seeds and the same resampling as the rest of the pipeline. The propensity model is
refitted once per replicate and all eight definitions are scored inside that
replicate, which is what keeps this to roughly half an hour rather than several
hours. Two gates: the observed-data estimate *and* the bootstrap interval for the
pre-specified rule must both reproduce the frozen `table7` entry to machine
precision. Writes `table27_cross_database_grid.csv`.

**`work/figure_definition_grid.py`** — draws Figure 1 from `table22_definition_grid.csv`
(matplotlib; PNG at 600 dpi plus a TIFF). It reads only cached results, so it needs
no database access.

**`work/dump_prepared_frames.py`** — auxiliary. The one thing that cannot be rebuilt
from the caches is the ventilation status refreshed from the raw database; this writes
the prepared analysis frame for each database to CSV so that later analyses can be run
without the raw data. Gates on reproducing the frozen analysis sample size.

## Step 12 (added 2026-10-15)

**`work/patient_important_anchors.py`** — compares the thresholded haemodynamic
surrogate against outcomes whose definitions contain no analyst-chosen vital-sign
threshold: new vasopressor initiation after the index time (in a cohort that
required no vasopressor at baseline), ICU mortality and hospital mortality. It
reports them descriptively under the identical weighting and resampling. Uses the
cached prepared frames when present, so it needs no raw-database access. Gate: the
pre-specified surrogate must reproduce frozen `table7`. Writes
`table28_anchor_outcomes.csv`.

Read the mortality rows as confounding by indication, not as drug effects — the
point of the table is that the surrogate and the hard outcomes disagree.
