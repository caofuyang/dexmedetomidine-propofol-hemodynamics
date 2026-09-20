# 分析代码 Version 1.0

本目录对应稿件中表1至表4、主要效应、跨数据库差异和敏感性分析的最终计算流程。代码已去除作者电脑上的个人绝对路径；统计定义、估计方法、bootstrap次数和随机种子均保持不变。

## 软件环境

- Python 3.12.14
- pandas 2.2.3
- NumPy 2.3.5

## 原始数据

代码需要依法取得的 eICU-CRD v2.0 与 MIMIC-IV v3.1。由于数据库使用协议限制，本资料包不包含患者级原始数据、派生患者级队列或任何可识别信息。

将环境变量 `SEDATION_DB_ROOT` 指向同时包含 `eICU-CRD` 与 `MIMIC-IV` 文件夹的目录。预期数据库子目录结构与脚本顶部的 `EICU`、`MIMIC` 路径一致。

## 运行

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SEDATION_DB_ROOT="/your/path/to/deidentified_databases"
python run_pipeline.py
```

`run_pipeline.py`按以下顺序执行：基础队列构建、MIMIC时间零点体重刷新、小时窗协调分析、诊断表、附加敏感性分析、事件级及时间零点血管活性药复核。完整重算会读取大型压缩数据库文件，运行时间和内存占用取决于硬件。

## 结果对应关系

- 表1：`table11_harmonized_flow.csv`
- 表2：`table17_harmonized_baseline.csv`
- 表3：`table7_harmonized_hourly_effects.csv`
- 表4：`table10_harmonized_hourly_sensitivity.csv`与`table21_additional_harmonized_sensitivity.csv`
- 跨数据库RD差：`table8_harmonized_hourly_heterogeneity.csv`
- 未协调事件级对照：`table19_unharmonized_event_level_same_cohort.csv`与`table20_unharmonized_event_level_heterogeneity.csv`

`reference_outputs`仅包含不含患者级信息的最终汇总结果，用于重算后核对。`MANIFEST_SHA256.txt`记录本版本全部代码、依赖文件及参考输出的校验值。
