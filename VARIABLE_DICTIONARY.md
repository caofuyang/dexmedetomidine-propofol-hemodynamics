# 变量字典 / Variable Dictionary

> 本字典与稿件方法学第 1.2–1.5 节一一对应，供审稿人和后续使用者核对变量定义。
> This dictionary corresponds to Methods sections 1.2–1.5 of the manuscript.

## 1. 暴露 / Exposure

| 项目 | 中文定义 | English definition | 取值 / Values |
|---|---|---|---|
| 初始镇静策略 | 首次 ICU 入住后 24 h 内，首先持续输注的研究药物 | First continuous infusion of the study drug within 24 h of first ICU admission | 右美托咪定 / 丙泊酚 (dexmedetomidine / propofol) |
| 时间零点 (index time) | 最早研究药物持续输注的起始时间 | Start time of the earliest continuous study-drug infusion | 时间戳 (timestamp) |
| 双药起始 ≤30 min | 两药起始时间相差 ≤30 min 者排除（无法可靠归入单一初始策略） | Excluded if the two drugs started within 30 min of each other | 排除规则 / exclusion rule |
| 换药/加药 | 24 h 内换药或加用另一研究药物不改变初始分组 | Cross-over/add-on within 24 h does not change initial group | 初始策略效应 (initial-strategy effect) |

暴露来源 / Source fields:
- eICU-CRD：`infusionDrug` 表
- MIMIC-IV：`inputevents` 表（仅保留持续给药、速率 >0 且结束时间晚于开始时间的记录）

## 2. 结局 / Outcomes

| 结局 | 中文定义 | English definition |
|---|---|---|
| 主要结局：重复性血流动力学异常 | 用药后 24 h 内，相邻 2 个小时窗均出现低血压或均出现心动过缓 | Hypotension (MAP < 65 mmHg or SBP < 90 mmHg) or bradycardia (HR < 50 beats/min) in two adjacent post-index hourly bins |
| 组成结局：重复性低血压 | 相邻 2 个小时窗均出现低血压 | Hypotension in two adjacent hourly bins |
| 组成结局：重复性心动过缓 | 相邻 2 个小时窗均出现心动过缓 | Bradycardia in two adjacent hourly bins |
| 探索性：任一小时窗异常 | 24 h 内任一小时窗出现异常 | Abnormality in any post-index hourly bin |
| 事件级未协调定义 | 原始记录中间隔 15–60 min 的至少 2 次异常（仅用于评价测量规则影响） | ≥ 2 abnormal raw records 15–60 min apart (measurement-rule sensitivity only) |

## 3. 协变量 / Covariates（时间零点前或时间零点可获得）

| 变量 | 中文 | English | 类型/单位 |
|---|---|---|---|
| 年龄 | 年龄 | Age | 连续 / 岁 |
| 性别 | 性别 | Sex | 二分类 |
| 体重 | 体重（eICU 用入 ICU 体重；MIMIC 用时间零点初始药物订单所附体重） | Weight | 连续 / kg |
| 基线 MAP | 基线平均动脉压 | Baseline mean arterial pressure | 连续 / mmHg |
| 基线心率 | 基线心率 | Baseline heart rate | 连续 / 次/min |
| 入 ICU 至用药时间 | 入 ICU 至用药时间 | Time from ICU admission to treatment | 连续 / h |
| 机械通气 | 用药时机械通气状态 | Mechanical ventilation at treatment | 二分类 |
| 用药前咪达唑仑 | 用药前咪达唑仑使用 | Pre-treatment midazolam | 二分类 |
| 用药前阿片类 | 用药前阿片类使用（MIMIC 含芬太尼、氢吗啡酮、硫酸吗啡等） | Pre-treatment opioid use | 二分类 |
| 用药前氯胺酮 | 用药前氯胺酮使用 | Pre-treatment ketamine | 二分类 |
| ICU 类型 | ICU 类型 | ICU type | 分类 |
| 入院来源 | 入院来源 | Admission source | 分类 |
| 基线血压有效小时窗数 | 基线血压有效小时窗数 | Number of valid baseline BP hourly bins | 计数 |
| 基线心率有效小时窗数 | 基线心率有效小时窗数 | Number of valid baseline HR hourly bins | 计数 |
| 年代组（仅 MIMIC-IV） | 年代组 | Calendar-era group | 分类 |
| 医院固定效应（仅 eICU-CRD） | 医院固定效应 | Hospital fixed effects | 分类 |

## 4. 规则与参数 / Rules and parameters

| 规则 | 中文 | English |
|---|---|---|
| 生理清洗范围 | 心率 20–250 次/min；SBP 30–300 mmHg；MAP 20–200 mmHg；RASS 仅 -5～4（敏感性） | HR 20–250 bpm; SBP 30–300 mmHg; MAP 20–200 mmHg; RASS −5 to 4 (sensitivity) |
| 时间窗 | 相对时间零点固定 1 h 窗；基线 = 用药前 6 h；随访 = 用药后 24 h；窗内取均值 | Fixed 1-h bins relative to index time; baseline = 6 h before; follow-up = 24 h after; within-bin mean |
| 时间零点记录归属 | 恰在时间零点的记录不归入任一窗口 | Records exactly at index time assigned to neither window |
| 血压来源优先级 | MAP 与 SBP 分别优先有创；该分量无有创时以无创补充；同一分量不把两种来源作为两个独立异常机会 | Invasive prioritized separately for MAP and SBP; noninvasive fills missing component; a single component never counts both sources as two events |
| 基线稳定性要求 | 用药前 6 h 至少 1 个有效血压窗 + 1 个有效心率窗；无 MAP<65 / SBP<90 / HR<50；基线及时间零点未用血管活性药 | ≥1 valid BP bin and ≥1 valid HR bin in 6-h baseline; no MAP<65, SBP<90, or HR<50; no vasopressor at baseline or index |
| 主分析排除 | 用药时已使用血管活性药者（敏感性分析保留） | Patients on vasopressors at treatment initiation excluded (retained in sensitivity) |
| 结局可评价要求 | 用药后 24 h 血压和心率各至少 2 个有效小时窗 | ≥2 valid BP bins and ≥2 valid HR bins in the 24-h follow-up |

## 5. 统计模型 / Statistical model

| 项目 | 说明 |
|---|---|
| 倾向评分模型 | 二元 logistic 回归（接受右美托咪定），主模型含连续变量线性项 + 分类虚拟变量；信息矩阵对角线加 1×10⁻⁷ 岭项 |
| 权重 | 重叠加权：右美托咪定组权重 = 1 − 倾向评分；丙泊酚组权重 = 倾向评分 |
| 缺失处理 | 模型协变量缺失者完整病例分析 |
| 平衡诊断 | 加权前后标准化差异（绝对值 < 0.10 可接受）、倾向评分分布、权重分布、有效样本量 |
| 效应量 | 加权 RD、RR（右美托咪定相对丙泊酚） |
| 置信区间 | 1000 次 bootstrap 百分位数区间；eICU-CRD 按医院整群重采样，MIMIC-IV 按患者重采样 |
| 跨库差异 | eICU-CRD RD − MIMIC-IV RD；CI 取两库独立 bootstrap 差值百分位数；P 值用差值 bootstrap 标准误的双侧正态近似 |
| 软件 | Python 3.12.14、pandas 2.2.3、NumPy 2.3.5 |
