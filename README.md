# 基于 Memento 与 Hope 架构的大模型作战体系方案生成与仿真研究

## 研究边界

本项目仅用于虚拟场景下的方案生成、仿真评估与论文实验研究，不用于真实作战指挥、现实目标攻击、武器部署或现实行动建议。所有场景、方案、指标和导出文件均服务于算法验证、系统架构分析和学术展示。

## 项目简介

本项目在本地 OpenAI-compatible 大模型服务之上，构建一个面向虚拟联合行动场景的“生成-评估-反思-再生成”研究管线。核心模块包括：

- `Memento`：案例记忆检索、正负样本提示、案例写回与去重。
- `Hope`：快慢权重融合、场景自适应能力调节和反馈更新。
- `Reflection`：基于仿真短板的 EvoPrompt 反思与变体生成。
- `Simulator`：基于五阶段 kill-chain 的仿真评估与 Monte Carlo 统计。
- `Exporters`：输出 DoDAF OV-5b、DoDAF OV-6c 与 C2SIM 结构化文件。

当前代码已经支持单场景消融、跨场景泛化、多种子复现实验、多后端 LLM 对比、外部 baseline 对照和运行时 manifest 记录。

## 环境依赖

推荐使用 Python 3.11+。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果使用 FAISS 语义检索，需要确保 `sentence-transformers` 和 `faiss-cpu` 可用；如果依赖或模型不可用，案例检索会自动降级为 keyword fallback。

## 本地模型启动

以 LM Studio 为例：

```powershell
$env:LOCAL_LLM_BASE_URL="http://localhost:1234/v1"
$env:LOCAL_LLM_API_KEY="lm-studio"
$env:LOCAL_LLM_MODEL="openai/gpt-oss-20b"
```

如需使用本地 embedding 模型：

```powershell
$env:CASE_EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
$env:CASE_EMBEDDING_LOCAL_ONLY="1"
```

## 单次运行

```powershell
python -X utf8 -m military_research.cli `
  --scenario data/sample_joint_operation.json `
  --case-bank data/military_case_bank_frozen_seed.jsonl `
  --iterations 4 `
  --sim-runs 20 `
  --seed 7 `
  --disable-writeback `
  --output-dir result/demo_run
```

常用消融开关：

- `--disable-memory`
- `--disable-hope`
- `--disable-reflection`
- `--disable-writeback`
- `--sim-runs`
- `--seed`

## 单场景消融实验

```powershell
.\run_ablation_suite.ps1 `
  -BaseOutputDir result/ablation_suite_seed7 `
  -Iterations 10 `
  -SimRuns 50 `
  -Seed 7 `
  -SummaryStyle chapter
```

手动汇总已有四组结果：

```powershell
python -X utf8 -m military_research.summarize_ablations `
  --pure-llm result/ablation_pure_llm/full_result.json `
  --memento result/ablation_memento/full_result.json `
  --hope result/ablation_hope/full_result.json `
  --full result/ablation_full/full_result.json `
  --style chapter `
  --output result/ablation_summary_chapter.md
```

## 跨场景泛化实验

场景文件位于：

```text
data/generalization_scenarios/
```

包含：

- `coastal_joint_assault.json`
- `urban_hub_defense.json`
- `mountain_corridor_recon.json`
- `river_crossing_breakthrough.json`
- `island_resupply_corridor.json`

### 严格任务族留出

该模式会排除与目标场景相同 `mission_type/family` 的全部案例，适合作为严格泛化对照：

```powershell
python -X utf8 -m military_research.export_generalization_case_banks `
  --input data/military_case_bank_generalization_v2.jsonl `
  --manifest data/generalization_scenarios/manifest.json `
  --output-dir data/generalization_case_banks_v2 `
  --mode leave-one-family-out `
  --min-cases 50
```

### 场景源留出

论文主跨场景证据建议使用 scene-out：只排除目标场景自身来源案例，保留同任务族异场景案例，避免过度排除导致 Memento 负迁移。

```powershell
python -X utf8 -m military_research.export_generalization_case_banks `
  --input data/military_case_bank_generalization_v2.jsonl `
  --manifest data/generalization_scenarios/manifest.json `
  --output-dir data/generalization_case_banks_v2_sceneout `
  --mode leave-one-source-scenario-out `
  --min-cases 50
```

运行 scene-out 泛化套件：

```powershell
.\run_generalization_suite.ps1 `
  -ScenarioDir data/generalization_scenarios `
  -CaseBankDir data/generalization_case_banks_v2_sceneout `
  -BaseOutputDir result/generalization_suite_seed7_v2_sceneout_iter20 `
  -Iterations 20 `
  -SimRuns 50 `
  -Seed 7 `
  -SummaryStyle chapter
```

50 轮结果可作为趋势辅助证据：

```powershell
.\run_generalization_suite.ps1 `
  -ScenarioDir data/generalization_scenarios `
  -CaseBankDir data/generalization_case_banks_v2_sceneout `
  -BaseOutputDir result/generalization_suite_seed7_v2_sceneout_iter50 `
  -Iterations 50 `
  -SimRuns 50 `
  -Seed 7 `
  -SummaryStyle chapter
```

## 多种子实验

默认参数多种子套件位于 `result/multiseed`，优化参数套件位于 `result/multiseed_optimized`。汇总优化参数 Full 与默认 Pure 的配对差值：

```powershell
python -X utf8 -m military_research.summarize_multiseed_ablation `
  --suite-dir result/multiseed_optimized `
  --baseline-suite-dir result/multiseed `
  --output result/multiseed_optimized/summary.md
```

## 输出文件说明

一次运行通常生成：

- `best_plan.json`
- `simulation.json`
- `full_result.json`
- `research_report.md`
- `dodaf_ov5b.json`
- `dodaf_ov6c.json`
- `c2sim.xml`

`full_result.json` 的关键字段包括：

- `experiment_config`
- `runtime_manifest`
- `best_plan`
- `best_simulation`
- `best_memory_hits`
- `optimization_history`

Monte Carlo 统计字段位于 `best_simulation.monte_carlo_stats`，包含 `mean/std/min/max/ci95_low/ci95_high/n`。

## 评分逻辑

仿真评分以 `PlanPhase.actions[*].action_type` 为主，中文关键词仅作为旧数据兼容 fallback。

- `detect`：主要依赖 `recon / c2 / ew`
- `disrupt`：主要依赖 `strike / ew / c2`
- `breach`：主要依赖 `mobility / strike / protection`
- `control`：主要依赖 `control / c2 / protection`
- `sustain`：主要依赖 `sustain / c2`

## 测试

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
$env:PYTHONPATH=(Get-Location).Path
python -m pytest tests -q
```

如果当前环境缺少 `pytest`，请先执行：

```powershell
python -m pip install -r requirements.txt
```

## 论文证据口径

建议论文主线采用：

- 单场景主消融：`result/ablation_suite_seed7_v3`
- 主跨场景泛化：`result/generalization_suite_seed7_v2_sceneout_iter20`
- 50 轮趋势辅助：`result/generalization_suite_seed7_v2_sceneout_iter50`
- 默认多种子：`result/multiseed`
- 优化参数多种子：`result/multiseed_optimized`
- 多后端对比：`result/multillm_backend`

解释实验结果时应强调：系统提升的是虚拟仿真评估指标，不代表现实行动有效性；Memento 的收益与案例覆盖、留出策略和检索质量有关，存在场景依赖。
