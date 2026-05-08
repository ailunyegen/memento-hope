# 基于 Memento 与 Hope 架构的大模型作战体系方案生成与仿真研究

## 研究边界
- 本系统仅用于虚拟场景下的方案生成与仿真评估研究。
- 不面向真实作战指挥、现实目标攻击、武器部署或现实行动建议。
- 所有实验结果仅用于算法验证、架构分析和论文展示。

## 项目简介
本项目在本地大模型底座之上，结合：
- `Memento`：案例记忆检索、经验写回、正负样本提示
- `Hope`：快慢权重融合与场景自适应能力调节
- `Reflection`：基于失败教训的 EvoPrompt 反思优化

当前主链路包括：
- 作战方案生成
- 基于五阶段杀伤链的仿真评估
- `action_type` 驱动的结构化动作评分
- Monte Carlo 多次仿真统计
- DoDAF / C2SIM 导出
- 对照实验与消融实验汇总

## 环境依赖
- Python 3.11+
- 本地 OpenAI-compatible LLM 服务，例如 LM Studio
- `requirements.txt` 中列出的依赖

推荐安装：

```bash
python -m pip install -r requirements.txt
```

## 本地模型启动
以 LM Studio 为例：

1. 启动 LM Studio 本地服务
2. 加载 OpenAI-compatible 模型
3. 设置环境变量

```bash
set LOCAL_LLM_BASE_URL=http://localhost:1234/v1
set LOCAL_LLM_API_KEY=lm-studio
set LOCAL_LLM_MODEL=openai/gpt-oss-20b
```

如果使用本地 embedding 模型：

```bash
set CASE_EMBEDDING_MODEL=your-local-embedding-model
set CASE_EMBEDDING_LOCAL_ONLY=1
```

## Memory 检索模式
案例记忆支持双模式：

- `faiss`：优先使用 `sentence-transformers + FAISS` 做语义检索
- `keyword`：当 FAISS 或 embedding 模型不可用时，自动降级为关键词检索

关键词回退模式仍会保留：
- `positive -> imitate`
- `negative -> avoid`

因此在无 FAISS 环境下，主 pipeline 仍可运行并完成基础实验。

## 运行命令
单次原型运行：

```bash
python -X utf8 -m military_research.cli ^
  --scenario data/sample_joint_operation.json ^
  --case-bank data/military_case_bank_frozen_seed.jsonl ^
  --iterations 4 ^
  --sim-runs 20 ^
  --seed 7 ^
  --output-dir result/demo_run
```

常用消融开关：
- `--disable-memory`
- `--disable-hope`
- `--disable-reflection`
- `--disable-writeback`

## 消融实验命令
标准四组消融：

```powershell
.\run_ablation_suite.ps1 `
  -BaseOutputDir result/ablation_suite_seed7 `
  -Iterations 3 `
  -SimRuns 50 `
  -Seed 7 `
  -SummaryStyle chapter
```

手动汇总四组消融：

```bash
python -X utf8 -m military_research.summarize_ablations ^
  --pure-llm result/ablation_pure_llm/full_result.json ^
  --memento result/ablation_memento/full_result.json ^
  --hope result/ablation_hope/full_result.json ^
  --full result/ablation_full/full_result.json ^
  --style chapter ^
  --output result/ablation_summary_chapter.md
```

两组结果对照：

```bash
python -X utf8 -m military_research.compare_results ^
  --baseline result/math_innovation_test/full_result.json ^
  --candidate result/math_latest_test1/full_result.json ^
  --output result/math_comparison.md
```

## 跨场景泛化实验
仓库内已提供一组基础泛化场景，位于：

```text
data/generalization_scenarios/
```

包含：
- `coastal_joint_assault.json`
- `urban_hub_defense.json`
- `mountain_corridor_recon.json`
- `river_crossing_breakthrough.json`
- `island_resupply_corridor.json`

批量运行跨场景四组消融：

```powershell
.\run_generalization_suite.ps1 `
  -ScenarioDir data/generalization_scenarios `
  -CaseBank data/military_case_bank_frozen_seed.jsonl `
  -BaseOutputDir result/generalization_suite_seed7 `
  -Iterations 6 `
  -SimRuns 30 `
  -Seed 7 `
  -SummaryStyle chapter
```

该脚本会为每个场景分别生成：
- `ablation_pure_llm`
- `ablation_memento`
- `ablation_hope`
- `ablation_full`

并在总目录下输出：

```text
generalization_summary.md
```

也可以单独汇总已有泛化实验目录：

```bash
python -X utf8 -m military_research.summarize_generalization ^
  --suite-dir result/generalization_suite_seed7 ^
  --manifest data/generalization_scenarios/manifest.json ^
  --style chapter ^
  --output result/generalization_suite_seed7/generalization_summary.md
```

## 输出文件说明
一次运行通常会生成：
- `best_plan.json`
- `simulation.json`
- `full_result.json`
- `research_report.md`
- `dodaf_ov5b.json`
- `dodaf_ov6c.json`
- `c2sim.xml`

`full_result.json` 中的关键字段包括：
- `best_plan`
- `best_simulation`
- `best_memory_hits`
- `optimization_history`
- `experiment_config`

`best_simulation.monte_carlo_stats` 当前输出：
- `mean`
- `std`
- `min`
- `max`
- `ci95_low`
- `ci95_high`
- `n`

## 评分逻辑说明
当前仿真评分以 `PlanPhase.actions[*].action_type` 为主：
- `detect`：主要依赖 `recon / c2 / ew`
- `disrupt`：主要依赖 `strike / ew / c2`
- `breach`：主要依赖 `mobility / strike / protection`
- `control`：主要依赖 `control / c2 / protection`
- `sustain`：主要依赖 `sustain / c2`

中文关键词只作为旧数据兼容时的 fallback，不再作为主评分来源。

## 案例库冻结流程
为保证论文实验公平，建议使用冻结案例库而不是持续增长的主案例库。

默认冻结案例库：

```text
data/military_case_bank_frozen_seed.jsonl
```

如需重新导出：

```bash
python -X utf8 -m military_research.export_seed_case_bank ^
  --input data/military_case_bank.jsonl ^
  --output data/military_case_bank_frozen_seed.jsonl
```

## 案例库扩充建议
当前冻结案例库只适合做单场景或小规模消融验证。若要支持“算法具有跨场景普适性”的论文结论，建议继续扩充多场景案例库。

详细蓝图见：

```text
docs/case_bank_expansion_blueprint.md
```

## 测试
运行测试：

```bash
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
set PYTHONPATH=%CD%
pytest tests -q
```

当前测试覆盖：
- `tests/test_domain.py`
- `tests/test_plan_repair.py`
- `tests/test_case_memory.py`
- `tests/test_simulator.py`
- `tests/test_exporters.py`

## 泛化案例库导出
如果希望做“留一类场景不入库”的跨场景泛化实验，可以先按场景导出冻结案例库：

```bash
python -X utf8 -m military_research.export_generalization_case_banks ^
  --input data/military_case_bank_frozen_seed.jsonl ^
  --manifest data/generalization_scenarios/manifest.json ^
  --output-dir data/generalization_case_banks ^
  --mode leave-one-family-out
```

导出后会生成：
- `data/generalization_case_banks/all_cases.jsonl`
- `data/generalization_case_banks/coastal_joint_assault.jsonl`
- `data/generalization_case_banks/urban_hub_defense.jsonl`
- `data/generalization_case_banks/mountain_corridor_recon.jsonl`
- `data/generalization_case_banks/river_crossing_breakthrough.jsonl`
- `data/generalization_case_banks/island_resupply_corridor.jsonl`
- `data/generalization_case_banks/summary.json`

随后可以让泛化实验脚本按场景自动选取对应案例库：

```powershell
.\run_generalization_suite.ps1 `
  -ScenarioDir data/generalization_scenarios `
  -CaseBankDir data/generalization_case_banks `
  -BaseOutputDir result/generalization_suite_seed7_lf1 `
  -Iterations 6 `
  -SimRuns 30 `
  -Seed 7 `
  -SummaryStyle chapter
```
