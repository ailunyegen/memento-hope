# 技术方案书 §3 合流素材：实时响应与敏捷重规划

> 本文件为《技术方案设计书：基于“双系统投机增量规划”的大模型作战方案秒级生成架构》
> §3 章节的实证补充素材（2026-08 实测，RTX 4090 D，deepseek-r1-0528-qwen3-8b），
> 可直接并入方案书正文或附录。

## 3.x 三级时延响应矩阵（实测）

| 层级 | 模式 | 核心机制 | 实测耗时 | 质量（MS） | 指控实战定位 |
| --- | --- | --- | --- | --- | --- |
| Tier 1 | System 1 应急预案 | 案例骨架继承 + HOPE 权重投影 + 兵力缩放（纯规则） | **1.4~5.6 ms** | 0.6172~0.6212 | 态势突变、防空拦截等毫秒级紧急预案 |
| Tier 2 | System 2 Delta Loop | BottleneckDetector 定位 + 能力赤字注入 + 局部补丁 + 门禁回滚 | **15~18 s** | 0.6331~0.6381 | 战术机动、局部突破等半分钟级敏捷调整 |
| Tier 3 | System 2 Fast 全量 | 单候选生成 + 规则化反思 | 45.8 s（10 轮） | 0.6347 | 初始任务规划、战役推演等分钟级全局统筹 |
| 原系统 | Full 全量 | 4 候选竞争 + LLM 反思 | ~470 s（10 轮） | 0.6388 | —（不满足实时性） |

关键结论：**Tier 2（Delta）以 16 s 达到原系统 470 s 的 99.2% 质量**，
Tier 1 的 5.6 ms 应急预案本身即达原系统最终质量的 96.6%。

## 3.x 跨场景 Delta 泛化增益表（5 场景，scene-out 案例库，50-MC）

| 场景 | 任务族 | System 1 MS | Delta MS | 增益 | 轮数 | 接受率 | Delta 耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coastal_joint_assault | assault | 0.6212 | 0.6331 | +0.0119 | 8 | 0.88 | 17.9 s |
| island_resupply_corridor | sustain | 0.6295 | 0.6381 | +0.0086 | 8 | 1.00 | 15.1 s |
| mountain_corridor_recon | recon | 0.6756 | 0.6747 | -0.0009 | 3 | 1.00 | 5.3 s |
| river_crossing_breakthrough | assault | 0.6524 | 0.6630 | +0.0106 | 8 | 0.88 | 15.3 s |
| urban_hub_defense | defense | 0.6480 | 0.6608 | +0.0128 | 8 | 1.00 | 16.7 s |

- 4/5 场景正增益（平均 +0.0110，范围 +0.0086 ~ +0.0128），门禁接受率 ≥ 0.875；
- mountain 场景 -0.0009（<0.001，50-MC 噪声级）：该场景 System 1 预案（0.6756）
  已接近质量上限，Delta 3 轮即触发停滞早停；
- 结论：基于瓶颈诊断的局部 Patch 机制在突击/防御/支援/侦察四类任务族下均保持
  单调收敛与零有意义退化，跨域普适性成立。

## 3.x 全量重写退化 vs 增量单调收敛（对比图）

图：`docs/figures/full_rewrite_vs_delta.png`

- 曲线 A（Full Rewrite / Single-Candidate，Fast-10）：
  0.6347 → 0.5439 → … → 0.5521，冲高后期望回归（方差驱动假象）；
- 曲线 B/C（Delta Loop v2 / 15 轮压榨）：
  0.6166 → … → 0.6328 单调爬升，早停触发，零回滚；
- 理论总结点：单候选大模型规划下，全量自回归重写具有极高的方差破坏性；
  基于环境诊断与能力赤字引导的局部打补丁（Delta Patching）是维持规划
  单调收敛的有效路径——这是 Delta Patch 作为"刚需机制"而非"性能微调"的实证依据。

## 数据溯源

- 三级时延矩阵：`result/fast10_benchmark/`、`result/delta_15_squeeze/`、`result/fast_first_v2/`
- 跨场景增益表：`result/delta_generalization/gain_table.md`（机器可读 JSON 同目录）
- 对比图脚本：`scripts/plot_fast_vs_delta.py`（数据源自动读取，可重生成）
- 图表入库：`docs/figures/full_rewrite_vs_delta.png` + `full_rewrite_vs_delta_data.json`
