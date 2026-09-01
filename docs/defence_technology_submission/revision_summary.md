# 改投 *Defence Technology* 说明（vs 原 SMPT 投稿）

> 本说明记录从 *Simulation Modelling Practice and Theory*（SMPT）改投
> *Defence Technology*（中国兵工学会主办、中科院 1 区 TOP）时的定位重塑与
> 全部改动要点，供投稿与后续审稿应答使用。

## 0. 改投背景

SMPT 拒稿的本质是**"仿真建模理论"不匹配**——该刊关注仿真模型本身的逼真度与
建模仿真理论，而本工作定位为"面向指控实战的智能筹划闭环"。改投 *Defence
Technology* 把重心从"评估仿真器逼真度"转向：**C2 实战需求的智能筹划闭环、
双系统实时响应机制、标准化体系集成**。

## 1. 定位重塑（Framing Shift）

| 维度 | 原 SMPT 叙事 | 改投 DT 叙事 |
| --- | --- | --- |
| 研究主题 | 大模型作战方案生成与仿真评估流水线 | 面向多域联合作战指控(C2)的实时智能任务筹划架构 |
| 杀伤链模型定位 | 独立提出的五阶段仿真评估模型 | 遵循 JP 3-60 / F2T2EA 规约的轻量级算法目标评估沙箱 |
| 双系统机制价值 | 算法层面推理加速实验 | 解决现代战场 OODA 环毫秒级应急/秒级重规划矛盾的关键突破 |
| 标准输出价值 | 格式转换工具 | 打通大模型方案向指控系统与联合仿真联邦落地的互操作底座 |

## 2. 题目 / 期刊 / 关键词

- **题目**：*A Dual-System Enhanced Large Language Model Architecture for Real-Time Joint-Operation Mission Planning in Command and Control Systems*（原：*A Memento-Hope-Reflection Enhanced LLM Pipeline for Simulated Joint-Operation Plan Generation and Evaluation*）
- **期刊**：`Simulation Modelling Practice and Theory` → `Defence Technology`
- **关键词**：`military command and control / large language models / mission planning / case-based reasoning / adaptive weighting / kill-chain optimization / C2SIM interoperability / real-time replanning`

## 3. 章节级改动清单

### 3.1 摘要（abstract.tex）
- 前置多域协同 + OODA 实时矛盾的实战背景；
- 显式列出三级时延矩阵：System 1 规则/参数投影（5.6 ms 应急出库）→ System 2 瓶颈导向增量修补（22.8 s 达 96.4% 质量）→ 全量深度优化（分钟级）；
- 固化标准化交付物（DoDAF OV-5b/OV-6c + SISO C2SIM），明确构建标准规划中枢、跨系统协同；
- 诚实边界：证据限定内部评估沙箱，不宣称外部有效性、不与 OneSAF/FLAMES 竞争。

### 3.2 引言（introduction.tex）
- 展开 Command-agent（liu2026commandagent）——LLM 指挥官重建战争仿真与指挥决策，作为"大模型重塑作战指挥决策"的核心支撑；
- 补全军事 LLM 智能体 / 兵棋推演 / 军事决策基准文献链；
- 新增**三大军事痛点锚定段**：① 战术经验先验高效复用（Memento）② 动态战场威胁下能力平衡（HOPE）③ 单候选全量重写的期望回归（Delta Patch 必要性，引用 Figure 6 实证）。

### 3.3 方法（method.tex）
- **§2.6 仿真去争议化**：定义评估器为**基于 JP 3-60 / F2T2EA 的算法目标评估沙箱（Task Sandbox / Algorithmic Objective Evaluator）**，为强化/反思闭环提供梯度与诊断信号；非 OneSAF/FLAMES 替代、不建模平台级弹道/通信/地形物理。
- **§2.7 导出层桥梁定位**：不替代 OneSAF/FLAMES，而是输出 **IEEE 1516 / SISO C2SIM** 标准格式，充当大模型通往现有指控系统与分布式联合仿真联邦（HLA/C2SIM）的**标准规划生成中枢**，引擎无关。
- **§2.8（原）双系统小节**：保留 Dual-system speculative incremental planning（System 1 毫秒应急 + System 2 瓶颈 Delta 打补丁闭合）。

### 3.4 讨论（discussion.tex）
- 将 breach（突击突破）与 sustain（支援保障）争夺 protection（防护）的现象，上升到**多域联合作战中的兵力/资源帕累托分配矛盾（Pareto Resource Allocation in Joint Operations）**：同一防护资源池分配于突击/保障的两难 → 标量化奖励只能折中 → 控制器收敛于帕累托前沿但无法沿前沿移动 → 战术对抗角度提升分析深度。

### 3.5 Cover Letter（docs/cover_letter_defence_technology.md）
- 开篇定性：人工智能与军事指挥控制(C2)结合的跨学科前沿应用；
- 三大核心防务贡献：① Evidence-Grounded Cognitive Architecture（Memento+HOPE+Reflection 深度耦合闭环）② Dual-System Real-Time Execution（5.6 ms 应急 + 22.8 s 增量修补解决高时延痛点）③ Standard-Compliant Interoperability（原生 DoDAF OV-5b/6c + C2SIM，工程就绪）；
- 审稿人建议：C2 决策 / 军事 LLM 智能体 / C2SIM-DoDAF 互操作方向专家。

## 4. 数据口径（已统一）

- 全稿 **4090 默认参数口径**（θ=0.5, ε=0.02），删除优化参数表（tab:multiseed-optimized）；
- Table 4/5/6/7/8/12 + baseline + cost-quality 全部为 RTX 4090 D 实测；
- 泛化 4/5（非 5/5）、sustain 阶段 Hope 领先（如实）、基线 30-MC 标注为补充参考；
- 3060 补充表（multillm/casebank/hyperparam/convergence-20）caption 标注 "recorded under the earlier RTX 3060 environment"。

## 5. 重投包内容（docs/defence_technology_submission/）

| 文件 | 说明 |
| --- | --- |
| `Defence_Technology_main.pdf` | 改投版终稿（已编译） |
| `main.tex` + `sections/*.tex` + `references.bib` | 源文件 |
| `Cover_Letter.md` | 投稿信 |
| `full_rewrite_vs_delta.png` | 全量重写退化 vs Delta 单调收敛图（Figure） |

## 6. 待办 / 可选

- [ ] 按 DT 投稿系统字段微调（摘要字数、作者信息、基金声明、图表编号）
- [ ] 若 R1 要求：补外部独立验证（专家盲评/独立规则评估）作为后续增强
- [ ] 旧 SMPT 投稿包（SMPT_Submission_Package/）去留决定
