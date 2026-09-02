# *Defence Technology* 投稿系统一页纸备忘单（Copy-Paste Cheat Sheet）

> 供网申系统逐字段复制粘贴。所有字段与 `docs/defence_technology_submission/` 内定稿一致。
> 生成日期：2026-09。

---

## 1. 稿件类型 / 分类

- **Article type**：Research Article（Original Full-length Article）
- **Suggested subject**：Artificial Intelligence / Military Command & Control / Operations Research
  - 关键词（提交系统勾选或自填）：Military Command and Control · Mission Planning · Large Language Models · C2SIM Interoperability · Kill-Chain Optimization · Real-Time Replanning

---

## 2. 标题（Title）

> **A Dual-System Enhanced Large Language Model Architecture for Real-Time Joint-Operation Mission Planning in Command and Control Systems**

（备选，若系统要求更短：*Real-Time Joint-Operation Mission Planning in Command and Control Systems via a Dual-System LLM Architecture*）

---

## 3. 摘要（Abstract）

> **Modern multi-domain joint operations** present a planning environment in which the battle-space situation evolves rapidly, capabilities must be balanced across competing mission objectives, and decision cycles are measured in seconds rather than minutes. Traditional manual course-of-action development and prompt-only large language model (LLM) assistants fall short on three practical fronts: aligning generated plans with tactical constraints, resisting the variance drift of autoregressive regeneration, and delivering a plan early enough to inform a command-and-control (C2) decision. This paper presents a dual-system enhanced LLM architecture that addresses these constraints as an integrated, evidence-grounded planning loop.
>
> The architecture couples three mechanism families into a single traceable pipeline. A Memento-style case-retrieval module reuses historical operation experience as positive/negative priors; a Hope adaptive controller projects scenario attributes (terrain, weather, electronic-warfare threat, time pressure) onto a bounded capability-weight state through a stochastic, bottleneck-aware update; and a reflection loop converts simulator feedback into local prompt repairs. The pipeline generates structured joint-operation plans, evaluates them in a lightweight closed-loop environment, and exports the selected plan into Department of Defense Architecture Framework (DoDAF) OV-5b, OV-6c, and Command and Control Systems -- Simulation Systems Interoperation (C2SIM) artifacts. On a reference scenario with 10 iterations and 50 Monte Carlo runs, the full loop improves mission success from 62.75 to 65.76 and overall effectiveness from 73.84 to 78.75 over the pure-LLM baseline, and a five-seed replication confirms the gain is statistically significant (paired $t(4)=3.75$, $p=0.02$, $d=1.68$). Across five scene-out generalization scenarios the full pipeline outperforms the pure baseline in four of five, increasing average mission success from 65.01 to 66.48.
>
> For real-time command-and-control use, the architecture adds a decoupled dual-system response path. A rule-based emergency tier produces a structurally complete plan and its DoDAF/C2SIM export in milliseconds (5.6 ms on the reference hardware, 1.4 ms warm) with no LLM call, and a bottleneck-driven delta-patching tier refines that anchor to 96.4% of the full-loop mission-success quality within 22.8 s, against 365 s for the complete loop. This yields a three-tier latency profile --- millisecond emergency plan, tens-of-seconds agile tactical refinement, minutes-level full optimization --- and the delta-patching mechanism preserves the high-quality anchor that full regeneration destroys. The pipeline does not replace dedicated physics-based simulation engines (e.g., OneSAF, FLAMES); instead, it emits IEEE 1516/SISO C2SIM-compatible outputs and acts as a standard-compliant planning hub that bridges LLM-generated courses of action into existing C2 systems and distributed simulation federations. The reported evidence is confined to this internal evaluation sandbox; the paper does not claim external validation against a calibrated field simulator, and the reported confidence intervals reflect within-seed Monte Carlo variance under the fixed seed rather than across-seed variance.

> 字数约 260 词（如系统限 250 词，可删末句"and the reported confidence intervals..."）。

---

## 4. 关键词（Keywords）

> military command and control; large language models; mission planning; case-based reasoning; adaptive weighting; kill-chain optimization; C2SIM interoperability; real-time replanning

---

## 5. Highlights（不超过 5 条，每条 ≤ 85 字符）

1. Dual-system LLM architecture for real-time joint-operation mission planning in C2
2. Evidence-grounded loop coupling case memory, adaptive control, and reflection repair
3. Full loop improves mission success over pure-LLM baseline (paired t(4)=3.75, p=0.02)
4. Three-tier response: 5.6 ms emergency plan, 22.8 s delta refinement to 96.4% quality
5. Native DoDAF OV-5b/OV-6c and SISO C2SIM exports as a standard interoperability hub

---

## 6. 作者与机构（Authors）

- **Changrui Zhang** — Beijing Institute of Technology, Beijing, China
- **Chengwei Yang** (∗ corresponding author) — Beijing Institute of Technology, Beijing, China
- **Corresponding email**：`yangchengwei@bit.edu.cn`

---

## 7. 基金 / 竞争利益 / CRediT

- **Funding**：This research received no external funding.
- **Competing interests**：The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
- **CRediT**：Changrui Zhang: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft. Chengwei Yang: Supervision, Validation, Resources, Writing – review & editing, Project administration, Funding acquisition.

---

## 8. 数据可用性（Data Availability）

> The structured scenario configurations, evaluation logs, and software artifacts supporting the findings of this study are available from the corresponding author upon reasonable request.

---

## 9. 推荐审稿人方向（Suggested Reviewers, 由作者按领域补充具体专家）

- Command-and-control decision-making / military operations research
- Large language model agents for military simulation and wargaming
- C2-simulation interoperability（C2SIM / DoDAF / HLA）

---

## 10. 投稿附件清单

| 附件 | 文件 |
| --- | --- |
| Manuscript (PDF) | `Defence_Technology_main.pdf` |
| Highlights | `Highlights.txt` |
| Cover Letter | `Cover_Letter.md` |
| LaTeX source + figures + bib | `main.tex`, `sections/*.tex`, `references.bib`, `full_rewrite_vs_delta.png` |
| （可选）Declaration | 见 archive/SMPT_Submission_Package/Declaration_of_Interest.txt 可复用模板 |

---

## 11. 投递前自检

- [ ] 期刊名已改为 Defence Technology（main.tex `\journal`）
- [ ] 摘要 ≤ 系统字数上限；Highlights ≤ 5 条
- [ ] 作者/通讯/基金/CRediT 与系统一致
- [ ] 推荐审稿人已填具体专家名
- [ ] 旧 SMPT 投稿包已归档于 `archive/SMPT_Submission_Package/`（本地上传前勿误传旧稿）
