# 论文 RTX 4090 D 全量替换包（data replacement package）

> 用途：将论文正文（manuscript/sections/*.tex）中的 RTX 3060 实验描述与结果
> 全量替换为 RTX 4090 D 实测数据，统一硬件口径，保证全稿数据链 100% 闭环。
> 生成日期：2026-08（数据源见每节末尾溯源）。
> 状态：⏳ 泛化套件复跑完成后填充 Table 6/7 与摘要泛化数字。

## 0. 口径决策（重要）

| 项 | 论文现状 | 4090 替换后 | 说明 |
| --- | --- | --- | --- |
| HOPE 参数 | Table 4/5 用优化参数（θ=0.9, ε=0.005），Table 12 用默认参数 | **全稿统一默认参数（θ=0.5, ε=0.02）** | 4090 复跑为默认参数口径；保留优化参数会形成混合口径 |
| 硬件 | RTX 3060 (12GB) / i7-13700F | **RTX 4090 D (24GB) / i7-13700KF** | runtime_manifest 实测 |
| 单套消融耗时 | 61 min | **6 min（10.2×）** | runtime_stats 实测 |
| 软件栈 | Python 3.12.11, torch 2.2.0+cu121, CUDA available | **Python 3.12.7, torch 2.5.1+cpu, CUDA not available（GPU 推理由本地 OpenAI-compatible 服务承担）** | 4090 runtime_manifest 实测 |
| 优化参数表（tab:multiseed-optimized） | 3060 优化参数 5-seed | **删除或改写为 4090 默认参数扩展** | 避免混合口径（建议删除，正文叙述合并进 Table 12 段） |
| Reflection-only 行 | 3060 单场景 0.6340 | **4090 补跑（⏳ 待补跑）** | 泛化套件跑完后再补（~2 min） |

---

## 1. 硬件与运行元数据替换

### 1.1 `experimental_setup.tex:79`

**旧：**
```
The refreshed canonical manuscript evidence was rerun with the local backend
model \texttt{deepseek-r1-0528-qwen3-8b} inside a dedicated Python environment.
The recorded runtime manifests report Python 3.12.11, PyTorch 2.2.0+cu121, an
OpenAI-compatible local API endpoint, a 13th Gen Intel(R) Core(TM) i7-13700F CPU,
and an NVIDIA GeForce RTX 3060 GPU with CUDA available. ...
```

**新：**
```
The refreshed canonical manuscript evidence was rerun with the local backend
model \texttt{deepseek-r1-0528-qwen3-8b} inside a dedicated Python environment.
The recorded runtime manifests report Python 3.12.7, PyTorch 2.5.1, an
OpenAI-compatible local API endpoint, a 13th Gen Intel(R) Core(TM) i7-13700KF CPU,
and an NVIDIA GeForce RTX 4090 D GPU (24\,GB). The Python-side torch build is
CPU-only; GPU inference is served by the local OpenAI-compatible endpoint.
...
```

### 1.2 `discussion.tex:22`

**旧：** `... on a 13th Gen Intel(R) Core(TM) i7-13700F CPU with an NVIDIA GeForce RTX 3060 GPU, using an OpenAI-compatible local endpoint.`

**新：** `... on a 13th Gen Intel(R) Core(TM) i7-13700KF CPU with an NVIDIA GeForce RTX 4090 D GPU (24\,GB), using an OpenAI-compatible local endpoint.`

> 溯源：`result/ablation_4090_multiseed/seed_7/ablation_full/full_result.json -> runtime_manifest`

---

## 2. Table 4（tab:ablation-main）替换（4090 默认参数）

**旧表（优化参数口径，含 Reflection 行）：**
```
Pure LLM & 62.60 & 73.28 & 1.66 & 11.41 & 60.96 \\
LLM + Memento & 62.71 & 72.21 & 1.63 & 14.59 & 60.96 \\
LLM + Hope & 62.29 & 75.72 & 1.76 & 9.93 & 71.56 \\
LLM + Reflection & 63.40 & 74.47 & 1.75 & 11.39 & 60.96 \\
Full & 67.07 & 80.11 & 2.27 & 10.98 & 75.33 \\
```

**新表（4090 默认参数，MS×100）：**
```
Pure LLM & 62.75 & 73.84 & 1.71 & 11.41 & 60.96 \\
LLM + Memento & 61.25 & 70.90 & 1.52 & 13.48 & 60.96 \\
LLM + Hope & 64.45 & 77.29 & 1.97 & 10.92 & 63.59 \\
LLM + Reflection & \textcolor{gray}{待补跑} & & & & \\
Full & 65.76 & 78.75 & 2.19 & 14.18 & 76.08 \\
```

**caption 改写建议：** 删除 "The Full setting uses optimized HOPE SDE parameters
($\theta=0.9$, $\epsilon=0.005$)..."，改为 "All groups use default HOPE parameters
($\theta=0.5$, $\epsilon=0.02$)."（全稿统一默认参数口径）

> 溯源：`result/ablation_suite_4090_rerun/ablation_{pure_llm,memento,hope,full}/full_result.json -> best_simulation`

---

## 3. Table 5（tab:ablation-stage）替换（4090 默认参数）

**新表（MS×100）：**
```
detect & 62.24 & 61.00 & 64.94 & — & 66.65 \\
disrupt & 57.31 & 61.26 & 57.78 & — & 65.69 \\
breach & 55.91 & 55.83 & 58.57 & — & 60.88 \\
control & 64.43 & 60.88 & 66.50 & — & 67.97 \\
sustain & 68.40 & 64.03 & 70.09 & — & 66.57 \\
```

**叙述改写要点：**
- "Full 最强阶段 detect(66.65)/control(67.97)，breach(60.88) 仍为最弱瓶颈"——与原稿
  "breach 结构性瓶颈"叙事一致；
- 删除 "optimized HOPE SDE parameters (θ=0.9, ε=0.005)" 表述；
- "Full 领先 every stage" 的叙述需按 4090 数据复核（4090 下 sustain 中 Full 66.57
  低于 Hope 70.09，叙述需改为 "Full 领先 detect/disrupt/breach/control 四阶段，
  sustain 由 Hope 领先"——**诚实修正，避免审稿漏洞**）。

> 溯源：`result/ablation_suite_4090_rerun/ablation_{...}/full_result.json -> best_simulation.stage_scores`

---

## 4. Table 12（tab:multiseed）替换（4090 5-seed 配对检验）

**旧叙述：** mean MS gain +4.14 (SD=1.66), t(4)=5.593, p<0.01, d=2.50

**新叙述：**
```
The Full pipeline improves mission success in every seed, with per-seed gains
ranging from +2.04 (seed 42) to +7.71 (seed 123). The mean mission-success gain is
+3.81 (SD=2.27), and a paired t-test across the five seeds yields t(4)=3.75, which
is significant at p<0.05 (two-tailed). Cohen's d=1.68 indicates a large effect
size. The mean overall-effectiveness gain is +4.85 (SD=1.84) with t(4)=5.87 and
d=2.63 (p<0.01), and the mean LER gain is +0.486 (t=6.60, p<0.01).
```

**新表（MS×100，per-seed）：**
```
seed & Pure & Full & \Delta MS \\
7 & 63.33 & 66.92 & +3.59 \\
42 & 63.92 & 65.96 & +2.04 \\
123 & 63.97 & 71.68 & +7.71 \\
999 & 62.73 & 65.99 & +3.26 \\
2024 & 65.64 & 68.08 & +2.44 \\
```

> 溯源：`result/ablation_4090_multiseed/summary.md`（per-seed 明细）
> p 值由 scipy 配对 t 检验复核：MS p=0.0199, OE p=0.0042, LER p=0.0027

---

## 5. 摘要（abstract.tex）重写要点（⏳ 泛化数据到达后定稿）

- 62.60→67.07（优化参数）改为 **62.75→65.76（4090 默认参数，ΔMS=+3.01 点）**；
- t(4)=5.593, p<0.01 → **t(4)=3.75, p=0.02（5-seed 4090）**；
- 泛化 65.95→68.28 → **⏳ 待 scene_out_4090 数据**；
- "optimized HOPE parameters" 相关表述删除。

---

## 6. Table 6/7（tab:generalization-summary / tab:generalization-scenes）替换

⏳ 等待 `result/scene_out_4090/` 套件完成（后台运行中，5 场景 × 20 轮 × 4 分支）。
完成后：读取各场景 ablation_{pure_llm,memento,hope,full}/full_result.json -> best_simulation，
生成 4090 泛化汇总表 + 场景明细表 + 摘要泛化数字。

---

## 7. 待办

- [ ] 泛化套件完成 → 填 Table 6/7 + 摘要泛化数字
- [ ] 4090 补跑 reflection-only 单场景（~2 min，勿与泛化并行）→ 填 Table 4/5 Reflection 行
- [ ] 决定 tab:multiseed-optimized 表去留（建议删除，正文合并进 Table 12 段）
- [ ] 生成最终 LaTeX 替换片段（含完整 table 环境代码）供合入
