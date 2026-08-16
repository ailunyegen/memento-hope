"""生成论文 RTX 4090 D 全量替换包 docs/paper_4090_replacement.tex。

从 4090 各套件 full_result.json 自动提取数据，生成可直接替换粘贴的
LaTeX 片段（Table 4/5/6/7/12 + 硬件元数据 + 摘要 + 叙述改写）。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ms(x: float) -> str:
    return f"{x * 100:.2f}"


def main() -> None:
    suite = ROOT / "result/ablation_suite_4090_rerun"
    variants = {
        "Pure LLM": "ablation_pure_llm",
        "LLM + Memento": "ablation_memento",
        "LLM + Hope": "ablation_hope",
        "Full": "ablation_full",
    }
    reflection = load(ROOT / "result/ablation_reflection_only_4090/full_result.json")["best_simulation"]
    data = {}
    for name, folder in variants.items():
        sim = load(suite / folder / "full_result.json")["best_simulation"]
        data[name] = sim

    def row(name):
        sim = data[name]
        return (
            f"{name} & {ms(sim['mission_success'])} & {ms(sim['overall_effectiveness'])} & "
            f"{sim['ler']:.2f} & {sim['completion_time_hours']:.2f} & {ms(sim['command_resilience'])} \\\\"
        )

    refl_row = (
        f"LLM + Reflection & {ms(reflection['mission_success'])} & {ms(reflection['overall_effectiveness'])} & "
        f"{reflection['ler']:.2f} & {reflection['completion_time_hours']:.2f} & {ms(reflection['command_resilience'])} \\\\"
    )

    # Table 5 stage rows
    stages = ["detect", "disrupt", "breach", "control", "sustain"]
    def stage_row(stage):
        cells = [f"{stage}"]
        for name in ["Pure LLM", "LLM + Memento", "LLM + Hope", "Full"]:
            cells.append(ms(data[name]["stage_scores"].get(stage, 0.0)))
        cells.append(ms(reflection["stage_scores"].get(stage, 0.0)))
        # 列序：Stage & Pure & Memento & Hope & Full & Reflection
        return " & ".join(cells) + r" \\"

    # multiseed per-seed
    ms_suite = ROOT / "result/ablation_4090_multiseed"
    seeds = [7, 42, 123, 999, 2024]
    per_seed = []
    for s in seeds:
        pure = load(ms_suite / f"seed_{s}" / "ablation_pure_llm" / "full_result.json")["best_simulation"]["mission_success"]
        full = load(ms_suite / f"seed_{s}" / "ablation_full" / "full_result.json")["best_simulation"]["mission_success"]
        per_seed.append(f"{s} & {ms(pure)} & {ms(full)} & {(full - pure) * 100:+.2f} \\\\")

    # generalization table content (from generated table file, re-emit)
    gen = (ROOT / "docs/paper_4090_tables_67.tex").read_text(encoding="utf-8")

    # aggregate numbers for abstract/narrative
    agg_ms_pure = sum(data[n]["mission_success"] for n in ["Pure LLM"]) + 0  # placeholder not used

    out = []
    out.append(r"""% ============================================================================
% 论文 RTX 4090 D 全量替换包（自动生成，2026-08）
% 应用方式：按节替换 manuscript/sections/*.tex 对应段落/表格。
% 口径：全稿统一默认 HOPE 参数（theta=0.5, eps=0.02）；已删除优化参数表
%       （tab:multiseed-optimized），避免混合口径。
% 数据源：result/ablation_suite_4090_rerun | ablation_4090_multiseed |
%         scene_out_4090 | ablation_reflection_only_4090（runtime_manifest 实测）
% ============================================================================

% ----------------------------------------------------------------------------
% 1) experimental_setup.tex:79 硬件与运行元数据（替换整段 "The refreshed
%    canonical manuscript evidence was rerun ..."）
% ----------------------------------------------------------------------------
\paragraph{替换文本：}
The refreshed canonical manuscript evidence was rerun with the local backend
model \texttt{deepseek-r1-0528-qwen3-8b} inside a dedicated Python environment.
The recorded runtime manifests report Python 3.12.7, PyTorch 2.5.1, an
OpenAI-compatible local API endpoint, a 13th Gen Intel(R) Core(TM) i7-13700KF
CPU, and an NVIDIA GeForce RTX 4090 D GPU (24\,GB). The Python-side torch build
is CPU-only; GPU inference is served by the local OpenAI-compatible endpoint.
Newly generated \texttt{full\_result.json} files emit this information through a
top-level \texttt{runtime\_manifest} ...

% ----------------------------------------------------------------------------
% 2) discussion.tex:22 硬件句（替换 "...i7-13700F CPU with an NVIDIA GeForce
%    RTX 3060 GPU..." 为下句）
% ----------------------------------------------------------------------------
\paragraph{替换文本：}
... on a 13th Gen Intel(R) Core(TM) i7-13700KF CPU with an NVIDIA GeForce RTX
4090 D GPU (24\,GB), using an OpenAI-compatible local endpoint.

% ----------------------------------------------------------------------------
% 3) Table 4（tab:ablation-main）替换（4090 默认参数，含 Reflection 行）
%    caption 改为：All groups use default HOPE parameters ($\theta=0.5$,
%    $\epsilon=0.02$). The reflection-only row uses the matched follow-up rerun.
% ----------------------------------------------------------------------------
\begin{table}[ht]
\centering
\caption{Main ablation results on the sample coastal joint-assault scenario
(RTX 4090 D). All groups use 10 planning iterations, 50 Monte Carlo runs,
seed 7, writeback disabled, and default HOPE parameters
($\theta=0.5$, $\epsilon=0.02$).}
\label{tab:ablation-main}
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lccccc}
\toprule
Setting & MS & OE & LER & Time (h) & Cmd.\ resilience \\
\midrule
""")
    for name in ["Pure LLM", "LLM + Memento", "LLM + Hope"]:
        out.append(row(name))
    out.append(refl_row)
    out.append(row("Full"))
    out.append(r"""\bottomrule
\end{tabular}
\end{table}

% ----------------------------------------------------------------------------
% 4) Table 5（tab:ablation-stage）替换（列序：Stage & Pure & Memento & Hope &
%    Full & Reflection）
% ----------------------------------------------------------------------------
\begin{table}[ht]
\centering
\caption{Five-stage kill-chain scores for the ablation study (RTX 4090 D).}
\label{tab:ablation-stage}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
Stage & Pure LLM & \shortstack{LLM +\\Memento} & \shortstack{LLM +\\Hope} & Full & \shortstack{LLM +\\Reflection} \\
\midrule
""")
    for st in stages:
        out.append(stage_row(st))
    out.append(r"""\bottomrule
\end{tabular}
\setlength{\tabcolsep}{6pt}
\end{table}

% ----------------------------------------------------------------------------
% 5) Table 12（tab:multiseed）替换（4090 5-seed 配对检验）
%    叙述：mean MS gain +3.81 (SD=2.27), t(4)=3.75, p=0.02, d=1.68;
%          OE gain +4.85 (t=5.87, p<0.01); LER gain +0.486 (t=6.60, p<0.01)
% ----------------------------------------------------------------------------
\begin{table}[ht]
\centering
\caption{Multi-seed replication (RTX 4090 D): Pure LLM vs.\ Full pipeline
across five seeds. All runs use 10 iterations, 50 Monte Carlo runs, default
HOPE parameters, and writeback disabled.}
\label{tab:multiseed}
\small
\begin{tabular}{lccc}
\toprule
Seed & Pure LLM & Full & $\Delta$MS \\
\midrule
""")
    out.extend(per_seed)
    out.append(r"""\bottomrule
\end{tabular}
\end{table}

% ----------------------------------------------------------------------------
% 6) Table 6/7（tab:generalization-summary / tab:generalization-scenes）
% ----------------------------------------------------------------------------
""")
    out.append(gen)

    out.append(r"""
% ----------------------------------------------------------------------------
% 7) 摘要（abstract.tex）重写要点
% ----------------------------------------------------------------------------
% - 单场景：Full 相对 Pure：MS 62.75 -> 65.76（Δ+3.01 点），OE 73.84 -> 78.75
%   （Δ+4.91 点），LER 1.71 -> 2.19（4090 默认参数）。
% - 多种子：5 seeds 配对 ΔMS = +3.81（SD=2.27），t(4)=3.75，p=0.02，
%   Cohen's d=1.68；ΔOE t(4)=5.87 p<0.01；ΔLER t(4)=6.60 p<0.01。
% - 泛化：5 场景中 Full 胜 Pure 4/5，平均 MS 65.01 -> 66.48（Δ+1.48 点），
%   平均 OE 72.49 -> 77.92（Δ+5.43 点）；coastal/river 场景 Hope 领先，
%   urban 场景 Pure 领先（seed 7 单次运行，如实报告）。
% - 删除所有 "optimized HOPE parameters (theta=0.9, eps=0.005)" 表述。

% ----------------------------------------------------------------------------
% 8) results.tex 叙述改写要点
% ----------------------------------------------------------------------------
% - Table 4 段：Full 相对 Pure 提升改为 +3.01 点 MS、+4.91 点 OE；删除
%   "The Full setting uses optimized HOPE SDE parameters" 表述。
% - Table 5 段：如实改写为 "Full 在 detect/disrupt/breach/control 四个阶段
%   领先，sustain 阶段 Hope 保持单项优势（70.09 vs 66.57）——该现象与
%   breach/sustain 竞争 protection 维度的结构性张力一致（见 4.5 节）"。
% - 泛化段：Full 胜 Pure 由 "all five scenarios" 改为 "4/5 scenarios
%   （urban-defense 除外）"；平均增益 +1.48 点；说明单 seed 单次运行方差。
% - 删除 tab:multiseed-optimized 表及 "optimized parameters" 相关段落。
% - Figure 数据（ablation_stage_profile.png 等）需按新表重新生成
%   （scripts/generate_paper_figures.py 数据源切换为 4090 套件）。
""")

    text = "\n".join(out) + "\n"
    out_path = ROOT / "docs/paper_4090_replacement.tex"
    out_path.write_text(text, encoding="utf-8")
    print(f"saved: {out_path} ({len(text)} bytes)")
    print("\n---- Table 4 preview ----")
    for name in ["Pure LLM", "LLM + Memento", "LLM + Hope", "Full"]:
        print("  " + row(name))
    print("  " + refl_row)
    print("\n---- Table 12 preview ----")
    print("\n".join("  " + r for r in per_seed))


if __name__ == "__main__":
    main()
