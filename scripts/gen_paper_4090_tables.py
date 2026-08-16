"""生成论文 Table 6/7 的 4090 LaTeX 替换片段（读取 scene_out_4090 套件）。

输出：docs/paper_4090_tables_67.tex（可直接替换粘贴）
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "result/scene_out_4090"
SCENES = [
    ("coastal_joint_assault", "Coastal joint assault", "assault"),
    ("island_resupply_corridor", "Island resupply corridor", "sustain"),
    ("mountain_corridor_recon", "Mountain corridor recon", "recon"),
    ("river_crossing_breakthrough", "River crossing breakthrough", "assault"),
    ("urban_hub_defense", "Urban hub defense", "defense"),
]
VARIANTS = ["pure_llm", "memento", "hope", "full"]


def load_best_sim(scene: str, variant: str) -> dict:
    path = SUITE / scene / f"ablation_{variant}" / "full_result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["best_simulation"]


def pct(x: float) -> str:
    return f"{x * 100:.2f}"


def main() -> None:
    rows = []
    for scene, label, mission in SCENES:
        sims = {}
        for v in VARIANTS:
            try:
                sims[v] = load_best_sim(scene, v)
            except FileNotFoundError:
                sims[v] = None
        if sims["pure_llm"] is None or sims["full"] is None:
            print(f"[skip] {scene}: missing pure/full result")
            continue
        rows.append((scene, label, mission, sims))

    # ---- Table 6: suite summary（aggregate）----
    n = len(rows)
    avg = {v: sum(r[3][v]["mission_success"] for r in rows) / n for v in VARIANTS}
    avg_oe = {v: sum(r[3][v]["overall_effectiveness"] for r in rows) / n for v in VARIANTS}
    full_wins = sum(1 for r in rows if r[3]["full"]["mission_success"] > r[3]["pure_llm"]["mission_success"])
    full_best = sum(
        1
        for r in rows
        if r[3]["full"]["mission_success"] >= max(
            (r[3][v]["mission_success"] for v in VARIANTS if r[3][v] is not None)
        )
    )

    out = []
    out.append("% ===== Table 6 (tab:generalization-summary) — 4090 scene-out suite =====")
    out.append(f"% 场景数 {n} | Full 胜 Pure {full_wins}/{n} | Full 不低于全部变体 {full_best}/{n}")
    out.append("\\begin{table}[ht]")
    out.append("\\centering")
    out.append("\\caption{Cross-scenario transfer summary (4090, 20-iteration scene-out suite)."
               " All settings use default HOPE parameters ($\\theta=0.5$, $\\epsilon=0.02$).}")
    out.append("\\label{tab:generalization-summary}")
    out.append("\\small")
    out.append("\\begin{tabular}{lcc}")
    out.append("\\toprule")
    out.append("Setting & Avg. MS & Avg. OE \\\\")
    out.append("\\midrule")
    for v, name in [("pure_llm", "Pure LLM"), ("memento", "LLM + Memento"),
                    ("hope", "LLM + Hope"), ("full", "Full")]:
        out.append(f"{name} & {pct(avg[v])} & {pct(avg_oe[v])} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")

    # ---- Table 7: per-scene rows ----
    out.append("% ===== Table 7 (tab:generalization-scenes) — 4090 per-scene =====")
    out.append("\\begin{table}[ht]")
    out.append("\\centering")
    out.append("\\caption{Per-scene mission success (4090, 20-iteration scene-out suite).}")
    out.append("\\label{tab:generalization-scenes}")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    out.append("\\begin{tabular}{lcccccc}")
    out.append("\\toprule")
    out.append("Scenario & Family & Pure & Memento & Hope & Full & $\\Delta$MS \\\\")
    out.append("\\midrule")
    for scene, label, mission, sims in rows:
        pure = sims["pure_llm"]["mission_success"]
        full = sims["full"]["mission_success"]
        mem = sims["memento"]["mission_success"] if sims["memento"] else None
        hope = sims["hope"]["mission_success"] if sims["hope"] else None
        out.append(
            f"{label} & {mission} & {pct(pure)} & {pct(mem) if mem else '--'} & "
            f"{pct(hope) if hope else '--'} & {pct(full)} & {full - pure:+.2f} \\\\"
        )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")

    out.append("% ===== 叙述要点（results.tex 段落替换参考） =====")
    out.append(f"% - Full 胜 Pure：{full_wins}/{n} 场景；平均 MS Pure={pct(avg['pure_llm'])} "
               f"Full={pct(avg['full'])}（Δ{(avg['full'] - avg['pure_llm']) * 100:+.2f} 点）")
    out.append(f"% - Full 不低于全部变体：{full_best}/{n} 场景")
    out.append("% - 若有场景非 Full 最优，列出（如 river 场景 Hope 领先），如实报告")
    out.append("")

    text = "\n".join(out) + "\n"
    out_path = ROOT / "docs/paper_4090_tables_67.tex"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
