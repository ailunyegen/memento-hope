"""HOPE weight convergence visualization and breach bottleneck analysis.

Generates:
1. Slow/fast weight trajectory plots across iterations
2. Stage-capability response heatmaps
3. Breach bottleneck persistence analysis (variance, weight competition)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_optimization_history(full_result_path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(full_result_path).read_text(encoding="utf-8"))
    return data.get("optimization_history", [])


def extract_weight_trajectories(
    history: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, float]]]:
    """Extract slow/fast/fused weight vectors from each iteration."""
    trajectories: Dict[str, List[Dict[str, float]]] = {
        "slow": [],
        "fast": [],
        "fused": [],
    }
    for entry in history:
        plan = entry.get("plan", {})
        for key in trajectories:
            weights = plan.get(f"{key}_weights" if key != "fused" else "fused_weights")
            if weights:
                trajectories[key].append(dict(weights))
    return trajectories


def extract_stage_trajectories(
    history: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    """Extract per-stage scores across iterations."""
    stages: Dict[str, List[float]] = defaultdict(list)
    for entry in history:
        sim = entry.get("simulation", {})
        stage_scores = sim.get("stage_scores", {})
        for stage_name in ["detect", "disrupt", "breach", "control", "sustain"]:
            stages[stage_name].append(stage_scores.get(stage_name, float("nan")))
    return dict(stages)


def analyze_breach_bottleneck(
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Diagnose why breach remains the persistent bottleneck.

    Returns:
        breach_variance_ratio: breach variance / mean variance of other stages
        correlated_stages: stages whose scores correlate with breach
        weight_competition: capability dimensions shared between breach and
            the highest-scoring stage, indicating weight competition
    """
    stage_vals: Dict[str, List[float]] = defaultdict(list)
    for entry in history:
        sim = entry.get("simulation", {})
        ss = sim.get("stage_scores", {})
        for s in ["detect", "disrupt", "breach", "control", "sustain"]:
            stage_vals[s].append(ss.get(s, float("nan")))

    # 1. Variance analysis
    variances = {}
    for s, vals in stage_vals.items():
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            variances[s] = sum((v - mean) ** 2 for v in vals) / len(vals)
        else:
            variances[s] = 0.0

    breach_var = variances.get("breach", 0.0)
    other_vars = [v for k, v in variances.items() if k != "breach"]
    mean_other_var = sum(other_vars) / max(1, len(other_vars))
    breach_variance_ratio = breach_var / max(1e-8, mean_other_var)

    # 2. Stage score correlations with breach
    correlations = {}
    breach_vals = stage_vals.get("breach", [])
    if len(breach_vals) >= 3:
        for s, vals in stage_vals.items():
            if s == "breach":
                continue
            n = min(len(breach_vals), len(vals))
            correlations[s] = _pearson_r(breach_vals[:n], vals[:n])

    # 3. Capability weight competition analysis
    # breach depends on: mobility(0.30), protection(0.25), fires(0.30)
    # detect depends on: awareness(0.45), c2(0.25), ew(0.15)
    # If detect/sustain get high weights and breach shares some capabilities with them,
    # there's competition for limited weight budget
    stage_to_cap = {
        "detect": {"awareness": 0.42, "c2": 0.20, "ew": 0.18},
        "disrupt": {"fires": 0.42, "ew": 0.20, "c2": 0.16},
        "breach": {"mobility": 0.30, "fires": 0.30, "protection": 0.25},
        "control": {"protection": 0.35, "c2": 0.35, "mobility": 0.30},
        "sustain": {"sustainment": 0.40, "c2": 0.30, "protection": 0.25},
    }

    # Find strongest non-breach stage
    mean_stage_scores = {}
    for s, vals in stage_vals.items():
        if vals:
            mean_stage_scores[s] = sum(vals) / len(vals)

    if mean_stage_scores:
        strongest = max(
            (s for s in mean_stage_scores if s != "breach"),
            key=lambda s: mean_stage_scores[s],
            default="sustain",
        )
        breach_caps = stage_to_cap.get("breach", {})
        strong_caps = stage_to_cap.get(strongest, {})
        shared_caps = set(breach_caps) & set(strong_caps)
        weight_competition = {
            "strongest_stage": strongest,
            "strongest_score": round(mean_stage_scores.get(strongest, 0), 4),
            "breach_score": round(mean_stage_scores.get("breach", 0), 4),
            "shared_capabilities": sorted(shared_caps),
            "breach_cap_weights": {k: round(v, 3) for k, v in breach_caps.items()},
            f"{strongest}_cap_weights": {k: round(v, 3) for k, v in strong_caps.items()},
        }
    else:
        weight_competition = {}

    return {
        "breach_variance_ratio": round(breach_variance_ratio, 4),
        "breach_var": round(breach_var, 6),
        "mean_other_var": round(mean_other_var, 6),
        "stage_variances": {k: round(v, 6) for k, v in variances.items()},
        "breach_correlations": {k: round(v, 4) for k, v in correlations.items()},
        "weight_competition": weight_competition,
        "interpretation": _breach_interpretation(breach_variance_ratio, correlations, weight_competition),
    }


def _breach_interpretation(
    variance_ratio: float,
    correlations: Dict[str, float],
    competition: Dict[str, Any],
) -> str:
    parts = []
    if variance_ratio > 1.5:
        parts.append(
            f"breach 阶段的得分方差是其他阶段均值的 {variance_ratio:.1f} 倍，"
            "说明该阶段存在本质上的高不确定性，可能来自场景中的非线性因素或敌我力量对比的随机波动。"
        )
    else:
        parts.append(
            f"breach 阶段的方差比 ({variance_ratio:.2f}) 与其他阶段接近，"
            "说明其低分是系统性的，而非随机波动。"
        )

    pos_corrs = {k: v for k, v in correlations.items() if v > 0.3}
    neg_corrs = {k: v for k, v in correlations.items() if v < -0.3}
    if pos_corrs:
        names = ", ".join(f"{k}({v:.2f})" for k, v in pos_corrs.items())
        parts.append(f"breach 与 {names} 正相关，这些阶段共享能力维度，同步提升。")
    if neg_corrs:
        names = ", ".join(f"{k}({v:.2f})" for k, v in neg_corrs.items())
        parts.append(f"breach 与 {names} 负相关，说明存在权重竞争关系。")

    if competition:
        shared = competition.get("shared_capabilities", [])
        strongest = competition.get("strongest_stage", "?")
        if shared:
            parts.append(
                f"breach 与最强阶段 {strongest} 共享能力维度 {shared}，"
                "在有限的能力权重预算下，强化 {strongest} 可能挤占 breach 所需的权重资源。"
            )

    return " ".join(parts) if parts else "未发现明确的瓶颈成因。"


def _pearson_r(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (sx * sy)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_convergence_report(
    full_result_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Generate a markdown report of HOPE convergence and breach analysis."""
    history = load_optimization_history(full_result_path)
    if not history:
        return "No optimization history found."

    trajectories = extract_weight_trajectories(history)
    stages = extract_stage_trajectories(history)
    breach_analysis = analyze_breach_bottleneck(history)

    n_iterations = len(history)
    lines: List[str] = []
    lines.append("# HOPE 控制器收敛行为分析\n")

    lines.append(f"## 1. 基本统计（{n_iterations} 次迭代）\n")

    # Slow weight convergence summary
    if trajectories["slow"]:
        first = trajectories["slow"][0]
        last = trajectories["slow"][-1]
        lines.append("### Slow Weights 变化\n")
        lines.append("| 能力维度 | 初始值 | 最终值 | Δ |")
        lines.append("|----------|--------|--------|---|")
        for key in first:
            delta = last[key] - first[key]
            lines.append(f"| {key} | {first[key]:.4f} | {last[key]:.4f} | {delta:+.4f} |")
        lines.append("")

    # Stage score summary
    if stages:
        lines.append("### 阶段得分变化\n")
        lines.append("| 阶段 | 初始分 | 最终分 | Δ |")
        lines.append("|------|--------|--------|---|")
        for stage, vals in stages.items():
            if vals:
                delta = vals[-1] - vals[0]
                lines.append(f"| {stage} | {vals[0]:.4f} | {vals[-1]:.4f} | {delta:+.4f} |")
        lines.append("")

    # Breach bottleneck analysis
    lines.append("## 2. Breach 瓶颈分析\n")
    ba = breach_analysis
    lines.append(f"- **方差比**: breach_var / mean_other_var = {ba['breach_variance_ratio']}")
    lines.append(f"- **breach 方差**: {ba['breach_var']}")
    lines.append(f"- **其他阶段平均方差**: {ba['mean_other_var']}")
    lines.append("")
    lines.append("### 阶段间相关性")
    lines.append("| 阶段对 | Pearson r |")
    lines.append("|--------|-----------|")
    for stage, r in ba.get("breach_correlations", {}).items():
        lines.append(f"| breach vs {stage} | {r:+.4f} |")
    lines.append("")

    if ba.get("weight_competition"):
        wc = ba["weight_competition"]
        lines.append("### 权重竞争分析")
        lines.append(f"- 最强非-breach阶段: **{wc.get('strongest_stage', '?')}** (得分 {wc.get('strongest_score', 0)})")
        lines.append(f"- breach 得分: {wc.get('breach_score', 0)}")
        lines.append(f"- 共享能力维度: {wc.get('shared_capabilities', [])}")
        lines.append("")

    lines.append(f"### 解释\n{ba['interpretation']}\n")

    # Data tables for plotting
    lines.append("## 3. 数据表（用于绘图）\n")

    if trajectories["slow"]:
        lines.append("### Slow Weight 轨迹\n")
        cap_keys = list(trajectories["slow"][0].keys())
        header = "| iter | " + " | ".join(cap_keys) + " |"
        lines.append(header)
        lines.append("|------|" + "|".join(["------" for _ in cap_keys]) + "|")
        for i, w in enumerate(trajectories["slow"]):
            vals = " | ".join(f"{w.get(k, 0):.4f}" for k in cap_keys)
            lines.append(f"| {i+1} | {vals} |")
        lines.append("")

    if stages:
        lines.append("### 阶段得分轨迹\n")
        stage_keys = list(stages.keys())
        header = "| iter | " + " | ".join(stage_keys) + " |"
        lines.append(header)
        lines.append("|------|" + "|".join(["------" for _ in stage_keys]) + "|")
        for i in range(n_iterations):
            vals = " | ".join(f"{stages[s][i]:.4f}" if i < len(stages[s]) else "-" for s in stage_keys)
            lines.append(f"| {i+1} | {vals} |")
        lines.append("")

    report = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m military_research.visualize_convergence <full_result.json> [output.md]")
        sys.exit(1)
    input_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    report = generate_convergence_report(input_path, output)
    print(report)
