from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


STAGES = ["detect", "disrupt", "breach", "control", "sustain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比两组军事研究实验结果。")
    parser.add_argument("--baseline", required=True, help="基线 full_result.json 路径")
    parser.add_argument("--candidate", required=True, help="候选 full_result.json 路径")
    parser.add_argument("--output", required=True, help="输出 Markdown 路径，或 '-' 直接打印")
    return parser.parse_args()


def load_result(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def best_simulation(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("best_simulation", {})


def best_iteration_by(history: List[Dict[str, Any]], metric: str) -> Tuple[int | None, Dict[str, Any]]:
    if not history:
        return None, {}
    valid = [
        item
        for item in history
        if isinstance(item, dict)
        and isinstance(item.get("simulation"), dict)
        and metric in item["simulation"]
    ]
    if not valid:
        return None, {}
    best = max(valid, key=lambda item: item["simulation"][metric])
    return int(best.get("iteration", 0) or 0), best["simulation"]


def metric_delta(candidate: float, baseline: float) -> str:
    return f"{candidate - baseline:+.4f}"


def stage_lines(baseline_scores: Dict[str, float], candidate_scores: Dict[str, float]) -> List[str]:
    lines: List[str] = []
    for key in STAGES:
        base = baseline_scores.get(key)
        cand = candidate_scores.get(key)
        if base is None and cand is None:
            lines.append(f"| {key} | N/A | N/A | N/A |")
        elif base is None:
            lines.append(f"| {key} | N/A | {cand:.4f} | N/A |")
        elif cand is None:
            lines.append(f"| {key} | {base:.4f} | N/A | N/A |")
        else:
            lines.append(f"| {key} | {base:.4f} | {cand:.4f} | {cand - base:+.4f} |")
    return lines


def bottleneck_text(stage_scores: Dict[str, float]) -> str:
    if not stage_scores:
        return "未提供五阶段得分"
    ordered = sorted(stage_scores.items(), key=lambda item: item[1])
    return "、".join(f"{name}={score:.4f}" for name, score in ordered[:2])


def iteration_text(history: List[Dict[str, Any]], best_result: Dict[str, Any]) -> str:
    best_ms_iter, best_ms = best_iteration_by(history, "mission_success")
    best_oe_iter, best_oe = best_iteration_by(history, "overall_effectiveness")
    if best_ms_iter is None or best_oe_iter is None:
        mission_success = best_result.get("mission_success", "N/A")
        overall = best_result.get("overall_effectiveness", "N/A")
        return (
            "未提供有效 optimization_history，直接采用 best_simulation；"
            f"mission_success={mission_success}，overall_effectiveness={overall}。"
        )
    return (
        f"共 {len(history)} 轮；最高 mission_success 出现在第 {best_ms_iter} 轮"
        f"（{best_ms.get('mission_success', 0.0):.4f}），"
        f"最高 overall_effectiveness 出现在第 {best_oe_iter} 轮"
        f"（{best_oe.get('overall_effectiveness', 0.0):.4f}）。"
    )


def monte_text(simulation: Dict[str, Any], metric: str) -> str:
    stats = simulation.get("monte_carlo_stats", {}).get(metric)
    if not stats:
        return "N/A"
    mean = stats.get("mean", "N/A")
    std = stats.get("std", "N/A")
    ci95_low = stats.get("ci95_low", "N/A")
    ci95_high = stats.get("ci95_high", "N/A")
    n = stats.get("n", "N/A")
    return f"{mean} ± {std} (95% CI: [{ci95_low}, {ci95_high}], n={n})"


def build_report(
    baseline_name: str,
    baseline: Dict[str, Any],
    candidate_name: str,
    candidate: Dict[str, Any],
) -> str:
    baseline_best = best_simulation(baseline)
    candidate_best = best_simulation(candidate)
    baseline_hist = baseline.get("optimization_history") or []
    candidate_hist = candidate.get("optimization_history") or []
    baseline_stage_scores = baseline_best.get("stage_scores", {}) or {}
    candidate_stage_scores = candidate_best.get("stage_scores", {}) or {}

    lines = [
        "# 毕设实验对照分析报告",
        "",
        "## 1. 实验对象",
        f"- 基线结果：{baseline_name}",
        f"- 候选结果：{candidate_name}",
        f"- 作战任务：{candidate.get('scenario', {}).get('name', 'N/A')}",
        f"- 作战目标：{candidate.get('scenario', {}).get('objective', 'N/A')}",
        "",
        "## 2. 最优指标对比",
        "",
        "| 指标 | 基线 | 候选 | 增量 |",
        "| --- | ---: | ---: | ---: |",
        f"| mission_success | {baseline_best.get('mission_success', 0.0):.4f} | {candidate_best.get('mission_success', 0.0):.4f} | {metric_delta(candidate_best.get('mission_success', 0.0), baseline_best.get('mission_success', 0.0))} |",
        f"| overall_effectiveness | {baseline_best.get('overall_effectiveness', 0.0):.4f} | {candidate_best.get('overall_effectiveness', 0.0):.4f} | {metric_delta(candidate_best.get('overall_effectiveness', 0.0), baseline_best.get('overall_effectiveness', 0.0))} |",
        f"| ler | {baseline_best.get('ler', 0.0):.4f} | {candidate_best.get('ler', 0.0):.4f} | {metric_delta(candidate_best.get('ler', 0.0), baseline_best.get('ler', 0.0))} |",
        f"| completion_time_hours | {baseline_best.get('completion_time_hours', 0.0):.2f} | {candidate_best.get('completion_time_hours', 0.0):.2f} | {metric_delta(candidate_best.get('completion_time_hours', 0.0), baseline_best.get('completion_time_hours', 0.0))} |",
        f"| survivability | {baseline_best.get('survivability', 0.0):.4f} | {candidate_best.get('survivability', 0.0):.4f} | {metric_delta(candidate_best.get('survivability', 0.0), baseline_best.get('survivability', 0.0))} |",
        f"| command_resilience | {baseline_best.get('command_resilience', 0.0):.4f} | {candidate_best.get('command_resilience', 0.0):.4f} | {metric_delta(candidate_best.get('command_resilience', 0.0), baseline_best.get('command_resilience', 0.0))} |",
        "",
        "## 3. Monte Carlo 稳定性",
        f"- 基线 mission_success：{monte_text(baseline_best, 'mission_success')}",
        f"- 候选 mission_success：{monte_text(candidate_best, 'mission_success')}",
        f"- 基线 overall_effectiveness：{monte_text(baseline_best, 'overall_effectiveness')}",
        f"- 候选 overall_effectiveness：{monte_text(candidate_best, 'overall_effectiveness')}",
        "",
        "## 4. 五阶段杀伤链得分对比",
        "",
        "| 阶段 | 基线 | 候选 | 增量 |",
        "| --- | ---: | ---: | ---: |",
        *stage_lines(baseline_stage_scores, candidate_stage_scores),
        "",
        "## 5. 迭代历史对比",
        f"- 基线：{iteration_text(baseline_hist, baseline_best)}",
        f"- 候选：{iteration_text(candidate_hist, candidate_best)}",
        "",
        "## 6. 关键发现",
    ]

    if candidate_best.get("mission_success", 0.0) > baseline_best.get("mission_success", 0.0):
        lines.append(
            f"- 候选结果的任务成功率相对基线提升了 "
            f"{candidate_best.get('mission_success', 0.0) - baseline_best.get('mission_success', 0.0):+.4f}。"
        )
    else:
        lines.append("- 候选结果的任务成功率未超过基线，说明当前增强尚未稳定转化为最终任务优势。")

    lines.extend(
        [
            f"- 基线主要瓶颈：{bottleneck_text(baseline_stage_scores)}",
            f"- 候选主要瓶颈：{bottleneck_text(candidate_stage_scores)}",
            "",
            "## 7. 论文写作建议",
            "- 先报告 mission_success、LER、overall_effectiveness 的整体变化，再用五阶段得分解释性能来源。",
            "- 对缺失 history 或 stage_scores 的旧结果，明确说明评价口径差异，避免做伪精确比较。",
            "- 将候选方案的最低分阶段作为后续优化重点，而不是只看整体均值。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    output_path = Path(args.output)

    baseline = load_result(baseline_path)
    candidate = load_result(candidate_path)
    report = build_report(
        baseline_path.parent.name,
        baseline,
        candidate_path.parent.name,
        candidate,
    )
    if args.output == "-":
        print(report)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": str(baseline_path),
                "candidate": str(candidate_path),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
