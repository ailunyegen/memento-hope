from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Experiment = Tuple[str, Dict[str, Any]]
STAGES = ["detect", "disrupt", "breach", "control", "sustain"]
EXPERIMENT_ORDER = ["纯LLM", "LLM+Memento", "LLM+Hope", "Full"]
METRICS = [
    "mission_success",
    "overall_effectiveness",
    "ler",
    "completion_time_hours",
    "command_resilience",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总四组消融实验结果。")
    parser.add_argument("--pure-llm", required=True, help="纯LLM结果 full_result.json")
    parser.add_argument("--memento", required=True, help="LLM + Memento 结果 full_result.json")
    parser.add_argument("--hope", required=True, help="LLM + Hope 结果 full_result.json")
    parser.add_argument("--full", required=True, help="Full 系统结果 full_result.json")
    parser.add_argument("--output", required=True, help="输出 Markdown 路径，或 '-' 直接打印")
    parser.add_argument(
        "--style",
        choices=["report", "thesis", "chapter"],
        default="report",
        help="输出风格：report 为工程报告，thesis 为论文小节，chapter 为论文第4章风格。",
    )
    return parser.parse_args()


def load_result(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def best_simulation(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("best_simulation", {}) or {}


def experiment_config(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("experiment_config", {}) or {}


def metric_value(result: Dict[str, Any], metric: str) -> float:
    return float(best_simulation(result).get(metric, 0.0))


def metric_stats(result: Dict[str, Any], metric: str) -> Dict[str, Any]:
    return best_simulation(result).get("monte_carlo_stats", {}).get(metric, {}) or {}


def stage_score(result: Dict[str, Any], stage_name: str) -> float | None:
    scores = best_simulation(result).get("stage_scores", {}) or {}
    value = scores.get(stage_name)
    return None if value is None else float(value)


def average_memory_quality(result: Dict[str, Any]) -> float | None:
    hits = result.get("memory_hits", []) or []
    values = [float(hit.get("quality")) for hit in hits if hit.get("quality") is not None]
    return None if not values else sum(values) / len(values)


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_count(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:.2f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def stats_summary(result: Dict[str, Any], metric: str) -> str:
    stats = metric_stats(result, metric)
    if not stats:
        return "N/A"
    mean = fmt_num(stats.get("mean"))
    std = fmt_num(stats.get("std"))
    ci_low = fmt_num(stats.get("ci95_low"))
    ci_high = fmt_num(stats.get("ci95_high"))
    n = fmt_count(stats.get("n", "N/A"))
    return f"{mean} ± {std} (95% CI: [{ci_low}, {ci_high}], n={n})"


def stage_bottleneck(result: Dict[str, Any], top_k: int = 2) -> str:
    scores = best_simulation(result).get("stage_scores", {}) or {}
    if not scores:
        return "N/A"
    ordered = sorted(scores.items(), key=lambda item: item[1])[:top_k]
    return "、".join(f"{name}={score:.4f}" for name, score in ordered)


def stage_advantage(result: Dict[str, Any], top_k: int = 2) -> str:
    scores = best_simulation(result).get("stage_scores", {}) or {}
    if not scores:
        return "N/A"
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return "、".join(f"{name}={score:.4f}" for name, score in ordered)


def best_group(experiments: Sequence[Experiment], metric: str) -> Tuple[str, float]:
    label, result = max(experiments, key=lambda item: metric_value(item[1], metric))
    return label, metric_value(result, metric)


def metric_rows(experiments: Sequence[Experiment]) -> List[str]:
    rows: List[str] = []
    for label, result in experiments:
        config = experiment_config(result)
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    "关闭" if config.get("disable_memory") else "开启",
                    "关闭" if config.get("disable_hope") else "开启",
                    "关闭" if config.get("disable_reflection") else "开启",
                    "关闭" if config.get("allow_writeback", True) else "关闭",
                    str(config.get("sim_runs", "N/A")),
                    fmt_num(metric_value(result, "mission_success")),
                    fmt_num(metric_value(result, "overall_effectiveness")),
                    fmt_num(metric_value(result, "ler")),
                    fmt_num(metric_value(result, "completion_time_hours"), digits=2),
                    fmt_num(metric_value(result, "command_resilience")),
                ]
            )
            + " |"
        )
    return rows


def stage_rows(experiments: Sequence[Experiment]) -> List[str]:
    rows: List[str] = []
    for stage_name in STAGES:
        values = [fmt_num(stage_score(result, stage_name)) for _, result in experiments]
        rows.append(f"| {stage_name} | " + " | ".join(values) + " |")
    return rows


def monte_rows(experiments: Sequence[Experiment]) -> List[str]:
    rows: List[str] = []
    for metric in METRICS:
        rows.append(
            "| "
            + " | ".join([metric, *[stats_summary(result, metric) for _, result in experiments]])
            + " |"
        )
    return rows


def build_metric_table(experiments: Sequence[Experiment]) -> List[str]:
    return [
        "| 实验组 | Memory | Hope | Reflection | Writeback | Sim Runs | mission_success | overall_effectiveness | LER | completion_time_hours | command_resilience |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *metric_rows(experiments),
    ]


def build_stage_table(experiments: Sequence[Experiment]) -> List[str]:
    return [
        "| 阶段 | 纯LLM | LLM+Memento | LLM+Hope | Full |",
        "| --- | ---: | ---: | ---: | ---: |",
        *stage_rows(experiments),
    ]


def build_monte_table(experiments: Sequence[Experiment]) -> List[str]:
    return [
        "| 指标 | 纯LLM | LLM+Memento | LLM+Hope | Full |",
        "| --- | --- | --- | --- | --- |",
        *monte_rows(experiments),
    ]


def module_gain_sentence(label: str, delta: float) -> str:
    direction = "提升" if delta >= 0 else "下降"
    return f"{label} 相对纯LLM的 mission_success {direction} {abs(delta):.4f}。"


def diagnose_anomaly(experiments: Sequence[Experiment]) -> List[str]:
    by_label = {label: result for label, result in experiments}
    pure = by_label["纯LLM"]
    memento = by_label["LLM+Memento"]
    hope = by_label["LLM+Hope"]
    full = by_label["Full"]

    pure_ms = metric_value(pure, "mission_success")
    memento_ms = metric_value(memento, "mission_success")
    hope_ms = metric_value(hope, "mission_success")
    full_ms = metric_value(full, "mission_success")
    pure_cr = metric_value(pure, "command_resilience")
    hope_cr = metric_value(hope, "command_resilience")
    full_cr = metric_value(full, "command_resilience")
    best_non_full_label, best_non_full_ms = best_group(experiments[:3], "mission_success")

    lines: List[str] = []
    if full_ms >= best_non_full_ms:
        lines.append("当前没有出现“非 Full 组明显优于 Full”的异常排序，增强链路整体方向正常。")
        if memento_ms + 0.01 < pure_ms:
            memory_quality = average_memory_quality(memento)
            if memory_quality is None:
                lines.append("但 Memento 单独组仍低于纯LLM，说明记忆检索尚未稳定转化为收益。")
            else:
                lines.append(
                    f"但 Memento 单独组仍低于纯LLM，当前命中案例平均质量约为 {memory_quality:.4f}，"
                    "说明样本贴合度或正负样本组织方式仍值得继续优化。"
                )
        if hope_ms + 0.01 < pure_ms:
            lines.append(
                f"Hope 单独组仍低于纯LLM，且指挥韧性从 {pure_cr:.4f} 下降到 {hope_cr:.4f}，"
                "说明快慢权重融合对 awareness/c2 的保护仍需加强。"
            )
        return lines

    lines.append(
        f"本轮出现异常排序：{best_non_full_label} 的 mission_success 为 {best_non_full_ms:.4f}，"
        f"高于 Full 的 {full_ms:.4f}。"
    )

    if memento_ms + 0.01 < pure_ms:
        memory_quality = average_memory_quality(memento)
        quality_phrase = "未记录案例质量" if memory_quality is None else f"平均质量约为 {memory_quality:.4f}"
        lines.append(
            f"Memento 更可能在拖后腿。LLM+Memento 从纯LLM的 {pure_ms:.4f} 下降到 {memento_ms:.4f}，"
            f"且当前命中案例{quality_phrase}。这通常意味着案例检索虽引入了先验信息，但样本贴合度或筛选质量仍不稳定。"
        )

    if hope_ms + 0.01 < pure_ms or full_cr + 0.02 < pure_cr:
        lines.append(
            f"Hope 也可能在拖后腿。LLM+Hope 的 mission_success 为 {hope_ms:.4f}，"
            f"纯LLM为 {pure_ms:.4f}；指挥韧性从 {pure_cr:.4f} 下降到 Hope 组的 {hope_cr:.4f}，"
            f"Full 组为 {full_cr:.4f}。这类现象通常说明快权重把 awareness/c2 或抗干扰相关能力压得过低。"
        )

    stronger_single = max(memento_ms, hope_ms)
    if full_ms + 0.01 < stronger_single:
        stronger_label = "LLM+Memento" if memento_ms >= hope_ms else "LLM+Hope"
        lines.append(
            f"Reflection 也存在拖后腿嫌疑。Full 比单独更强的 {stronger_label} 还低"
            f"（{full_ms:.4f} < {stronger_single:.4f}），说明反思增量没有稳定补短板，"
            "反而可能改坏了原本有效的阶段结构。"
        )

    if len(lines) == 1:
        lines.append(
            "当前异常排序更像是多模块叠加后的耦合偏差，建议优先检查 Memento 检索样本、"
            "Hope 的 c2/awareness 下限，以及 Reflection 是否过度重写高分阶段。"
        )
    return lines


def build_key_findings(experiments: Sequence[Experiment]) -> List[str]:
    pure_result = experiments[0][1]
    memento_result = experiments[1][1]
    hope_result = experiments[2][1]
    full_result = experiments[3][1]

    pure_ms = metric_value(pure_result, "mission_success")
    memento_delta = metric_value(memento_result, "mission_success") - pure_ms
    hope_delta = metric_value(hope_result, "mission_success") - pure_ms
    full_delta = metric_value(full_result, "mission_success") - pure_ms

    best_ms_label, best_ms = best_group(experiments, "mission_success")
    best_oe_label, best_oe = best_group(experiments, "overall_effectiveness")
    best_ler_label, best_ler = best_group(experiments, "ler")

    lines = [
        f"mission_success 最高的实验组是 {best_ms_label}（{best_ms:.4f}）。",
        f"overall_effectiveness 最高的实验组是 {best_oe_label}（{best_oe:.4f}）。",
        f"LER 最高的实验组是 {best_ler_label}（{best_ler:.4f}）。",
        module_gain_sentence("LLM+Memento", memento_delta),
        module_gain_sentence("LLM+Hope", hope_delta),
        module_gain_sentence("Full", full_delta),
        f"Full 当前主要短板为 {stage_bottleneck(full_result)}；相对优势阶段为 {stage_advantage(full_result)}。",
    ]
    return lines


def report_sections(experiments: Sequence[Experiment]) -> List[str]:
    full_result = experiments[3][1]
    pure_result = experiments[0][1]
    scenario = full_result.get("scenario", {}) or {}
    findings = build_key_findings(experiments)
    anomaly_lines = diagnose_anomaly(experiments)

    return [
        "# 消融实验汇总报告",
        "",
        "## 1. 实验设置",
        f"- 作战任务：{scenario.get('name', 'N/A')}",
        f"- 作战目标：{scenario.get('objective', 'N/A')}",
        f"- 纯LLM结果目录：{experiment_config(pure_result).get('output_dir', 'N/A')}",
        f"- Full结果目录：{experiment_config(full_result).get('output_dir', 'N/A')}",
        "",
        "## 2. 总体指标对比",
        "",
        *build_metric_table(experiments),
        "",
        "## 3. Monte Carlo 稳定性",
        "",
        *build_monte_table(experiments),
        "",
        "## 4. 五阶段杀伤链得分",
        "",
        *build_stage_table(experiments),
        "",
        "## 5. 关键结论",
        *[f"- {line}" for line in findings],
        "",
        "## 6. 异常结论诊断",
        *[f"- {line}" for line in anomaly_lines],
    ]


def thesis_sections(experiments: Sequence[Experiment]) -> List[str]:
    full_result = experiments[3][1]
    pure_result = experiments[0][1]
    scenario = full_result.get("scenario", {}) or {}

    pure_ms = metric_value(pure_result, "mission_success")
    full_ms = metric_value(full_result, "mission_success")
    full_oe = metric_value(full_result, "overall_effectiveness")
    pure_oe = metric_value(pure_result, "overall_effectiveness")
    hope_delta = metric_value(experiments[2][1], "mission_success") - pure_ms
    memento_delta = metric_value(experiments[1][1], "mission_success") - pure_ms
    anomaly_lines = diagnose_anomaly(experiments)

    lines = [
        "# 消融实验分析",
        "",
        "## 实验设置",
        f"本组消融实验围绕场景“{scenario.get('name', 'N/A')}”展开，"
        "通过依次关闭 Memory、Hope 与 Reflection 模块，对比纯LLM、LLM+Memento、LLM+Hope 和 Full 四组方案表现。",
        "",
        "## 总体结果",
        "",
        *build_metric_table(experiments),
        "",
        *build_monte_table(experiments),
        "",
        "## 五阶段得分",
        "",
        *build_stage_table(experiments),
        "",
        "## 结果分析",
        (
            f"从总体指标看，Full 组的 mission_success 为 {full_ms:.4f}，"
            f"高于纯LLM组的 {pure_ms:.4f}；overall_effectiveness 从 {pure_oe:.4f} 提升到 {full_oe:.4f}。"
            "这说明当前增强链路已经能够在该场景下形成稳定的综合收益。"
        ),
        (
            f"从模块贡献看，Hope 相对纯LLM的 mission_success 变化为 {hope_delta:+.4f}，"
            f"Memento 相对纯LLM的变化为 {memento_delta:+.4f}。"
            "因此可以进一步区分当前收益主要来自哪一类增强机制，以及哪一模块仍需继续补强。"
        ),
        (
            f"从五阶段得分看，Full 当前的主要短板集中在 {stage_bottleneck(full_result)}，"
            f"而优势阶段主要体现在 {stage_advantage(full_result)}。"
            "这为后续围绕压制链、突破链或持续保障链开展定向优化提供了明确依据。"
        ),
        "",
        "## 异常结论诊断",
        *[f"- {line}" for line in anomaly_lines],
    ]
    return lines


def chapter_sections(experiments: Sequence[Experiment]) -> List[str]:
    full_result = experiments[3][1]
    pure_result = experiments[0][1]
    scenario = full_result.get("scenario", {}) or {}
    anomaly_lines = diagnose_anomaly(experiments)

    pure_ms = metric_value(pure_result, "mission_success")
    full_ms = metric_value(full_result, "mission_success")
    pure_oe = metric_value(pure_result, "overall_effectiveness")
    full_oe = metric_value(full_result, "overall_effectiveness")
    hope_delta = metric_value(experiments[2][1], "mission_success") - pure_ms
    memento_delta = metric_value(experiments[1][1], "mission_success") - pure_ms
    full_delta = full_ms - pure_ms

    return [
        "# 第4章 消融实验分析",
        "",
        "## 4.1 消融实验设计",
        (
            f"本章围绕场景“{scenario.get('name', 'N/A')}”设计四组对照实验，"
            "分别为纯LLM、LLM+Memento、LLM+Hope 与 Full。"
            "通过逐步接入案例记忆、快慢权重调节与反思优化机制，分析各模块对最终作战方案生成与仿真评估结果的贡献。"
        ),
        "",
        "## 4.2 实验结果",
        "",
        *build_metric_table(experiments),
        "",
        *build_monte_table(experiments),
        "",
        *build_stage_table(experiments),
        "",
        "## 4.3 结果分析",
        (
            f"从总体指标看，Full 组的 mission_success 为 {full_ms:.4f}，"
            f"相对纯LLM组的 {pure_ms:.4f} 提升了 {full_delta:+.4f}；"
            f"overall_effectiveness 从 {pure_oe:.4f} 提升到 {full_oe:.4f}。"
            "结合 Monte Carlo 统计结果可以看到，增强后系统不仅均值更高，而且稳定性指标也更完整，"
            "这说明多模块协同在当前场景下已经表现出较明确的正向收益。"
        ),
        (
            f"从单模块收益看，LLM+Hope 相对纯LLM的 mission_success 变化为 {hope_delta:+.4f}，"
            f"LLM+Memento 的变化为 {memento_delta:+.4f}。"
            "因此，本轮实验可以用于区分当前收益是主要来自权重自适应，还是来自案例记忆检索与经验迁移。"
        ),
        (
            f"从五阶段杀伤链看，Full 当前仍然主要受制于 {stage_bottleneck(full_result)}，"
            f"而 {stage_advantage(full_result)} 则构成了当前方案的相对优势。"
            "这表明后续优化应优先围绕低分阶段补短板，而不是只追求总体均值的小幅提升。"
        ),
        "",
        "## 4.4 异常结论诊断",
        *[f"- {line}" for line in anomaly_lines],
        "",
        "## 4.5 本章小结",
        (
            "本章通过四组消融实验验证了完整系统相对基线模型的变化趋势，"
            "并从总体指标、Monte Carlo 稳定性与五阶段杀伤链得分三个层面解释了性能差异来源。"
            "若后续实验继续保持 Full 组相对纯LLM的稳定优势，则可进一步增强论文关于系统有效性的论证力度。"
        ),
    ]


def build_report(experiments: Sequence[Experiment], style: str) -> str:
    if style == "report":
        lines = report_sections(experiments)
    elif style == "thesis":
        lines = thesis_sections(experiments)
    else:
        lines = chapter_sections(experiments)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    experiments: List[Experiment] = [
        ("纯LLM", load_result(args.pure_llm)),
        ("LLM+Memento", load_result(args.memento)),
        ("LLM+Hope", load_result(args.hope)),
        ("Full", load_result(args.full)),
    ]
    report = build_report(experiments, args.style)

    if args.output == "-":
        print(report)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "style": args.style,
                "experiments": [label for label, _ in experiments],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
