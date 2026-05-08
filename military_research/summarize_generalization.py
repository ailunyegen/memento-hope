from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


STAGES = ["detect", "disrupt", "breach", "control", "sustain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总多场景泛化实验结果。")
    parser.add_argument("--suite-dir", required=True, help="泛化实验总目录")
    parser.add_argument(
        "--manifest",
        default="data/generalization_scenarios/manifest.json",
        help="场景清单 manifest.json 路径，用于补充 family / terrain_family 信息。",
    )
    parser.add_argument(
        "--style",
        choices=["report", "thesis", "chapter"],
        default="report",
        help="输出风格：report 为工程报告，thesis 为论文小节，chapter 为论文第4章风格。",
    )
    parser.add_argument("--output", required=True, help="输出 Markdown 路径，或 '-' 直接打印")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_scene_dirs(suite_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "ablation_full" / "full_result.json").exists()
    )


def best_sim(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("best_simulation", {}) or {}


def metric(result: Dict[str, Any], key: str) -> float:
    return float(best_sim(result).get(key, 0.0))


def metric_stats(result: Dict[str, Any], key: str) -> Dict[str, Any]:
    return best_sim(result).get("monte_carlo_stats", {}).get(key, {}) or {}


def stage_scores(result: Dict[str, Any]) -> Dict[str, float]:
    raw = best_sim(result).get("stage_scores", {}) or {}
    return {key: float(value) for key, value in raw.items()}


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


def stats_summary(result: Dict[str, Any], key: str) -> str:
    stats = metric_stats(result, key)
    if not stats:
        return "N/A"
    mean = fmt_num(stats.get("mean"))
    std = fmt_num(stats.get("std"))
    ci_low = fmt_num(stats.get("ci95_low"))
    ci_high = fmt_num(stats.get("ci95_high"))
    n = fmt_count(stats.get("n", "N/A"))
    return f"{mean} ± {std} (95% CI: [{ci_low}, {ci_high}], n={n})"


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_manifest_index(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    manifest = load_json(path)
    scenarios = manifest.get("scenarios", []) or []
    return {Path(item.get("file", "")).stem: item for item in scenarios if item.get("file")}


def family_label(scene_name: str, manifest_index: Dict[str, Dict[str, Any]]) -> str:
    return str(manifest_index.get(scene_name, {}).get("family", "unknown"))


def terrain_family_label(scene_name: str, manifest_index: Dict[str, Dict[str, Any]]) -> str:
    return str(manifest_index.get(scene_name, {}).get("terrain_family", "unknown"))


def weakest_stages(result: Dict[str, Any], top_k: int = 2) -> str:
    scores = stage_scores(result)
    if not scores:
        return "N/A"
    ordered = sorted(scores.items(), key=lambda item: item[1])[:top_k]
    return "、".join(f"{name}={score:.4f}" for name, score in ordered)


def full_beats_all(pure: Dict[str, Any], memento: Dict[str, Any], hope: Dict[str, Any], full: Dict[str, Any]) -> bool:
    full_ms = metric(full, "mission_success")
    return full_ms >= max(metric(pure, "mission_success"), metric(memento, "mission_success"), metric(hope, "mission_success"))


def collect_scene_rows(scene_dirs: Sequence[Path], manifest_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene_dir in scene_dirs:
        scene_key = scene_dir.name
        pure = load_json(scene_dir / "ablation_pure_llm" / "full_result.json")
        memento = load_json(scene_dir / "ablation_memento" / "full_result.json")
        hope = load_json(scene_dir / "ablation_hope" / "full_result.json")
        full = load_json(scene_dir / "ablation_full" / "full_result.json")

        row = {
            "scene_key": scene_key,
            "scene_name": full.get("scenario", {}).get("name", scene_key),
            "family": family_label(scene_key, manifest_index),
            "terrain_family": terrain_family_label(scene_key, manifest_index),
            "pure": pure,
            "memento": memento,
            "hope": hope,
            "full": full,
        }
        rows.append(row)
    return rows


def build_scene_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| 场景 | 任务族 | 地形族 | 纯LLM MS | Memento MS | Hope MS | Full MS | Full-纯LLM MS | Full-纯LLM OE | Full MS 统计 | Full短板阶段 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        pure = row["pure"]
        memento = row["memento"]
        hope = row["hope"]
        full = row["full"]
        pure_ms = metric(pure, "mission_success")
        full_ms = metric(full, "mission_success")
        pure_oe = metric(pure, "overall_effectiveness")
        full_oe = metric(full, "overall_effectiveness")
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scene_name"],
                    row["family"],
                    row["terrain_family"],
                    fmt_num(pure_ms),
                    fmt_num(metric(memento, "mission_success")),
                    fmt_num(metric(hope, "mission_success")),
                    fmt_num(full_ms),
                    f"{full_ms - pure_ms:+.4f}",
                    f"{full_oe - pure_oe:+.4f}",
                    stats_summary(full, "mission_success"),
                    weakest_stages(full),
                ]
            )
            + " |"
        )
    return lines


def build_family_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)

    lines = [
        "| 任务族 | 场景数 | Full胜场 | 平均纯LLM MS | 平均Full MS | 平均增量 | 平均Full OE增量 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, family_rows in sorted(grouped.items()):
        pure_values = [metric(item["pure"], "mission_success") for item in family_rows]
        full_values = [metric(item["full"], "mission_success") for item in family_rows]
        oe_deltas = [
            metric(item["full"], "overall_effectiveness") - metric(item["pure"], "overall_effectiveness")
            for item in family_rows
        ]
        wins = sum(1 for item in family_rows if metric(item["full"], "mission_success") > metric(item["pure"], "mission_success"))
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    str(len(family_rows)),
                    str(wins),
                    fmt_num(average(pure_values)),
                    fmt_num(average(full_values)),
                    f"{average(full_values) - average(pure_values):+.4f}",
                    f"{average(oe_deltas):+.4f}",
                ]
            )
            + " |"
        )
    return lines


def anomaly_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    failed = [row for row in rows if metric(row["full"], "mission_success") <= metric(row["pure"], "mission_success")]
    if not failed:
        return ["当前没有出现 Full 低于纯LLM 的失败场景，泛化方向整体正常。"]

    lines = []
    for row in failed:
        pure_ms = metric(row["pure"], "mission_success")
        memento_ms = metric(row["memento"], "mission_success")
        hope_ms = metric(row["hope"], "mission_success")
        full_ms = metric(row["full"], "mission_success")
        reasons: List[str] = []
        if memento_ms + 0.01 < pure_ms:
            reasons.append("Memento 检索样本对该场景族的迁移支持不足")
        if hope_ms + 0.01 < pure_ms:
            reasons.append("Hope 权重在该场景中可能压低了关键能力维度")
        if full_ms + 0.01 < max(memento_ms, hope_ms):
            reasons.append("Reflection 可能改坏了原本更有效的阶段结构")
        if not reasons:
            reasons.append("多模块叠加后的耦合偏差仍需进一步诊断")
        lines.append(
            f"{row['scene_name']}：Full={full_ms:.4f}，纯LLM={pure_ms:.4f}，"
            f"主要怀疑原因包括：{'；'.join(reasons)}。"
        )
    return lines


def summary_numbers(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    pure_ms_values = [metric(row["pure"], "mission_success") for row in rows]
    full_ms_values = [metric(row["full"], "mission_success") for row in rows]
    pure_oe_values = [metric(row["pure"], "overall_effectiveness") for row in rows]
    full_oe_values = [metric(row["full"], "overall_effectiveness") for row in rows]
    wins = sum(1 for row in rows if metric(row["full"], "mission_success") > metric(row["pure"], "mission_success"))
    strict_wins = sum(1 for row in rows if full_beats_all(row["pure"], row["memento"], row["hope"], row["full"]))
    return {
        "scene_count": len(rows),
        "wins": wins,
        "strict_wins": strict_wins,
        "avg_pure_ms": average(pure_ms_values),
        "avg_full_ms": average(full_ms_values),
        "avg_pure_oe": average(pure_oe_values),
        "avg_full_oe": average(full_oe_values),
    }


def report_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    summary = summary_numbers(rows)
    lines = [
        "# 跨场景泛化实验汇总",
        "",
        f"- 场景数量：{summary['scene_count']}",
        f"- Full 相对纯LLM的胜场：{summary['wins']}/{summary['scene_count']}",
        f"- Full 同时不低于纯LLM、Memento、Hope 的场景数：{summary['strict_wins']}/{summary['scene_count']}",
        f"- 平均 mission_success：纯LLM={summary['avg_pure_ms']:.4f}，Full={summary['avg_full_ms']:.4f}，增量={summary['avg_full_ms'] - summary['avg_pure_ms']:+.4f}",
        f"- 平均 overall_effectiveness：纯LLM={summary['avg_pure_oe']:.4f}，Full={summary['avg_full_oe']:.4f}，增量={summary['avg_full_oe'] - summary['avg_pure_oe']:+.4f}",
        "",
        "## 场景级结果",
        "",
        *build_scene_table(rows),
        "",
        "## 任务族汇总",
        "",
        *build_family_table(rows),
        "",
        "## 异常场景诊断",
        *[f"- {line}" for line in anomaly_lines(rows)],
    ]
    return lines


def thesis_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    summary = summary_numbers(rows)
    lines = [
        "# 跨场景泛化实验分析",
        "",
        "## 实验设计",
        (
            "本组实验在多个任务族与地形族场景上重复四组消融流程，"
            "用于评估完整系统相对纯LLM基线的跨场景迁移能力，并进一步区分 Memento、Hope 与 Reflection 的模块化贡献。"
        ),
        "",
        "## 场景级结果",
        "",
        *build_scene_table(rows),
        "",
        "## 任务族汇总",
        "",
        *build_family_table(rows),
        "",
        "## 结果分析",
        (
            f"从总体统计看，Full 在 {summary['scene_count']} 个场景中有 {summary['wins']} 个场景优于纯LLM，"
            f"平均 mission_success 从 {summary['avg_pure_ms']:.4f} 提升到 {summary['avg_full_ms']:.4f}，"
            f"平均 overall_effectiveness 从 {summary['avg_pure_oe']:.4f} 提升到 {summary['avg_full_oe']:.4f}。"
            "这可以用来判断系统是否已经具备初步的跨场景泛化能力。"
        ),
        (
            f"进一步看，Full 在 {summary['strict_wins']} 个场景中同时不低于纯LLM、Memento 与 Hope，"
            "这能帮助区分当前收益是来自单一增强模块，还是来自多模块协同后的结构性提升。"
        ),
        (
            "从任务族维度看，不同场景族的收益差异能够反映当前案例库覆盖度与模块适配性的强弱；"
            "从阶段短板看，若某一类场景持续在 disrupt 或 breach 上失分，则后续优化应优先围绕该场景族的压制链或突破链展开。"
        ),
        "",
        "## 异常场景诊断",
        *[f"- {line}" for line in anomaly_lines(rows)],
    ]
    return lines


def chapter_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    summary = summary_numbers(rows)
    lines = [
        "# 第4章 跨场景泛化实验分析",
        "",
        "## 4.1 泛化实验设计",
        (
            "为验证系统在不同任务类型、不同地形条件与不同作战压力下的迁移能力，"
            "本研究选取多组代表性场景构建跨场景泛化实验。每个场景均重复纯LLM、LLM+Memento、LLM+Hope 与 Full 四组对照实验，"
            "并基于 Monte Carlo 统计结果比较系统在不同场景族上的稳定性与收益来源。"
        ),
        "",
        "## 4.2 场景级实验结果",
        "",
        *build_scene_table(rows),
        "",
        "## 4.3 任务族汇总结果",
        "",
        *build_family_table(rows),
        "",
        "## 4.4 结果分析",
        (
            f"从总体表现看，Full 在 {summary['scene_count']} 个场景中有 {summary['wins']} 个场景优于纯LLM，"
            f"平均 mission_success 从 {summary['avg_pure_ms']:.4f} 提升到 {summary['avg_full_ms']:.4f}，"
            f"平均 overall_effectiveness 从 {summary['avg_pure_oe']:.4f} 提升到 {summary['avg_full_oe']:.4f}。"
            "若这一优势在更多场景上持续保持，则可作为系统具备跨场景稳定收益的重要证据。"
        ),
        (
            f"同时，Full 在 {summary['strict_wins']} 个场景中同时不低于纯LLM、Memento 与 Hope，"
            "说明当前收益并不完全依赖某一单模块，而是部分来自多模块叠加后的协同增强。"
        ),
        (
            "从任务族维度进一步观察，若某些场景族的平均增量显著高于其他场景族，"
            "则说明当前案例库与权重调节机制对该类任务的适配度更高；反之，则提示该场景族仍需补充样本或单独调优。"
        ),
        "",
        "## 4.5 异常场景诊断",
        *[f"- {line}" for line in anomaly_lines(rows)],
        "",
        "## 4.6 本章小结",
        (
            "本章从场景级结果、任务族汇总与异常场景诊断三个层面分析了系统的跨场景泛化表现。"
            "后续若继续扩充案例库并在更多未入库场景上复现实验，则能够进一步增强论文关于系统普适性的论证力度。"
        ),
    ]
    return lines


def build_report(rows: Sequence[Dict[str, Any]], style: str) -> str:
    if style == "report":
        lines = report_lines(rows)
    elif style == "thesis":
        lines = thesis_lines(rows)
    else:
        lines = chapter_lines(rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    if not suite_dir.exists():
        raise SystemExit(f"泛化实验目录不存在：{suite_dir}")

    scene_dirs = discover_scene_dirs(suite_dir)
    if not scene_dirs:
        raise SystemExit(f"未在 {suite_dir} 下发现可用的泛化实验结果目录。")

    manifest_index = load_manifest_index(Path(args.manifest))
    rows = collect_scene_rows(scene_dirs, manifest_index)
    report = build_report(rows, args.style)

    if args.output == "-":
        print(report)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "suite_dir": str(suite_dir),
                "output": str(output_path),
                "style": args.style,
                "scenarios": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
