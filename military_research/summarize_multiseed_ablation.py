from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List


METRICS = [
    "mission_success",
    "overall_effectiveness",
    "ler",
    "completion_time_hours",
    "command_resilience",
]

GROUPS = [
    ("ablation_pure_llm", "Pure LLM"),
    ("ablation_memento", "LLM + Memento"),
    ("ablation_hope", "LLM + Hope"),
    ("ablation_full", "Full"),
    ("ablation_reflection_only", "Reflection only"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed ablation suites.")
    parser.add_argument("--suite-dir", required=True, help="Directory produced by run_ablation_multiseed.ps1")
    parser.add_argument("--output", required=True, help="Markdown output path or '-' to print")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_metric(result: Dict[str, Any], metric: str) -> float:
    return float(result.get("best_simulation", {}).get(metric, 0.0))


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.4f}"


def discover_seed_dirs(suite_dir: Path) -> List[Path]:
    return sorted(path for path in suite_dir.iterdir() if path.is_dir() and path.name.startswith("seed_"))


def load_seed_results(seed_dir: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for group_dir, _label in GROUPS:
        result_path = seed_dir / group_dir / "full_result.json"
        if result_path.exists():
            results[group_dir] = load_json(result_path)
    return results


def summarize_values(values: Iterable[float]) -> Dict[str, float] | None:
    items = list(values)
    if not items:
        return None
    return {
        "mean": mean(items),
        "std": stdev(items) if len(items) > 1 else 0.0,
        "min": min(items),
        "max": max(items),
        "n": len(items),
    }


def build_metric_rows(seed_payloads: List[Dict[str, Dict[str, Any]]]) -> List[str]:
    rows: List[str] = []
    for metric in METRICS:
        parts = [metric]
        for group_dir, _label in GROUPS:
            values = [safe_metric(seed_payload[group_dir], metric) for seed_payload in seed_payloads if group_dir in seed_payload]
            stats = summarize_values(values)
            if stats is None:
                parts.append("N/A")
            else:
                parts.append(
                    f"{fmt_num(stats['mean'])} ± {fmt_num(stats['std'])} "
                    f"(min={fmt_num(stats['min'])}, max={fmt_num(stats['max'])}, n={int(stats['n'])})"
                )
        rows.append("| " + " | ".join(parts) + " |")
    return rows


def build_delta_rows(seed_payloads: List[Dict[str, Dict[str, Any]]]) -> List[str]:
    rows: List[str] = []
    comparisons = [
        ("Full - Pure", "ablation_full", "ablation_pure_llm"),
        ("Reflection - Pure", "ablation_reflection_only", "ablation_pure_llm"),
        ("Full - Hope", "ablation_full", "ablation_hope"),
        ("Full - Memento", "ablation_full", "ablation_memento"),
    ]
    for metric in METRICS:
        parts = [metric]
        for _label, left_group, right_group in comparisons:
            deltas = [
                safe_metric(seed_payload[left_group], metric) - safe_metric(seed_payload[right_group], metric)
                for seed_payload in seed_payloads
                if left_group in seed_payload and right_group in seed_payload
            ]
            stats = summarize_values(deltas)
            if stats is None:
                parts.append("N/A")
            else:
                parts.append(
                    f"{fmt_delta(stats['mean'])} ± {fmt_num(stats['std'])} "
                    f"(min={fmt_delta(stats['min'])}, max={fmt_delta(stats['max'])}, n={int(stats['n'])})"
                )
        rows.append("| " + " | ".join(parts) + " |")
    return rows


def build_report(suite_dir: Path, seed_dirs: List[Path], seed_payloads: List[Dict[str, Dict[str, Any]]]) -> str:
    header = [
        "# Multi-seed ablation summary",
        "",
        f"- Suite directory: `{suite_dir}`",
        f"- Seed count discovered: {len(seed_dirs)}",
        f"- Seeds: {', '.join(path.name.replace('seed_', '') for path in seed_dirs)}",
        "",
        "## Per-group metric summary",
        "",
        "| Metric | Pure LLM | LLM + Memento | LLM + Hope | Full | Reflection only |",
        "| --- | --- | --- | --- | --- | --- |",
        *build_metric_rows(seed_payloads),
        "",
        "## Paired seed deltas",
        "",
        "| Metric | Full - Pure | Reflection - Pure | Full - Hope | Full - Memento |",
        "| --- | --- | --- | --- | --- |",
        *build_delta_rows(seed_payloads),
        "",
        "## Notes",
        "",
        "- This script reports descriptive multi-seed summaries only.",
        "- It does not perform significance testing; users can layer formal tests on top of the exported per-seed result bundles if needed.",
    ]
    return "\n".join(header) + "\n"


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    if not suite_dir.exists():
        raise SystemExit(f"Suite directory not found: {suite_dir}")

    seed_dirs = discover_seed_dirs(suite_dir)
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories found under: {suite_dir}")

    seed_payloads = [load_seed_results(seed_dir) for seed_dir in seed_dirs]
    report = build_report(suite_dir, seed_dirs, seed_payloads)

    if args.output == "-":
        print(report)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"suite_dir": str(suite_dir), "output": str(output_path), "seed_count": len(seed_dirs)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
