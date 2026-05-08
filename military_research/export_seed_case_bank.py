from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a frozen high-quality seed case bank for fair ablation studies.")
    parser.add_argument("--input", default="data/military_case_bank.jsonl", help="Source case-bank JSONL path.")
    parser.add_argument("--output", default="data/military_case_bank_frozen_seed.jsonl", help="Output frozen seed JSONL path.")
    parser.add_argument("--min-mission-success", type=float, default=0.58, help="Minimum mission_success for auto-generated cases.")
    parser.add_argument("--min-overall-effectiveness", type=float, default=0.70, help="Minimum overall_effectiveness for auto-generated cases.")
    parser.add_argument("--min-stage-score", type=float, default=0.50, help="Minimum stage_* score for auto-generated cases when present.")
    parser.add_argument("--min-quality", type=float, default=0.60, help="Minimum blended quality score for auto-generated cases.")
    parser.add_argument("--max-auto", type=int, default=12, help="Maximum number of AUTO-* records to keep after ranking.")
    parser.add_argument("--keep-negative-curated", action="store_true", help="Keep curated CASE-* negative examples for diagnostic analysis.")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Case bank not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def quality_score(record: Dict[str, Any]) -> float:
    metrics = record.get("metrics", {}) or {}
    mission_success = float(metrics.get("mission_success", 0.0))
    overall = float(metrics.get("overall_effectiveness", 0.0))
    ler = float(metrics.get("ler", 1.0))
    command_resilience = float(metrics.get("command_resilience", 0.58))
    stage_scores = [
        float(value)
        for key, value in metrics.items()
        if str(key).startswith("stage_")
    ]
    stage_quality = min(stage_scores) if stage_scores else 0.55
    update_bonus = min(max(int(record.get("update_count", 1)), 1), 4) * 0.015
    auto_penalty = 0.04 if str(record.get("case_id", "")).startswith("AUTO-") else 0.0
    negative_penalty = 0.12 if record.get("outcome") != "positive" else 0.0
    ler_score = min(ler / 2.2, 1.0)
    quality = (
        0.38 * mission_success
        + 0.24 * overall
        + 0.16 * stage_quality
        + 0.12 * ler_score
        + 0.10 * command_resilience
        + update_bonus
        - auto_penalty
        - negative_penalty
    )
    return max(0.0, min(1.0, quality))


def record_signature(record: Dict[str, Any]) -> Tuple[str, str, str, Tuple[str, ...], Tuple[str, ...]]:
    return (
        normalize_text(record.get("mission_type", "")),
        normalize_text(record.get("terrain", "")),
        normalize_text(record.get("objective", "")),
        tuple(sorted(normalize_text(role) for role in record.get("friendly_roles", []) or [])),
        tuple(sorted(normalize_text(role) for role in record.get("enemy_roles", []) or [])),
    )


def stage_floor(record: Dict[str, Any]) -> float:
    metrics = record.get("metrics", {}) or {}
    stage_scores = [
        float(value)
        for key, value in metrics.items()
        if str(key).startswith("stage_")
    ]
    return min(stage_scores) if stage_scores else 0.55


def curated_priority(record: Dict[str, Any]) -> int:
    case_id = str(record.get("case_id", ""))
    if case_id.startswith("CASE-"):
        return 2
    if case_id.startswith("AUTO-"):
        return 1
    return 0


def should_keep_curated(record: Dict[str, Any], keep_negative_curated: bool) -> bool:
    case_id = str(record.get("case_id", ""))
    if not case_id.startswith("CASE-"):
        return False
    if record.get("outcome") == "positive":
        return True
    return keep_negative_curated


def should_keep_auto(record: Dict[str, Any], args: argparse.Namespace) -> bool:
    metrics = record.get("metrics", {}) or {}
    mission_success = float(metrics.get("mission_success", 0.0))
    overall = float(metrics.get("overall_effectiveness", 0.0))
    min_stage = stage_floor(record)
    quality = quality_score(record)
    if record.get("outcome") != "positive":
        return False
    if mission_success < args.min_mission_success:
        return False
    if overall < args.min_overall_effectiveness:
        return False
    if min_stage < args.min_stage_score:
        return False
    if quality < args.min_quality:
        return False
    return True


def select_records(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    curated: List[Dict[str, Any]] = []
    autos: List[Dict[str, Any]] = []

    for record in records:
        case_id = str(record.get("case_id", ""))
        if should_keep_curated(record, args.keep_negative_curated):
            curated.append(record)
        elif case_id.startswith("AUTO-") and should_keep_auto(record, args):
            autos.append(record)

    autos.sort(
        key=lambda item: (
            quality_score(item),
            float(item.get("metrics", {}).get("mission_success", 0.0)),
            float(item.get("metrics", {}).get("overall_effectiveness", 0.0)),
            int(item.get("update_count", 1)),
        ),
        reverse=True,
    )
    autos = autos[: max(0, args.max_auto)]

    best_by_signature: Dict[Tuple[str, str, str, Tuple[str, ...], Tuple[str, ...]], Dict[str, Any]] = {}
    for record in curated + autos:
        signature = record_signature(record)
        incumbent = best_by_signature.get(signature)
        if incumbent is None:
            best_by_signature[signature] = record
            continue
        incumbent_key = (
            curated_priority(incumbent),
            quality_score(incumbent),
            float(incumbent.get("metrics", {}).get("mission_success", 0.0)),
            float(incumbent.get("metrics", {}).get("overall_effectiveness", 0.0)),
        )
        candidate_key = (
            curated_priority(record),
            quality_score(record),
            float(record.get("metrics", {}).get("mission_success", 0.0)),
            float(record.get("metrics", {}).get("overall_effectiveness", 0.0)),
        )
        if candidate_key > incumbent_key:
            best_by_signature[signature] = record

    selected = list(best_by_signature.values())
    selected.sort(
        key=lambda item: (
            curated_priority(item),
            quality_score(item),
            float(item.get("metrics", {}).get("mission_success", 0.0)),
            float(item.get("metrics", {}).get("overall_effectiveness", 0.0)),
        ),
        reverse=True,
    )
    return selected


def write_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    curated_count = sum(1 for item in records if str(item.get("case_id", "")).startswith("CASE-"))
    auto_count = sum(1 for item in records if str(item.get("case_id", "")).startswith("AUTO-"))
    top_records = [
        {
            "case_id": item.get("case_id"),
            "quality": round(quality_score(item), 4),
            "mission_success": item.get("metrics", {}).get("mission_success"),
            "overall_effectiveness": item.get("metrics", {}).get("overall_effectiveness"),
        }
        for item in records[:5]
    ]
    return {
        "total_records": len(records),
        "curated_case_records": curated_count,
        "auto_case_records": auto_count,
        "top_records": top_records,
    }


def main() -> None:
    args = parse_args()
    source_path = Path(args.input)
    output_path = Path(args.output)
    records = load_records(source_path)
    selected = select_records(records, args)
    write_records(output_path, selected)
    print(
        json.dumps(
            {
                "input": str(source_path),
                "output": str(output_path),
                **summarize(selected),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
