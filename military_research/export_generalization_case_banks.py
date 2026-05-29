from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为跨场景泛化实验导出按场景留出的冻结案例库。")
    parser.add_argument("--input", default="data/military_case_bank_frozen_seed.jsonl", help="输入冻结案例库 JSONL 路径。")
    parser.add_argument("--manifest", default="data/generalization_scenarios/manifest.json", help="泛化场景清单 manifest.json 路径。")
    parser.add_argument("--output-dir", default="data/generalization_case_banks", help="输出案例库目录。")
    parser.add_argument(
        "--mode",
        choices=["leave-one-family-out", "leave-one-scene-out", "leave-one-source-scenario-out"],
        default="leave-one-family-out",
        help="导出模式。leave-one-family-out 会排除同任务族；leave-one-scene-out 额外可排除同地形族。",
    )
    parser.add_argument(
        "--exclude-terrain-family",
        action="store_true",
        help="在 leave-one-scene-out 模式下额外排除与目标场景同地形族的案例。",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=2,
        help="若导出结果少于该值，则在 summary 中给出 warning。",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"案例库不存在：{path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest 不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", "-").split())


def infer_record_family(record: Dict[str, Any]) -> str:
    mission_type = normalize_text(record.get("mission_type"))
    tags = {normalize_text(tag) for tag in record.get("tags", []) or []}
    objective = normalize_text(record.get("objective"))

    if mission_type in {"assault", "defense", "recon", "sustain"}:
        return mission_type
    if "resupply" in objective or "sustain" in objective or "support" in tags:
        return "sustain"
    if "defense" in objective or "defense" in tags:
        return "defense"
    if "recon" in objective or "isr" in tags:
        return "recon"
    return "assault"


def infer_terrain_family(record: Dict[str, Any]) -> str:
    terrain = normalize_text(record.get("terrain"))
    mapping = {
        "coastal-urban": "coastal",
        "coastal": "coastal",
        "urban": "urban",
        "mountain": "mountain",
        "river": "river",
        "river-crossing": "river",
        "maritime": "maritime",
        "island": "maritime",
    }
    return mapping.get(terrain, terrain or "unknown")


def should_keep_record(
    record: Dict[str, Any],
    scenario_spec: Dict[str, Any],
    mode: str,
    exclude_terrain_family: bool,
) -> bool:
    record_family = infer_record_family(record)
    record_terrain = infer_terrain_family(record)
    target_family = normalize_text(scenario_spec.get("family"))
    target_terrain = normalize_text(scenario_spec.get("terrain_family"))
    target_scene = normalize_text(Path(str(scenario_spec.get("file", ""))).stem)

    if mode == "leave-one-source-scenario-out":
        record_sources = {normalize_text(item) for item in record.get("source_scenarios", []) or []}
        record_tags = {normalize_text(tag) for tag in record.get("tags", []) or []}
        tagged_sources = {
            tag.split(":", 1)[1]
            for tag in record_tags
            if tag.startswith("group:") and ":" in tag
        }
        return target_scene not in (record_sources | tagged_sources)

    if mode == "leave-one-family-out" and record_family == target_family:
        return False

    if mode == "leave-one-scene-out":
        if record_family == target_family:
            return False
        if exclude_terrain_family and record_terrain == target_terrain:
            return False

    return True


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_case_banks(
    records: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    output_dir: Path,
    mode: str,
    exclude_terrain_family: bool,
    min_cases: int,
) -> Dict[str, Any]:
    scenarios = manifest.get("scenarios", []) or []
    output_dir.mkdir(parents=True, exist_ok=True)
    per_scene: List[Dict[str, Any]] = []

    all_cases_path = output_dir / "all_cases.jsonl"
    write_jsonl(all_cases_path, records)

    for spec in scenarios:
        file_name = str(spec.get("file", "")).strip()
        scene_stem = Path(file_name).stem
        selected = [
            record
            for record in records
            if should_keep_record(record, spec, mode=mode, exclude_terrain_family=exclude_terrain_family)
        ]
        output_path = output_dir / f"{scene_stem}.jsonl"
        count = write_jsonl(output_path, selected)
        warning = None
        if count < min_cases:
            warning = f"仅导出 {count} 条案例，低于建议下限 {min_cases}"
        per_scene.append(
            {
                "scene": scene_stem,
                "family": spec.get("family"),
                "terrain_family": spec.get("terrain_family"),
                "output": str(output_path),
                "case_count": count,
                "warning": warning,
            }
        )

    summary = {
        "mode": mode,
        "exclude_terrain_family": exclude_terrain_family,
        "source_case_count": len(records),
        "output_dir": str(output_dir),
        "all_cases": str(all_cases_path),
        "per_scene": per_scene,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    records = load_jsonl(input_path)
    manifest = load_manifest(manifest_path)
    summary = export_case_banks(
        records=records,
        manifest=manifest,
        output_dir=output_dir,
        mode=args.mode,
        exclude_terrain_family=args.exclude_terrain_family,
        min_cases=args.min_cases,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
