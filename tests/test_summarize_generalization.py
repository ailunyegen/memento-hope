from __future__ import annotations

import json
from pathlib import Path

from military_research.summarize_generalization import build_report, collect_scene_rows


def _write_result(path: Path, mission_success: float, overall_effectiveness: float) -> None:
    payload = {
        "scenario": {"name": path.parent.parent.name},
        "best_simulation": {
            "mission_success": mission_success,
            "overall_effectiveness": overall_effectiveness,
            "stage_scores": {
                "detect": 0.62,
                "disrupt": 0.58,
                "breach": 0.57,
                "control": 0.64,
                "sustain": 0.66,
            },
            "monte_carlo_stats": {
                "mission_success": {
                    "mean": mission_success,
                    "std": 0.01,
                    "ci95_low": mission_success - 0.005,
                    "ci95_high": mission_success + 0.005,
                    "n": 20,
                }
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_generalization_report_contains_family_summary(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    scene_dir = suite_dir / "coastal_joint_assault"

    _write_result(scene_dir / "ablation_pure_llm" / "full_result.json", 0.61, 0.72)
    _write_result(scene_dir / "ablation_memento" / "full_result.json", 0.62, 0.73)
    _write_result(scene_dir / "ablation_hope" / "full_result.json", 0.64, 0.77)
    _write_result(scene_dir / "ablation_full" / "full_result.json", 0.67, 0.80)

    manifest_index = {
        "coastal_joint_assault": {"family": "assault", "terrain_family": "coastal"}
    }
    rows = collect_scene_rows([scene_dir], manifest_index)
    report = build_report(rows, "chapter")

    assert "## 4.3 任务族汇总结果" in report
    assert "assault" in report
    assert "coastal" in report
    assert "0.6700" in report
