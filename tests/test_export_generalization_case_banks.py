from __future__ import annotations

import json
from pathlib import Path

from military_research.export_generalization_case_banks import export_case_banks


def test_export_case_banks_leave_one_family_out(tmp_path: Path) -> None:
    records = [
        {
            "case_id": "CASE-001",
            "mission_type": "assault",
            "terrain": "coastal-urban",
            "objective": "secure beachhead",
            "tags": ["joint"],
        },
        {
            "case_id": "CASE-002",
            "mission_type": "defense",
            "terrain": "urban",
            "objective": "hold logistics hub",
            "tags": ["urban"],
        },
        {
            "case_id": "CASE-003",
            "mission_type": "recon",
            "terrain": "mountain",
            "objective": "conduct corridor reconnaissance",
            "tags": ["isr"],
        },
    ]
    manifest = {
        "scenarios": [
            {"file": "coastal_joint_assault.json", "family": "assault", "terrain_family": "coastal"},
            {"file": "urban_hub_defense.json", "family": "defense", "terrain_family": "urban"},
        ]
    }

    summary = export_case_banks(
        records=records,
        manifest=manifest,
        output_dir=tmp_path,
        mode="leave-one-family-out",
        exclude_terrain_family=False,
        min_cases=1,
    )

    coastal_cases = [json.loads(line) for line in (tmp_path / "coastal_joint_assault.jsonl").read_text(encoding="utf-8").splitlines()]
    urban_cases = [json.loads(line) for line in (tmp_path / "urban_hub_defense.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(coastal_cases) == 2
    assert all(case["mission_type"] != "assault" for case in coastal_cases)
    assert len(urban_cases) == 2
    assert all(case["mission_type"] != "defense" for case in urban_cases)
    assert (tmp_path / "summary.json").exists()
    assert summary["source_case_count"] == 3
