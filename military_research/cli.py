from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domain import Scenario
from .engine import MilitaryResearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memento + Hope military research prototype")
    parser.add_argument("--scenario", default="data/sample_joint_operation.json", help="Path to the scenario JSON file.")
    parser.add_argument("--case-bank", default="data/military_case_bank.jsonl", help="Path to the military case bank JSONL file.")
    parser.add_argument("--output-dir", default="result/military_research_demo", help="Directory for generated outputs.")
    parser.add_argument("--iterations", type=int, default=3, help="Optimization loop count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_payload = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    scenario = Scenario.from_dict(scenario_payload)
    pipeline = MilitaryResearchPipeline(args.case_bank)
    result = pipeline.run(scenario, iterations=args.iterations)
    pipeline.write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "operation": result["best_plan"]["title"],
                "mission_success": result["best_simulation"]["mission_success"],
                "ler": result["best_simulation"]["ler"],
                "completion_time_hours": result["best_simulation"]["completion_time_hours"],
                "overall_effectiveness": result["best_simulation"]["overall_effectiveness"],
                "output_dir": args.output_dir,
                "report": str(Path(args.output_dir) / "research_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
