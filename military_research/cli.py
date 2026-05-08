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
    parser.add_argument("--seed", type=int, default=7, help="Random seed for Python and PyTorch reproducibility.")
    parser.add_argument("--disable-memory", action="store_true", help="Disable Memento memory retrieval and write-back.")
    parser.add_argument("--disable-hope", action="store_true", help="Disable HOPE fast/slow adaptive weighting.")
    parser.add_argument("--disable-reflection", action="store_true", help="Disable EvoPrompt reflection between iterations.")
    parser.add_argument("--disable-writeback", action="store_true", help="Disable case-bank write-back for fair ablations or frozen evaluations.")
    parser.add_argument("--sim-runs", type=int, default=1, help="Number of simulation runs for Monte Carlo averaging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_payload = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    scenario = Scenario.from_dict(scenario_payload)
    pipeline = MilitaryResearchPipeline(args.case_bank, seed=args.seed)
    result = pipeline.run(
        scenario,
        iterations=args.iterations,
        disable_memory=args.disable_memory,
        disable_hope=args.disable_hope,
        disable_reflection=args.disable_reflection,
        sim_runs=args.sim_runs,
        allow_writeback=not args.disable_writeback,
    )
    result.setdefault("experiment_config", {})
    result["experiment_config"]["scenario_path"] = str(Path(args.scenario))
    result["experiment_config"]["output_dir"] = str(Path(args.output_dir))
    pipeline.write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "operation": result["best_plan"]["title"],
                "seed": args.seed,
                "disable_memory": args.disable_memory,
                "disable_hope": args.disable_hope,
                "disable_reflection": args.disable_reflection,
                "disable_writeback": args.disable_writeback,
                "sim_runs": args.sim_runs,
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
