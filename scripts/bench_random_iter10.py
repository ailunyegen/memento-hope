"""10 轮迭代随机搜索基线（对齐算力预算，回应审稿人基线不对称）。

与单次(1-pass)随机搜索不同：本脚本迭代 10 轮，每轮以随机扰动的融合权重
生成 1 个候选方案，仿真评估后以 previous_best_plan 保优传递，最终输出
best-so-far。算力预算与主实验的 10 轮迭代框架对齐。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from military_research.case_memory import CaseBank  # noqa: E402
from military_research.domain import CAPABILITY_KEYS, Scenario, clamp_value  # noqa: E402
from military_research.engine import PlanGenerator, PlanSimulator, SLOW_WEIGHT_LIBRARY  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_PATH = ROOT / "data/sample_joint_operation.json"
CASE_BANK = ROOT / "data/military_case_bank_frozen_seed.jsonl"
OUT = ROOT / "result/baselines_4090/random_iter10/full_result.json"
SEED = 7
SIM_RUNS = 50
ITERATIONS = 10
NOISE = 0.15


def aggregate(sim: PlanSimulator, scenario: Scenario, plan, runs: int):
    if runs <= 1:
        return sim.run(scenario, plan, stochastic=False)
    rr = [sim.run(scenario, plan, stochastic=True) for _ in range(runs)]
    best = max(rr, key=lambda r: r.mission_success)
    return type(rr[0])(
        mission_success=round(sum(r.mission_success for r in rr) / len(rr), 4),
        ler=round(sum(r.ler for r in rr) / len(rr), 4),
        completion_time_hours=round(sum(r.completion_time_hours for r in rr) / len(rr), 2),
        survivability=round(sum(r.survivability for r in rr) / len(rr), 4),
        command_resilience=round(sum(r.command_resilience for r in rr) / len(rr), 4),
        overall_effectiveness=round(sum(r.overall_effectiveness for r in rr) / len(rr), 4),
        stage_scores={n: round(sum(r.stage_scores.get(n, 0.0) for r in rr) / len(rr), 4) for n in (rr[0].stage_scores or {})},
        notes=best.notes + [f"10 轮迭代随机搜索（第 {len(rr)} 次仿真均值）。"],
        phases=best.phases,
        recommendations=best.recommendations,
    )


def main() -> None:
    scenario = Scenario.from_dict(json.loads(SCENARIO_PATH.read_text(encoding="utf-8")))
    bank = CaseBank(CASE_BANK)
    gen = PlanGenerator()
    sim = PlanSimulator(random.Random(SEED))
    rng = random.Random(SEED)

    profession = scenario.doctrine_profile
    base_slow = SLOW_WEIGHT_LIBRARY.get(profession, SLOW_WEIGHT_LIBRARY["balanced_joint"])
    best_plan = None
    best_result = None
    history = []

    for i in range(1, ITERATIONS + 1):
        fused = {
            k: round(clamp_value(base_slow.get(k, 0.65) + rng.uniform(-NOISE, NOISE), 0.24, 1.0), 4)
            for k in CAPABILITY_KEYS
        }
        fast = {k: 0.0 for k in CAPABILITY_KEYS}
        plan = gen.generate(
            scenario=scenario,
            fused_weights=fused,
            fast_weights=fast,
            memory_hits=[],
            iteration_index=i - 1,
            reflection=None,
            previous_best_plan=best_plan,
        )
        result = aggregate(sim, scenario, plan, SIM_RUNS)
        if best_result is None or result.mission_success > best_result.mission_success:
            best_plan = plan
            best_result = result
        history.append(
            {
                "iteration": i,
                "ms": result.mission_success,
                "best_ms": best_result.mission_success,
            }
        )
        print(f"iter {i}: ms={result.mission_success:.4f} best={best_result.mission_success:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": "random_iter10",
        "experiment_config": {
            "scenario_path": str(SCENARIO_PATH),
            "case_bank_path": str(CASE_BANK),
            "iterations": ITERATIONS,
            "sim_runs": SIM_RUNS,
            "seed": SEED,
            "noise": NOISE,
        },
        "best_plan": best_plan.to_dict(),
        "best_simulation": best_result.to_dict(),
        "optimization_history": history,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresult_id未生成（baseline 脚本，无需 runtime_manifest）；best MS = {best_result.mission_success:.4f}")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
