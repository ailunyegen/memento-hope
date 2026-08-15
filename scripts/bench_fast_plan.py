"""Phase 0 基准测速脚本：验证双系统提速效果。

测量对象（对应技术方案书 Phase 0 三步）：
  1. System 1（快道）：Top-1 案例 + HOPE 权重投影的应急预案出库耗时（零 LLM）；
  2. System 1 预案的 50-MC 仿真质量与耗时；
  3. （可选，需本地 LLM）Fast 模式（single_candidate + rule_based_reflection）
     与现有 Full 模式的单轮耗时对比。

用法：
  python -X utf8 scripts/bench_fast_plan.py --scenario data/sample_joint_operation.json ^
      --case-bank data/military_case_bank_frozen_seed.jsonl
  # 显式开启 LLM 对比（需本地 OpenAI-compatible 服务在运行）：
  python -X utf8 scripts/bench_fast_plan.py --llm
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# 保证从任意工作目录运行时都能导入仓库包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from military_research.case_memory import CaseBank  # noqa: E402
from military_research.domain import Scenario  # noqa: E402
from military_research.engine import MilitaryResearchPipeline  # noqa: E402
from military_research.fast_pipeline import System1FastPlanner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 双系统提速基准测速")
    parser.add_argument("--scenario", default="data/sample_joint_operation.json")
    parser.add_argument("--case-bank", default="data/military_case_bank_frozen_seed.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mc-runs", type=int, default=50, help="System 1 预案质量核验的 MC 次数")
    parser.add_argument("--llm", action="store_true", help="开启 LLM 对比测速（需本地模型服务）")
    parser.add_argument("--fast-iterations", type=int, default=3, help="Fast 模式迭代轮数")
    return parser.parse_args()


def _load(args) -> tuple[Scenario, CaseBank]:
    scenario = Scenario.from_dict(json.loads(Path(args.scenario).read_text(encoding="utf-8")))
    bank = CaseBank(args.case_bank)
    return scenario, bank


def _llm_available() -> bool:
    from military_research.engine import LocalLLMPlanner

    try:
        planner = LocalLLMPlanner()
        planner.client.models.list()
        return True
    except Exception as exc:  # noqa: PERF203
        print(f"[info] 本地 LLM 不可用（{type(exc).__name__}），跳过 LLM 对比测速。")
        print("      请先启动 LM Studio / OpenAI-compatible 服务并设置环境变量：")
        print("      LOCAL_LLM_BASE_URL / LOCAL_LLM_API_KEY / LOCAL_LLM_MODEL")
        return False


def bench_system1(scenario: Scenario, bank: CaseBank, args) -> dict:
    planner = System1FastPlanner(bank, seed=args.seed)

    # 出库计时（多次取中位数，避免单次抖动）
    build_times: list[float] = []
    export_times: list[float] = []
    plan = None
    warped = scenario
    for _ in range(5):
        t0 = time.perf_counter()
        plan, warped = planner.build_emergency_plan(scenario)
        build_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        artifacts = planner.export(plan, warped)
        export_times.append(time.perf_counter() - t0)

    # 质量核验：50-MC 仿真
    t0 = time.perf_counter()
    sim = planner.simulate(plan, warped, runs=args.mc_runs)
    sim_time = time.perf_counter() - t0

    return {
        "plan": plan,
        "artifacts": artifacts,
        "build_median_ms": statistics.median(build_times) * 1000,
        "export_median_ms": statistics.median(export_times) * 1000,
        "sim_ms": sim_time * 1000,
        "mission_success": sim.mission_success,
        "overall_effectiveness": sim.overall_effectiveness,
        "ler": sim.ler,
        "stage_scores": sim.stage_scores,
        "c2sim_len": len(artifacts["c2sim_xml"]),
    }


def bench_llm_modes(scenario: Scenario, bank: CaseBank, args) -> dict:
    """对比 Full（4 候选 + LLM 反思）与 Fast（单候选 + 规则反思）的单轮耗时。"""
    out: dict = {}
    pipeline = MilitaryResearchPipeline(bank.path, seed=args.seed)

    # Full：1 轮，观察单轮耗时（4 候选 + 1 反思）
    print("\n[llm] 运行 Full 模式 1 轮（4 候选 + LLM 反思）...")
    t0 = time.perf_counter()
    full_result = pipeline.run(scenario, iterations=1, sim_runs=1, allow_writeback=False)
    full_wall = time.perf_counter() - t0
    out["full"] = {
        "one_iteration_sec": full_result["runtime_stats"]["avg_iteration_wall_time_sec"],
        "wall_sec": full_wall,
        "candidates": full_result["runtime_stats"]["total_candidate_evaluations"],
        "mission_success": full_result["best_simulation"]["mission_success"],
    }

    # Fast：N 轮
    print(f"\n[llm] 运行 Fast 模式 {args.fast_iterations} 轮（单候选 + 规则反思）...")
    t0 = time.perf_counter()
    fast_result = pipeline.run(
        scenario,
        iterations=args.fast_iterations,
        sim_runs=1,
        allow_writeback=False,
        single_candidate=True,
        rule_based_reflection=True,
    )
    fast_wall = time.perf_counter() - t0
    out["fast"] = {
        "iterations": args.fast_iterations,
        "one_iteration_sec": fast_result["runtime_stats"]["avg_iteration_wall_time_sec"],
        "wall_sec": fast_wall,
        "candidates": fast_result["runtime_stats"]["total_candidate_evaluations"],
        "mission_success": fast_result["best_simulation"]["mission_success"],
        "overall_effectiveness": fast_result["best_simulation"]["overall_effectiveness"],
        "config": fast_result["experiment_config"],
    }
    return out


def main() -> None:
    args = parse_args()
    scenario, bank = _load(args)
    print(f"场景: {scenario.name} | mission_type={scenario.mission_type} | terrain={scenario.terrain}")
    print(f"案例库: {args.case_bank} (records={len(bank.records)}, backend={bank.backend})")

    print("\n===== System 1（快道）应急出库 =====")
    s1 = bench_system1(scenario, bank, args)
    print(f"  方案构建中位数 : {s1['build_median_ms']:.2f} ms")
    print(f"  DoDAF/C2SIM导出 : {s1['export_median_ms']:.2f} ms")
    print(f"  总出库耗时      : {s1['build_median_ms'] + s1['export_median_ms']:.2f} ms")
    print(f"  C2SIM 字节数    : {s1['c2sim_len']}")
    print(f"  预案质量({args.mc_runs} MC) : MS={s1['mission_success']:.4f} "
          f"OE={s1['overall_effectiveness']:.4f} LER={s1['ler']:.4f} 耗时={s1['sim_ms']:.1f} ms")
    print(f"  五阶段          : {json.dumps({k: round(v, 4) for k, v in s1['stage_scores'].items()}, ensure_ascii=False)}")

    if args.llm and _llm_available():
        print("\n===== LLM 模式对比（需本地模型） =====")
        llm = bench_llm_modes(scenario, bank, args)
        full = llm["full"]
        fast = llm["fast"]
        speedup = full["one_iteration_sec"] / max(fast["one_iteration_sec"], 1e-9)
        print(f"  Full 单轮耗时 : {full['one_iteration_sec']:.1f}s (候选={full['candidates']})")
        print(f"  Fast 单轮耗时 : {fast['one_iteration_sec']:.1f}s (候选={fast['candidates']})")
        print(f"  单轮加速比    : {speedup:.1f}x")
        print(f"  Fast {fast['iterations']} 轮 MS = {fast['mission_success']:.4f}, OE = {fast['overall_effectiveness']:.4f}")
        est_full_10 = full["one_iteration_sec"] * 10
        est_fast_10 = fast["one_iteration_sec"] * 10
        print(f"  外推 10 轮: Full ≈ {est_full_10:.0f}s vs Fast ≈ {est_fast_10:.0f}s")
    else:
        print("\n[skip] 未开启 LLM 对比（加 --llm 可测 Fast/Full 单轮耗时）。")


if __name__ == "__main__":
    main()
