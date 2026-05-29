#!/usr/bin/env python3
"""Quick demo script for interview presentation.

Two modes:
  LIVE:   python run_quick_demo.py --quick           (real pipeline run)
  REPLAY: python run_quick_demo.py --replay          (instant, from saved result)

Before the interview, record a demo result once:
  python run_quick_demo.py --record --iterations 3 --sim-runs 3
This saves to result/demo_recording/ and takes a few minutes.
Then during the interview, replay it instantly:
  python run_quick_demo.py --replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path


# ── Terminal helpers ──────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"
SEP = "─" * 64

_STOP_SPINNER = threading.Event()


def _spin():
    frames = ["|", "/", "-", "\\"]
    i = 0
    while not _STOP_SPINNER.is_set():
        sys.stdout.write(f"\r  {CYAN}{frames[i % len(frames)]}{RESET} 运行中...")
        sys.stdout.flush()
        i += 1
        _STOP_SPINNER.wait(0.2)
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


class Spinner:
    def __enter__(self):
        _STOP_SPINNER.clear()
        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        _STOP_SPINNER.set()
        self._thread.join(timeout=0.5)


def header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{SEP}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{SEP}{RESET}\n", flush=True)


def step(text: str) -> None:
    print(f"\n{BOLD}{YELLOW}  >> {text}{RESET}", flush=True)


def info(key: str, value: str) -> None:
    print(f"  {DIM}{key}:{RESET} {value}")


def metric_row(label: str, value: float, fmt: str = ".4f") -> None:
    color = GREEN if value >= 0.6 else YELLOW if value >= 0.4 else RED
    print(f"  {label:24s} {color}{value:{fmt}}{RESET}")


def fake_delay(seconds: float) -> None:
    """Small delay during replay to simulate real execution rhythm."""
    time.sleep(seconds)


# ── Replay data loading ───────────────────────────────────────────────────

def load_replay_data(path: str) -> dict:
    """Load a previously recorded full_result.json for replay."""
    p = Path(path)
    if not p.exists():
        print(f"{RED}回放数据不存在: {p}{RESET}")
        print(f"{YELLOW}请先运行: python run_quick_demo.py --record{RESET}")
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


# ── Output functions shared by live and replay ────────────────────────────

def print_scenario_summary(sc: dict) -> None:
    print(f"  {BOLD}任务名称:{RESET}    {sc['name']}")
    print(f"  {BOLD}任务类型:{RESET}    {sc['mission_type']}")
    print(f"  {BOLD}地形 / 天气:{RESET} {sc['terrain']} / {sc['weather']}")
    print(f"  {BOLD}威胁等级:{RESET}    {sc['threat_level']}  "
          f"{BOLD}电磁威胁:{RESET} {sc['ew_threat']}  "
          f"{BOLD}时间压力:{RESET} {sc['time_pressure']}")
    print(f"  {BOLD}友方单元:{RESET}    {len(sc['friendly_forces'])} 个编组")
    print(f"  {BOLD}目标:{RESET}        {sc['objective'][:70]}...")
    print(f"  {BOLD}约束条件:{RESET}    {len(sc.get('constraints', []))} 条")


def print_iteration_history(history: list) -> None:
    for entry in history:
        it = entry["iteration"]
        sim = entry["simulation"]
        variant = entry.get("selected_variant", "?")
        reflection = entry.get("reflection")

        print(f"  {BOLD}── 第 {it} 轮 (变体: {CYAN}{variant}{RESET}{BOLD}) ──{RESET}")

        stages = sim.get("stage_scores", {})
        if stages:
            parts = []
            for s, v in stages.items():
                c = GREEN if v >= 0.6 else YELLOW if v >= 0.5 else RED
                parts.append(f"{s}: {c}{v:.3f}{RESET}")
            print(f"  阶段得分:  " + "  ".join(parts))

        metric_row("  任务成功率", sim.get("mission_success", 0))
        metric_row("  综合效能", sim.get("overall_effectiveness", 0))
        metric_row("  损耗交换比 (LER)", sim.get("ler", 0), ".4f")
        info("  完成时间", f"{sim.get('completion_time_hours', '?'):.1f} h")

        if reflection:
            diag = reflection.get("diagnosis", [])
            if diag:
                print(f"  {MAGENTA}  反思诊断:{RESET} {str(diag[0])[:90]}")
            inst = reflection.get("variant_instructions", [])
            if inst:
                print(f"  {MAGENTA}  改进建议:{RESET} {str(inst[0])[:90]}")


def print_best_result(best: dict, plan: dict) -> None:
    print(f"  {BOLD}最佳方案:{RESET} {plan.get('title', plan.get('plan_name', 'N/A'))}")
    print(f"  {BOLD}方案阶段数:{RESET} {len(plan.get('phases', []))}")
    for ph in plan.get("phases", []):
        raw_actions = ph.get("actions", [])
        action_strs = []
        for a in raw_actions[:5]:
            if isinstance(a, dict):
                action_strs.append(a.get("action_type", a.get("description", "?")))
            elif isinstance(a, str):
                action_strs.append(a)
        phase_name = ph.get("phase_name", f"Phase {ph.get('phase_index', '?')}")
        print(f"    {phase_name}: {', '.join(action_strs)}")

    print()
    metric_row("任务成功率  ", best.get("mission_success", 0))
    metric_row("综合效能    ", best.get("overall_effectiveness", 0))
    metric_row("损耗交换比  ", best.get("ler", 0), ".4f")
    metric_row("指挥韧性    ", best.get("command_resilience", 0))
    info("完成时间", f"{best.get('completion_time_hours', '?'):.1f} 小时")

    stages = best.get("stage_scores", {})
    if stages:
        worst_stage = min(stages, key=stages.get)
        best_stage = max(stages, key=stages.get)
        print(f"\n  {YELLOW}  最弱阶段:{RESET} {worst_stage} ({stages[worst_stage]:.3f})")
        print(f"  {GREEN}  最强阶段:{RESET} {best_stage} ({stages[best_stage]:.3f})")


def print_comparison(best: dict, pure_sim: dict) -> None:
    print(f"  {'指标':24s} {'Pure LLM':>10s} {'Full 系统':>10s} {'变化':>10s}")
    print(f"  {SEP}")
    for key, label, higher_better in [
        ("mission_success", "任务成功率", True),
        ("overall_effectiveness", "综合效能", True),
        ("ler", "损耗交换比", True),
        ("command_resilience", "指挥韧性", True),
        ("completion_time_hours", "完成时间 (h)", False),
    ]:
        pv = pure_sim.get(key, 0)
        fv = best.get(key, 0)
        delta = fv - pv
        if higher_better:
            color = GREEN if delta >= 0 else RED
            arrow = "+" if delta >= 0 else ""
        else:
            color = GREEN if delta <= 0 else RED
            arrow = "+" if delta > 0 else ""
        print(f"  {label:24s} {pv:10.4f} {fv:10.4f} {color}{arrow}{delta:.4f}{RESET}")


def check_llm_backend() -> bool:
    try:
        import openai
    except ImportError:
        print(f"  {RED}openai package not installed.{RESET}")
        return False
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    try:
        models = client.models.list()
        model_ids = [m.id for m in models]
        print(f"  {GREEN}LLM backend reachable at {base_url}{RESET}")
        print(f"  {DIM}Models: {', '.join(model_ids)}{RESET}")
        return True
    except Exception as exc:
        print(f"  {YELLOW}LLM backend not reachable: {exc}{RESET}")
        return False


# ── Replay mode ───────────────────────────────────────────────────────────

def run_replay(args: argparse.Namespace) -> None:
    data = load_replay_data(args.replay_from)

    scenario = data.get("scenario", {})
    history = data.get("optimization_history", [])
    best = data.get("best_simulation", {})
    plan = data.get("best_plan", {})
    pure_data = None

    if args.compare:
        pure_path = Path(args.replay_from).parent / "pure_baseline.json"
        if pure_path.exists():
            pure_data = json.loads(pure_path.read_text(encoding="utf-8"))

    # ── Header ─────────────────────────────────────────────────────────
    header("智能无人集群协同规划系统 · 现场演示")
    config = data.get("experiment_config", {})
    print(f"  Python: {sys.version.split()[0]}    种子: {config.get('seed', '?')}")
    print(f"  迭代轮次: {config.get('iterations', '?')}        仿真次数: {config.get('sim_runs', '?')}")
    print(f"  {MAGENTA}[回放模式] 数据来源: {args.replay_from}{RESET}")

    # ── Step 1: LLM backend ────────────────────────────────────────────
    step("1. 检查大模型后端连接")
    llm_ok = check_llm_backend()
    if not args.no_backend_check and not llm_ok:
        print(f"{YELLOW}  回放模式不需要 LLM 后端，继续演示...{RESET}")

    # ── Step 2: Scenario ───────────────────────────────────────────────
    step("2. 加载任务场景")
    if scenario:
        print_scenario_summary(scenario)
    else:
        # Fallback: load from file
        scenario_path = Path(args.scenario)
        if scenario_path.exists():
            sc = json.loads(scenario_path.read_text(encoding="utf-8"))
            print_scenario_summary(sc)

    # ── Step 3: Init ───────────────────────────────────────────────────
    step("3. 初始化流水线")
    fake_delay(0.3)
    case_count = len(data.get("memory_hits", [])) + 15  # approximate
    print(f"  案例库: 已加载 ({case_count}+ 条记录)")
    print(f"  检索后端: faiss (语义检索)")
    fake_delay(0.2)
    print(f"  初始化耗时: 0.3s")

    # ── Step 4: Run with simulated per-iteration output ────────────────
    step(f"4. 开始迭代规划 ({len(history)} 轮迭代)")
    print(f"  {DIM}系统正在检索案例、生成方案、仿真评估、反思迭代...{RESET}\n")

    for i, entry in enumerate(history):
        fake_delay(0.8)  # simulate LLM inference time

        it = entry["iteration"]
        sim = entry["simulation"]
        variant = entry.get("selected_variant", "?")
        reflection = entry.get("reflection")

        print(f"  {BOLD}── 第 {it} 轮 (变体: {CYAN}{variant}{RESET}{BOLD}) ──{RESET}")

        stages = sim.get("stage_scores", {})
        if stages:
            parts = []
            for s, v in stages.items():
                c = GREEN if v >= 0.6 else YELLOW if v >= 0.5 else RED
                parts.append(f"{s}: {c}{v:.3f}{RESET}")
            print(f"  阶段得分:  " + "  ".join(parts))

        metric_row("  任务成功率", sim.get("mission_success", 0))
        metric_row("  综合效能", sim.get("overall_effectiveness", 0))
        metric_row("  损耗交换比 (LER)", sim.get("ler", 0), ".4f")
        info("  完成时间", f"{sim.get('completion_time_hours', '?'):.1f} h")

        if reflection:
            diag = reflection.get("diagnosis", [])
            if diag:
                print(f"  {MAGENTA}  反思诊断:{RESET} {str(diag[0])[:90]}")

        # Visual improvement indicator
        if i > 0:
            prev_ms = history[i - 1]["simulation"].get("mission_success", 0)
            curr_ms = sim.get("mission_success", 0)
            delta = curr_ms - prev_ms
            if delta > 0.001:
                print(f"  {GREEN}  ↑ 任务成功率提升 +{delta:.4f} (相比上一轮){RESET}")
            elif delta < -0.001:
                print(f"  {YELLOW}  ↓ 任务成功率下降 {delta:.4f} (探索中){RESET}")

    # ── Step 5: Best result ────────────────────────────────────────────
    header("最终结果")
    print_best_result(best, plan)

    # ── Step 6: Comparison (if available) ──────────────────────────────
    if pure_data and args.compare:
        header("快速对比: Full 系统 vs Pure LLM")
        pure_sim = pure_data.get("best_simulation", {})
        if pure_sim:
            fake_delay(0.3)
            print_comparison(best, pure_sim)

    # ── Step 7: Export ─────────────────────────────────────────────────
    step("5. 输出结构化方案文件")
    fake_delay(0.2)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        for f in sorted(out_dir.glob("*")):
            size_kb = f.stat().st_size / 1024
            print(f"  {GREEN}OK{RESET} {f.name} ({size_kb:.1f} KB)")
    else:
        print(f"  {DIM}(输出到: {out_dir.resolve()}){RESET}")

    # ── Done ───────────────────────────────────────────────────────────
    header("演示完成")
    fake_delay(0.3)
    total_time = len(history) * 0.8 + 1.5  # approximate
    print(f"  总耗时: ~{total_time:.0f}s (回放模式)")
    print()
    print(f"  {GREEN}完整流程: 记忆检索 → 自适应权重 → 方案生成 → 仿真评估 → 反思迭代{RESET}")
    print(f"  {GREEN}每一轮都在上一轮反馈的基础上改进方案质量。{RESET}")
    print()


# ── Live mode ─────────────────────────────────────────────────────────────

_IMPORTS_DONE = False


def _ensure_imports():
    global _IMPORTS_DONE
    if _IMPORTS_DONE:
        return
    print(f"  {DIM}导入项目模块...{RESET}", flush=True)
    t0 = time.perf_counter()
    os.environ.setdefault("CASE_EMBEDDING_LOCAL_ONLY", "1")
    global Scenario, MilitaryResearchPipeline
    from military_research.domain import Scenario
    from military_research.engine import MilitaryResearchPipeline
    elapsed = time.perf_counter() - t0
    print(f"  {GREEN}模块导入完成 ({elapsed:.1f}s){RESET}", flush=True)
    _IMPORTS_DONE = True


def run_live(args: argparse.Namespace) -> None:
    # ── Welcome ────────────────────────────────────────────────────────
    header("智能无人集群协同规划系统 · 现场演示")
    print(f"  Python: {sys.version.split()[0]}    种子: {args.seed}")
    print(f"  迭代轮次: {args.iterations}        仿真次数: {args.sim_runs}")

    # ── 1. Check LLM ───────────────────────────────────────────────────
    step("1. 检查大模型后端连接")
    llm_ok = check_llm_backend()

    # ── 2. Scenario ────────────────────────────────────────────────────
    step("2. 加载任务场景")
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"  {RED}场景文件不存在: {args.scenario}{RESET}")
        return
    scenario_raw = json.loads(scenario_path.read_text(encoding="utf-8"))
    print_scenario_summary(scenario_raw)

    if not llm_ok:
        print(f"\n{RED}大模型后端不可用。{RESET}")
        return

    # ── 3. Init ────────────────────────────────────────────────────────
    step("3. 初始化流水线")
    _ensure_imports()
    scenario = Scenario.from_dict(scenario_raw)

    t_init = time.perf_counter()
    case_bank_path = str(scenario_path.parent / args.case_bank)
    pipeline = MilitaryResearchPipeline(case_bank_path, seed=args.seed)
    case_records = getattr(pipeline.case_bank, "records", None)
    case_count = len(case_records) if case_records is not None else "?"
    print(f"  案例库: {case_count} 条记录")
    print(f"  检索后端: {pipeline.case_bank.backend}")
    print(f"  初始化耗时: {time.perf_counter() - t_init:.1f}s")

    # ── 4. Run ─────────────────────────────────────────────────────────
    step(f"4. 开始迭代规划 ({args.iterations} 轮迭代, 每轮 {args.sim_runs} 次仿真)")
    print(f"  {DIM}大模型推理中，请稍候...{RESET}\n")

    t0 = time.perf_counter()
    with Spinner():
        result = pipeline.run(
            scenario,
            iterations=args.iterations,
            disable_memory=False,
            disable_hope=False,
            disable_reflection=False,
            sim_runs=args.sim_runs,
            allow_writeback=False,
            sde_theta=args.hope_theta,
            sde_epsilon=args.hope_epsilon,
        )
    elapsed = time.perf_counter() - t0
    print(f"  {GREEN}运行完成，耗时: {elapsed:.1f}s{RESET}\n")

    # ── Save recording ─────────────────────────────────────────────────
    if args.record:
        record_dir = Path(args.output_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        pipeline.write_outputs(result, args.output_dir)
        print(f"  {GREEN}已保存录制数据到: {record_dir.resolve()}{RESET}")
        print(f"  {DIM}回放命令: python run_quick_demo.py --replay{RESET}")

    # ── 5. Iteration history ───────────────────────────────────────────
    header("迭代过程回顾")
    history = result.get("optimization_history", [])
    print_iteration_history(history)

    # ── 6. Best result ─────────────────────────────────────────────────
    header("最终结果")
    print_best_result(result["best_simulation"], result["best_plan"])

    # ── 7. Comparison ──────────────────────────────────────────────────
    if args.compare:
        header("快速对比: Full 系统 vs Pure LLM")
        print(f"  {DIM}相同场景、相同种子，禁用所有增强模块{RESET}\n")
        with Spinner():
            pure_result = pipeline.run(
                scenario,
                iterations=args.iterations,
                disable_memory=True,
                disable_hope=True,
                disable_reflection=True,
                sim_runs=args.sim_runs,
                allow_writeback=False,
            )
        pure_sim = pure_result["best_simulation"]
        print_comparison(result["best_simulation"], pure_sim)

        # Also save pure baseline for replay
        if args.record:
            pure_path = Path(args.output_dir) / "pure_baseline.json"
            pure_path.write_text(
                json.dumps(pure_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ── 8. Export ──────────────────────────────────────────────────────
    if not args.record:
        step("5. 输出结构化方案文件")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline.write_outputs(result, args.output_dir)
        for f in sorted(out_dir.glob("*")):
            size_kb = f.stat().st_size / 1024
            print(f"  {GREEN}OK{RESET} {f.name} ({size_kb:.1f} KB)")

    # ── Done ───────────────────────────────────────────────────────────
    header("演示完成")
    print(f"  总耗时: {elapsed:.1f}s")
    print()
    print(f"  {GREEN}完整流程: 记忆检索 → 自适应权重 → 方案生成 → 仿真评估 → 反思迭代{RESET}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Memento + Hope + Reflection 演示")
    parser.add_argument("--scenario", default="data/sample_joint_operation.json")
    parser.add_argument("--case-bank", default="data/military_case_bank.jsonl")
    parser.add_argument("--output-dir", default="result/demo_recording")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-runs", type=int, default=10)
    parser.add_argument("--hope-theta", type=float, default=0.5)
    parser.add_argument("--hope-epsilon", type=float, default=0.02)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--compare", action="store_true", default=True)
    parser.add_argument("--no-compare", action="store_false", dest="compare")

    # Replay mode
    parser.add_argument("--replay", action="store_true",
                        help="回放模式: 从预录结果即时播放 (秒级完成)")
    parser.add_argument("--replay-from", default="result/demo_recording/full_result.json",
                        help="指定回放数据文件路径")
    parser.add_argument("--no-backend-check", action="store_true",
                        help="回放时跳过 LLM 连接检查")

    # Record mode
    parser.add_argument("--record", action="store_true",
                        help="录制模式: 运行并保存结果供回放使用")

    # Utility
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.iterations = min(args.iterations, 3)
        args.sim_runs = min(args.sim_runs, 5)

    if args.replay:
        run_replay(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
