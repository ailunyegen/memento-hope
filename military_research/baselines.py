"""External baselines for the Memento-Hope-Reflection pipeline.

Implements three lightweight baselines that require no external framework
dependencies, as requested by the reviewer:

1. **Few-shot CoT** – 5 example plans as a chain-of-thought prompt, single-pass.
2. **Random search** – N randomly perturbed plans; best by simulation score.
3. **Single-pass Memento** – memory retrieval only, no iteration / no Hope.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List

from .case_memory import CaseBank
from .domain import CAPABILITY_KEYS, CombatPlan, Scenario, SimulationResult
from .engine import (
    SLOW_WEIGHT_LIBRARY,
    PlanGenerator,
    PlanSimulator,
)


# ---------------------------------------------------------------------------
# Few-shot chain-of-thought examples (abbreviated, bilingual, structure-focused)
# ---------------------------------------------------------------------------
FEWSHOT_COT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "title": "沿海联合突击方案（示例A）",
        "commander_intent": "先期侦察塑形，精确火力瘫痪岸防节点，两栖机动开辟登陆窗口，夺控后转入纵深防御。",
        "theory_of_victory": "以侦-打-突-控四段闭环，集中优势火力与机动于登陆窗薄弱处，速决制胜。",
        "phases": [
            {
                "phase_id": "P1", "name": "侦察塑形",
                "intent": "多源侦察锁定敌岸防与防空节点，电子压制掩护侦察行动。",
                "actions": [
                    {"action_type": "recon", "description": "无人机/卫星侦察定位敌岸防导弹与雷达阵地"},
                    {"action_type": "ew", "description": "电子干扰压制敌预警雷达和通信链路"},
                    {"action_type": "c2", "description": "合成情报融合形成统一战场态势图"},
                ],
                "allocated_units": ["侦察营", "电子对抗连"],
                "decision_points": ["确认敌主要火力点坐标", "评估电磁压制效果"],
                "expected_effects": ["敌岸防体系透明化", "我方侦察安全通道建立"],
            },
            {
                "phase_id": "P2", "name": "火力瘫痪",
                "intent": "精确打击摧毁敌防空和岸防关键节点，削弱敌反登陆能力。",
                "actions": [
                    {"action_type": "strike", "description": "海空联合精确打击摧毁敌防空导弹阵地"},
                    {"action_type": "ew", "description": "持续电磁压制配合火力打击"},
                    {"action_type": "mobility", "description": "两栖突击群向登陆展开线机动"},
                ],
                "allocated_units": ["战斗机中队", "驱逐舰编队", "电子对抗连"],
                "decision_points": ["评估火力毁伤效果", "确定登陆窗口开启时机"],
                "expected_effects": ["敌防空/岸防能力降至30%以下", "登陆通道火力走廊建立"],
            },
            {
                "phase_id": "P3", "name": "突破上陆",
                "intent": "多波次两栖突击，快速开辟并巩固登陆场。",
                "actions": [
                    {"action_type": "mobility", "description": "两栖突击车与气垫登陆艇多波次冲击"},
                    {"action_type": "strike", "description": "舰炮与航空兵近距火力支援"},
                    {"action_type": "protection", "description": "防空掩护与烟幕遮蔽登陆场"},
                ],
                "allocated_units": ["两栖突击营", "驱逐舰编队", "防空连"],
                "decision_points": ["登陆场正面宽度调整", "预备队投入时机"],
                "expected_effects": ["登陆场正面≥2km", "第一梯队完成上陆展开"],
            },
            {
                "phase_id": "P4", "name": "夺控稳局",
                "intent": "向内陆突贯夺占关键地形，建立防御态势，保障后续梯队投送。",
                "actions": [
                    {"action_type": "control", "description": "夺占港口设施与交通枢纽"},
                    {"action_type": "sustain", "description": "建立滩头补给基地与医疗后送链路"},
                    {"action_type": "c2", "description": "指挥所前移，整合后续梯队指挥关系"},
                ],
                "allocated_units": ["机械化步兵营", "后勤保障营"],
                "decision_points": ["纵深推进停止线", "后续梯队投送优先序"],
                "expected_effects": ["关键港口/枢纽控制", "持续投送能力建立"],
            },
        ],
    },
    {
        "title": "城市防御作战方案（示例B）",
        "commander_intent": "外围逐次抗击，核心城区弹性防御，保存有生力量待援反冲击。",
        "theory_of_victory": "以空间换时间，利用城市复杂地形消耗敌进攻锐势，在关键节点集中预备队反冲击。",
        "phases": [
            {
                "phase_id": "P1", "name": "外围预警",
                "intent": "远距离侦察预警，迟滞敌装甲前锋，为主防御地带争取准备时间。",
                "actions": [
                    {"action_type": "recon", "description": "无人机/侦察兵前出监视敌装甲纵队动向"},
                    {"action_type": "c2", "description": "建立多级预警信息分发链路"},
                ],
                "allocated_units": ["侦察营"],
                "decision_points": ["敌主攻方向判断", "外围迟滞作战发起时机"],
                "expected_effects": ["获取≥6小时预警时间", "敌装甲前锋行进速度下降50%"],
            },
            {
                "phase_id": "P2", "name": "火力拦阻",
                "intent": "在敌接近路上以火力层层拦阻，重点打击敌装甲和工兵。",
                "actions": [
                    {"action_type": "strike", "description": "远程炮兵与反坦克导弹在预设歼敌区精确打击"},
                    {"action_type": "ew", "description": "干扰敌通信与导航，制造指挥混乱"},
                    {"action_type": "protection", "description": "工兵设障与爆破封锁关键道路"},
                ],
                "allocated_units": ["炮兵营", "反坦克连", "工兵连"],
                "decision_points": ["火力转移时机", "预设歼敌区调整"],
                "expected_effects": ["敌装甲损失≥30%", "敌进攻队形被打散"],
            },
            {
                "phase_id": "P3", "name": "城区坚守",
                "intent": "依托城市建筑群开展要点防守，逐楼逐街消耗敌步兵。",
                "actions": [
                    {"action_type": "protection", "description": "核心建筑加固与地下通道利用"},
                    {"action_type": "control", "description": "分区防守，保持预备队机动能力"},
                    {"action_type": "mobility", "description": "预备队在关键缺口快速机动反冲击"},
                ],
                "allocated_units": ["机械化步兵营", "反坦克连"],
                "decision_points": ["预备队投入方向", "防御重心转移"],
                "expected_effects": ["核心城区保持控制", "敌24小时内无法突破"],
            },
            {
                "phase_id": "P4", "name": "反击恢复",
                "intent": "在我方增援到达后发起协调反冲击，恢复防御态势。",
                "actions": [
                    {"action_type": "mobility", "description": "装甲预备队沿隐蔽路线向突破口机动"},
                    {"action_type": "strike", "description": "集中全部火力支援反冲击方向"},
                    {"action_type": "sustain", "description": "战场抢救抢修与弹药紧急前送"},
                ],
                "allocated_units": ["坦克营", "炮兵营", "后勤保障营"],
                "decision_points": ["反冲击发起时刻", "追击终止线"],
                "expected_effects": ["恢复原防御线", "敌被逐出城区"],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def run_fewshot_cot_baseline(
    scenario: Scenario,
    rng: random.Random,
    sim_runs: int = 50,
) -> Dict[str, Any]:
    """Few-shot CoT: inject example plans directly into prompt, single generation."""
    generator = PlanGenerator()
    simulator = PlanSimulator(rng)

    # Use PlanGenerator.generate with the few-shot examples embedded via a
    # custom reflection-like patch that carries the CoT examples.
    cot_patch = {
        "diagnosis": [
            "请参考以下示例方案的推理模式与结构深度，为目标场景生成一个同等质量的方案。"
        ],
        "variant_instructions": [
            json.dumps(ex, ensure_ascii=False)
            for ex in FEWSHOT_COT_EXAMPLES
        ],
        "meta_prompt_patch": (
            "请模仿提供的示例方案的推理深度、阶段分解方式和动作设计模式。"
            "示例中展示了如何根据场景特点设计侦察-火力-机动-保障的闭环链路。"
        ),
    }

    doctrine = SLOW_WEIGHT_LIBRARY.get(
        scenario.doctrine_profile, SLOW_WEIGHT_LIBRARY["balanced_joint"]
    )
    fused = {key: round(value, 4) for key, value in doctrine.items()}
    fast = {key: 0.0 for key in CAPABILITY_KEYS}

    plan = generator.generate(
        scenario=scenario,
        fused_weights=fused,
        fast_weights=fast,
        memory_hits=[],
        iteration_index=0,
        reflection=cot_patch,
        previous_best_plan=None,
    )
    result = _evaluate_with_mc(simulator, scenario, plan, sim_runs, rng)
    return {
        "baseline_type": "fewshot_cot",
        "plan": plan.to_dict(),
        "simulation": result.to_dict(),
    }


def run_random_search_baseline(
    scenario: Scenario,
    rng: random.Random,
    candidates: int = 20,
    sim_runs: int = 50,
) -> Dict[str, Any]:
    """Random search: generate *candidates* plans with random weight perturbations,
    evaluate all, return the best."""
    generator = PlanGenerator()
    simulator = PlanSimulator(rng)

    profession = scenario.doctrine_profile
    base_slow = SLOW_WEIGHT_LIBRARY.get(profession, SLOW_WEIGHT_LIBRARY["balanced_joint"])

    best_plan: CombatPlan | None = None
    best_result: SimulationResult | None = None
    all_results: List[Dict[str, Any]] = []

    for i in range(candidates):
        # Randomly perturb capability weights
        fused = {}
        for key in CAPABILITY_KEYS:
            noise = rng.uniform(-0.15, 0.15)
            fused[key] = round(clamp_value(base_slow.get(key, 0.65) + noise, 0.24, 1.0), 4)
        fast = {key: 0.0 for key in CAPABILITY_KEYS}

        plan = generator.generate(
            scenario=scenario,
            fused_weights=fused,
            fast_weights=fast,
            memory_hits=[],
            iteration_index=i,
            reflection=None,
            previous_best_plan=best_plan,
        )
        result = _evaluate_with_mc(simulator, scenario, plan, sim_runs, rng)
        all_results.append(
            {
                "candidate": i,
                "mission_success": result.mission_success,
                "overall_effectiveness": result.overall_effectiveness,
                "ler": result.ler,
            }
        )
        if best_result is None or result.overall_effectiveness > best_result.overall_effectiveness:
            best_plan = plan
            best_result = result

    assert best_plan is not None and best_result is not None
    return {
        "baseline_type": "random_search",
        "candidates_evaluated": candidates,
        "all_candidate_summaries": all_results,
        "plan": best_plan.to_dict(),
        "simulation": best_result.to_dict(),
    }


def run_singlepass_memento_baseline(
    scenario: Scenario,
    case_bank: CaseBank,
    rng: random.Random,
    sim_runs: int = 50,
) -> Dict[str, Any]:
    """Single-pass Memento: retrieve memory, generate one plan, no iteration."""
    generator = PlanGenerator()
    simulator = PlanSimulator(rng)

    raw_hits = case_bank.retrieve(scenario, top_k=3)
    memory_hits = _simple_select(raw_hits)

    fused = {
        key: round(value, 4)
        for key, value in SLOW_WEIGHT_LIBRARY.get(
            scenario.doctrine_profile,
            SLOW_WEIGHT_LIBRARY["balanced_joint"],
        ).items()
    }
    fast = {key: 0.0 for key in CAPABILITY_KEYS}

    plan = generator.generate(
        scenario=scenario,
        fused_weights=fused,
        fast_weights=fast,
        memory_hits=memory_hits,
        iteration_index=0,
        reflection=None,
        previous_best_plan=None,
    )
    result = _evaluate_with_mc(simulator, scenario, plan, sim_runs, rng)
    return {
        "baseline_type": "singlepass_memento",
        "memory_hits_used": [
            {
                "score": h["score"],
                "confidence": h["confidence"],
                "usage_mode": h.get("usage_mode"),
                "record_id": h["record"].case_id,
            }
            for h in memory_hits
        ],
        "plan": plan.to_dict(),
        "simulation": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_with_mc(
    simulator: PlanSimulator,
    scenario: Scenario,
    plan: CombatPlan,
    sim_runs: int,
    rng: random.Random,
) -> SimulationResult:
    if sim_runs <= 1:
        return simulator.run(scenario, plan, stochastic=False)
    runs = [simulator.run(scenario, plan, stochastic=True) for _ in range(sim_runs)]
    return _aggregate_runs(runs)


def _aggregate_runs(runs: List[SimulationResult]) -> SimulationResult:
    if len(runs) == 1:
        return runs[0]
    n = len(runs)
    ms = sum(r.mission_success for r in runs) / n
    oe = sum(r.overall_effectiveness for r in runs) / n
    ler_vals = [r.ler for r in runs]
    ler = sum(ler_vals) / n
    ct = sum(r.completion_time_hours for r in runs) / n
    cr = sum(r.command_resilience for r in runs) / n
    ss: Dict[str, float] = {}
    if runs[0].stage_scores:
        for stage in runs[0].stage_scores:
            ss[stage] = sum(r.stage_scores.get(stage, 0.0) for r in runs) / n

    mc_stats = {
        "mission_success": _summary([r.mission_success for r in runs]),
        "overall_effectiveness": _summary([r.overall_effectiveness for r in runs]),
        "ler": _summary(ler_vals),
        "completion_time_hours": _summary([r.completion_time_hours for r in runs]),
        "command_resilience": _summary([r.command_resilience for r in runs]),
    }
    return SimulationResult(
        mission_success=round(ms, 4),
        overall_effectiveness=round(oe, 4),
        ler=round(ler, 4),
        completion_time_hours=round(ct, 4),
        command_resilience=round(cr, 4),
        survivability=round(sum(r.survivability for r in runs) / n, 4),
        phases=runs[0].phases,
        stage_scores=ss,
        notes=[],
        recommendations=[],
        monte_carlo_stats=mc_stats,
    )


def _summary(values: List[float]) -> Dict[str, float]:
    mean = sum(values) / len(values)
    # Use sample variance (n-1) so confidence intervals are honest at small n.
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
    std = math.sqrt(variance)
    ci = 1.96 * std / math.sqrt(len(values))
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "ci95_low": round(mean - ci, 4),
        "ci95_high": round(mean + ci, 4),
        "n": len(values),
    }


def _simple_select(raw_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple memory selection for single-pass baseline (no strict filters).

    Returns new dicts so the caller's *raw_hits* are never mutated.
    """
    selected: List[Dict[str, Any]] = []
    for hit in raw_hits:
        rec = hit["record"]
        if rec.outcome == "success" and hit["score"] >= 0.35:
            hit_copy = dict(hit)
            hit_copy["usage_mode"] = "imitate"
            selected.append(hit_copy)
        elif rec.outcome != "success" and hit["score"] >= 0.30:
            hit_copy = dict(hit)
            hit_copy["usage_mode"] = "avoid"
            selected.append(hit_copy)
        if len(selected) >= 2:
            break
    return selected


def clamp_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
