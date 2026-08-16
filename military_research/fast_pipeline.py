"""System 1（快道）应急方案快速出库原型。

对应《技术方案设计书：基于“双系统投机增量规划”的大模型作战方案秒级生成架构》
中的 System 1（参数化模板形变引擎）：

1. 骨架继承：Memento 检索 Top-1 历史最佳案例，继承其阶段划分与动作类型序列；
2. 权重投影：HOPE 控制器输出当前场景融合权重，覆盖方案能力权重；
3. 兵力配比映射：按威胁等级差线性缩放兵力数量
   （``UnitCount * CurrentThreat / PriorThreat``）；
4. 纯 Python 规则实现，零 LLM 调用，目标 < 1s 完成 DoDAF/C2SIM 出库。

本模块是原型代码，不修改现有核心流水线（``engine.MilitaryResearchPipeline``）
的任何行为；System 2（慢道增量演化）与 Delta-Patch 另行实现。
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, List, Tuple

from .case_memory import CaseBank, CaseRecord
from .domain import (
    ACTION_TYPES,
    CAPABILITY_KEYS,
    ActionItem,
    CombatPlan,
    ForceUnit,
    PlanPhase,
    Scenario,
    SimulationResult,
    clamp_value,
    infer_action_type,
)
from .engine import (
    SLOW_WEIGHT_LIBRARY,
    BottleneckDetector,
    DualMemoryController,
    LocalLLMPlanner,
    PlanGenerator,
    PlanSimulator,
)
from .exporters import build_c2sim_xml, build_ov5b, build_ov6c

# kill-chain 阶段 → 支持的动作类型集合（与 engine.PlanSimulator 的 action_map 对齐）
STAGE_ACTION_TYPES: Dict[str, set[str]] = {
    "detect": {"recon", "c2", "ew"},
    "disrupt": {"strike", "ew", "c2"},
    "breach": {"mobility", "strike", "protection"},
    "control": {"control", "c2", "protection"},
    "sustain": {"sustain", "c2"},
}

STAGE_NAMES = ["detect", "disrupt", "breach", "control"]

# kill-chain 阶段 → 核心能力维度（与 engine.BottleneckDetector.STAGE_TO_CAP 对齐）
STAGE_TOP_CAPS: Dict[str, List[str]] = {
    "detect": ["awareness", "c2"],
    "disrupt": ["fires", "ew"],
    "breach": ["mobility", "fires"],
    "control": ["protection", "c2"],
    "sustain": ["sustainment", "c2"],
}

# 能力维度 → 动作类型（能力赤字注入时建议 LLM 使用的 action_type）
CAP_TO_ACTION: Dict[str, str] = {
    "fires": "strike",
    "mobility": "mobility",
    "protection": "protection",
    "awareness": "recon",
    "ew": "ew",
    "c2": "c2",
    "sustainment": "sustain",
}


def _normalize_patch_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """将 LLM 输出的补丁动作降维为域模型所需的 ``{action_type, description}``。

    现有 ``ActionItem`` 仅消费 ``action_type`` 与 ``description`` 两个字段
    （仿真器与 DoDAF/C2SIM 导出器都只读它们），``action_id`` / ``target`` /
    ``capability_weight`` 等语义字段若不并入 description 会在下游解析时被
    静默丢弃，因此这里统一折叠进描述文本。
    """
    action_type = str(action.get("action_type") or "other").strip().lower()
    if action_type not in ACTION_TYPES:
        action_type = "other"
    description = str(action.get("description") or "").strip()
    if not description:
        target = str(action.get("target") or "").strip()
        description = f"针对{target}的{action_type}行动" if target else f"{action_type}行动（补丁生成）"
    return {"action_type": action_type, "description": description}


def apply_plan_patch(base_plan: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """以原方案为锚点，原地局部替换指定 Phase 的 actions 列表。

    下游 ``CombatPlan`` / ``PlanPhase`` 解析与仿真器输入完全兼容（零侵入）：
    - 仅修改 ``patch["phase_id"]`` 对应阶段的 ``actions``，其余阶段保持不动；
    - 补丁说明 ``rationale`` 并入该阶段 ``intent``（``PlanPhase`` 无 ``notes`` 字段，
      直接写入会导致下游解析 TypeError）；
    - ``phase_id`` 未匹配时返回原始方案（容错，调用方决定是否回滚）。
    """
    updated_plan = copy.deepcopy(base_plan)
    target_phase_id = patch.get("phase_id")
    new_actions = patch.get("actions", [])

    phase_found = False
    for phase in updated_plan.get("phases", []):
        if phase.get("phase_id") == target_phase_id:
            phase["actions"] = [_normalize_patch_action(action) for action in new_actions]
            rationale = str(patch.get("rationale") or "").strip()
            if rationale:
                intent = str(phase.get("intent") or "").strip()
                phase["intent"] = (
                    f"{intent}（局部补丁：{rationale}）" if intent else f"局部补丁：{rationale}"
                )
            phase_found = True
            break

    if not phase_found:
        return base_plan
    return updated_plan


class System1FastPlanner:
    """基于 Top-1 案例 + HOPE 权重投影的零 LLM 应急方案生成器。"""

    def __init__(self, case_bank: CaseBank, seed: int = 7):
        self.case_bank = case_bank
        self.seed = seed
        self._generator = PlanGenerator()  # 仅复用模板方法，不触发 LLM 调用

    # ------------------------------------------------------------------
    def build_emergency_plan(
        self,
        scenario: Scenario,
        run_hope: bool = True,
    ) -> Tuple[CombatPlan, Scenario]:
        """构造 System 1 应急预案。

        返回 ``(plan, warped_scenario)``：
        - ``plan``：由案例骨架 + 权重投影得到的结构化方案；
        - ``warped_scenario``：兵力数量按威胁比缩放后的场景副本（供仿真评估用）。
        """
        case = self._retrieve_top_case(scenario)
        warped_scenario = self._warp_scenario(scenario, case)

        controller = DualMemoryController(scenario.doctrine_profile, seed=self.seed)
        if run_hope:
            memory_metrics = [case.metrics] if case is not None and case.metrics else None
            fast_weights = controller.fast_weights(scenario, memory_metrics=memory_metrics)
            fused_weights = controller.fuse(fast_weights)
        else:
            fast_weights = {key: 0.0 for key in CAPABILITY_KEYS}
            fused_weights = dict(SLOW_WEIGHT_LIBRARY.get(
                scenario.doctrine_profile, SLOW_WEIGHT_LIBRARY["balanced_joint"]
            ))

        phases = self._build_phases(scenario, case)
        plan = CombatPlan(
            title=self._title(scenario, case),
            commander_intent=self._commander_intent(scenario, case),
            theory_of_victory=self._theory_of_victory(scenario, case),
            doctrine_weights=dict(SLOW_WEIGHT_LIBRARY.get(
                scenario.doctrine_profile, SLOW_WEIGHT_LIBRARY["balanced_joint"]
            )),
            fast_weights=fast_weights,
            fused_weights=fused_weights,
            memory_insights=self._memory_insights(case),
            risk_controls=[
                "应急预案：基于历史案例骨架的确定性形变，未经深度优化。",
                "应急出库后由 System 2 慢道异步增量优化接力。",
                "兵力配比已按当前/历史威胁比线性缩放。",
            ],
            assessment_metrics=[
                "战损交换比(LER)",
                "任务完成时间",
                "五阶段杀伤链得分",
                "指挥链路完整度",
                "后续行动可持续性",
            ],
            phases=phases,
        )
        return plan, warped_scenario

    # ------------------------------------------------------------------
    def export(self, plan: CombatPlan, scenario: Scenario) -> Dict[str, Any]:
        """秒级出库：一次性生成 DoDAF OV-5b / OV-6c 与 C2SIM XML。"""
        return {
            "dodaf_ov5b": build_ov5b(plan, scenario),
            "dodaf_ov6c": build_ov6c(plan),
            "c2sim_xml": build_c2sim_xml(plan, scenario),
        }

    # ------------------------------------------------------------------
    def simulate(self, plan: CombatPlan, scenario: Scenario, runs: int = 1):
        """对应急预案做蒙特卡洛仿真评估（10ms 级，用于质量快速核验）。"""
        import random

        from .domain import SimulationResult

        simulator = PlanSimulator(random.Random(self.seed))
        if runs <= 1:
            return simulator.run(scenario, plan, stochastic=False)
        runs_list = [simulator.run(scenario, plan, stochastic=True) for _ in range(runs)]
        best = max(runs_list, key=lambda r: r.mission_success)
        return SimulationResult(
            mission_success=round(sum(r.mission_success for r in runs_list) / len(runs_list), 4),
            ler=round(sum(r.ler for r in runs_list) / len(runs_list), 4),
            completion_time_hours=round(sum(r.completion_time_hours for r in runs_list) / len(runs_list), 2),
            survivability=round(sum(r.survivability for r in runs_list) / len(runs_list), 4),
            command_resilience=round(sum(r.command_resilience for r in runs_list) / len(runs_list), 4),
            overall_effectiveness=round(sum(r.overall_effectiveness for r in runs_list) / len(runs_list), 4),
            stage_scores={
                name: round(sum(r.stage_scores.get(name, 0.0) for r in runs_list) / len(runs_list), 4)
                for name in (runs_list[0].stage_scores or {})
            },
            notes=best.notes + [f"System 1 预案，{len(runs_list)} 次 MC 均值统计。"],
            phases=best.phases,
            recommendations=best.recommendations,
        )

    # ------------------------------------------------------------------
    def warmup(self, scenario: Scenario | None = None) -> float:
        """服务常驻预热：预跑一次完整出库链路，锁定在线推理为纯热态。

        首次构建的 ~0.9s 主要来自 PyTorch/CUDA context 初始化与 HOPE 权重
        装载，属于服务生命周期管理范畴。FastAPI / gRPC / 指控后台守护进程
        启动时调用一次本方法，即可将后续每次出库严格控制在毫秒级。
        返回预热耗时（毫秒）。
        """
        dummy = scenario or self._dummy_scenario()
        started = time.perf_counter()
        self.build_emergency_plan(dummy)
        return round((time.perf_counter() - started) * 1000, 1)

    @staticmethod
    def _dummy_scenario() -> Scenario:
        from .domain import ForceUnit

        return Scenario(
            name="warmup-dummy",
            mission_type="assault",
            objective="warmup",
            terrain="coastal",
            weather="clear",
            threat_level=0.5,
            ew_threat=0.3,
            civilian_presence=0.2,
            time_pressure=0.5,
            friendly_forces=[
                ForceUnit(name="Unit_A", role="recon", domain="land"),
                ForceUnit(name="Unit_B", role="fires", domain="air"),
            ],
            enemy_forces=[
                ForceUnit(name="Enemy_A", role="defense", domain="land"),
            ],
        )

    # ------------------------------------------------------------------
    def _retrieve_top_case(self, scenario: Scenario) -> CaseRecord | None:
        hits = self.case_bank.retrieve(scenario, top_k=1)
        if hits:
            return hits[0]["record"]
        # 兜底：场景先验案例（与 engine 的 synthetic prior 对齐的最小化版本）
        return None

    def _warp_scenario(self, scenario: Scenario, case: CaseRecord | None) -> Scenario:
        """兵力配比映射：``UnitCount * CurrentThreat / PriorThreat``。"""
        prior_threat = 0.5
        if case is not None and case.metrics:
            prior_threat = float(case.metrics.get("threat_level", 0.5) or 0.5)
        scale = clamp_value(scenario.threat_level / max(prior_threat, 1e-3), 0.5, 2.0)
        if abs(scale - 1.0) < 1e-6:
            return scenario
        warped_units = []
        for unit in scenario.friendly_forces:
            warped_units.append(
                ForceUnit(
                    name=unit.name,
                    role=unit.role,
                    domain=unit.domain,
                    quantity=max(1, round(unit.quantity * scale)),
                    readiness=unit.readiness,
                    fires=unit.fires,
                    mobility=unit.mobility,
                    protection=unit.protection,
                    awareness=unit.awareness,
                    ew=unit.ew,
                    sustainment=unit.sustainment,
                    c2=unit.c2,
                )
            )
        data = scenario.to_dict()
        data["friendly_forces"] = [u.to_dict() for u in warped_units]
        return Scenario.from_dict(data)

    def _build_phases(self, scenario: Scenario, case: CaseRecord | None) -> List[PlanPhase]:
        fallback_units = self._generator._phase_fallback_units(scenario)
        # 按全部 kill-chain 阶段归集案例动作；sustain 类型动作并入 control 阶段
        # （4 阶段方案无独立 sustain 阶段，与现有 CombatPlan 结构一致）。
        case_actions: Dict[str, List[ActionItem]] = {stage: [] for stage in STAGE_ACTION_TYPES}
        if case is not None:
            for text in case.key_actions:
                action_type = infer_action_type(text)
                for stage, types in STAGE_ACTION_TYPES.items():
                    if action_type in types:
                        case_actions[stage].append(ActionItem(action_type=action_type, description=text))
                        break
        case_actions["control"].extend(case_actions.get("sustain", []))

        phases: List[PlanPhase] = []
        friendly_names = [unit.name for unit in scenario.friendly_forces]
        for idx, stage_name in enumerate(STAGE_NAMES):
            phase_id = f"P{idx + 1}"
            allocated = fallback_units.get(stage_name, [])[:3]
            if len(allocated) < 2:
                for name in friendly_names:
                    if name not in allocated:
                        allocated.append(name)
                    if len(allocated) >= 2:
                        break
            if len(allocated) < 2:  # 极端兜底：单元不足时复制名称占位
                allocated = (friendly_names * 2)[:2]
            phase = PlanPhase(
                phase_id=phase_id,
                name=stage_name,
                intent=self._phase_intent(stage_name, scenario),
                actions=[],
                allocated_units=allocated,
                decision_points=self._generator._default_decision_points(stage_name, scenario),
                expected_effects=self._generator._default_expected_effects(stage_name, scenario),
            )
            # 案例动作优先，模板动作兜底到至少 2 条
            seen: set[Tuple[str, str]] = set()
            merged: List[ActionItem] = []
            for action in case_actions.get(stage_name, []) + self._generator._default_actions(stage_name, phase, scenario):
                key = (action.action_type, action.description)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(action)
                if len(merged) >= 4:
                    break
            if len(merged) < 2:
                merged.extend(self._generator._default_actions(stage_name, phase, scenario)[: 2 - len(merged)])
            phase.actions = merged
            phases.append(phase)
        return phases

    # ------------------------------------------------------------------
    def _phase_intent(self, stage_name: str, scenario: Scenario) -> str:
        intents = {
            "detect": f"多源侦察塑形，锁定{scenario.objective}相关关键节点并建立统一态势图。",
            "disrupt": f"对敌关键节点实施火力与电磁压制，为后续行动创造窗口。",
            "breach": f"沿主攻轴快速机动突破，依托压制窗口达成{scenario.objective}。",
            "control": "完成关键地域夺控后展开稳控部署，封控敌方反扑通道。",
        }
        return intents[stage_name]

    def _title(self, scenario: Scenario, case: CaseRecord | None) -> str:
        if case is not None:
            return f"{scenario.name}应急方案（基于案例 {case.case_id} 形变）"
        return f"{scenario.name}应急方案"

    def _commander_intent(self, scenario: Scenario, case: CaseRecord | None) -> str:
        if case is not None and case.lessons:
            return f"在{scenario.time_pressure:.0%}时间压力下，继承历史经验完成{scenario.objective}；要点：{'；'.join(case.lessons[:2])}"
        return f"在{scenario.time_pressure:.0%}时间压力与{scenario.threat_level:.0%}威胁强度下，围绕{scenario.objective}组织多域协同与稳控。"

    def _theory_of_victory(self, scenario: Scenario, case: CaseRecord | None) -> str:
        return f"以侦察塑形、联合火力、机动夺控和持续保障形成闭环，实现{scenario.objective}。"

    def _memory_insights(self, case: CaseRecord | None) -> List[str]:
        if case is None:
            return ["未检索到高置信案例，本预案基于默认作战模板实例化。"]
        prefix = "正样本" if case.outcome == "positive" else "负样本"
        return [f"{prefix} {case.case_id} 提示：{'；'.join(case.lessons[:2]) or '按案例骨架执行'}"]


class DeltaRefiner:
    """System 2（慢道）增量优化闭环（Delta Loop）。

    以应急预案（Anchor）为锚点，避免全量重写破坏高质量案例骨架：

    1. 50-MC 仿真当前方案，``BottleneckDetector`` 锁定最弱 kill-chain 阶段；
    2. LLM 仅针对该阶段对应的 Phase 生成局部补丁（~150 tokens），
       Prompt 指令为“保持其他阶段完全不变，仅重构 [Phase ID] 的 actions”；
    3. ``apply_plan_patch`` 内存合并；
    4. 门禁更新（Acceptance Gate）：补丁后 50-MC 仿真，
       ``MS_new >= MS_current`` 才接受，否则回滚，彻底消灭负向退化。
    """

    # kill-chain 阶段 → 4 阶段方案中的 Phase id（与 PlanGenerator._infer_stage_name 对齐）
    STAGE_TO_PHASE: Dict[str, str] = {
        "detect": "P1",
        "disrupt": "P2",
        "breach": "P3",
        "control": "P4",
        "sustain": "P4",
    }

    def __init__(self, llm: LocalLLMPlanner | None = None, seed: int = 7, sim_runs: int = 50):
        self.llm = llm or LocalLLMPlanner()
        self.seed = seed
        self.sim_runs = max(1, int(sim_runs))
        self.simulator = PlanSimulator(random.Random(seed))
        self.detector = BottleneckDetector(window_size=6)
        self.controller: DualMemoryController | None = None
        self.fused_weights: Dict[str, float] | None = None

    # ------------------------------------------------------------------
    def refine(
        self,
        scenario: Scenario,
        anchor_plan: CombatPlan,
        iterations: int = 5,
        stream_callback=None,
        explore_iters: int = 3,
        tolerance: float = 0.008,
        early_stop_ms: float = 0.67,
        stagnation_rounds: int = 3,
    ) -> Tuple[CombatPlan, List[Dict[str, Any]]]:
        """从应急预案出发执行增量闭环，返回（全局最优方案, 迭代历史）。

        - 探索期（前 ``explore_iters`` 轮）：允许 ``MS_new >= MS_cur - tolerance``
          的小幅负向扰动，给 LLM 重构动作组合的空间；
        - 收敛期（其后）：严格要求 ``MS_new >= MS_cur``；
        - 双轨协同：每轮 50-MC 后调用 ``controller.integrate_feedback``，
          快慢权重伴随补丁迭代实时演化，并写回候选方案的 fused_weights；
        - 动态早停：best-so-far 达到 ``early_stop_ms``，或连续
          ``stagnation_rounds`` 轮无正向突破（ΔMS < 0.001）即截断；
        - 最终交付全局 best-so-far 方案，中间状态无论接受与否都不影响终值。
        """
        self.controller = DualMemoryController(scenario.doctrine_profile, seed=self.seed)
        fast = self.controller.fast_weights(scenario)
        fused = self.controller.fuse(fast)

        current_plan = self._apply_fused(anchor_plan, fused)
        current_result = self._simulate(scenario, current_plan)
        best_plan = current_plan
        best_result = current_result
        history: List[Dict[str, Any]] = []
        no_improve_streak = 0
        last_best_ms = best_result.mission_success

        for iteration in range(1, iterations + 1):
            stage_scores = current_result.stage_scores or {}
            self.detector.update(stage_scores)
            bottleneck_stage, severity, _ = self.detector.identify()
            phase_id = self.STAGE_TO_PHASE.get(bottleneck_stage, "P1")

            deficits, suggested_types = self._capability_deficits(bottleneck_stage, phase_id, current_plan)
            patch = self._llm_generate_patch(
                scenario, current_plan, phase_id, bottleneck_stage, severity,
                deficits=deficits, suggested_types=suggested_types, fused_weights=fused,
            )
            if patch is None:
                history.append({"iteration": iteration, "bottleneck_stage": bottleneck_stage,
                                "accepted": False, "note": "patch_generation_failed"})
                break

            candidate_plan = CombatPlan(**apply_plan_patch(current_plan.to_dict(), patch))
            candidate_plan = self._apply_fused(candidate_plan, fused)  # 参数轨道写回
            candidate_result = self._simulate(scenario, candidate_plan)

            # ── HOPE 双轨反馈：无论接受与否，权重随本轮仿真结果演化 ──
            self.controller.integrate_feedback(candidate_plan, candidate_result)
            fused = self.controller.fuse(self.controller.fast_weights(scenario))

            # ── 门禁：探索容忍（前 explore_iters 轮）→ 严格贪心（其后）──
            ms_current = current_result.mission_success
            threshold = ms_current - tolerance if iteration <= explore_iters else ms_current
            accepted = candidate_result.mission_success >= threshold
            if accepted:
                current_plan = candidate_plan
                current_result = candidate_result
            if candidate_result.mission_success >= best_result.mission_success:
                best_plan = candidate_plan
                best_result = candidate_result

            # ── 停滞检测：连续无正向突破计数 ──
            if best_result.mission_success > last_best_ms + 0.001:
                no_improve_streak = 0
            else:
                no_improve_streak += 1
            last_best_ms = best_result.mission_success

            record = {
                "iteration": iteration,
                "bottleneck_stage": bottleneck_stage,
                "severity": round(severity, 4),
                "phase_id": phase_id,
                "deficits": deficits,
                "suggested_types": suggested_types,
                "rationale": patch.get("rationale", ""),
                "ms_current": round(ms_current, 4),
                "ms_candidate": round(candidate_result.mission_success, 4),
                "threshold": round(threshold, 4),
                "accepted": accepted,
                "best_ms": round(best_result.mission_success, 4),
                "no_improve_streak": no_improve_streak,
            }
            history.append(record)
            if stream_callback is not None:
                stream_callback(
                    {
                        "tier": "System-2-Delta",
                        "iteration": iteration,
                        "accepted": accepted,
                        "ms_current": record["ms_current"],
                        "ms_candidate": record["ms_candidate"],
                        "best_ms": record["best_ms"],
                        "bottleneck_stage": bottleneck_stage,
                        "deficits": deficits,
                    }
                )
            # ── 动态早停 ──
            if best_result.mission_success >= early_stop_ms and iteration >= explore_iters:
                break  # 达到预设质量门限
            if no_improve_streak >= stagnation_rounds:
                break  # 连续 stagnation_rounds 轮无突破

        return best_plan, history

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_fused(plan: CombatPlan, fused: Dict[str, float]) -> CombatPlan:
        """将当前 HOPE 融合权重写回方案（参数轨道实时生效）。"""
        plan.fused_weights = {key: round(float(value), 4) for key, value in fused.items()}
        return plan

    # ------------------------------------------------------------------
    def _capability_deficits(
        self, bottleneck_stage: str, phase_id: str, plan: CombatPlan
    ) -> Tuple[List[str], List[str]]:
        """提取瓶颈阶段的核心能力赤字与建议补齐的 action_type。"""
        top_caps = list(STAGE_TOP_CAPS.get(bottleneck_stage, ["mobility", "fires"]))
        phase = next((p for p in plan.phases if p.phase_id == phase_id), None)
        current_types = {a.action_type for a in phase.actions} if phase else set()
        suggested = [
            CAP_TO_ACTION[cap]
            for cap in top_caps
            if CAP_TO_ACTION.get(cap) and CAP_TO_ACTION[cap] not in current_types
        ]
        return top_caps, suggested

    # ------------------------------------------------------------------
    def _simulate(self, scenario: Scenario, plan: CombatPlan) -> SimulationResult:
        runs = [self.simulator.run(scenario, plan, stochastic=True) for _ in range(self.sim_runs)]
        best = max(runs, key=lambda r: r.mission_success)
        return SimulationResult(
            mission_success=round(sum(r.mission_success for r in runs) / len(runs), 4),
            ler=round(sum(r.ler for r in runs) / len(runs), 4),
            completion_time_hours=round(sum(r.completion_time_hours for r in runs) / len(runs), 2),
            survivability=round(sum(r.survivability for r in runs) / len(runs), 4),
            command_resilience=round(sum(r.command_resilience for r in runs) / len(runs), 4),
            overall_effectiveness=round(sum(r.overall_effectiveness for r in runs) / len(runs), 4),
            stage_scores={
                name: round(sum(r.stage_scores.get(name, 0.0) for r in runs) / len(runs), 4)
                for name in (runs[0].stage_scores or {})
            },
            notes=best.notes + [f"Delta 门禁 50-MC 统计（n={len(runs)}）。"],
            phases=best.phases,
            recommendations=best.recommendations,
        )

    # ------------------------------------------------------------------
    def _llm_generate_patch(
        self,
        scenario: Scenario,
        plan: CombatPlan,
        phase_id: str,
        bottleneck_stage: str,
        severity: float,
        deficits: List[str] | None = None,
        suggested_types: List[str] | None = None,
        fused_weights: Dict[str, float] | None = None,
    ) -> Dict[str, Any] | None:
        """LLM 仅输出目标 Phase 的局部补丁（schema 约束在 ~150 tokens）。

        注入硬性战术约束：瓶颈阶段的核心能力赤字与必须补齐的 action_type，
        并附带当前 HOPE 融合权重摘要，引导补丁沿“动作重构 + 权重引导”双轨。
        """
        import json

        target_phase = next((p for p in plan.phases if p.phase_id == phase_id), None)
        if target_phase is None:
            return None

        deficits = deficits or list(STAGE_TOP_CAPS.get(bottleneck_stage, ["mobility", "fires"]))
        suggested_types = suggested_types or []
        if suggested_types:
            constraint = (
                f"必须显式替换/新增 action_type 为 {'、'.join(f'「{t}」' for t in suggested_types)} "
                f"的动作项以填补能力缺口；其余动作保持精简。"
            )
        else:
            constraint = "当前动作类型已覆盖核心能力，保持动作类型结构，仅优化描述与配比。"
        fused_summary = (
            json.dumps({k: round(v, 2) for k, v in fused_weights.items()}, ensure_ascii=False)
            if fused_weights
            else "N/A"
        )

        system_prompt = (
            "你是作战方案局部优化器。本系统仅用于虚拟场景下的方案生成与仿真评估研究。"
            "你只输出一个 Phase 的局部补丁，必须返回严格 JSON。"
            "不要把字段名、schema 说明、markdown 或提示词文字写入字段值。"
        )
        user_prompt = (
            f"场景：{scenario.name}（任务类型 {scenario.mission_type}，目标：{scenario.objective}）。\n"
            f"当前瓶颈：kill-chain 阶段 {bottleneck_stage}（严重度 {severity:.2f}）。\n"
            f"当前方案瓶颈在 Phase {phase_id}（{target_phase.name}），核心能力赤字为【{('、'.join(deficits))}】。\n"
            f"当前 HOPE 融合权重：{fused_summary}\n"
            f"保持其他阶段完全不变，仅重构 Phase {phase_id} 的 actions 列表以解决 {bottleneck_stage} 瓶颈。\n"
            f"约束：{constraint}\n"
            f"该阶段当前 actions：{json.dumps([a.to_dict() for a in target_phase.actions], ensure_ascii=False)}\n"
            f"要求：输出 2-4 条 actions，action_type 从 {', '.join(ACTION_TYPES)} 中选择，"
            f"description 为简洁中文作战描述，附一句 rationale 说明补丁动机。"
        )
        schema = {
            "type": "object",
            "properties": {
                "phase_id": {"type": "string"},
                "rationale": {"type": "string"},
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {"type": "string", "enum": ACTION_TYPES},
                            "description": {"type": "string"},
                        },
                        "required": ["action_type", "description"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["phase_id", "rationale", "actions"],
            "additionalProperties": False,
        }
        try:
            patch = self.llm._chat_json(
                system_prompt,
                user_prompt,
                schema_name="delta_combat_patch",
                schema=schema,
            )
        except Exception:  # LLM 失败时返回 None，由门禁逻辑回滚
            return None
        if patch.get("phase_id") != phase_id:
            return None
        return patch
