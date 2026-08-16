"""System 1（快道）应急预案原型单元测试。

仅覆盖零 LLM 的确定性路径：方案构建合法性、DoDAF/C2SIM 出库、仿真核验、
常驻预热。不依赖外部 LLM 服务，也不写入任何文件。
"""

from __future__ import annotations

import json
from pathlib import Path

from military_research.case_memory import CaseBank
from military_research.domain import CombatPlan, Scenario
from military_research.fast_pipeline import System1FastPlanner, apply_plan_patch

SCENARIO_PATH = Path("data/sample_joint_operation.json")
CASE_BANK_PATH = Path("data/military_case_bank_frozen_seed.jsonl")


def _scenario() -> Scenario:
    return Scenario.from_dict(json.loads(SCENARIO_PATH.read_text(encoding="utf-8")))


def _planner() -> System1FastPlanner:
    return System1FastPlanner(CaseBank(CASE_BANK_PATH), seed=7)


def test_system1_builds_valid_plan():
    scenario = _scenario()
    plan, _ = _planner().build_emergency_plan(scenario)
    assert len(plan.phases) >= 4
    units = {unit.name for unit in scenario.friendly_forces}
    for phase in plan.phases:
        assert len(phase.actions) >= 2
        assert len(phase.allocated_units) >= 2
        assert set(phase.allocated_units) <= units


def test_system1_exports_artifacts():
    planner = _planner()
    scenario = _scenario()
    plan, warped = planner.build_emergency_plan(scenario)
    artifacts = planner.export(plan, warped)
    assert artifacts["dodaf_ov5b"]["view"] == "OV-5b"
    assert artifacts["dodaf_ov6c"]["view"] == "OV-6c"
    assert "<C2SIM_Message" in artifacts["c2sim_xml"]
    assert artifacts["c2sim_xml"].strip()


def test_system1_simulate_returns_bounded_result():
    planner = _planner()
    scenario = _scenario()
    plan, warped = planner.build_emergency_plan(scenario)
    sim = planner.simulate(plan, warped, runs=5)
    assert 0.0 <= sim.mission_success <= 1.0
    assert sim.overall_effectiveness >= 0.0
    assert sim.ler >= 0.0


def test_system1_warmup_returns_positive_ms():
    planner = _planner()
    elapsed = planner.warmup()
    assert elapsed > 0.0


def test_system1_warp_scenario_scales_quantity():
    planner = _planner()
    scenario = _scenario()
    # 高威胁场景应放大兵力数量（UnitCount * CurrentThreat / PriorThreat）
    plan, warped = planner.build_emergency_plan(scenario)
    total_before = sum(unit.quantity for unit in scenario.friendly_forces)
    total_after = sum(unit.quantity for unit in warped.friendly_forces)
    assert total_after >= total_before


# ---------------------------------------------------------------------------
# Delta Patch 合并算子
# ---------------------------------------------------------------------------

def _sample_plan_dict() -> dict:
    plan, _ = _planner().build_emergency_plan(_scenario())
    return plan.to_dict()


def test_apply_plan_patch_replaces_target_phase_only():
    base = _sample_plan_dict()
    p2_actions_before = [p for p in base["phases"] if p["phase_id"] == "P2"][0]["actions"]
    patch = {
        "phase_id": "P2",
        "rationale": "Strengthen breach mobility and artillery support.",
        "actions": [
            {"action_type": "strike", "description": "远程火力群对突破口两翼实施持续压制。"},
            {"action_type": "mobility", "description": "两栖突击群沿主突击轴快速机动突入。"},
        ],
    }
    updated = apply_plan_patch(base, patch)
    # P2 被替换
    for phase in updated["phases"]:
        if phase["phase_id"] == "P2":
            assert [a["action_type"] for a in phase["actions"]] == ["strike", "mobility"]
            assert "Strengthen breach" in phase["intent"]
        else:
            # 其他阶段完全不变
            assert phase["actions"] == [p for p in base["phases"] if p["phase_id"] == phase["phase_id"]][0]["actions"]
    assert p2_actions_before != updated["phases"][1]["actions"]
    # 合并结果可被域模型重新解析（零侵入）
    CombatPlan(**updated)


def test_apply_plan_patch_fallback_on_missing_phase():
    base = _sample_plan_dict()
    patch = {"phase_id": "P99", "rationale": "x", "actions": [{"action_type": "strike", "description": "y"}]}
    updated = apply_plan_patch(base, patch)
    assert updated is base  # 未匹配时返回原方案（容错）


def test_apply_plan_patch_normalizes_actions_without_description():
    base = _sample_plan_dict()
    patch = {
        "phase_id": "P3",
        "rationale": "",
        "actions": [{"action_type": "strike", "target": "shore_defense_nodes"}],
    }
    updated = apply_plan_patch(base, patch)
    phase3 = [p for p in updated["phases"] if p["phase_id"] == "P3"][0]
    assert phase3["actions"][0]["action_type"] == "strike"
    assert "shore_defense_nodes" in phase3["actions"][0]["description"]
    # 无 rationale 时 intent 保持不变
    assert "局部补丁" not in phase3["intent"]
    CombatPlan(**updated)
