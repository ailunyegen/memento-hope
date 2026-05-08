from __future__ import annotations

from military_research.case_memory import CaseRecord
from military_research.domain import ForceUnit, PlanPhase, Scenario
from military_research.engine import PlanGenerator


def make_scenario() -> Scenario:
    return Scenario(
        name="demo",
        mission_type="assault",
        objective="夺控登陆窗口",
        terrain="coastal-urban",
        weather="rain",
        threat_level=0.7,
        ew_threat=0.6,
        civilian_presence=0.2,
        time_pressure=0.5,
        doctrine_profile="offensive_breakthrough",
        friendly_forces=[
            ForceUnit(name="侦察分队", role="recon", domain="air", awareness=0.9, c2=0.7),
            ForceUnit(name="火力分队", role="fires", domain="land", fires=0.9),
            ForceUnit(name="突击分队", role="assault", domain="land", mobility=0.8, protection=0.7),
        ],
        enemy_forces=[ForceUnit(name="敌防御群", role="defense", domain="land")],
    )


def test_repair_plan_structure_adds_action_types_and_covers_units() -> None:
    generator = PlanGenerator.__new__(PlanGenerator)
    scenario = make_scenario()
    phases = [
        PlanPhase(
            phase_id="P1",
            name="侦察塑形",
            intent="建立态势",
            actions=["前出侦察"],
            allocated_units=["侦察分队"],
            decision_points=["是否继续侦察"],
            expected_effects=["形成目标图谱"],
        ),
        PlanPhase(
            phase_id="P2",
            name="火力压制",
            intent="瘫痪关键节点",
            actions=["火力压制"],
            allocated_units=["火力分队"],
            decision_points=["是否追加火力"],
            expected_effects=["敌火力下降"],
        ),
    ]
    repaired = generator._repair_plan_structure(phases, scenario)
    assert all(len(phase.actions) >= 2 for phase in repaired)
    assert all(hasattr(action, "action_type") for phase in repaired for action in phase.actions)
    assigned = {unit for phase in repaired for unit in phase.allocated_units}
    assert {"侦察分队", "火力分队", "突击分队"}.issubset(assigned)


def test_build_phase_recovers_blank_phase_id_and_name() -> None:
    generator = PlanGenerator.__new__(PlanGenerator)
    phase = generator._build_phase(
        {
            "phase_id": "   ",
            "name": "   ",
            "intent": "",
            "actions": [{"action_type": "recon", "description": "前出侦察"}],
            "allocated_units": ["侦察分队"],
            "decision_points": ["是否继续侦察"],
            "expected_effects": ["形成目标图谱"],
        },
        3,
    )
    assert phase.phase_id == "P3"
    assert phase.name


def test_memory_action_guidance_strengthens_coastal_phases() -> None:
    generator = PlanGenerator.__new__(PlanGenerator)
    scenario = make_scenario()
    phases = generator._repair_plan_structure(
        [
            PlanPhase(
                phase_id="P1",
                name="phase-1",
                intent="intent-1",
                actions=["recon-step"],
                allocated_units=["侦察分队"],
                decision_points=["dp-1"],
                expected_effects=["effect-1"],
            ),
            PlanPhase(
                phase_id="P2",
                name="phase-2",
                intent="intent-2",
                actions=["strike-step"],
                allocated_units=["火力分队"],
                decision_points=["dp-2"],
                expected_effects=["effect-2"],
            ),
            PlanPhase(
                phase_id="P3",
                name="phase-3",
                intent="intent-3",
                actions=["mobility-step"],
                allocated_units=["突击分队"],
                decision_points=["dp-3"],
                expected_effects=["effect-3"],
            ),
            PlanPhase(
                phase_id="P4",
                name="phase-4",
                intent="intent-4",
                actions=["control-step"],
                allocated_units=["火力分队"],
                decision_points=["dp-4"],
                expected_effects=["effect-4"],
            ),
        ],
        scenario,
    )
    memory_hits = [
        {
            "record": CaseRecord(
                case_id="PRIOR-COASTAL-ASSAULT",
                mission_type="assault",
                terrain="coastal-urban",
                objective="obj",
                friendly_roles=["recon"],
                enemy_roles=["defense"],
                key_actions=["a"],
                lessons=["l"],
                outcome="positive",
                metrics={"stage_disrupt": 0.66, "stage_breach": 0.63},
            ),
            "score": 0.5,
            "confidence": 0.5,
            "quality": 0.6,
            "usage_mode": "imitate",
        }
    ]
    guided = generator._apply_memory_action_guidance(phases, scenario, memory_hits)
    phase2_types = {action.action_type for action in guided[1].actions}
    phase3_types = {action.action_type for action in guided[2].actions}
    assert {"strike", "ew"}.issubset(phase2_types)
    assert {"mobility", "strike"}.issubset(phase3_types)
