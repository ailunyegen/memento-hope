from __future__ import annotations

import random

from military_research.domain import ActionItem, CombatPlan, ForceUnit, PlanPhase, Scenario
from military_research.engine import PlanSimulator


def make_scenario() -> Scenario:
    return Scenario(
        name="demo",
        mission_type="assault",
        objective="夺控登陆窗口",
        terrain="coastal-urban",
        weather="rain",
        threat_level=0.72,
        ew_threat=0.65,
        civilian_presence=0.2,
        time_pressure=0.5,
        doctrine_profile="offensive_breakthrough",
        friendly_forces=[
            ForceUnit(name="侦察分队", role="recon", domain="air", awareness=0.9, c2=0.7, ew=0.7),
            ForceUnit(name="火力分队", role="fires", domain="land", fires=0.9),
            ForceUnit(name="突击分队", role="assault", domain="land", mobility=0.8, protection=0.7),
            ForceUnit(name="保障分队", role="sustain", domain="land", sustainment=0.8),
        ],
        enemy_forces=[ForceUnit(name="敌岸防群", role="defense", domain="land")],
    )


def make_plan(primary_breach_type: str = "mobility") -> CombatPlan:
    phases = []
    stage_names = ["侦察塑形", "火力压制", "突破夺控", "持续稳控"]
    stage_actions = [
        [ActionItem("recon", "侦察分队前出侦察"), ActionItem("c2", "建立共享链路")],
        [ActionItem("strike", "火力分队压制敌关键节点"), ActionItem("ew", "电子干扰主轴")],
        [ActionItem(primary_breach_type, "突击分队沿主轴快速推进"), ActionItem("strike", "火力协同开辟通路")],
        [ActionItem("control", "完成稳控部署"), ActionItem("sustain", "组织后续补给")],
    ]
    units = [
        ["侦察分队", "火力分队"],
        ["火力分队", "侦察分队"],
        ["突击分队", "火力分队"],
        ["保障分队", "突击分队"],
    ]
    for idx, name in enumerate(stage_names, start=1):
        phases.append(
            PlanPhase(
                phase_id=f"P{idx}",
                name=name,
                intent=name,
                actions=stage_actions[idx - 1],
                allocated_units=units[idx - 1],
                decision_points=["d1", "d2"],
                expected_effects=["e1", "e2"],
            )
        )
    return CombatPlan(
        title="demo",
        commander_intent="intent",
        theory_of_victory="victory",
        doctrine_weights={"fires": 0.8, "mobility": 0.7, "protection": 0.6, "awareness": 0.75, "ew": 0.7, "sustainment": 0.65, "c2": 0.76},
        fast_weights={"fires": 0.0, "mobility": 0.0, "protection": 0.0, "awareness": 0.0, "ew": 0.0, "sustainment": 0.0, "c2": 0.0},
        fused_weights={"fires": 0.8, "mobility": 0.7, "protection": 0.6, "awareness": 0.75, "ew": 0.7, "sustainment": 0.65, "c2": 0.76},
        memory_insights=[],
        risk_controls=["r1"],
        assessment_metrics=["m1"],
        phases=phases,
    )


def test_simulator_outputs_monte_carlo_stats() -> None:
    simulator = PlanSimulator(random.Random(7))
    result = simulator.run(make_scenario(), make_plan(), stochastic=False)
    assert "mission_success" in result.monte_carlo_stats
    assert result.monte_carlo_stats["mission_success"]["std"] == 0.0
    assert result.monte_carlo_stats["mission_success"]["ci95_low"] == result.monte_carlo_stats["mission_success"]["mean"]


def test_action_type_changes_stage_scores() -> None:
    simulator = PlanSimulator(random.Random(7))
    scenario = make_scenario()
    mobility_result = simulator.run(scenario, make_plan(primary_breach_type="mobility"), stochastic=False)
    control_result = simulator.run(scenario, make_plan(primary_breach_type="control"), stochastic=False)
    assert mobility_result.stage_scores["breach"] >= control_result.stage_scores["breach"]
