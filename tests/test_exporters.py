from __future__ import annotations

from military_research.domain import ActionItem, CombatPlan, ForceUnit, PlanPhase, Scenario
from military_research.exporters import build_c2sim_xml, build_ov5b, build_ov6c


def make_scenario() -> Scenario:
    return Scenario(
        name="demo",
        mission_type="assault",
        objective="夺控登陆窗口",
        terrain="coastal",
        weather="rain",
        threat_level=0.7,
        ew_threat=0.6,
        civilian_presence=0.2,
        time_pressure=0.4,
        doctrine_profile="offensive_breakthrough",
        friendly_forces=[ForceUnit(name="A", role="assault", domain="land")],
        enemy_forces=[ForceUnit(name="B", role="defense", domain="land")],
    )


def make_plan() -> CombatPlan:
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
        phases=[
            PlanPhase(
                phase_id="P1",
                name="侦察塑形",
                intent="建立态势",
                actions=[ActionItem("recon", "前出侦察"), ActionItem("c2", "共享情报")],
                allocated_units=["A"],
                decision_points=["d1"],
                expected_effects=["e1"],
            )
        ],
    )


def test_exporters_support_structured_actions() -> None:
    plan = make_plan()
    scenario = make_scenario()
    ov5b = build_ov5b(plan, scenario)
    ov6c = build_ov6c(plan)
    xml = build_c2sim_xml(plan, scenario)
    assert ov5b["activities"][0]["name"] == "侦察塑形"
    assert ov6c["event_traces"][0]["to"] == "侦察塑形"
    assert 'type="recon"' in xml
