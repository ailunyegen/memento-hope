from __future__ import annotations

import pytest

from military_research.domain import ActionItem, ForceUnit, PlanPhase, Scenario, SimulationResult


def test_force_unit_clamps_ratio_fields() -> None:
    unit = ForceUnit(
        name="A",
        role="recon",
        domain="land",
        quantity=2,
        readiness=1.2,
        fires=-0.3,
        mobility=0.6,
        protection=0.7,
        awareness=1.5,
        ew=-1.0,
        sustainment=0.4,
        c2=2.0,
    )
    assert unit.readiness == 1.0
    assert unit.fires == 0.0
    assert unit.awareness == 1.0
    assert unit.ew == 0.0
    assert unit.c2 == 1.0


def test_scenario_rejects_unknown_doctrine() -> None:
    with pytest.raises(ValueError):
        Scenario(
            name="demo",
            mission_type="assault",
            objective="夺控桥头堡",
            terrain="coastal",
            weather="rain",
            threat_level=0.6,
            ew_threat=0.6,
            civilian_presence=0.2,
            time_pressure=0.4,
            doctrine_profile="unknown_profile",
            friendly_forces=[ForceUnit(name="A", role="recon", domain="land")],
            enemy_forces=[ForceUnit(name="B", role="defense", domain="land")],
        )


def test_plan_phase_accepts_legacy_string_actions() -> None:
    phase = PlanPhase(
        phase_id="P1",
        name="侦察塑形",
        intent="建立态势",
        actions=["无人机前出侦察", {"action_type": "c2", "description": "同步共享目标信息"}],
        allocated_units=["U1", "U2"],
        decision_points=["是否延长侦察窗口"],
        expected_effects=["形成目标图谱"],
    )
    assert all(isinstance(action, ActionItem) for action in phase.actions)
    assert phase.actions[0].description == "无人机前出侦察"


def test_simulation_result_normalizes_stats() -> None:
    result = SimulationResult(
        mission_success=0.7,
        ler=1.3,
        completion_time_hours=8.5,
        survivability=0.8,
        command_resilience=0.75,
        overall_effectiveness=0.72,
        stage_scores={"detect": 0.6},
        notes=["ok"],
        phases=[],
        recommendations=["keep"],
        monte_carlo_stats={
            "mission_success": {
                "mean": 0.7,
                "std": 0.1,
                "min": 0.5,
                "max": 0.9,
                "ci95_low": 0.5,
                "ci95_high": 0.9,
                "n": 5,
            }
        },
    )
    assert result.monte_carlo_stats["mission_success"]["std"] == 0.1
    assert result.monte_carlo_stats["mission_success"]["n"] == 5
