from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from .domain import CombatPlan, Scenario


def build_ov5b(plan: CombatPlan, scenario: Scenario) -> Dict[str, Any]:
    activities: List[Dict[str, Any]] = []
    for index, phase in enumerate(plan.phases, 1):
        activities.append(
            {
                "activity_id": f"OV5B-ACT-{index:02d}",
                "name": phase.name,
                "intent": phase.intent,
                "performers": phase.allocated_units,
                "inputs": [scenario.objective, *scenario.constraints],
                "outputs": phase.expected_effects,
                "governing_rules": plan.risk_controls,
            }
        )
    return {
        "view": "OV-5b",
        "operation_name": plan.title,
        "mission_type": scenario.mission_type,
        "activities": activities,
    }


def build_ov6c(plan: CombatPlan) -> Dict[str, Any]:
    traces: List[Dict[str, Any]] = []
    previous = "CommandAuthority"
    for index, phase in enumerate(plan.phases, 1):
        event_id = f"OV6C-EVT-{index:02d}"
        traces.append(
            {
                "event_id": event_id,
                "from": previous,
                "to": phase.name,
                "trigger": phase.intent,
                "effects": phase.expected_effects,
                "decision_points": phase.decision_points,
            }
        )
        previous = phase.name
    traces.append(
        {
            "event_id": f"OV6C-EVT-{len(plan.phases) + 1:02d}",
            "from": previous,
            "to": "BattleDamageAssessment",
            "trigger": "Collect mission metrics and close the loop",
            "effects": plan.assessment_metrics,
            "decision_points": ["Write simulation feedback into memory"],
        }
    )
    return {
        "view": "OV-6c",
        "operation_name": plan.title,
        "event_traces": traces,
    }


def build_c2sim_xml(plan: CombatPlan, scenario: Scenario) -> str:
    root = ET.Element("C2SIM_Message", attrib={"xmlns": "urn:sisostds:c2sim:std:019:2020"})
    header = ET.SubElement(root, "Header")
    ET.SubElement(header, "MessageType").text = "Order"
    ET.SubElement(header, "OperationName").text = plan.title

    body = ET.SubElement(root, "Body")
    order = ET.SubElement(body, "Order")
    ET.SubElement(order, "MissionType").text = scenario.mission_type
    ET.SubElement(order, "Objective").text = scenario.objective
    ET.SubElement(order, "CommanderIntent").text = plan.commander_intent
    ET.SubElement(order, "DesiredEndState").text = scenario.desired_end_state or scenario.objective

    control = ET.SubElement(order, "ControlMeasures")
    for item in plan.risk_controls:
        ET.SubElement(control, "Measure").text = item

    for phase in plan.phases:
        task = ET.SubElement(order, "Task", attrib={"id": phase.phase_id})
        ET.SubElement(task, "Name").text = phase.name
        ET.SubElement(task, "Intent").text = phase.intent
        ET.SubElement(task, "AllocatedUnits").text = ", ".join(phase.allocated_units)

        action_list = ET.SubElement(task, "ActionList")
        for action in phase.actions:
            ET.SubElement(action_list, "Action").text = action

        effect_list = ET.SubElement(task, "ExpectedEffects")
        for effect in phase.expected_effects:
            ET.SubElement(effect_list, "Effect").text = effect

        decision_list = ET.SubElement(task, "DecisionPoints")
        for item in phase.decision_points:
            ET.SubElement(decision_list, "Decision").text = item

    return ET.tostring(root, encoding="unicode")
