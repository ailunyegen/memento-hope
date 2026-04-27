from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


CAPABILITY_KEYS = [
    "fires",
    "mobility",
    "protection",
    "awareness",
    "ew",
    "sustainment",
    "c2",
]


def clamp_value(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class ForceUnit:
    name: str
    role: str
    domain: str
    quantity: int = 1
    readiness: float = 0.8
    fires: float = 0.5
    mobility: float = 0.5
    protection: float = 0.5
    awareness: float = 0.5
    ew: float = 0.3
    sustainment: float = 0.5
    c2: float = 0.5

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ForceUnit":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Scenario:
    name: str
    mission_type: str
    objective: str
    terrain: str
    weather: str
    threat_level: float
    ew_threat: float
    civilian_presence: float
    time_pressure: float
    doctrine_profile: str = "balanced_joint"
    desired_end_state: str = ""
    constraints: List[str] = field(default_factory=list)
    friendly_forces: List[ForceUnit] = field(default_factory=list)
    enemy_forces: List[ForceUnit] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Scenario":
        friendly = [ForceUnit.from_dict(item) for item in payload.get("friendly_forces", [])]
        enemy = [ForceUnit.from_dict(item) for item in payload.get("enemy_forces", [])]
        data = dict(payload)
        data["friendly_forces"] = friendly
        data["enemy_forces"] = enemy
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["friendly_forces"] = [unit.to_dict() for unit in self.friendly_forces]
        data["enemy_forces"] = [unit.to_dict() for unit in self.enemy_forces]
        return data


@dataclass
class PlanPhase:
    phase_id: str
    name: str
    intent: str
    actions: List[str]
    allocated_units: List[str]
    decision_points: List[str]
    expected_effects: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CombatPlan:
    title: str
    commander_intent: str
    theory_of_victory: str
    doctrine_weights: Dict[str, float]
    fast_weights: Dict[str, float]
    fused_weights: Dict[str, float]
    memory_insights: List[str]
    risk_controls: List[str]
    assessment_metrics: List[str]
    phases: List[PlanPhase]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["phases"] = [phase.to_dict() for phase in self.phases]
        return data


@dataclass
class PhaseSimulation:
    phase_id: str
    phase_name: str
    success_score: float
    duration_hours: float
    friendly_loss: float
    enemy_loss: float
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    mission_success: float
    ler: float
    completion_time_hours: float
    survivability: float
    command_resilience: float
    overall_effectiveness: float
    stage_scores: Dict[str, float]
    notes: List[str]
    phases: List[PhaseSimulation]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["phases"] = [phase.to_dict() for phase in self.phases]
        return data
