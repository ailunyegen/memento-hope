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

ACTION_TYPES = [
    "recon",
    "strike",
    "mobility",
    "control",
    "sustain",
    "c2",
    "ew",
    "other",
]

DOCTRINE_PROFILES = {"balanced_joint", "offensive_breakthrough", "defense_in_depth"}


def clamp_value(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _require_non_empty_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} 不能为空。")
    return text


def _coerce_ratio(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数值。") from exc
    return clamp_value(numeric)


def _coerce_positive_int(value: Any, field_name: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是正整数。") from exc
    if numeric <= 0:
        raise ValueError(f"{field_name} 必须大于 0。")
    return numeric


@dataclass
class ActionItem:
    action_type: str
    description: str

    def __post_init__(self) -> None:
        action_type = str(self.action_type or "").strip().lower()
        description = _require_non_empty_text(self.description, "ActionItem.description")
        if action_type not in ACTION_TYPES:
            action_type = "other"
        self.action_type = action_type
        self.description = description

    @classmethod
    def from_any(cls, payload: Any) -> "ActionItem":
        if isinstance(payload, ActionItem):
            return payload
        if isinstance(payload, dict):
            return cls(
                action_type=payload.get("action_type", "other"),
                description=payload.get("description", ""),
            )
        description = _require_non_empty_text(payload, "ActionItem.description")
        return cls(action_type=infer_action_type(description), description=description)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def infer_action_type(description: str) -> str:
    lower = description.lower()
    if any(token in description for token in ["侦察", "侦搜", "侦收", "态势", "目标确认", "识别"]) or "recon" in lower:
        return "recon"
    if any(token in description for token in ["压制", "打击", "火力", "摧毁", "点杀"]) or "strike" in lower:
        return "strike"
    if any(token in description for token in ["机动", "突击", "突破", "前出", "登陆"]) or "mobility" in lower:
        return "mobility"
    if any(token in description for token in ["夺控", "稳控", "封控", "控制", "占领"]) or "control" in lower:
        return "control"
    if any(token in description for token in ["补给", "恢复", "后送", "增援", "持续"]) or "sustain" in lower:
        return "sustain"
    if any(token in description for token in ["指挥", "协同", "通信", "链路"]) or lower.startswith("c2"):
        return "c2"
    if any(token in description for token in ["电子", "电磁", "干扰", "压制频谱"]) or lower == "ew":
        return "ew"
    return "other"


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

    def __post_init__(self) -> None:
        self.name = _require_non_empty_text(self.name, "ForceUnit.name")
        self.role = _require_non_empty_text(self.role, "ForceUnit.role")
        self.domain = _require_non_empty_text(self.domain, "ForceUnit.domain")
        self.quantity = _coerce_positive_int(self.quantity, "ForceUnit.quantity")
        for key in CAPABILITY_KEYS + ["readiness"]:
            setattr(self, key, _coerce_ratio(getattr(self, key), f"ForceUnit.{key}"))

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

    def __post_init__(self) -> None:
        self.name = _require_non_empty_text(self.name, "Scenario.name")
        self.mission_type = _require_non_empty_text(self.mission_type, "Scenario.mission_type")
        self.objective = _require_non_empty_text(self.objective, "Scenario.objective")
        self.terrain = _require_non_empty_text(self.terrain, "Scenario.terrain")
        self.weather = _require_non_empty_text(self.weather, "Scenario.weather")
        self.threat_level = _coerce_ratio(self.threat_level, "Scenario.threat_level")
        self.ew_threat = _coerce_ratio(self.ew_threat, "Scenario.ew_threat")
        self.civilian_presence = _coerce_ratio(self.civilian_presence, "Scenario.civilian_presence")
        self.time_pressure = _coerce_ratio(self.time_pressure, "Scenario.time_pressure")
        self.doctrine_profile = _require_non_empty_text(self.doctrine_profile, "Scenario.doctrine_profile")
        if self.doctrine_profile not in DOCTRINE_PROFILES:
            raise ValueError(
                f"未知 doctrine_profile: {self.doctrine_profile}。可选值为 {sorted(DOCTRINE_PROFILES)}。"
            )
        self.desired_end_state = str(self.desired_end_state or "").strip()
        self.constraints = [_require_non_empty_text(item, "Scenario.constraints") for item in self.constraints if str(item).strip()]
        self.friendly_forces = [ForceUnit.from_dict(item) if isinstance(item, dict) else item for item in self.friendly_forces]
        self.enemy_forces = [ForceUnit.from_dict(item) if isinstance(item, dict) else item for item in self.enemy_forces]
        if not self.friendly_forces:
            raise ValueError("Scenario.friendly_forces 不能为空。")
        if not self.enemy_forces:
            raise ValueError("Scenario.enemy_forces 不能为空。")

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
    actions: List[ActionItem]
    allocated_units: List[str]
    decision_points: List[str]
    expected_effects: List[str]

    def __post_init__(self) -> None:
        self.phase_id = _require_non_empty_text(self.phase_id, "PlanPhase.phase_id")
        self.name = _require_non_empty_text(self.name, "PlanPhase.name")
        self.intent = _require_non_empty_text(self.intent, "PlanPhase.intent")
        self.actions = [ActionItem.from_any(item) for item in self.actions]
        self.allocated_units = [
            _require_non_empty_text(item, "PlanPhase.allocated_units")
            for item in self.allocated_units
            if str(item).strip()
        ]
        self.decision_points = [
            _require_non_empty_text(item, "PlanPhase.decision_points")
            for item in self.decision_points
            if str(item).strip()
        ]
        self.expected_effects = [
            _require_non_empty_text(item, "PlanPhase.expected_effects")
            for item in self.expected_effects
            if str(item).strip()
        ]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlanPhase":
        data = dict(payload)
        data["actions"] = [ActionItem.from_any(item) for item in payload.get("actions", [])]
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["actions"] = [action.to_dict() for action in self.actions]
        return data


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

    def __post_init__(self) -> None:
        self.title = _require_non_empty_text(self.title, "CombatPlan.title")
        self.commander_intent = _require_non_empty_text(self.commander_intent, "CombatPlan.commander_intent")
        self.theory_of_victory = _require_non_empty_text(self.theory_of_victory, "CombatPlan.theory_of_victory")
        self.doctrine_weights = {key: _coerce_ratio(value, f"CombatPlan.doctrine_weights.{key}") for key, value in self.doctrine_weights.items()}
        self.fast_weights = {key: float(value) for key, value in self.fast_weights.items()}
        self.fused_weights = {key: _coerce_ratio(value, f"CombatPlan.fused_weights.{key}") for key, value in self.fused_weights.items()}
        self.memory_insights = [_require_non_empty_text(item, "CombatPlan.memory_insights") for item in self.memory_insights if str(item).strip()]
        self.risk_controls = [_require_non_empty_text(item, "CombatPlan.risk_controls") for item in self.risk_controls if str(item).strip()]
        self.assessment_metrics = [_require_non_empty_text(item, "CombatPlan.assessment_metrics") for item in self.assessment_metrics if str(item).strip()]
        self.phases = [PlanPhase.from_dict(item) if isinstance(item, dict) else item for item in self.phases]
        if not self.phases:
            raise ValueError("CombatPlan.phases 不能为空。")

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

    def __post_init__(self) -> None:
        self.phase_id = _require_non_empty_text(self.phase_id, "PhaseSimulation.phase_id")
        self.phase_name = _require_non_empty_text(self.phase_name, "PhaseSimulation.phase_name")
        self.success_score = _coerce_ratio(self.success_score, "PhaseSimulation.success_score")
        self.duration_hours = max(0.0, float(self.duration_hours))
        self.friendly_loss = clamp_value(self.friendly_loss, 0.0, 1.0)
        self.enemy_loss = clamp_value(self.enemy_loss, 0.0, 1.0)
        self.notes = [_require_non_empty_text(item, "PhaseSimulation.notes") for item in self.notes if str(item).strip()]

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
    monte_carlo_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mission_success = _coerce_ratio(self.mission_success, "SimulationResult.mission_success")
        self.ler = max(0.0, float(self.ler))
        self.completion_time_hours = max(0.0, float(self.completion_time_hours))
        self.survivability = _coerce_ratio(self.survivability, "SimulationResult.survivability")
        self.command_resilience = _coerce_ratio(self.command_resilience, "SimulationResult.command_resilience")
        self.overall_effectiveness = _coerce_ratio(self.overall_effectiveness, "SimulationResult.overall_effectiveness")
        self.stage_scores = {key: _coerce_ratio(value, f"SimulationResult.stage_scores.{key}") for key, value in self.stage_scores.items()}
        self.notes = [_require_non_empty_text(item, "SimulationResult.notes") for item in self.notes if str(item).strip()]
        self.phases = [PhaseSimulation(**item) if isinstance(item, dict) else item for item in self.phases]
        self.recommendations = [
            _require_non_empty_text(item, "SimulationResult.recommendations")
            for item in self.recommendations
            if str(item).strip()
        ]
        self.monte_carlo_stats = _normalize_monte_carlo_stats(self.monte_carlo_stats)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["phases"] = [phase.to_dict() for phase in self.phases]
        return data


def _normalize_monte_carlo_stats(value: Dict[str, Dict[str, float]] | None) -> Dict[str, Dict[str, float]]:
    if not value:
        return {}
    normalized: Dict[str, Dict[str, float]] = {}
    for metric_name, stats in value.items():
        metric_stats: Dict[str, float] = {}
        for stat_name in ("mean", "std", "min", "max", "ci95_low", "ci95_high", "n"):
            if stat_name in stats:
                metric_stats[stat_name] = float(stats[stat_name])
        if metric_stats:
            normalized[str(metric_name)] = metric_stats
    return normalized
