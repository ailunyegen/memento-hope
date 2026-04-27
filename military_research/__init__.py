"""Research prototype for military scheme generation and simulation."""

from .case_memory import CaseBank, CaseRecord
from .domain import CombatPlan, ForceUnit, Scenario, SimulationResult
from .engine import MilitaryResearchPipeline

__all__ = [
    "CaseBank",
    "CaseRecord",
    "CombatPlan",
    "ForceUnit",
    "MilitaryResearchPipeline",
    "Scenario",
    "SimulationResult",
]
