"""Research prototype for military scheme generation and simulation.

This package keeps top-level imports lightweight so utility modules such as
`summarize_ablations` and `compare_results` can run without requiring the full
simulation dependency stack at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ActionItem",
    "CaseBank",
    "CaseRecord",
    "CombatPlan",
    "ForceUnit",
    "MilitaryResearchPipeline",
    "Scenario",
    "SimulationResult",
]


def __getattr__(name: str) -> Any:
    if name in {"CaseBank", "CaseRecord"}:
        module = import_module(".case_memory", __name__)
        return getattr(module, name)
    if name in {"ActionItem", "CombatPlan", "ForceUnit", "Scenario", "SimulationResult"}:
        module = import_module(".domain", __name__)
        return getattr(module, name)
    if name == "MilitaryResearchPipeline":
        module = import_module(".engine", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
