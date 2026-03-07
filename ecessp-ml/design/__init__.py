# design/__init__.py
# ============================================================
# Design Package Boundary
# ============================================================
# Purpose:
#   - Define design as a cohesive design-time layer
#   - Control what is importable from "design"
#   - Prevent accidental coupling with backend / models
#
# The design package:
#   - Owns system templates and generators
#   - Exposes design-time utilities
#   - Never exposes runtime or ML internals
# ============================================================

from .system_template import BatterySystem
from .system_generator import SystemGenerator
from .system_scorer import SystemScorer
from .system_reasoner import SystemReasoner
from .objective_schema import BatteryObjective

__all__ = [
    "BatterySystem",
    "SystemGenerator",
    "SystemScorer",
    "SystemReasoner",
    "BatteryObjective",
]
