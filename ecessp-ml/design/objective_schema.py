# design/objective_schema.py

"""
Battery System Objective Schema
================================

This file defines the COMPLETE and CANONICAL objective specification
used by:

- Web frontend (user prompts)
- Backend design & optimization engine
- Postprocessing, ranking, and explainability
- Future generative and reasoning layers

IMPORTANT:
- This schema NEVER directly modifies the dataset.
- It ONLY constrains, scores, and guides discovery.
- It aligns strictly with system-level targets predicted by HGT.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any


# ------------------------------------------------------------
# Core Objective Schema
# ------------------------------------------------------------
@dataclass
class BatteryObjective:
    """
    User-defined objectives for battery SYSTEM discovery.

    All fields are OPTIONAL.
    Unspecified fields are treated as unconstrained.

    This schema is intentionally explicit and verbose to:
    - Avoid ambiguity
    - Enable explainability
    - Support future generative models
    """

    # ========================================================
    # 1. Electrochemical Performance (HGT TARGETS)
    # ========================================================

    # Average operating voltage (V)
    min_average_voltage: Optional[float] = None
    max_average_voltage: Optional[float] = None

    # Capacity
    min_capacity_grav: Optional[float] = None   # mAh/g
    min_capacity_vol: Optional[float] = None    # mAh/cm^3

    # Energy density
    min_energy_grav: Optional[float] = None     # Wh/kg
    min_energy_vol: Optional[float] = None      # Wh/L

    # ========================================================
    # 2. Structural & Mechanical Constraints
    # ========================================================

    # Maximum allowable volume expansion
    max_delta_volume: Optional[float] = None    # fraction (e.g., 0.25)

    # ========================================================
    # 3. Thermodynamic / Stability Preferences
    # ========================================================

    # Stability thresholds (dimensionless or eV-based, dataset-defined)
    min_stability_charge: Optional[float] = None
    min_stability_discharge: Optional[float] = None

    # Which regime matters more for ranking
    stability_focus: Literal[
        "charge",
        "discharge",
        "balanced"
    ] = "balanced"

    # ========================================================
    # 4. Chemistry & System Identity
    # ========================================================

    # Working ion (must exist in dataset)
    working_ion: Optional[str] = None  # e.g. "Li", "Na", "Mg"

    # Battery system category
    battery_type: Optional[str] = None  # e.g. insertion, conversion

    # Host / framework constraints
    framework: Optional[str] = None
    framework_formula: Optional[str] = None

    # ========================================================
    # 5. Design & Discovery Behavior
    # ========================================================

    # Allow interpolated systems (between known systems)
    allow_interpolation: bool = True

    # Allow extrapolated systems (outside dataset envelope)
    allow_extrapolation: bool = False

    # Optimization preference flags
    safety_priority: bool = False
    energy_priority: bool = True
    capacity_priority: bool = False

    # ========================================================
    # 6. Output Control (Frontend / UX)
    # ========================================================

    # Number of candidate systems to return
    top_k: int = 5

    # Whether to return closest known dataset system if no improvement
    fallback_to_known: bool = True

    # Whether to include explanations
    return_explanations: bool = True

    # ========================================================
    # 7. Provenance & Metadata (DO NOT USE FOR LEARNING)
    # ========================================================

    # Free-form user intent (used ONLY for explanation / logging)
    user_intent: Optional[str] = None

    # Frontend-provided context (ignored by ML)
    frontend_metadata: Optional[Dict[str, Any]] = None
