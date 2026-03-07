# design/system_template.py
"""
Battery System Template
=======================

Canonical representation of a COMPLETE electrochemical battery system.

This object:
- Represents ONE system (real, inferred, interpolated, or generated)
- Is used AFTER prediction, NEVER during training
- Is frontend-safe, backend-safe, and explainability-safe
- Is mutable by design (generator & predictor safe)

This is the central object passed between:
- Predictors
- Optimizers
- Generators
- Explainers
- Web API
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
import uuid


# ------------------------------------------------------------
# Battery System (Canonical Object)
# ------------------------------------------------------------
@dataclass
class BatterySystem:
    """
    Canonical battery system representation.

    A BatterySystem may be:
    - Dataset-derived
    - Interpolated
    - Extrapolated
    - User-seeded (e.g. CIF upload)
    - Optimized variant

    This object is the SINGLE source of truth for system-level properties.
    """

    # ========================================================
    # 0. Identity & Provenance
    # ========================================================

    battery_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    provenance: Literal[
        "dataset",
        "interpolated",
        "extrapolated",
        "user_material",
        "optimized",
        "generated",
    ] = "dataset"

    parent_battery_id: Optional[str] = None

    # ========================================================
    # 1. System Identity (Dataset-Aligned)
    # ========================================================

    battery_type: Optional[str] = None
    working_ion: Optional[str] = None

    framework: Optional[str] = None
    framework_formula: Optional[str] = None
    host_structure: Optional[str] = None

    battery_formula: Optional[str] = None
    chemsys: Optional[str] = None
    elements: Optional[List[str]] = None
    nelements: Optional[int] = None

    # Explicit system components for frontend/rendering consistency
    cathode_material: Optional[str] = None
    anode_material: Optional[str] = None
    electrolyte: Optional[str] = None
    separator_material: Optional[str] = None
    additive_material: Optional[str] = None

    # ========================================================
    # 2. Electrochemical State (System-Level)
    # ========================================================

    average_voltage: Optional[float] = None

    capacity_grav: Optional[float] = None     # mAh/g
    capacity_vol: Optional[float] = None      # mAh/cm^3

    energy_grav: Optional[float] = None       # Wh/kg
    energy_vol: Optional[float] = None        # Wh/L

    # ========================================================
    # 3. Stability & Mechanics
    # ========================================================

    stability_charge: Optional[float] = None
    stability_discharge: Optional[float] = None

    max_delta_volume: Optional[float] = None

    # ========================================================
    # 4. Structural & Graph Metadata
    # ========================================================

    # Reference to graph node index (dataset-derived only)
    graph_node_index: Optional[int] = None

    # Edge influence summary (explainability only)
    relation_contributions: Optional[Dict[str, float]] = None

    # ========================================================
    # 5. Prediction Confidence & Uncertainty
    # ========================================================

    # Deterministic proxy (optional, model-dependent)
    confidence_score: Optional[float] = None

    # Placeholder for future uncertainty models
    uncertainty: Optional[Dict[str, Any]] = None

    # Explicit material vs cell-level properties for UI clarity
    material_level: Optional[Dict[str, Optional[float]]] = None
    cell_level: Optional[Dict[str, Optional[float]]] = None

    # ========================================================
    # 6. Explainability & Reasoning
    # ========================================================

    explanation: Optional[str] = None
    reasoning_trace: Optional[List[str]] = None

    # ========================================================
    # 7. Frontend Presentation Helpers
    # ========================================================

    is_feasible: bool = True
    constraint_violations: Optional[List[str]] = None

    # ========================================================
    # 8. Raw Attachments (Read-only payloads)
    # ========================================================

    # CIF, POSCAR, or uploaded material info (if any)
    attached_materials: Optional[List[Dict[str, Any]]] = None

    # Original dataset row snapshot (if applicable)
    dataset_snapshot: Optional[Dict[str, Any]] = None

    # ========================================================
    # Utility Methods
    # ========================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system to dictionary representation."""
        return {
            "battery_id": self.battery_id,
            "provenance": self.provenance,
            "parent_battery_id": self.parent_battery_id,
            "battery_type": self.battery_type,
            "working_ion": self.working_ion,
            "framework": self.framework,
            "framework_formula": self.framework_formula,
            "host_structure": self.host_structure,
            "battery_formula": self.battery_formula,
            "chemsys": self.chemsys,
            "elements": self.elements,
            "nelements": self.nelements,
            "cathode_material": self.cathode_material,
            "anode_material": self.anode_material,
            "electrolyte": self.electrolyte,
            "separator_material": self.separator_material,
            "additive_material": self.additive_material,
            "average_voltage": self.average_voltage,
            "capacity_grav": self.capacity_grav,
            "capacity_vol": self.capacity_vol,
            "energy_grav": self.energy_grav,
            "energy_vol": self.energy_vol,
            "stability_charge": self.stability_charge,
            "stability_discharge": self.stability_discharge,
            "max_delta_volume": self.max_delta_volume,
            "graph_node_index": self.graph_node_index,
            "relation_contributions": self.relation_contributions,
            "confidence_score": self.confidence_score,
            "uncertainty": self.uncertainty,
            "material_level": self.material_level,
            "cell_level": self.cell_level,
            "explanation": self.explanation,
            "reasoning_trace": self.reasoning_trace,
            "is_feasible": self.is_feasible,
            "constraint_violations": self.constraint_violations,
            "attached_materials": self.attached_materials,
            "dataset_snapshot": self.dataset_snapshot,
        }

