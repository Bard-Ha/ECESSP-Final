# ============================================================
# Response Mapper (ECESSP – API Boundary Layer)
# ============================================================
# Purpose:
#   Convert internal BatterySystem objects into
#   strict API schema responses.
#
# Rules:
#   - No ML logic
#   - No generation
#   - No scoring
#   - No constraint evaluation
#   - Pure data transformation
# ============================================================

from typing import Dict, Any, Optional

from design.system_template import BatterySystem
from backend.api.schemas import (
    BatterySystemResponse,
    SystemProperties,
    ConstraintReport,
    ConstraintSummary,
)


# ============================================================
# Constraint Mapping
# ============================================================

def _map_constraints(constraints: Dict[str, Dict]) -> ConstraintSummary:
    """
    Convert evaluate_system() dict into ConstraintSummary schema.
    """

    return ConstraintSummary(
        overall_valid=constraints.get("overall_valid", False),
        physical=ConstraintReport(**constraints.get("physical", {})),
        chemical=ConstraintReport(**constraints.get("chemical", {})),
        performance=ConstraintReport(**constraints.get("performance", {})),
    )


# ============================================================
# System Mapping
# ============================================================

def map_system_to_response(
    system: BatterySystem,
    constraints: Optional[Dict[str, Dict]] = None,
    score: Optional[float] = None,
    speculative: bool = False,
) -> BatterySystemResponse:
    """
    Convert BatterySystem → BatterySystemResponse.
    """

    properties = SystemProperties(
        average_voltage=system.average_voltage,
        capacity_grav=system.capacity_grav,
        capacity_vol=system.capacity_vol,
        energy_grav=system.energy_grav,
        energy_vol=system.energy_vol,
        max_delta_volume=system.max_delta_volume,
        stability_charge=system.stability_charge,
        stability_discharge=system.stability_discharge,
    )

    constraint_summary = None
    if constraints is not None:
        constraint_summary = _map_constraints(constraints)

    return BatterySystemResponse(
        battery_id=system.battery_id,
        working_ion=system.working_ion,
        elements=system.elements or [],
        properties=properties,
        constraints=constraint_summary,
        score=score,
        speculative=speculative,
    )