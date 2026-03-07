# materials/material_to_system.py
# ============================================================
# Material → System-Level Feature Mapping (FINAL — BULLETPROOF)
# ============================================================
# Purpose:
#   - Convert material-level descriptors into system-level priors
#   - Provide a stable adapter for backend services
#
# Design Principles:
#   - NO predictions
#   - NO hard constraints
#   - Priors only (HGT / generator may override)
#   - Backend-facing API MUST remain stable
# ============================================================

from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path
import math

# ------------------------------------------------------------
# Optional CIF-based pathway (preferred when available)
# ------------------------------------------------------------
try:
    # Canonical, stable API from cif_parser
    from materials.cif_parser import parse_cif_text
except Exception:  # pragma: no cover
    parse_cif_text = None


# ============================================================
# CORE: Material → System Feature Priors
# ============================================================

def material_to_system_features(descriptors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert material descriptors into system-level feature priors.

    Parameters
    ----------
    descriptors : Dict
        Output from material_descriptors.py

    Returns
    -------
    Dict
        System-level priors aligned with BatterySystem fields
    """

    system: Dict[str, Any] = {}

    # --------------------------------------------------------
    # 1. Gravimetric Capacity PRIOR (mAh/g)
    # --------------------------------------------------------
    nelements = int(descriptors.get("nelements", 1))

    system["capacity_grav"] = float(
        120.0 + 35.0 * math.log1p(max(nelements, 1))
    )

    # --------------------------------------------------------
    # 2. Average Voltage PRIOR (V)
    # --------------------------------------------------------
    Z_mean = float(descriptors.get("atomic_number_mean", 20.0))

    voltage = 2.3 + 0.012 * Z_mean
    system["average_voltage"] = float(
        max(2.0, min(voltage, 4.5))
    )

    # --------------------------------------------------------
    # 3. Volume Change PRIOR (fraction)
    # --------------------------------------------------------
    vpa = float(descriptors.get("volume_per_atom", 15.0))
    is_layered = bool(descriptors.get("is_layered", False))

    base_delta_v = 0.015 + 0.004 * vpa
    if is_layered:
        base_delta_v *= 0.6

    system["max_delta_volume"] = float(
        min(base_delta_v, 0.25)
    )

    # --------------------------------------------------------
    # 4. Cycling Stability PRIOR
    # --------------------------------------------------------
    density = float(descriptors.get("density", 3.5))

    stability = 0.0
    if is_layered:
        stability += 0.12

    if density > 4.5:
        stability += 0.08
    elif density < 2.5:
        stability -= 0.08

    stability = max(-0.3, min(stability, 0.3))
    system["stability_charge"] = float(stability)
    system["stability_discharge"] = float(stability)

    # --------------------------------------------------------
    # 5. Gravimetric Energy Density PRIOR (Wh/kg)
    # --------------------------------------------------------
    system["energy_grav"] = float(
        system["capacity_grav"]
        * system["average_voltage"]
        / 1000.0
    )

    # --------------------------------------------------------
    # 6. Volumetric quantities intentionally unset
    # --------------------------------------------------------
    system["capacity_vol"] = None
    system["energy_vol"] = None

    return system


# ============================================================
# ✅ REQUIRED BACKEND ADAPTER (CANONICAL)
# ============================================================

def material_to_battery_system(
    *,
    battery_id: str,
    cif_text: Optional[str] = None,
    material_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Backend-facing adapter.

    This function MUST exist and MUST remain stable.
    backend.services.cif_service depends on it.

    Resolution order:
        1. CIF text → BatterySystem (authoritative)
        2. Material descriptors → priors only
    """

    # --------------------------------------------------------
    # 1. CIF-based path (authoritative)
    # --------------------------------------------------------
    if cif_text is not None:
        if parse_cif_text is None:
            raise RuntimeError(
                "CIF parsing requested but cif_parser is unavailable."
            )

        system = parse_cif_text(cif_text)

        # Always return JSON-safe payload
        if hasattr(system, "to_dict"):
            system_payload = system.to_dict()
        else:
            system_payload = system.__dict__

        return {
            "battery_id": battery_id,
            "system": system_payload,
            "source": "cif_upload",
        }

    # --------------------------------------------------------
    # 2. Descriptor-based fallback (NO structure)
    # --------------------------------------------------------
    if material_data is not None:
        priors = material_to_system_features(material_data)

        return {
            "battery_id": battery_id,
            "priors": priors,
            "source": "material_descriptors_only",
        }

    # --------------------------------------------------------
    # 3. Safety net (never silent)
    # --------------------------------------------------------
    raise ValueError(
        "material_to_battery_system requires either "
        "`cif_text` or `material_data`."
    )
