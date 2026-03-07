# materials/role_inference.py
# ============================================================
# Material Role Inference
# ============================================================
# Purpose:
#   Infer likely electrochemical roles of a material
#   based on:
#     - material descriptors
#     - system-level feature priors
#
# This module:
#   - DOES NOT predict performance
#   - DOES NOT invent chemistry
#   - Produces interpretable, explainable roles
# ============================================================

from typing import Dict, List


def infer_material_roles_from_descriptors(
    descriptors: Dict,
    system_features: Dict,
) -> Dict:
    """
    Infer electrochemical roles for a material.

    Parameters
    ----------
    descriptors : Dict
        Output of material_descriptors.py

    system_features : Dict
        Output of material_to_system.py

    Returns
    -------
    Dict
        {
            "roles": List[str],
            "confidence": float,
            "rationale": Dict[str, str]
        }
    """

    roles: List[str] = []
    rationale: Dict[str, str] = {}

    # =========================================================
    # Cathode candidate
    # =========================================================
    if system_features.get("average_voltage", 0) >= 3.5:
        roles.append("Cathode candidate")
        rationale["Cathode candidate"] = (
            "High inferred operating voltage suggests "
            "favorable redox potential for cathodic operation."
        )

    # =========================================================
    # High-capacity insertion host
    # =========================================================
    if system_features.get("capacity_grav", 0) >= 200:
        roles.append("High-capacity insertion host")
        rationale["High-capacity insertion host"] = (
            "Gravimetric capacity prior exceeds typical intercalation thresholds."
        )

    # =========================================================
    # Intercalation framework
    # =========================================================
    if descriptors.get("is_layered", False):
        roles.append("Intercalation-friendly framework")
        rationale["Intercalation-friendly framework"] = (
            "Layered crystal structure favors reversible ion insertion."
        )

    # =========================================================
    # Structurally stable host
    # =========================================================
    if system_features.get("max_delta_volume", 1.0) <= 0.06:
        roles.append("Structurally stable host")
        rationale["Structurally stable host"] = (
            "Low predicted volume change suggests mechanical robustness."
        )

    # =========================================================
    # Energy-dense active material
    # =========================================================
    if (
        system_features.get("energy_grav", 0) >= 250
        and system_features.get("average_voltage", 0) >= 3.0
    ):
        roles.append("Energy-dense active material")
        rationale["Energy-dense active material"] = (
            "Combination of voltage and capacity implies high energy density."
        )

    # =========================================================
    # Fallback classification
    # =========================================================
    if not roles:
        roles.append("Electrochemically inert or support framework")
        rationale["Electrochemically inert or support framework"] = (
            "No strong electrochemical indicators detected; "
            "material may act as a structural or conductive component."
        )

    # =========================================================
    # Confidence estimation
    # =========================================================
    confidence = min(1.0, 0.25 * len(roles))

    return {
        "roles": roles,
        "confidence": confidence,
        "rationale": rationale,
    }
