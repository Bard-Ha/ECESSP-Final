# materials/cif_parser.py
# ============================================================
# CIF → Battery System Inference Module (FINAL — PRODUCTION)
# ============================================================
# Purpose:
#   - Parse CIF uploads (TEXT or FILE)
#   - Infer material & electrochemical priors
#   - Convert into BatterySystem template
#   - Feed generator / HGT / reasoner
#
# DESIGN RULES:
#   - Deterministic
#   - No ML inference
#   - No mutation of global state
# ============================================================

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import uuid
import tempfile

import numpy as np

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from design.system_template import BatterySystem

# ------------------------------------------------------------
# System safety limits (robust import)
# ------------------------------------------------------------

try:
    from design.system_constraints import SYSTEM_LIMITS
except Exception:
    SYSTEM_LIMITS = {
        "max_voltage": 5.0,
        "max_delta_volume": 0.15,
    }


# ============================================================
# Domain Knowledge
# ============================================================

ION_LIBRARY = {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al"}

COMMON_CATHODE_ELEMENTS = {
    "Co", "Ni", "Mn", "Fe", "V", "Cr", "Ti", "Mo"
}

COMMON_ANODE_ELEMENTS = {
    "C", "Si", "Sn", "Sb", "Ge", "P"
}


# ============================================================
# CIF Loading (LOW LEVEL)
# ============================================================

def _load_structure_from_file(path: Path) -> Structure:
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")
    return Structure.from_file(path)


def _load_structure_from_text(cif_text: str) -> Structure:
    """
    Safely load CIF text by writing to a temporary file
    (pymatgen requirement).
    """
    if not cif_text or not cif_text.strip():
        raise ValueError("Empty CIF text")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".cif",
        delete=True,
    ) as tmp:
        tmp.write(cif_text)
        tmp.flush()
        return Structure.from_file(tmp.name)


# ============================================================
# Structural Feature Extraction
# ============================================================

def extract_structural_features(structure: Structure) -> Dict:
    analyzer = SpacegroupAnalyzer(structure)
    comp = structure.composition

    elements = sorted(el.symbol for el in comp.elements)

    return {
        "formula": comp.reduced_formula,
        "elements": elements,
        "nelements": len(elements),
        "volume": float(structure.volume),
        "density": float(structure.density),
        "spacegroup": analyzer.get_space_group_symbol(),
        "crystal_system": analyzer.get_crystal_system(),
        "avg_coordination": float(
            np.mean(
                [len(structure.get_neighbors(site, 3.0)) for site in structure]
            )
        ),
    }


# ============================================================
# Electrochemical Heuristics (PRIORS ONLY)
# ============================================================

def infer_working_ion(elements: List[str]) -> Optional[str]:
    for el in elements:
        if el in ION_LIBRARY:
            return el
    return None


def infer_material_role(elements: List[str]) -> str:
    if any(el in COMMON_CATHODE_ELEMENTS for el in elements):
        return "cathode_candidate"
    if any(el in COMMON_ANODE_ELEMENTS for el in elements):
        return "anode_candidate"
    return "framework_or_electrolyte_candidate"


def infer_voltage_proxy(elements: List[str]) -> float:
    voltage = 2.5
    if {"Co", "Ni"} & set(elements):
        voltage += 1.0
    if {"Mn", "Fe"} & set(elements):
        voltage += 0.5
    return min(voltage, 4.5)


def infer_volume_stability(structure: Structure) -> float:
    density = structure.density
    if density > 5.0:
        return 0.03
    if density > 3.0:
        return 0.06
    return 0.12


# ============================================================
# System Projection
# ============================================================

def structure_to_battery_system(
    structure: Structure,
    battery_id: str,
) -> BatterySystem:
    features = extract_structural_features(structure)
    elements = features["elements"]

    system = BatterySystem(
        battery_id=battery_id,
        battery_type="generated_from_cif",
        battery_formula=features["formula"],
        framework=features["crystal_system"],
        framework_formula=features["formula"],
        elements=elements,
        nelements=features["nelements"],
        chemsys="-".join(elements),
        working_ion=infer_working_ion(elements),
        host_structure=infer_material_role(elements),

        # -------- Generator priors --------
        average_voltage=infer_voltage_proxy(elements),
        max_delta_volume=infer_volume_stability(structure),

        # -------- Unknowns --------
        capacity_grav=None,
        capacity_vol=None,
        energy_grav=None,
        energy_vol=None,
        stability_charge=None,
        stability_discharge=None,
    )

    # Always present (prevents downstream crashes)
    system.flags = []
    system.provenance = {}

    return system


# ============================================================
# Validation
# ============================================================

def validate_system(system: BatterySystem) -> Dict:
    violations: List[str] = []

    if (
        system.max_delta_volume is not None
        and system.max_delta_volume > SYSTEM_LIMITS["max_delta_volume"]
    ):
        violations.append("Excessive volume expansion risk")

    if (
        system.average_voltage is not None
        and system.average_voltage > SYSTEM_LIMITS["max_voltage"]
    ):
        violations.append("Voltage exceeds safety window")

    return {
        "battery_id": system.battery_id,
        "valid": len(violations) == 0,
        "violations": violations,
    }


# ============================================================
# 🔥 PUBLIC CANONICAL API (INTERNAL USE)
# ============================================================

def parse_cif_text(cif_text: str) -> BatterySystem:
    """
    Internal canonical entrypoint.
    Returns a BatterySystem object.
    """

    battery_id = f"cif-{uuid.uuid4().hex[:10]}"

    structure = _load_structure_from_text(cif_text)
    system = structure_to_battery_system(structure, battery_id)

    validation = validate_system(system)
    if not validation["valid"]:
        system.flags.extend(validation["violations"])

    system.provenance.update({
        "source": "cif_upload",
        "pipeline": "cif → structure → heuristics → system",
    })

    return system


# ============================================================
# ✅ PUBLIC API (FASTAPI / JSON SAFE)
# ============================================================

def parse_cif_text_safe(cif_text: str) -> Dict:
    """
    JSON-serializable API wrapper.
    This is what backend services SHOULD call.
    """

    system = parse_cif_text(cif_text)

    if hasattr(system, "to_dict"):
        payload = system.to_dict()
    else:
        payload = system.__dict__

    return {
        "status": "ok",
        "system": payload,
    }
