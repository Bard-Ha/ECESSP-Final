# ============================================================
# materials/material_descriptors.py
# ============================================================
# Purpose:
#   Robust, scalable, physics-informed material descriptor engine
#   for battery system discovery and generation.
#
# Design Principles:
#   - Deterministic (no randomness)
#   - Numerically stable (no NaNs / infs)
#   - Flat feature dictionary (ML-ready)
#   - Physics-meaningful
#   - CIF → Graph → HGT → Generator compatible
# ============================================================

from __future__ import annotations
from typing import Dict, Any
import math
import numpy as np

from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ============================================================
# -------------------- CONSTANTS ------------------------------
# ============================================================

ION_LIBRARY = {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al"}

# Safe numerical bounds
EPS = 1e-8
MAX_CLIP = 1e4


# ============================================================
# -------------------- UTILITIES ------------------------------
# ============================================================

def safe_mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else 0.0

def safe_std(values):
    values = [v for v in values if v is not None]
    return float(np.std(values)) if values else 0.0

def safe_log(x):
    return math.log(max(x, EPS))

def clip(x, lo=-MAX_CLIP, hi=MAX_CLIP):
    return float(np.clip(x, lo, hi))


# ============================================================
# ---------------- COMPOSITION DESCRIPTORS -------------------
# ============================================================

def composition_descriptors(structure: Structure) -> Dict[str, float]:
    comp = structure.composition
    elements = comp.elements
    fractions = comp.fractional_composition

    atomic_numbers = []
    atomic_masses = []
    electronegativities = []

    for el in elements:
        atomic_numbers.append(el.Z)
        atomic_masses.append(el.atomic_mass)
        electronegativities.append(el.X if el.X is not None else 0.0)

    return {
        # Size & diversity
        "nelements": float(len(elements)),

        # Atomic statistics
        "avg_atomic_number": safe_mean(atomic_numbers),
        "std_atomic_number": safe_std(atomic_numbers),
        "avg_atomic_mass": safe_mean(atomic_masses),
        "std_atomic_mass": safe_std(atomic_masses),

        # Chemical bonding
        "avg_electronegativity": safe_mean(electronegativities),
        "std_electronegativity": safe_std(electronegativities),

        # Entropy proxy
        "composition_entropy": -sum(
            f * safe_log(f) for f in fractions.values()
        ),
    }


# ============================================================
# ---------------- STRUCTURAL DESCRIPTORS --------------------
# ============================================================

def structural_descriptors(structure: Structure) -> Dict[str, float]:
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    volume = structure.volume
    nsites = len(structure)

    return {
        # Global geometry
        "density": clip(structure.density),
        "volume": clip(volume),
        "volume_per_atom": clip(volume / max(nsites, 1)),

        # Symmetry
        "spacegroup_number": float(sga.get_space_group_number()),
        "crystal_system_id": float(
            [
                "triclinic", "monoclinic", "orthorhombic",
                "tetragonal", "trigonal", "hexagonal", "cubic"
            ].index(sga.get_crystal_system())
        ),
    }


# ============================================================
# ----------- LOCAL COORDINATION DESCRIPTORS -----------------
# ============================================================

def coordination_descriptors(structure: Structure) -> Dict[str, float]:
    cnn = CrystalNN()
    coord_nums = []

    for i in range(len(structure)):
        try:
            coord_nums.append(len(cnn.get_nn_info(structure, i)))
        except Exception:
            continue

    return {
        "avg_coordination": safe_mean(coord_nums),
        "std_coordination": safe_std(coord_nums),
        "max_coordination": max(coord_nums) if coord_nums else 0.0,
        "min_coordination": min(coord_nums) if coord_nums else 0.0,
    }


# ============================================================
# ----------- ION TRANSPORT / INSERTION PROXIES --------------
# ============================================================

def ion_transport_descriptors(structure: Structure) -> Dict[str, float]:
    elements = {el.symbol for el in structure.composition.elements}
    nsites = len(structure)

    contains_working_ion = any(el in ION_LIBRARY for el in elements)

    return {
        "contains_working_ion": float(contains_working_ion),

        # Framework openness proxy
        "framework_openness": clip(structure.volume / max(nsites, 1)),

        # Ion fraction proxy
        "working_ion_fraction": float(
            sum(
                structure.composition.get_atomic_fraction(el)
                for el in ION_LIBRARY
                if el in structure.composition
            )
        ),
    }


# ============================================================
# ----------- MECHANICAL / STABILITY PROXIES -----------------
# ============================================================

def stability_descriptors(structure: Structure) -> Dict[str, float]:
    density = structure.density

    # Heuristic mechanical rigidity proxy
    rigidity = (
        1.0 / density if density > 0 else 0.0
    )

    return {
        "rigidity_proxy": clip(rigidity),
        "volume_stability_proxy": clip(density / 10.0),
    }


# ============================================================
# ----------------- MASTER INTERFACE -------------------------
# ============================================================

def extract_material_descriptors(
    structure: Structure,
    *,
    strict: bool = True,
) -> Dict[str, float]:
    """
    Master descriptor extractor.

    Returns:
        Flat dictionary of ML-ready features.
    """

    descriptors: Dict[str, float] = {}

    descriptors.update(composition_descriptors(structure))
    descriptors.update(structural_descriptors(structure))
    descriptors.update(coordination_descriptors(structure))
    descriptors.update(ion_transport_descriptors(structure))
    descriptors.update(stability_descriptors(structure))

    # ---------------- SAFETY PASS ----------------
    for k, v in descriptors.items():
        if not np.isfinite(v):
            if strict:
                raise ValueError(f"Non-finite descriptor: {k} = {v}")
            descriptors[k] = 0.0

        descriptors[k] = float(clip(v))

    return descriptors


# ============================================================
# ---------------- PUBLIC API (BACKEND) ----------------------
# ============================================================

def describe_structure(structure: Structure) -> Dict[str, Any]:
    """
    Website / API-safe wrapper.
    """
    return {
        "descriptors": extract_material_descriptors(structure),
        "num_features": len(extract_material_descriptors(structure)),
        "status": "ok",
    }
