# design/system_constraints.py
# ============================================================
# Battery System Constraints (FINAL  CANONICAL & RIGOROUS)
# ============================================================
# Purpose:
#   - Define HARD limits (never violated)
#   - Define SOFT limits (penalized, not rejected)
#   - Explicitly mark SPECULATIVE regimes
#   - Preserve discovery beyond known chemistry
#
# This module is used by:
#   - Generators
#   - Scorers
#   - Reasoners
#   - CIF parsing heuristics
#   - Frontend safety & labeling
#
#  NO ML CODE belongs here
# ============================================================

from __future__ import annotations

from typing import Dict, List
import math
import json
from pathlib import Path

from .system_template import BatterySystem
from .physics_chemistry import (
    TRANSITION_METALS,
    parse_formula,
    solve_oxidation_states,
    theoretical_capacity_mAh_per_g,
)


# ============================================================
#  CANONICAL NUMERIC LIMITS (BACKWARD COMPATIBILITY)
# ============================================================
# NOTE:
#   These are SIMPLE, HARD-REFERENCED limits used by:
#     - cif_parser
#     - heuristics
#     - early validation
#
#   They are NOT the full constraint logic.
# ============================================================

SYSTEM_LIMITS: Dict[str, float] = {
    "max_voltage": 4.4,          # default ceiling for conventional electrolytes
    "recommended_voltage": 4.2,  # beyond this = speculative
    "max_delta_volume": 0.25,    # default volume expansion limit (25%)
    "recommended_delta_volume": 0.20,
    "min_energy_grav": 50.0,     # Wh/kg (soft)
    "max_energy_grav": 450.0,    # Wh/kg (hard cap for cell-level realism)
    "max_energy_vol": 1200.0,    # Wh/L (hard cap for cell-level realism)
    "max_capacity_grav": 350.0,  # mAh/g (hard cap for cell-level realism)
}

PENALIZED_ELEMENTS = {"co", "sc", "ga", "in", "te"}
_PHYSICS_SPEC_PATH = Path(__file__).resolve().parent / "physics_first_spec.json"


def _load_physics_spec() -> dict:
    try:
        with _PHYSICS_SPEC_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Keep runtime safe even if spec file is missing/corrupt.
        return {}


PHYSICS_FIRST_SPEC = _load_physics_spec()


def _normalize_formula_for_physics(system: BatterySystem) -> str:
    """
    Choose a chemistry formula suitable for oxidation-state solving.

    Preferred source is framework/cathode formula because full-cell strings like
    "Na-NaFePO4|Hard carbon" are not parseable by the formula parser.
    """
    framework = (system.framework_formula or "").strip()
    if framework:
        return framework

    raw = (system.battery_formula or "").strip()
    if not raw:
        return ""

    # Full-cell schema frequently uses "<ion>-<cathode>|<anode>".
    # Use the cathode side for oxidation-state and theoretical-capacity checks.
    left = raw.split("|", 1)[0].strip()
    if "-" in left:
        parts = [p.strip() for p in left.split("-", 1)]
        if len(parts) == 2 and parts[1]:
            return parts[1]
    return left


def _cell_or_system_value(system: BatterySystem, field: str) -> float | None:
    if system.cell_level and field in system.cell_level and system.cell_level[field] is not None:
        return float(system.cell_level[field])
    if system.uncertainty and isinstance(system.uncertainty, dict):
        cell_level = system.uncertainty.get("cell_level")
        if isinstance(cell_level, dict) and cell_level.get(field) is not None:
            return float(cell_level[field])
    raw = getattr(system, field, None)
    return float(raw) if raw is not None else None


def _physics_limits() -> dict[str, float]:
    hard = PHYSICS_FIRST_SPEC.get("hard_physical_limits", {}) if isinstance(PHYSICS_FIRST_SPEC, dict) else {}
    return {
        "max_energy_grav": float(hard.get("max_gravimetric_energy_Wh_per_kg", SYSTEM_LIMITS["max_energy_grav"])),
        "max_energy_vol": float(hard.get("max_volumetric_energy_Wh_per_L", SYSTEM_LIMITS["max_energy_vol"])),
        "max_capacity_grav": float(hard.get("max_gravimetric_capacity_mAh_per_g", SYSTEM_LIMITS["max_capacity_grav"])),
        "max_voltage_li": float(hard.get("max_voltage_liquid_electrolyte_V", SYSTEM_LIMITS["max_voltage"])),
        "max_voltage_na": float(hard.get("max_voltage_sodium_liquid_V", SYSTEM_LIMITS["max_voltage"])),
        "min_voltage": float(hard.get("minimum_stable_voltage_V", 1.0)),
    }


def _expected_output_fields_report(system: BatterySystem) -> dict:
    output_req = PHYSICS_FIRST_SPEC.get("output_requirements", {}) if isinstance(PHYSICS_FIRST_SPEC, dict) else {}
    expected = output_req.get("each_candidate_must_report", [])
    if not isinstance(expected, list):
        expected = []

    # Map canonical output fields to current BatterySystem keys where possible.
    mapping = {
        "formula": "battery_formula",
        "voltage": "average_voltage",
        "cell_capacity": "capacity_grav",
        "cell_energy": "energy_grav",
        "volume_change": "max_delta_volume",
        "uncertainty": "uncertainty",
        "constraint_validation_report": "__generated__",
    }

    payload = system.to_dict() if hasattr(system, "to_dict") else {}
    missing: list[str] = []
    available: list[str] = []
    for field in expected:
        key = mapping.get(field, field)
        if key == "__generated__":
            available.append(field)
            continue
        value = payload.get(key)
        if value is None:
            missing.append(field)
        else:
            available.append(field)

    return {
        "expected_fields": expected,
        "available_fields": available,
        "missing_fields": missing,
    }


def evaluate_physics_first(system: BatterySystem) -> Dict[str, object]:
    """
    Explicit physics-first hard-gate checks derived from physics_first_spec.json.
    Only checks computable from current BatterySystem fields are enforced here.
    """
    limits = _physics_limits()
    hard_violations: list[str] = []
    warnings: list[str] = []
    derived: dict[str, object] = {}
    physics_penalty = 0.0

    v = system.average_voltage
    if v is not None:
        if v < limits["min_voltage"]:
            hard_violations.append(
                f"average_voltage={v:.3f}V below minimum_stable_voltage={limits['min_voltage']:.3f}V"
            )
        max_v = limits["max_voltage_na"] if (system.working_ion or "").strip().lower() == "na" else limits["max_voltage_li"]
        if v > max_v:
            hard_violations.append(
                f"average_voltage={v:.3f}V exceeds electrolyte stability window max={max_v:.3f}V"
            )

    cap_grav = _cell_or_system_value(system, "capacity_grav")
    if cap_grav is not None and cap_grav > limits["max_capacity_grav"]:
        hard_violations.append(
            f"capacity_grav={cap_grav:.3f} exceeds max_gravimetric_capacity={limits['max_capacity_grav']:.3f}"
        )

    energy_grav = _cell_or_system_value(system, "energy_grav")
    if energy_grav is not None and energy_grav > limits["max_energy_grav"]:
        hard_violations.append(
            f"energy_grav={energy_grav:.3f} exceeds max_gravimetric_energy={limits['max_energy_grav']:.3f}"
        )

    energy_vol = _cell_or_system_value(system, "energy_vol")
    if energy_vol is not None and energy_vol > limits["max_energy_vol"]:
        hard_violations.append(
            f"energy_vol={energy_vol:.3f} exceeds max_volumetric_energy={limits['max_energy_vol']:.3f}"
        )

    # Enforce energy relation when all required values are present:
    # E_cell (Wh/kg) ~= V_avg * C_cell (mAh/g)
    if v is not None and cap_grav is not None and energy_grav is not None:
        expected = float(v) * float(cap_grav)
        if not math.isclose(float(energy_grav), expected, rel_tol=0.30):
            hard_violations.append(
                "cell_energy inconsistent with energy relation (E ~= V * C)"
            )

    # Formula-based oxidation-state solver and theoretical-capacity bound.
    formula = _normalize_formula_for_physics(system)
    if formula:
        ox = solve_oxidation_states(formula)
        derived["formula"] = formula
        derived["composition"] = ox.composition
        derived["oxidation_states"] = ox.oxidation_states
        derived["redox_active_elements"] = ox.redox_active_elements
        derived["n_electrons_max"] = ox.n_electrons_max
        derived["molar_mass_g_per_mol"] = ox.molar_mass
        derived["oxidation_solver_errors"] = ox.errors

        if not ox.valid:
            hard_violations.append("no valid oxidation-state assignment (charge neutrality failed)")
        else:
            if ox.molar_mass is not None and ox.molar_mass > 0 and ox.n_electrons_max > 0:
                c_theoretical = theoretical_capacity_mAh_per_g(ox.n_electrons_max, ox.molar_mass)
                derived["C_theoretical_mAh_per_g"] = c_theoretical
                if cap_grav is not None and cap_grav > c_theoretical:
                    # No-override policy: clip predicted capacity to theoretical max.
                    system.capacity_grav = float(c_theoretical)
                    if isinstance(system.cell_level, dict):
                        system.cell_level["capacity_grav"] = float(c_theoretical)
                    if isinstance(system.material_level, dict):
                        system.material_level["capacity_grav"] = min(
                            float(system.material_level.get("capacity_grav") or c_theoretical),
                            float(c_theoretical),
                        )
                    derived["capacity_grav_original"] = float(cap_grav)
                    derived["capacity_grav_clipped"] = float(c_theoretical)
                    physics_penalty += 0.4
            else:
                warnings.append("theoretical capacity not computed (missing molar mass or redox electrons)")
    else:
        warnings.append("formula missing: oxidation-state and theoretical-capacity checks skipped")

    if getattr(system, "uncertainty", None) is None:
        warnings.append("uncertainty_and_ood hard rejection not active for this candidate")

    return {
        "enforced": True,
        "hard_valid": len(hard_violations) == 0,
        "hard_violations": hard_violations,
        "warnings": warnings,
        "limits_used": limits,
        "output_requirements": _expected_output_fields_report(system),
        "derived": derived,
        "physics_penalty": round(min(1.0, physics_penalty), 4),
    }


# ============================================================
# Constraint Result Object
# ============================================================
class ConstraintResult:
    """
    Standardized constraint evaluation output.
    """

    def __init__(
        self,
        valid: bool,
        violations: List[str],
        score_penalty: float = 0.0,
        speculative: bool = False,
    ):
        self.valid = bool(valid)
        self.violations = list(violations)
        self.score_penalty = float(score_penalty)
        self.speculative = bool(speculative)

    def to_dict(self) -> Dict[str, object]:
        return {
            "valid": self.valid,
            "violations": self.violations,
            "score_penalty": round(self.score_penalty, 4),
            "speculative": self.speculative,
        }


# ============================================================
# HARD PHYSICAL CONSTRAINTS
# ============================================================
def check_physical_constraints(system: BatterySystem) -> ConstraintResult:
    """
    Absolute physical impossibilities.
    Violating ANY of these invalidates the system.
    """

    violations: List[str] = []

    # --------------------------------------------------------
    # Voltage bounds
    # --------------------------------------------------------
    if system.average_voltage is not None:
        if system.average_voltage <= 0:
            violations.append("average_voltage must be positive")

        if system.average_voltage > SYSTEM_LIMITS["max_voltage"]:
            violations.append(
                f"average_voltage={system.average_voltage:.2f} "
                "exceeds hard electrochemical limit"
            )

    # --------------------------------------------------------
    # Capacities must be non-negative
    # --------------------------------------------------------
    for name in ("capacity_grav", "capacity_vol"):
        value = _cell_or_system_value(system, name)
        if value is not None and value < 0:
            violations.append(f"{name} < 0")

    cap_grav = _cell_or_system_value(system, "capacity_grav")
    if cap_grav is not None and cap_grav > SYSTEM_LIMITS["max_capacity_grav"]:
        violations.append("capacity_grav exceeds hard realism cap")

    # --------------------------------------------------------
    # Energy consistency: E  V  Q
    # (very loose tolerance)
    # --------------------------------------------------------
    if (
        system.average_voltage is not None
        and _cell_or_system_value(system, "capacity_grav") is not None
        and _cell_or_system_value(system, "energy_grav") is not None
    ):
        expected = system.average_voltage * _cell_or_system_value(system, "capacity_grav")
        energy_grav = _cell_or_system_value(system, "energy_grav")
        if energy_grav is not None and not math.isclose(energy_grav, expected, rel_tol=0.5):
            violations.append(
                "energy_grav inconsistent with voltage * capacity"
            )

    # --------------------------------------------------------
    # Hard caps for cell-level realism (protect against
    # material-level leakage in discovery outputs)
    # --------------------------------------------------------
    energy_grav = _cell_or_system_value(system, "energy_grav")
    if energy_grav is not None:
        if energy_grav > SYSTEM_LIMITS["max_energy_grav"]:
            violations.append("energy_grav exceeds hard realism cap")
    energy_vol = _cell_or_system_value(system, "energy_vol")
    if energy_vol is not None:
        if energy_vol > SYSTEM_LIMITS["max_energy_vol"]:
            violations.append("energy_vol exceeds hard realism cap")

    # --------------------------------------------------------
    # Mechanical failure
    # --------------------------------------------------------
    if system.max_delta_volume is not None:
        if system.max_delta_volume > SYSTEM_LIMITS["max_delta_volume"]:
            violations.append("excessive volume expansion (>60%)")

    return ConstraintResult(
        valid=len(violations) == 0,
        violations=violations,
    )


# ============================================================
# CHEMICAL CONSISTENCY CONSTRAINTS
# ============================================================
def check_chemical_constraints(system: BatterySystem) -> ConstraintResult:
    """
    Chemistry-level consistency checks.
    """

    violations: List[str] = []

    if system.working_ion and system.elements:
        if system.working_ion not in system.elements:
            violations.append("working ion not present in element set")

    if system.nelements is not None and system.elements is not None:
        if system.nelements != len(system.elements):
            violations.append("nelements does not match element list")

    for name, value in (
        ("stability_charge", system.stability_charge),
        ("stability_discharge", system.stability_discharge),
    ):
        if value is not None and value < -0.5:
            violations.append(f"{name} indicates unstable phase")

    # --------------------------------------------------------
    # User-specified component chemistry checks
    # (strict for non-generated systems)
    # --------------------------------------------------------
    if str(system.provenance or "").strip().lower() != "generated":
        anode_formula = str(system.anode_material or "").strip()
        cathode_formula = str(system.cathode_material or system.framework_formula or "").strip()

        def oxidation_summary(formula: str) -> dict[str, object]:
            if not formula:
                return {"present": False}
            ox = solve_oxidation_states(formula)
            if not ox.valid:
                return {
                    "present": True,
                    "valid": False,
                    "reason": ";".join(ox.errors or ["no_valid_oxidation_solution"]),
                }
            tm_states = {
                el: int(state)
                for el, state in (ox.oxidation_states or {}).items()
                if el in TRANSITION_METALS
            }
            comp = ox.composition or {}
            has_polyanion = bool(comp.get("P", 0.0) or comp.get("S", 0.0) or comp.get("Si", 0.0))
            alkali_count = float(comp.get("Li", 0.0) + comp.get("Na", 0.0) + comp.get("K", 0.0))
            return {
                "present": True,
                "valid": True,
                "tm_states": tm_states,
                "max_tm_ox": max(tm_states.values()) if tm_states else None,
                "tm_count": len(tm_states),
                "has_polyanion": has_polyanion,
                "alkali_count": alkali_count,
            }

        anode = oxidation_summary(anode_formula)
        cathode = oxidation_summary(cathode_formula)

        if anode.get("present") and not anode.get("valid", False):
            violations.append(f"anode oxidation-state unsolved: {anode.get('reason')}")
        if cathode.get("present") and not cathode.get("valid", False):
            violations.append(f"cathode oxidation-state unsolved: {cathode.get('reason')}")

        if anode.get("valid", False):
            anode_max = anode.get("max_tm_ox")
            if isinstance(anode_max, int):
                if anode_max >= 4:
                    violations.append(
                        "anode contains high-valence transition metal (>=+4), cathode-like role mismatch"
                    )
                elif anode_max >= 3 and bool(anode.get("has_polyanion")):
                    violations.append(
                        "anode shows high-valence polyanion transition-metal chemistry, likely cathode-like"
                    )

        if cathode.get("valid", False):
            cathode_max = cathode.get("max_tm_ox")
            if isinstance(cathode_max, int) and cathode_max >= 6:
                violations.append(
                    "cathode requires extreme transition-metal oxidation state (>=+6), insertion stability doubtful"
                )

            # Heuristic rejector for highly mixed simple oxides with unknown insertion topology.
            tm_count = cathode.get("tm_count")
            has_polyanion = bool(cathode.get("has_polyanion"))
            alkali_count = cathode.get("alkali_count")
            if (
                isinstance(tm_count, int)
                and tm_count >= 3
                and not has_polyanion
                and isinstance(alkali_count, float)
                and alkali_count <= 1e-9
            ):
                violations.append(
                    "cathode framework appears as high-complexity mixed oxide without known insertion topology"
                )
                violations.append(
                    "structure classifier: unknown insertion framework"
                )

        # Voltage-role consistency heuristic for user-defined full-cell pairs.
        if anode.get("valid", False) and cathode.get("valid", False):
            anode_max = anode.get("max_tm_ox")
            cathode_max = cathode.get("max_tm_ox")
            if isinstance(anode_max, int) and isinstance(cathode_max, int):
                if anode_max >= 4 and cathode_max >= 4:
                    violations.append(
                        "voltage alignment poor: both electrodes likely high-potential vs Li/Li+"
                    )

        # Coarse insertion-likelihood screen for user-entered formulas.
        insertion_score = 1.0
        if anode.get("valid", False):
            anode_max = anode.get("max_tm_ox")
            if isinstance(anode_max, int) and anode_max >= 4:
                insertion_score -= 0.45
            if isinstance(anode_max, int) and anode_max >= 3 and bool(anode.get("has_polyanion")):
                insertion_score -= 0.20
        if cathode.get("valid", False):
            cathode_max = cathode.get("max_tm_ox")
            if isinstance(cathode_max, int) and cathode_max >= 6:
                insertion_score -= 0.25
            tm_count = cathode.get("tm_count")
            has_polyanion = bool(cathode.get("has_polyanion"))
            alkali_count = cathode.get("alkali_count")
            if (
                isinstance(tm_count, int)
                and tm_count >= 3
                and not has_polyanion
                and isinstance(alkali_count, float)
                and alkali_count <= 1e-9
            ):
                insertion_score -= 0.30
        if insertion_score < 0.45:
            violations.append(
                "insertion probability below threshold"
            )

    return ConstraintResult(
        valid=len(violations) == 0,
        violations=violations,
    )


# ============================================================
# SOFT PERFORMANCE CONSTRAINTS
# ============================================================
def check_performance_constraints(system: BatterySystem) -> ConstraintResult:
    """
    Soft constraints:
    - NEVER invalidate a system
    - Affect ranking, confidence & labeling
    """

    violations: List[str] = []
    penalty: float = 0.0
    speculative: bool = False

    energy_grav = _cell_or_system_value(system, "energy_grav")
    if energy_grav is not None:
        if energy_grav < SYSTEM_LIMITS["min_energy_grav"]:
            penalty += 0.25
            violations.append("low gravimetric energy")

    if system.average_voltage is not None:
        if SYSTEM_LIMITS["recommended_voltage"] < system.average_voltage <= SYSTEM_LIMITS["max_voltage"]:
            speculative = True
            penalty += 0.2
            violations.append("voltage in speculative regime")

    if system.max_delta_volume is not None:
        if system.max_delta_volume > SYSTEM_LIMITS["recommended_delta_volume"]:
            penalty += min(0.3, system.max_delta_volume)
            violations.append("large volume expansion")

    # Silicon expansion heuristic
    anode = (system.anode_material or "").lower()
    cap_grav = _cell_or_system_value(system, "capacity_grav")
    if "si" in anode and cap_grav is not None:
        if cap_grav > 250:
            penalty += 0.25
            violations.append("si anode expansion risk at high capacity")

    # Metal anode dendrite risk penalty
    if any(token in anode for token in ("li metal", "na metal", "mg metal", "zn metal", "al metal", "k metal")):
        penalty += 0.2
        violations.append("metal anode dendrite risk")

    # Abundance / supply risk penalty
    elements = [e.lower() for e in (system.elements or []) if isinstance(e, str)]
    if any(e in PENALIZED_ELEMENTS for e in elements):
        penalty += 0.2
        violations.append("supply risk elements present")

    # --------------------------------------------------------
    # Chemistry-aware realism gates (soft, not hard reject)
    # --------------------------------------------------------
    ion = (system.working_ion or "").strip().lower()
    formula_text = " ".join(
        [
            str(system.battery_formula or ""),
            str(system.framework_formula or ""),
            str(system.chemsys or ""),
        ]
    ).lower().replace(" ", "")

    # Na-ion practical full-cell realism guidance.
    # These are soft plausibility warnings for ranking/explanation.
    if ion == "na":
        if energy_grav is not None:
            if energy_grav > 180:
                speculative = True
                penalty += 0.22
                violations.append("na-ion gravimetric energy in optimistic regime (>180 Wh/kg)")
            if energy_grav > 220:
                speculative = True
                penalty += 0.20
                violations.append("na-ion gravimetric energy statistically implausible (>220 Wh/kg)")

    # NaFePO4 specific realism guidance.
    # Typical average voltage near ~3.2-3.4V and practical full-cell energy below ~170 Wh/kg.
    if "nafepo4" in formula_text:
        if system.average_voltage is not None and system.average_voltage > 3.5:
            speculative = True
            penalty += 0.15
            violations.append("NaFePO4 average voltage above typical practical band (>3.5V)")
        if energy_grav is not None and energy_grav > 170:
            speculative = True
            penalty += 0.18
            violations.append("NaFePO4 full-cell gravimetric energy above typical practical band (>170 Wh/kg)")

    return ConstraintResult(
        valid=True,
        violations=violations,
        score_penalty=min(penalty, 0.7),
        speculative=speculative,
    )


# ============================================================
# MASTER CONSTRAINT EVALUATION
# ============================================================
def evaluate_system(system: BatterySystem) -> Dict[str, Dict]:
    """
    Full constraint evaluation pipeline.
    """

    physical = check_physical_constraints(system)
    chemical = check_chemical_constraints(system)
    performance = check_performance_constraints(system)
    physics_first = evaluate_physics_first(system)

    return {
        "overall_valid": physical.valid and chemical.valid and bool(physics_first.get("hard_valid", False)),
        "physical": physical.to_dict(),
        "chemical": chemical.to_dict(),
        "performance": performance.to_dict(),
        "physics_first": physics_first,
    }

