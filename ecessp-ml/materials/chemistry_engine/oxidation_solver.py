from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

from design.physics_chemistry import (
    parse_formula,
    compute_molar_mass,
    theoretical_capacity_mAh_per_g,
    ALLOWED_OX_STATES as BASE_ALLOWED_OX_STATES,
)


@dataclass
class OxidationSolveReport:
    valid: bool
    unique_solution: bool
    composition: Dict[str, float]
    oxidation_states: Dict[str, int]
    n_electrons_max: float
    molar_mass: Optional[float]
    c_theoretical_mAh_g: Optional[float]
    reasons: List[str] = field(default_factory=list)


class StrictOxidationSolver:
    """
    Strict integer oxidation solver with bounded element-state domains.

    Rules:
    - exact charge neutrality
    - reject ambiguous multi-solution assignments
    - enforce element-specific oxidation bounds
    """

    # Tight chemistry-aware ranges requested in directive.
    _STATE_BOUNDS: Dict[str, List[int]] = {
        "Li": [1],
        "Na": [1],
        "K": [1],
        "Mg": [2],
        "Ca": [2],
        "Zn": [2],
        "Al": [3],
        "O": [-2],
        "F": [-1],
        "Cl": [-1],
        "Br": [-1],
        "I": [-1],
        "P": [5],
        "S": [6],
        "B": [3],
        "Si": [4],
        "C": [4, -4],
        "Mn": [2, 3, 4],
        "V": [3, 4, 5],
        "Ni": [2, 3, 4],
        "Cr": [2, 3, 4],
        "Fe": [2, 3],
        "Cu": [1, 2],
        "Ti": [3, 4],
        "Co": [2, 3, 4],
    }

    _REDOX_SPAN_FALLBACK: Dict[str, int] = {
        "Mn": 2,
        "V": 2,
        "Ni": 2,
        "Cr": 2,
        "Fe": 1,
        "Cu": 1,
        "Ti": 1,
        "Co": 1,
    }

    def __init__(self, max_solutions_to_track: int = 2):
        self.max_solutions_to_track = max(1, int(max_solutions_to_track))

    @classmethod
    def _domain_for(cls, element: str) -> List[int]:
        if element in cls._STATE_BOUNDS:
            return list(cls._STATE_BOUNDS[element])
        if element in BASE_ALLOWED_OX_STATES:
            return [int(v) for v in BASE_ALLOWED_OX_STATES[element]]
        # Conservative fallback for unseen elements.
        return [0, 1, 2, 3, 4]

    @staticmethod
    def _nearly_zero(value: float, tol: float = 1e-8) -> bool:
        return abs(float(value)) <= tol

    def _solve_states(self, composition: Dict[str, float]) -> List[Dict[str, int]]:
        elements = sorted(composition.keys(), key=lambda e: len(self._domain_for(e)))
        domains = {el: self._domain_for(el) for el in elements}
        counts = {el: float(composition[el]) for el in elements}

        suffix_min = [0.0] * (len(elements) + 1)
        suffix_max = [0.0] * (len(elements) + 1)
        for i in range(len(elements) - 1, -1, -1):
            el = elements[i]
            c = counts[el]
            d = domains[el]
            suffix_min[i] = suffix_min[i + 1] + c * float(min(d))
            suffix_max[i] = suffix_max[i + 1] + c * float(max(d))

        assign: Dict[str, int] = {}
        out: list[Dict[str, int]] = []

        def backtrack(idx: int, charge_sum: float) -> None:
            if len(out) >= self.max_solutions_to_track:
                return
            if idx == len(elements):
                if self._nearly_zero(charge_sum):
                    out.append(dict(assign))
                return

            if charge_sum + suffix_min[idx] > 0.0:
                return
            if charge_sum + suffix_max[idx] < 0.0:
                return

            el = elements[idx]
            c = counts[el]
            for ox in domains[el]:
                assign[el] = int(ox)
                backtrack(idx + 1, charge_sum + c * float(ox))
                if len(out) >= self.max_solutions_to_track:
                    return
            assign.pop(el, None)

        backtrack(0, 0.0)
        return out

    def solve_formula(self, formula: str) -> OxidationSolveReport:
        reasons: list[str] = []
        try:
            composition = parse_formula(str(formula or "").strip())
        except Exception as exc:
            return OxidationSolveReport(
                valid=False,
                unique_solution=False,
                composition={},
                oxidation_states={},
                n_electrons_max=0.0,
                molar_mass=None,
                c_theoretical_mAh_g=None,
                reasons=[f"formula_parse_failed:{exc}"],
            )

        if not composition:
            return OxidationSolveReport(
                valid=False,
                unique_solution=False,
                composition={},
                oxidation_states={},
                n_electrons_max=0.0,
                molar_mass=None,
                c_theoretical_mAh_g=None,
                reasons=["empty_composition"],
            )

        solutions = self._solve_states(composition)
        if not solutions:
            reasons.append("no_charge_neutral_solution")
            return OxidationSolveReport(
                valid=False,
                unique_solution=False,
                composition=composition,
                oxidation_states={},
                n_electrons_max=0.0,
                molar_mass=compute_molar_mass(composition),
                c_theoretical_mAh_g=None,
                reasons=reasons,
            )
        if len(solutions) > 1:
            # Preserve strict ambiguity rejection unless ambiguity is equivalent on
            # electrochemically relevant elements.
            key_elements = {
                el
                for el in composition.keys()
                if el in {"Mn", "V", "Ni", "Cr", "Fe", "Cu", "Ti", "Co", "O", "F", "P", "S"}
            }
            signatures = {
                tuple((el, int(sol.get(el, 0))) for el in sorted(key_elements))
                for sol in solutions
            }
            if len(signatures) > 1:
                reasons.append("ambiguous_oxidation_assignments")
                return OxidationSolveReport(
                    valid=False,
                    unique_solution=False,
                    composition=composition,
                    oxidation_states={},
                    n_electrons_max=0.0,
                    molar_mass=compute_molar_mass(composition),
                    c_theoretical_mAh_g=None,
                    reasons=reasons,
                )
            reasons.append("ambiguous_but_equivalent_redox_signature")

        ox = solutions[0]
        n_electrons_max = 0.0
        for el, amt in composition.items():
            span = self._REDOX_SPAN_FALLBACK.get(el, 0)
            n_electrons_max += max(0.0, float(amt) * float(span))

        molar_mass = compute_molar_mass(composition)
        if molar_mass is None or not math.isfinite(molar_mass) or molar_mass <= 0:
            c_theory = None
            reasons.append("molar_mass_unavailable")
        else:
            c_theory = float(theoretical_capacity_mAh_per_g(n_electrons_max, molar_mass))

        return OxidationSolveReport(
            valid=True,
            unique_solution=True,
            composition=composition,
            oxidation_states=ox,
            n_electrons_max=float(max(0.0, n_electrons_max)),
            molar_mass=float(molar_mass) if molar_mass is not None else None,
            c_theoretical_mAh_g=c_theory,
            reasons=reasons,
        )
