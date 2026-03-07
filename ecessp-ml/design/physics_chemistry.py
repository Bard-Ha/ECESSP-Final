from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional


FARADAY_CONSTANT = 96485.0


# Minimal practical table for battery-relevant chemistry.
ATOMIC_WEIGHTS: Dict[str, float] = {
    "H": 1.00794,
    "Li": 6.941,
    "Be": 9.0122,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984,
    "Na": 22.9898,
    "Mg": 24.305,
    "Al": 26.9815,
    "Si": 28.0855,
    "P": 30.9738,
    "S": 32.065,
    "Cl": 35.453,
    "K": 39.0983,
    "Ca": 40.078,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.9332,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.64,
    "As": 74.9216,
    "Se": 78.96,
    "Br": 79.904,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.9059,
    "Zr": 91.224,
    "Nb": 92.9064,
    "Mo": 95.96,
    "Ag": 107.8682,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.6,
    "I": 126.9045,
    "Ba": 137.327,
    "La": 138.9055,
    "Ce": 140.116,
    "Pr": 140.9077,
    "Nd": 144.242,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.9253,
    "Dy": 162.5,
    "Ho": 164.9303,
    "Er": 167.259,
    "Tm": 168.9342,
    "Yb": 173.054,
    "Lu": 174.9668,
    "Hf": 178.49,
    "Ta": 180.9479,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.9666,
    "Hg": 200.59,
    "Pb": 207.2,
}


# Allowed oxidation states constrained to known stable chemistry ranges.
ALLOWED_OX_STATES: Dict[str, List[int]] = {
    "Li": [1],
    "Na": [1],
    "K": [1],
    "Mg": [2],
    "Ca": [2],
    "Zn": [2],
    "Al": [3],
    "B": [3],
    "C": [-4, 2, 4],
    "N": [-3, 3, 5],
    "O": [-2],
    "F": [-1],
    "P": [3, 5],
    "S": [-2, 4, 6],
    "Cl": [-1, 1, 3, 5, 7],
    "Br": [-1, 1, 3, 5],
    "I": [-1, 1, 5, 7],
    "Si": [4],
    "Ti": [2, 3, 4],
    "V": [2, 3, 4, 5],
    "Cr": [2, 3, 6],
    "Mn": [2, 3, 4, 7],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2, 3, 4],
    "Cu": [1, 2],
    "Mo": [4, 5, 6],
    "W": [4, 5, 6],
    "Nb": [3, 4, 5],
    "Zr": [4],
    "Sn": [2, 4],
    "Sb": [3, 5],
    "Te": [-2, 4, 6],
    "Ge": [2, 4],
    "Pb": [2, 4],
}

TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}


@dataclass
class OxidationSolveResult:
    valid: bool
    composition: Dict[str, float]
    oxidation_states: Dict[str, int]
    redox_active_elements: List[str]
    n_electrons_max: float
    molar_mass: Optional[float]
    errors: List[str]


_TOKEN_RE = re.compile(r"([A-Z][a-z]?|\d+(?:\.\d+)?|[()\[\]{}])")


def _merge_comp(dst: Dict[str, float], src: Dict[str, float], mul: float) -> None:
    for el, amt in src.items():
        dst[el] = dst.get(el, 0.0) + amt * mul


def _parse_formula_segment(segment: str) -> Dict[str, float]:
    if not segment:
        return {}

    tokens = _TOKEN_RE.findall(segment)
    if "".join(tokens) != segment:
        raise ValueError(f"Unsupported formula tokenization: {segment}")

    stack: List[Dict[str, float]] = [{}]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("(", "[", "{"):
            stack.append({})
            i += 1
            continue
        if tok in (")", "]", "}"):
            if len(stack) < 2:
                raise ValueError(f"Unbalanced grouping in formula: {segment}")
            group = stack.pop()
            mul = 1.0
            if i + 1 < len(tokens) and re.fullmatch(r"\d+(?:\.\d+)?", tokens[i + 1]):
                mul = float(tokens[i + 1])
                i += 1
            _merge_comp(stack[-1], group, mul)
            i += 1
            continue
        if re.fullmatch(r"[A-Z][a-z]?", tok):
            el = tok
            amt = 1.0
            if i + 1 < len(tokens) and re.fullmatch(r"\d+(?:\.\d+)?", tokens[i + 1]):
                amt = float(tokens[i + 1])
                i += 1
            stack[-1][el] = stack[-1].get(el, 0.0) + amt
            i += 1
            continue
        # Standalone numeric token is invalid here.
        raise ValueError(f"Unexpected token in formula: {tok}")

    if len(stack) != 1:
        raise ValueError(f"Unbalanced grouping in formula: {segment}")
    return stack[0]


def parse_formula(formula: str) -> Dict[str, float]:
    """
    Parse formula into elemental composition map.
    Supports parenthesis/brackets and hydrate separator ('.' or '·').
    """
    if not formula or not isinstance(formula, str):
        raise ValueError("Formula must be a non-empty string")

    cleaned = formula.replace(" ", "").replace("·", ".")
    parts = [p for p in cleaned.split(".") if p]
    total: Dict[str, float] = {}
    for part in parts:
        m = re.match(r"^(\d+(?:\.\d+)?)([A-Z].*)$", part)
        if m:
            part_mul = float(m.group(1))
            part_formula = m.group(2)
        else:
            part_mul = 1.0
            part_formula = part
        comp = _parse_formula_segment(part_formula)
        _merge_comp(total, comp, part_mul)
    return total


def compute_molar_mass(composition: Dict[str, float]) -> Optional[float]:
    total = 0.0
    for el, amt in composition.items():
        w = ATOMIC_WEIGHTS.get(el)
        if w is None:
            return None
        total += w * float(amt)
    return total


def _allowed_states_for_element(el: str) -> List[int]:
    # Conservative fallback for unknowns to prevent unbounded nonsense.
    if el in ALLOWED_OX_STATES:
        return ALLOWED_OX_STATES[el]
    if el in ("H",):
        return [-1, 1]
    return [0]


def _solve_charge_neutrality(composition: Dict[str, float]) -> Optional[Dict[str, int]]:
    elements = sorted(composition.keys(), key=lambda e: len(_allowed_states_for_element(e)))
    domains = {el: _allowed_states_for_element(el) for el in elements}
    counts = {el: float(composition[el]) for el in elements}

    suffix_min: List[float] = [0.0] * (len(elements) + 1)
    suffix_max: List[float] = [0.0] * (len(elements) + 1)
    for idx in range(len(elements) - 1, -1, -1):
        el = elements[idx]
        c = counts[el]
        d = domains[el]
        suffix_min[idx] = suffix_min[idx + 1] + c * min(d)
        suffix_max[idx] = suffix_max[idx + 1] + c * max(d)

    assign: Dict[str, int] = {}

    def backtrack(i: int, total: float) -> Optional[Dict[str, int]]:
        if i == len(elements):
            return dict(assign) if abs(total) < 1e-9 else None
        if total + suffix_min[i] > 0.0:
            return None
        if total + suffix_max[i] < 0.0:
            return None

        el = elements[i]
        c = counts[el]
        for ox in domains[el]:
            assign[el] = int(ox)
            out = backtrack(i + 1, total + c * ox)
            if out is not None:
                return out
        assign.pop(el, None)
        return None

    return backtrack(0, 0.0)


def solve_oxidation_states(formula: str) -> OxidationSolveResult:
    errors: List[str] = []
    try:
        composition = parse_formula(formula)
    except Exception as exc:
        return OxidationSolveResult(
            valid=False,
            composition={},
            oxidation_states={},
            redox_active_elements=[],
            n_electrons_max=0.0,
            molar_mass=None,
            errors=[f"formula_parse_failed:{exc}"],
        )

    solution = _solve_charge_neutrality(composition)
    if solution is None:
        errors.append("no_valid_oxidation_solution")
        return OxidationSolveResult(
            valid=False,
            composition=composition,
            oxidation_states={},
            redox_active_elements=[],
            n_electrons_max=0.0,
            molar_mass=compute_molar_mass(composition),
            errors=errors,
        )

    redox_active: List[str] = []
    n_electrons_max = 0.0
    for el, amt in composition.items():
        domain = _allowed_states_for_element(el)
        if el in TRANSITION_METALS and len(domain) > 1:
            redox_active.append(el)
            n_electrons_max += float(amt) * float(max(domain) - min(domain))

    return OxidationSolveResult(
        valid=True,
        composition=composition,
        oxidation_states=solution,
        redox_active_elements=sorted(redox_active),
        n_electrons_max=float(max(0.0, n_electrons_max)),
        molar_mass=compute_molar_mass(composition),
        errors=[],
    )


def theoretical_capacity_mAh_per_g(n_electrons: float, molar_mass_g_per_mol: float) -> float:
    if molar_mass_g_per_mol <= 0:
        raise ValueError("molar_mass must be > 0")
    if n_electrons < 0:
        raise ValueError("n_electrons must be >= 0")
    return float((n_electrons * FARADAY_CONSTANT) / (3.6 * molar_mass_g_per_mol))
