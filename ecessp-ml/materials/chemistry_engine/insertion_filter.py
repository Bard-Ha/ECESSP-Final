from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from design.physics_chemistry import parse_formula, TRANSITION_METALS


@dataclass
class InsertionFilterReport:
    valid: bool
    insertion_probability: float = 0.0
    reasons: List[str] = field(default_factory=list)


class InsertionFilter:
    """
    Deterministic insertion-vs-non-insertion screening.
    """

    _REJECT_ALLOY_HOSTS = {"Sn", "Si", "Al"}

    @classmethod
    def evaluate_formula(
        cls,
        formula: str,
        *,
        structure_family: Optional[str] = None,
        min_insertion_probability: float = 0.70,
    ) -> InsertionFilterReport:
        text = str(formula or "").replace(" ", "")
        if not text:
            return InsertionFilterReport(valid=False, insertion_probability=0.0, reasons=["missing_formula"])
        try:
            comp = parse_formula(text)
        except Exception as exc:
            return InsertionFilterReport(valid=False, insertion_probability=0.0, reasons=[f"formula_parse_failed:{exc}"])

        reasons: list[str] = []
        elements = {el for el, amt in comp.items() if float(amt) > 0.0}

        # Reject pure metallic hosts in insertion-only mode.
        if "O" not in elements and len(elements) == 1:
            reasons.append("pure_metal_host_rejected")

        # Reject simple binary oxides with no open-framework chemistry.
        if "O" in elements and len(elements) == 2:
            reasons.append("simple_binary_oxide_no_open_framework")

        # Reject pure/alloy-like frameworks dominated by Sn/Si/Al.
        if elements and elements.issubset(cls._REJECT_ALLOY_HOSTS | {"Li", "Na", "K", "Mg", "Ca", "Zn"}):
            reasons.append("alloy_type_host_rejected")

        # Reject simple fluorides without oxygen/polyanion framework.
        if "F" in elements and "O" not in elements and len(elements) <= 3:
            reasons.append("simple_fluoride_no_open_framework")

        # Require plausible diffusion topology proxy:
        # oxygen framework + at least one transition metal OR recognized structural family.
        has_tm = any(el in TRANSITION_METALS for el in elements)
        if structure_family is None and not ("O" in elements and has_tm):
            reasons.append("missing_diffusion_topology_proxy")
        if structure_family is not None and str(structure_family).strip() == "":
            reasons.append("missing_diffusion_topology_proxy")

        insertion_probability = 0.30
        if "O" in elements and has_tm:
            insertion_probability += 0.35
        if structure_family is not None and str(structure_family).strip():
            insertion_probability += 0.30
        if any(
            r in reasons
            for r in {
                "pure_metal_host_rejected",
                "simple_binary_oxide_no_open_framework",
                "alloy_type_host_rejected",
                "simple_fluoride_no_open_framework",
            }
        ):
            insertion_probability -= 0.35
        insertion_probability = max(0.0, min(1.0, insertion_probability))

        threshold = max(0.0, min(1.0, float(min_insertion_probability)))
        if insertion_probability < threshold:
            reasons.append("insertion_probability_below_threshold")

        return InsertionFilterReport(
            valid=(len(reasons) == 0),
            insertion_probability=float(insertion_probability),
            reasons=reasons,
        )
