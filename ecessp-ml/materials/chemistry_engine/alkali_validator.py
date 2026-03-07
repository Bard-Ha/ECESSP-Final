from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from design.physics_chemistry import parse_formula


@dataclass
class AlkaliValidationReport:
    valid: bool
    alkali_elements_present: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class AlkaliValidator:
    """
    Validate working-ion consistency (mixed-alkali rejection by default).
    """

    _ALKALI_SET: Set[str] = {"Li", "Na", "K", "Rb", "Cs"}

    @classmethod
    def validate_formula(
        cls,
        formula: str,
        working_ion: str,
        *,
        dual_ion_mode: bool = False,
    ) -> AlkaliValidationReport:
        text = str(formula or "").replace(" ", "")
        ion = str(working_ion or "").strip()
        if not text:
            return AlkaliValidationReport(valid=False, reasons=["missing_formula"])
        try:
            comp = parse_formula(text)
        except Exception as exc:
            return AlkaliValidationReport(valid=False, reasons=[f"formula_parse_failed:{exc}"])

        present = sorted([el for el in cls._ALKALI_SET if float(comp.get(el, 0.0)) > 0.0])
        reasons: list[str] = []

        if ion and not dual_ion_mode:
            others = [el for el in present if el != ion]
            if others:
                reasons.append("mixed_alkali_inconsistency")
            # Explicit Na-vs-Li rule from directive.
            if ion == "Na" and "Li" in present:
                reasons.append("na_system_contains_li_framework")

        return AlkaliValidationReport(
            valid=(len(reasons) == 0),
            alkali_elements_present=present,
            reasons=reasons,
        )
