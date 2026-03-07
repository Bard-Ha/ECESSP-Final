from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

from design.physics_chemistry import parse_formula, TRANSITION_METALS
from materials.structure_engine import StructurePrototypeRegistry


@dataclass
class StructureClassification:
    valid: bool
    family: Optional[str]
    prototype_name: Optional[str] = None
    supported_working_ions: List[str] = field(default_factory=list)
    diffusion_dimensionality: Optional[str] = None
    typical_voltage_range: Optional[Tuple[float, float]] = None
    reasons: List[str] = field(default_factory=list)


class StructureClassifier:
    """
    Heuristic insertion-family classifier.
    """

    ALLOWED_FAMILIES = (
        "Layered oxide",
        "Spinel",
        "Olivine",
        "NASICON",
        "Prussian Blue",
        "Polyanion framework",
    )
    _registry = StructurePrototypeRegistry()

    @staticmethod
    def _tm_count(comp: Dict[str, float]) -> int:
        return sum(1 for el in comp if el in TRANSITION_METALS and float(comp.get(el, 0.0)) > 0.0)

    @staticmethod
    def _has_any(comp: Dict[str, float], elements: set[str]) -> bool:
        return any(el in comp and float(comp[el]) > 0 for el in elements)

    @classmethod
    def classify_formula(
        cls,
        formula: str,
        *,
        working_ion: Optional[str] = None,
    ) -> StructureClassification:
        text = str(formula or "").replace(" ", "")
        if not text:
            return StructureClassification(valid=False, family=None, reasons=["missing_formula"])

        try:
            comp = parse_formula(text)
        except Exception as exc:
            return StructureClassification(valid=False, family=None, reasons=[f"formula_parse_failed:{exc}"])

        registry_match = cls._registry.match_formula(formula=text, working_ion=working_ion)
        if registry_match.valid:
            return StructureClassification(
                valid=True,
                family=registry_match.family,
                prototype_name=registry_match.prototype_name,
                supported_working_ions=list(registry_match.supported_working_ions),
                diffusion_dimensionality=registry_match.diffusion_dimensionality,
                typical_voltage_range=registry_match.typical_voltage_range,
                reasons=[],
            )
        if "working_ion_not_supported_by_prototype" in registry_match.reasons:
            return StructureClassification(
                valid=False,
                family=None,
                reasons=["working_ion_not_supported_by_prototype"],
            )

        o_count = float(comp.get("O", 0.0))
        has_oxygen = o_count > 0.0
        tm_count = cls._tm_count(comp)
        has_polyanion_centers = cls._has_any(comp, {"P", "S", "Si", "B", "C"})

        # Fallback classifier if registry match misses a valid known family.
        cation_count = sum(float(v) for k, v in comp.items() if k != "O")
        if has_oxygen and tm_count >= 1 and 3.2 <= o_count <= 4.8 and cation_count > 0:
            ratio = o_count / max(cation_count, 1e-6)
            if 1.1 <= ratio <= 1.8:
                return StructureClassification(valid=True, family="Spinel")

        # Layered oxide fallback for oxygen-rich TM frameworks.
        if has_oxygen and tm_count >= 1 and o_count >= 2.0:
            return StructureClassification(valid=True, family="Layered oxide")

        if has_polyanion_centers and has_oxygen and tm_count >= 1:
            return StructureClassification(valid=True, family="Polyanion framework")

        return StructureClassification(valid=False, family=None, reasons=["structure_family_unknown"])
