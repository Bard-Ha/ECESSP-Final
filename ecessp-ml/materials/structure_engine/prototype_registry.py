from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

from design.physics_chemistry import parse_formula, TRANSITION_METALS


@dataclass(frozen=True)
class PrototypeEntry:
    prototype_name: str
    family: str
    general_formula_pattern: str
    allowed_substitution_elements: Tuple[str, ...]
    supported_working_ions: Tuple[str, ...]
    diffusion_dimensionality: str
    typical_voltage_range: Tuple[float, float]


@dataclass
class PrototypeMatch:
    valid: bool
    prototype_name: Optional[str] = None
    family: Optional[str] = None
    supported_working_ions: List[str] = field(default_factory=list)
    diffusion_dimensionality: Optional[str] = None
    typical_voltage_range: Optional[Tuple[float, float]] = None
    reasons: List[str] = field(default_factory=list)


class StructurePrototypeRegistry:
    """
    Prototype registry for known insertion structural families.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        root = Path(__file__).resolve().parent
        self._registry_path = registry_path or (root / "structure_prototype_registry.json")
        self._entries = self._load_entries()

    def _load_entries(self) -> List[PrototypeEntry]:
        if not self._registry_path.exists():
            return []
        try:
            raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []

        entries: list[PrototypeEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("prototype_name") or "").strip()
            family = str(item.get("family") or "").strip()
            if not name or not family:
                continue
            ions = tuple(str(v).strip() for v in (item.get("supported_working_ions") or []) if str(v).strip())
            subs = tuple(
                str(v).strip()
                for v in (item.get("allowed_substitution_elements") or [])
                if str(v).strip()
            )
            tvr = item.get("typical_voltage_range") or [0.0, 0.0]
            lo = float(tvr[0]) if isinstance(tvr, (list, tuple)) and len(tvr) >= 1 else 0.0
            hi = float(tvr[1]) if isinstance(tvr, (list, tuple)) and len(tvr) >= 2 else lo
            entries.append(
                PrototypeEntry(
                    prototype_name=name,
                    family=family,
                    general_formula_pattern=str(item.get("general_formula_pattern") or "").strip(),
                    allowed_substitution_elements=subs,
                    supported_working_ions=ions,
                    diffusion_dimensionality=str(item.get("diffusion_dimensionality") or "").strip(),
                    typical_voltage_range=(lo, hi),
                )
            )
        return entries

    @staticmethod
    def _tm_count(comp: Dict[str, float]) -> int:
        return sum(1 for el, amt in comp.items() if el in TRANSITION_METALS and float(amt) > 0.0)

    @staticmethod
    def _family_candidates(formula: str) -> List[str]:
        text = str(formula or "").replace(" ", "")
        try:
            comp = parse_formula(text)
        except Exception:
            return []
        if not comp:
            return []
        o = float(comp.get("O", 0.0))
        tm = StructurePrototypeRegistry._tm_count(comp)
        out: list[str] = []

        if "C" in comp and "N" in comp and tm >= 1:
            out.append("Prussian Blue")
        if "P" in comp and o >= 10.0 and tm >= 1:
            out.append("NASICON")
        if "P" in comp and 3.5 <= o <= 8.5 and tm >= 1:
            out.append("Olivine")
        if o >= 3.2 and tm >= 1:
            cations = sum(float(v) for k, v in comp.items() if k != "O")
            ratio = o / max(cations, 1e-6)
            if 1.1 <= ratio <= 1.8:
                out.append("Spinel")
        if o >= 2.0 and tm >= 1:
            out.append("Layered oxide")
        if ("P" in comp or "S" in comp or "B" in comp or "Si" in comp or "C" in comp) and o > 0.0 and tm >= 1:
            out.append("Polyanion framework")

        return out

    def match_formula(self, *, formula: str, working_ion: Optional[str] = None) -> PrototypeMatch:
        families = self._family_candidates(formula)
        if not families:
            return PrototypeMatch(valid=False, reasons=["structure_family_unknown"])

        ion = str(working_ion or "").strip()
        for fam in families:
            for entry in self._entries:
                if entry.family != fam:
                    continue
                if ion and entry.supported_working_ions and ion not in entry.supported_working_ions:
                    continue
                return PrototypeMatch(
                    valid=True,
                    prototype_name=entry.prototype_name,
                    family=entry.family,
                    supported_working_ions=list(entry.supported_working_ions),
                    diffusion_dimensionality=entry.diffusion_dimensionality,
                    typical_voltage_range=entry.typical_voltage_range,
                    reasons=[],
                )

        # Family matched but not for requested ion.
        if ion:
            return PrototypeMatch(valid=False, reasons=["working_ion_not_supported_by_prototype"])
        return PrototypeMatch(valid=False, reasons=["structure_family_unknown"])

