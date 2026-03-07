from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from design.physics_chemistry import parse_formula, solve_oxidation_states
from materials.chemistry_engine import (
    StrictOxidationSolver,
    PolyanionLibrary,
    StructureClassifier,
    InsertionFilter,
    AlkaliValidator,
)
from materials.material_generator import MaterialCandidate


@dataclass
class ChemistryValidationReport:
    candidate_id: str
    valid: bool
    normalized_formula: str | None = None
    oxidation_state_solvable: bool = False
    duplicate: bool = False
    reasons: list[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class ChemistryValidator:
    """
    Stage chemistry checks executed after material generation.
    """
    _REDOX_ACTIVE_METALS = {
        "Ti", "V", "Mn", "Fe", "Co", "Ni", "Cu", "Sn", "Mo", "W", "Nb", "Cr",
    }
    _ELECTRODE_FRAMEWORK_WHITELIST = {
        "layered oxide",
        "spinel",
        "olivine",
        "nasicon",
        "prussian blue",
        "polyanion framework",
        "sulfide",
        "silicate",
    }
    _ROLE_VOLTAGE_WINDOWS_BY_ION = {
        "Li": {"anode": (0.01, 1.5), "cathode": (2.5, 4.5)},
        "Na": {"anode": (0.01, 1.8), "cathode": (2.0, 4.2)},
        "K": {"anode": (0.01, 1.8), "cathode": (2.0, 4.1)},
        "Mg": {"anode": (0.10, 2.0), "cathode": (1.5, 3.8)},
        "Ca": {"anode": (0.10, 2.0), "cathode": (1.7, 4.0)},
        "Zn": {"anode": (0.10, 1.8), "cathode": (1.2, 2.4)},
        "Al": {"anode": (0.20, 2.0), "cathode": (1.8, 3.2)},
        "Y": {"anode": (0.20, 2.0), "cathode": (1.8, 3.8)},
    }

    def __init__(self, max_working_ion_count: float = 6.0):
        self.max_working_ion_count = float(max_working_ion_count)
        self.strict_oxidation_solver = StrictOxidationSolver()
        self.polyanion_library = PolyanionLibrary()
        self.structure_classifier = StructureClassifier()
        self.insertion_filter = InsertionFilter()
        self.alkali_validator = AlkaliValidator()

    @staticmethod
    def _role_label(candidate: MaterialCandidate) -> str:
        meta = dict(getattr(candidate, "metadata", {}) or {})
        role = str(meta.get("role_condition") or meta.get("component_class") or "").strip().lower()
        return role

    @classmethod
    def _is_electrode_role(cls, candidate: MaterialCandidate) -> bool:
        return cls._role_label(candidate) in {"anode", "cathode", "electrode"}

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _contains_carbonate_group(composition: Dict[str, float]) -> bool:
        c = float(composition.get("C", 0.0) or 0.0)
        o = float(composition.get("O", 0.0) or 0.0)
        if c <= 0.0 or o <= 0.0:
            return False
        ratio = o / max(c, 1e-9)
        return 2.8 <= ratio <= 3.2

    @staticmethod
    def _class_blacklist_hits(formula: str, composition: Dict[str, float]) -> list[str]:
        text = str(formula or "").replace(" ", "").upper()
        hits: list[str] = []
        n = float(composition.get("N", 0.0) or 0.0)
        cl = float(composition.get("Cl", 0.0) or 0.0)
        o = float(composition.get("O", 0.0) or 0.0)
        if n > 0.0 and o / max(n, 1e-9) >= 2.5:
            hits.append("blacklist:nitrate")
        if cl > 0.0 and o / max(cl, 1e-9) >= 3.5:
            hits.append("blacklist:perchlorate")
        if "OH" in text:
            hits.append("blacklist:hydroxide")
        if "C" in composition and "H" in composition:
            hits.append("blacklist:organic_salt")
        return hits

    @classmethod
    def _gas_evolution_proxy_reasons(cls, composition: Dict[str, float]) -> list[str]:
        reasons: list[str] = []
        if cls._contains_carbonate_group(composition):
            reasons.append("gas_evolution_proxy:co2_risk")
        s = float(composition.get("S", 0.0) or 0.0)
        o = float(composition.get("O", 0.0) or 0.0)
        if s > 0.0 and o / max(s, 1e-9) >= 3.5:
            reasons.append("gas_evolution_proxy:sox_risk")
        return reasons

    @staticmethod
    def _format_coeff(value: float) -> str:
        rounded = round(float(value), 6)
        if abs(rounded - round(rounded)) < 1e-8:
            rounded = int(round(rounded))
            return "" if rounded == 1 else str(rounded)
        text = f"{rounded:.6f}".rstrip("0").rstrip(".")
        return text

    @classmethod
    def _normalize_formula(cls, composition: Dict[str, float]) -> str:
        ordered = sorted(composition.items(), key=lambda kv: kv[0])
        return "".join(f"{el}{cls._format_coeff(amount)}" for el, amount in ordered)

    def validate_candidates(
        self,
        candidates: List[MaterialCandidate],
    ) -> tuple[List[MaterialCandidate], List[ChemistryValidationReport]]:
        seen_signatures: Set[str] = set()
        reports: list[ChemistryValidationReport] = []

        for candidate in candidates:
            reasons: list[str] = []
            details: Dict[str, Any] = {}
            normalized_formula: str | None = None
            duplicate = False
            oxidation_state_solvable = False
            structure_family: str | None = None

            formula = str(candidate.framework_formula or "").strip()
            if not formula:
                reasons.append("missing_formula")
            else:
                try:
                    composition = parse_formula(formula)
                    normalized_formula = self._normalize_formula(composition)
                    details["composition"] = composition

                    working_ion = str(candidate.working_ion or "").strip()
                    if working_ion:
                        ion_count = float(composition.get(working_ion, 0.0))
                        details["working_ion_count"] = ion_count
                        if ion_count < 0:
                            reasons.append("working_ion_negative_count")
                        if ion_count > self.max_working_ion_count:
                            reasons.append("working_ion_count_out_of_bounds")

                    ox = solve_oxidation_states(formula)
                    oxidation_state_solvable = bool(ox.valid)
                    details["oxidation_states"] = dict(ox.oxidation_states)
                    details["oxidation_solver_errors"] = list(ox.errors)
                    if not ox.valid:
                        reasons.append("oxidation_state_unsolved")

                    strict_ox = self.strict_oxidation_solver.solve_formula(formula)
                    details["strict_oxidation"] = {
                        "valid": bool(strict_ox.valid),
                        "unique_solution": bool(strict_ox.unique_solution),
                        "oxidation_states": dict(strict_ox.oxidation_states),
                        "n_electrons_max": float(strict_ox.n_electrons_max),
                        "molar_mass": strict_ox.molar_mass,
                        "c_theoretical_mAh_g": strict_ox.c_theoretical_mAh_g,
                        "reasons": list(strict_ox.reasons),
                    }
                    if not strict_ox.valid:
                        reasons.append("strict_oxidation_unsolved")

                    polyanion = self.polyanion_library.validate_formula(formula)
                    details["polyanion"] = {
                        "valid": bool(polyanion.valid),
                        "recognized_units": list(polyanion.recognized_units),
                        "reasons": list(polyanion.reasons),
                    }
                    if not polyanion.valid:
                        reasons.extend(list(polyanion.reasons))

                    structure = self.structure_classifier.classify_formula(
                        formula,
                        working_ion=str(candidate.working_ion or "").strip() or None,
                    )
                    structure_family = structure.family
                    details["structure"] = {
                        "valid": bool(structure.valid),
                        "family": structure.family,
                        "prototype_name": structure.prototype_name,
                        "supported_working_ions": list(structure.supported_working_ions),
                        "diffusion_dimensionality": structure.diffusion_dimensionality,
                        "typical_voltage_range": list(structure.typical_voltage_range)
                        if structure.typical_voltage_range is not None
                        else None,
                        "reasons": list(structure.reasons),
                    }
                    if not structure.valid:
                        reasons.extend(list(structure.reasons))

                    insertion = self.insertion_filter.evaluate_formula(
                        formula,
                        structure_family=structure.family,
                    )
                    details["insertion_filter"] = {
                        "valid": bool(insertion.valid),
                        "insertion_probability": float(insertion.insertion_probability),
                        "reasons": list(insertion.reasons),
                    }
                    if not insertion.valid:
                        reasons.extend(list(insertion.reasons))

                    alkali = self.alkali_validator.validate_formula(
                        formula,
                        working_ion=str(candidate.working_ion or "").strip(),
                    )
                    details["alkali_validator"] = {
                        "valid": bool(alkali.valid),
                        "alkali_elements_present": list(alkali.alkali_elements_present),
                        "reasons": list(alkali.reasons),
                    }
                    if not alkali.valid:
                        reasons.extend(list(alkali.reasons))

                    if self._is_electrode_role(candidate):
                        role = self._role_label(candidate)
                        details["electrochemistry_filter"] = {
                            "role": role,
                            "framework_whitelist": sorted(self._ELECTRODE_FRAMEWORK_WHITELIST),
                            "redox_active_metals": sorted(self._REDOX_ACTIVE_METALS),
                        }
                        ion_key = str(candidate.working_ion or "Li").strip() or "Li"
                        windows_by_role = self._ROLE_VOLTAGE_WINDOWS_BY_ION.get(
                            ion_key,
                            self._ROLE_VOLTAGE_WINDOWS_BY_ION["Li"],
                        )
                        details["electrochemistry_filter"]["working_ion"] = ion_key
                        details["electrochemistry_filter"]["voltage_windows"] = {
                            k: list(v) for k, v in windows_by_role.items()
                        }

                        if self._contains_carbonate_group(composition):
                            reasons.append("carbonate_electrode_forbidden")

                        for cls_reason in self._class_blacklist_hits(formula, composition):
                            reasons.append(cls_reason)

                        gas_risks = self._gas_evolution_proxy_reasons(composition)
                        if gas_risks:
                            reasons.extend(gas_risks)

                        family_key = str(structure.family or "").strip().lower()
                        if family_key not in self._ELECTRODE_FRAMEWORK_WHITELIST:
                            reasons.append("framework_not_whitelisted_for_electrode")

                        elements = {str(k) for k, v in composition.items() if float(v) > 0.0}
                        if not any(el in self._REDOX_ACTIVE_METALS for el in elements):
                            reasons.append("missing_redox_active_metal")

                        reference_voltage = self._float_or_none(
                            (dict(getattr(candidate, "metadata", {}) or {})).get("reference_voltage")
                        )
                        if reference_voltage is not None and role in windows_by_role:
                            vmin, vmax = windows_by_role[role]
                            details["electrochemistry_filter"]["reference_voltage"] = float(reference_voltage)
                            if not (float(vmin) <= float(reference_voltage) <= float(vmax)):
                                reasons.append("role_voltage_window_violation")
                except Exception as exc:
                    reasons.append("formula_parse_failed")
                    details["parse_error"] = str(exc)

            signature = f"{candidate.working_ion}:{(normalized_formula or formula).lower()}"
            if signature in seen_signatures:
                duplicate = True
                reasons.append("duplicate_formula")
            else:
                seen_signatures.add(signature)

            valid = len(reasons) == 0
            candidate.valid = bool(valid)
            candidate.valid_reasons = list(reasons)
            candidate.metadata = dict(candidate.metadata)
            candidate.metadata["chemistry_validator"] = {
                "normalized_formula": normalized_formula,
                "oxidation_state_solvable": oxidation_state_solvable,
                "structure_family": structure_family,
                "duplicate": duplicate,
                "reasons": list(reasons),
            }
            if normalized_formula:
                candidate.metadata["normalized_formula"] = normalized_formula

            reports.append(
                ChemistryValidationReport(
                    candidate_id=candidate.candidate_id,
                    valid=valid,
                    normalized_formula=normalized_formula,
                    oxidation_state_solvable=oxidation_state_solvable,
                    duplicate=duplicate,
                    reasons=list(reasons),
                    details=details,
                )
            )

        return candidates, reports
