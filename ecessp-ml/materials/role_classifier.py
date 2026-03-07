from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from design.physics_chemistry import parse_formula
from materials.material_generator import MaterialCandidate
from materials.role_inference import infer_material_roles_from_descriptors


TRANSITION_METAL_SET = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

ELECTROLYTE_HINT_SET = {"F", "Cl", "Br", "I", "P", "S"}


@dataclass
class RoleAssignment:
    candidate_id: str
    role_probabilities: Dict[str, float]
    confidence_score: float
    selected_roles: List[str]
    used_fallback: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "role_probabilities": dict(self.role_probabilities),
            "confidence_score": float(self.confidence_score),
            "selected_roles": list(self.selected_roles),
            "used_fallback": bool(self.used_fallback),
        }


class RoleClassifier:
    """
    Stage-2 role assignment.

    This is a deterministic proxy for the future supervised head and preserves
    an explicit fallback path when confidence is low.
    """

    ROLE_NAMES = ("cathode", "anode", "electrolyte_candidate")

    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = float(confidence_threshold)

    @staticmethod
    def _normalize_probs(raw_probs: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, float(v)) for v in raw_probs.values())
        if total <= 0:
            return {"cathode": 1.0 / 3.0, "anode": 1.0 / 3.0, "electrolyte_candidate": 1.0 / 3.0}
        return {k: max(0.0, float(v)) / total for k, v in raw_probs.items()}

    @staticmethod
    def _elements(formula: str) -> list[str]:
        try:
            return sorted(parse_formula(formula).keys())
        except Exception:
            return []

    def _primary_probs(
        self,
        candidate: MaterialCandidate,
        target_property_vector: Dict[str, float],
    ) -> Dict[str, float]:
        formula = str(candidate.framework_formula or "")
        elements = self._elements(formula)
        voltage = float(target_property_vector.get("average_voltage", 3.5) or 3.5)
        capacity = float(target_property_vector.get("capacity_grav", 170.0) or 170.0)

        has_transition_metal = any(el in TRANSITION_METAL_SET for el in elements)
        electrolyte_like = any(el in ELECTROLYTE_HINT_SET for el in elements)

        cathode = 0.45 + max(0.0, min(0.35, (voltage - 3.2) / 2.0))
        anode = 0.40 + max(0.0, min(0.35, (3.3 - voltage) / 2.0))
        electrolyte = 0.15

        if has_transition_metal:
            cathode += 0.20
        else:
            anode += 0.10

        if capacity >= 220:
            anode += 0.12
        if electrolyte_like:
            electrolyte += 0.32
            cathode -= 0.10
            anode -= 0.10

        return self._normalize_probs(
            {
                "cathode": cathode,
                "anode": anode,
                "electrolyte_candidate": electrolyte,
            }
        )

    @staticmethod
    def _fallback_probs(
        candidate: MaterialCandidate,
        target_property_vector: Dict[str, float],
    ) -> Dict[str, float]:
        descriptors = {"is_layered": False}
        system_features = {
            "average_voltage": float(target_property_vector.get("average_voltage", 3.5) or 3.5),
            "capacity_grav": float(target_property_vector.get("capacity_grav", 170.0) or 170.0),
            "energy_grav": float(target_property_vector.get("energy_grav", 500.0) or 500.0),
            "max_delta_volume": float(target_property_vector.get("max_delta_volume", 0.12) or 0.12),
        }
        inferred = infer_material_roles_from_descriptors(descriptors, system_features)
        roles = set(inferred.get("roles", []))

        raw = {
            "cathode": 0.33,
            "anode": 0.33,
            "electrolyte_candidate": 0.34,
        }
        if "Cathode candidate" in roles:
            raw["cathode"] += 0.25
        if "High-capacity insertion host" in roles:
            raw["anode"] += 0.20
        if "Electrochemically inert or support framework" in roles:
            raw["electrolyte_candidate"] += 0.20
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    @staticmethod
    def _selected_roles(role_probabilities: Dict[str, float]) -> list[str]:
        ranked = sorted(role_probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top_role = ranked[0][0] if ranked else "electrolyte_candidate"
        selected = [role for role, prob in ranked if prob >= 0.45]
        if not selected:
            selected = [top_role]
        return selected

    def classify_candidates(
        self,
        candidates: List[MaterialCandidate],
        *,
        target_property_vector: Dict[str, float],
    ) -> Dict[str, RoleAssignment]:
        assignments: Dict[str, RoleAssignment] = {}
        for candidate in candidates:
            probs = self._primary_probs(candidate, target_property_vector)
            confidence = max(probs.values())
            used_fallback = False
            if confidence < self.confidence_threshold:
                probs = self._fallback_probs(candidate, target_property_vector)
                confidence = max(probs.values())
                used_fallback = True

            selected = self._selected_roles(probs)
            assignment = RoleAssignment(
                candidate_id=candidate.candidate_id,
                role_probabilities=probs,
                confidence_score=float(confidence),
                selected_roles=selected,
                used_fallback=used_fallback,
            )
            candidate.metadata = dict(candidate.metadata)
            candidate.metadata["role_assignment"] = assignment.to_dict()
            assignments[candidate.candidate_id] = assignment
        return assignments
