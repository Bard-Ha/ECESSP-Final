from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from design.physics_chemistry import parse_formula
from materials.material_generator import MaterialCandidate
from materials.role_classifier import RoleAssignment


@dataclass
class CompatibilityRecord:
    cathode: MaterialCandidate
    anode: MaterialCandidate
    electrolyte: MaterialCandidate
    voltage_window_overlap_score: float
    chemical_stability_score: float
    mechanical_strain_risk: float
    interface_risk_reason_codes: List[str]
    hard_valid: bool

    def aggregate_score(self) -> float:
        base = (
            0.45 * self.voltage_window_overlap_score
            + 0.45 * self.chemical_stability_score
            + 0.10 * (1.0 - self.mechanical_strain_risk)
        )
        if not self.hard_valid:
            base *= 0.2
        return max(0.0, min(1.0, float(base)))


class CompatibilityModel:
    """
    Stage-3 pair/triple compatibility model (deterministic proxy).
    """

    def __init__(self, max_pairs: int = 400):
        self.max_pairs = int(max_pairs)

    @staticmethod
    def _elements(candidate: MaterialCandidate) -> set[str]:
        try:
            return set(parse_formula(candidate.framework_formula).keys())
        except Exception:
            return set()

    @staticmethod
    def _role_prob(assignment: RoleAssignment, role: str) -> float:
        return float(assignment.role_probabilities.get(role, 0.0))

    @classmethod
    def _compatibility_scores(
        cls,
        cathode: MaterialCandidate,
        anode: MaterialCandidate,
        electrolyte: MaterialCandidate,
        a_cath: RoleAssignment,
        a_anode: RoleAssignment,
        a_el: RoleAssignment,
    ) -> Tuple[float, float, float, List[str]]:
        risks: list[str] = []

        cathode_strength = cls._role_prob(a_cath, "cathode")
        anode_strength = cls._role_prob(a_anode, "anode")
        electrolyte_strength = cls._role_prob(a_el, "electrolyte_candidate")

        overlap = 1.0 - abs(cathode_strength - anode_strength)
        overlap *= 0.8 + 0.2 * electrolyte_strength
        overlap = max(0.0, min(1.0, overlap))

        cathode_elements = cls._elements(cathode)
        anode_elements = cls._elements(anode)
        electrolyte_elements = cls._elements(electrolyte)

        heavy_overlap = len((cathode_elements & anode_elements) - {cathode.working_ion})
        ionic_overlap = len((cathode_elements | anode_elements) & electrolyte_elements)
        chemical_stability = 0.55 + 0.10 * ionic_overlap - 0.08 * heavy_overlap
        chemical_stability = max(0.0, min(1.0, chemical_stability))

        element_span = abs(len(cathode_elements) - len(anode_elements))
        strain_risk = 0.20 + 0.12 * element_span + 0.40 * (1.0 - electrolyte_strength)
        strain_risk = max(0.0, min(1.0, strain_risk))

        if overlap < 0.35:
            risks.append("voltage_window_overlap_low")
        if chemical_stability < 0.40:
            risks.append("chemical_stability_low")
        if strain_risk > 0.75:
            risks.append("mechanical_strain_risk_high")
        if cathode.working_ion != anode.working_ion or cathode.working_ion != electrolyte.working_ion:
            risks.append("working_ion_mismatch")

        return overlap, chemical_stability, strain_risk, risks

    def score_triples(
        self,
        *,
        candidates: Iterable[MaterialCandidate],
        assignments: Dict[str, RoleAssignment],
    ) -> list[CompatibilityRecord]:
        cathodes: list[MaterialCandidate] = []
        anodes: list[MaterialCandidate] = []
        electrolytes: list[MaterialCandidate] = []
        all_candidates: list[MaterialCandidate] = []

        for candidate in candidates:
            assignment = assignments.get(candidate.candidate_id)
            if assignment is None:
                continue
            all_candidates.append(candidate)
            if "cathode" in assignment.selected_roles:
                cathodes.append(candidate)
            if "anode" in assignment.selected_roles:
                anodes.append(candidate)
            if "electrolyte_candidate" in assignment.selected_roles:
                electrolytes.append(candidate)

        def fill_missing_role(
            bucket: list[MaterialCandidate],
            role_name: str,
            fallback_count: int = 10,
        ) -> None:
            if bucket:
                return
            ranked = sorted(
                all_candidates,
                key=lambda c: self._role_prob(assignments[c.candidate_id], role_name),
                reverse=True,
            )
            for candidate in ranked[:fallback_count]:
                if candidate not in bucket:
                    bucket.append(candidate)

        fill_missing_role(cathodes, "cathode")
        fill_missing_role(anodes, "anode")
        fill_missing_role(electrolytes, "electrolyte_candidate")

        if not cathodes or not anodes or not electrolytes:
            return []

        records: list[CompatibilityRecord] = []
        for cathode in cathodes:
            for anode in anodes:
                for electrolyte in electrolytes:
                    if len(records) >= self.max_pairs:
                        break
                    if cathode.candidate_id == anode.candidate_id:
                        continue
                    if cathode.candidate_id == electrolyte.candidate_id:
                        continue
                    if anode.candidate_id == electrolyte.candidate_id:
                        continue

                    a_cath = assignments[cathode.candidate_id]
                    a_anode = assignments[anode.candidate_id]
                    a_el = assignments[electrolyte.candidate_id]
                    overlap, stability, strain, risks = self._compatibility_scores(
                        cathode,
                        anode,
                        electrolyte,
                        a_cath,
                        a_anode,
                        a_el,
                    )
                    hard_valid = (
                        overlap >= 0.30
                        and stability >= 0.35
                        and strain <= 0.85
                        and cathode.working_ion == anode.working_ion == electrolyte.working_ion
                    )
                    records.append(
                        CompatibilityRecord(
                            cathode=cathode,
                            anode=anode,
                            electrolyte=electrolyte,
                            voltage_window_overlap_score=overlap,
                            chemical_stability_score=stability,
                            mechanical_strain_risk=strain,
                            interface_risk_reason_codes=risks,
                            hard_valid=hard_valid,
                        )
                    )
                if len(records) >= self.max_pairs:
                    break
            if len(records) >= self.max_pairs:
                break

        records.sort(key=lambda r: r.aggregate_score(), reverse=True)
        return records
