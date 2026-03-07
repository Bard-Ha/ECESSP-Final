from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import math
import sys

from .physics_chemistry import parse_formula, solve_oxidation_states, theoretical_capacity_mAh_per_g


@dataclass
class MaterialCandidate:
    formula: str
    cathode_label: str
    family: str
    novelty_score: float
    max_similarity: float
    thermodynamic_proxy: float
    uncertainty_proxy: float
    oxidation_valid: bool
    n_electrons_max: float
    c_theoretical: Optional[float]


class MaterialMutationEngine:
    """
    Controlled material mutation engine for discovery-time cathode proposal.

    Design goals:
    - Deterministic candidate generation from known insertion families.
    - Physics gate via oxidation-state solver + theoretical-capacity check.
    - Novelty gate against known formulas sourced from battery + material catalogs.
    - Lightweight uncertainty proxy for runtime filtering.
    """

    _NOVELTY_REJECT_SIMILARITY = 0.97
    _UNCERTAINTY_REJECT_THRESHOLD = 0.88

    def __init__(self, project_root: Optional[Path] = None):
        root = project_root or Path(__file__).resolve().parents[1]
        self.project_root = root
        self.known_formula_strings: set[str] = set()
        self.known_compositions: List[Dict[str, float]] = self._load_known_compositions(root)

    @staticmethod
    def _to_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _composition_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        a2 = 0.0
        b2 = 0.0
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            av = float(a.get(k, 0.0))
            bv = float(b.get(k, 0.0))
            dot += av * bv
            a2 += av * av
            b2 += bv * bv
        if a2 <= 0.0 or b2 <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (math.sqrt(a2) * math.sqrt(b2))))

    @staticmethod
    def _normalize_comp(comp: Dict[str, float]) -> Dict[str, float]:
        total = sum(float(v) for v in comp.values())
        if total <= 0.0:
            return {}
        return {k: float(v) / total for k, v in comp.items() if float(v) > 0.0}

    def _max_similarity_to_known(self, formula: str) -> float:
        clean_formula = "".join(str(formula or "").split())
        try:
            comp = self._normalize_comp(parse_formula(clean_formula))
        except Exception:
            return 1.0
        if not self.known_compositions:
            return 0.0
        max_comp_sim = 0.0
        for known in self.known_compositions:
            sim = self._composition_cosine(comp, known)
            if sim > max_comp_sim:
                max_comp_sim = sim
            if max_comp_sim >= 0.999:
                break

        exact_match = 1.0 if clean_formula in self.known_formula_strings else 0.0
        # Blend exact-formula identity with composition similarity so slight
        # stoichiometric mutations are not treated as duplicates.
        blended = 0.72 * max_comp_sim + 0.28 * exact_match
        return float(max(0.0, min(1.0, blended)))

    def _load_known_compositions(self, root: Path) -> List[Dict[str, float]]:
        rows: List[str] = []
        material_catalog = root / "data" / "processed" / "material_catalog.csv"
        if material_catalog.exists():
            with material_catalog.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    formula = (r.get("formula") or "").strip()
                    if formula:
                        rows.append(formula)

        # Prefer compact ML-curated file first to avoid extremely large serialized fields.
        battery_formula_sources = [
            root / "data" / "processed" / "batteries_ml_curated.csv",
            root / "data" / "processed" / "batteries_parsed_curated.csv",
        ]
        for batteries_curated in battery_formula_sources:
            if not batteries_curated.exists():
                continue
            with batteries_curated.open("r", encoding="utf-8", newline="") as f:
                # Some CSVs contain very large serialized objects.
                limit = sys.maxsize
                while limit > 131072:
                    try:
                        csv.field_size_limit(limit)
                        break
                    except OverflowError:
                        limit //= 10
                reader = csv.DictReader(f)
                for r in reader:
                    formula = (r.get("framework_formula") or "").strip()
                    if formula:
                        rows.append(formula)
            # ML curated is sufficient; only fall back if nothing was collected.
            if rows:
                break

        out: List[Dict[str, float]] = []
        seen: set[str] = set()
        for formula in rows:
            clean_formula = "".join(str(formula or "").split())
            if clean_formula in seen:
                continue
            seen.add(clean_formula)
            self.known_formula_strings.add(clean_formula)
            try:
                comp = self._normalize_comp(parse_formula(clean_formula))
            except Exception:
                continue
            if comp:
                out.append(comp)
        return out

    @staticmethod
    def _template_pool(working_ion: str, target_voltage: float, target_capacity: float) -> List[Dict[str, str]]:
        ion = (working_ion or "Li").strip().capitalize()

        li_pool = [
            {"formula": "LiFePO4", "family": "olivine", "label": "LFP"},
            {"formula": "Li2FeSiO4", "family": "polyanion", "label": "Li2FeSiO4"},
            {"formula": "Li3Fe2(PO4)3", "family": "polyanion", "label": "LFP3 mutation"},
            {"formula": "Li2FeP2O7", "family": "polyanion", "label": "pyrophosphate mutation"},
            {"formula": "Li2NiMnO4", "family": "spinel", "label": "LiNiMnO4 mutation"},
            {"formula": "Li2FeMnO4", "family": "spinel", "label": "LiFeMnO4 mutation"},
            {"formula": "Li2NiCoO4", "family": "spinel", "label": "LiNiCoO4 mutation"},
            {"formula": "Li2MnSiO4", "family": "silicate", "label": "Li2MnSiO4 mutation"},
        ]

        na_pool = [
            {"formula": "NaFePO4", "family": "olivine", "label": "NaFePO4"},
            {"formula": "Na3Fe2(PO4)3", "family": "nasicon", "label": "NaFePO4 mutation"},
            {"formula": "Na2FeMnO4", "family": "spinel", "label": "NaFeMnO4 mutation"},
            {"formula": "Na2TiMnO4", "family": "spinel", "label": "NaTiMnO4 mutation"},
        ]

        mg_pool = [
            {"formula": "MgMn2O4", "family": "spinel", "label": "MgMn2O4"},
            {"formula": "Mg2MnO4", "family": "spinel", "label": "Mg2MnO4 mutation"},
            {"formula": "Mg2FeMnO4", "family": "spinel", "label": "MgFeMnO4 mutation"},
            {"formula": "Mg2TiMnO4", "family": "spinel", "label": "MgTiMnO4 mutation"},
        ]

        zn_pool = [
            {"formula": "ZnMn2O4", "family": "spinel", "label": "ZnMn2O4"},
            {"formula": "Zn2MnO4", "family": "spinel", "label": "Zn2MnO4 mutation"},
            {"formula": "Zn2FeMnO4", "family": "spinel", "label": "ZnFeMnO4 mutation"},
        ]

        if ion == "Na":
            pool = na_pool
        elif ion == "Mg":
            pool = mg_pool
        elif ion == "Zn":
            pool = zn_pool
        else:
            pool = li_pool

        # Bias family ordering by objective regime.
        if target_capacity >= 220:
            preferred = {"layered", "spinel"}
        elif target_voltage <= 3.2:
            preferred = {"olivine", "nasicon", "polyanion"}
        else:
            preferred = {"layered", "olivine", "spinel"}

        ranked = sorted(
            pool,
            key=lambda x: (0 if x["family"] in preferred else 1, x["label"]),
        )
        return ranked

    def _score_formula_candidate(self, formula: str, family: str, label: str) -> MaterialCandidate:
        ox = solve_oxidation_states(formula)
        c_theoretical: Optional[float] = None
        if ox.valid and ox.molar_mass and ox.molar_mass > 0 and ox.n_electrons_max > 0:
            try:
                c_theoretical = theoretical_capacity_mAh_per_g(ox.n_electrons_max, ox.molar_mass)
            except Exception:
                c_theoretical = None

        max_similarity = self._max_similarity_to_known(formula)
        novelty_score = max(0.0, 1.0 - max_similarity)

        thermo = 0.62
        if "P" in ox.composition:
            thermo += 0.08
        if "O" in ox.composition:
            thermo += 0.06
        if ox.valid:
            thermo += 0.08
        if c_theoretical is not None and c_theoretical > 0:
            # Slightly favor practical-theoretical window.
            if 90.0 <= c_theoretical <= 320.0:
                thermo += 0.07
            elif c_theoretical > 360.0:
                thermo -= 0.12
        thermo = max(0.0, min(1.0, thermo))

        uncertainty = 0.24 + 0.58 * novelty_score + 0.18 * (1.0 - thermo)
        uncertainty = max(0.0, min(1.0, uncertainty))

        return MaterialCandidate(
            formula=formula,
            cathode_label=label,
            family=family,
            novelty_score=float(novelty_score),
            max_similarity=float(max_similarity),
            thermodynamic_proxy=float(thermo),
            uncertainty_proxy=float(uncertainty),
            oxidation_valid=bool(ox.valid),
            n_electrons_max=float(ox.n_electrons_max),
            c_theoretical=float(c_theoretical) if c_theoretical is not None else None,
        )

    def select_material_profile(
        self,
        *,
        working_ion: str,
        target_objectives: Dict[str, float],
        seed: int,
    ) -> Dict[str, Any]:
        target_voltage = self._to_float(target_objectives.get("average_voltage"), 3.7)
        target_capacity = self._to_float(target_objectives.get("capacity_grav"), 180.0)

        templates = self._template_pool(working_ion, target_voltage, target_capacity)
        scored = [self._score_formula_candidate(t["formula"], t["family"], t["label"]) for t in templates]

        valid_candidates: List[MaterialCandidate] = []
        for cand in scored:
            if not cand.oxidation_valid:
                continue
            if cand.max_similarity >= self._NOVELTY_REJECT_SIMILARITY:
                continue
            if cand.uncertainty_proxy > self._UNCERTAINTY_REJECT_THRESHOLD:
                continue
            if cand.c_theoretical is not None and cand.c_theoretical < 50.0:
                continue
            valid_candidates.append(cand)

        if not valid_candidates:
            # Fallback to physics-valid candidates, prioritizing novelty.
            fallback = [c for c in scored if c.oxidation_valid] or scored
            fallback.sort(key=lambda c: (c.novelty_score, c.thermodynamic_proxy), reverse=True)
            valid_candidates = fallback

        valid_candidates.sort(
            key=lambda c: (
                0.45 * c.thermodynamic_proxy
                + 0.40 * c.novelty_score
                + 0.15 * (1.0 - c.uncertainty_proxy)
            ),
            reverse=True,
        )

        idx = abs(int(seed)) % max(1, min(5, len(valid_candidates)))
        selected = valid_candidates[idx]

        # Component suggestions include composite options for high-capacity regimes.
        if target_capacity >= 240:
            anode = "Si-C composite (85:15)"
        elif target_capacity >= 180:
            anode = "Graphite-Si blend (92:8)"
        else:
            anode = "Graphite"

        if selected.family in {"olivine", "nasicon", "polyanion"}:
            electrolyte = "EC:DEC with FEC additive"
        elif selected.family in {"layered", "spinel"} and target_voltage > 4.1:
            electrolyte = "High-voltage carbonate + FEC"
        else:
            electrolyte = "LP30 carbonate"

        formula_comp = parse_formula(selected.formula)
        chemsys_elements = sorted({e for e in formula_comp.keys()} | {(working_ion or "Li").strip().capitalize()})

        return {
            "framework_formula": selected.formula,
            "cathode_material": f"{selected.cathode_label} ({selected.formula})",
            "anode_material": anode,
            "electrolyte": electrolyte,
            "separator_material": "Ceramic-coated PE",
            "additive_material": "FEC/VC blend",
            "chemsys": "-".join(chemsys_elements),
            "material_generation": {
                "method": "controlled_mutation",
                "family": selected.family,
                "mutation_source": selected.cathode_label,
                "novelty_score": round(selected.novelty_score, 6),
                "max_similarity_to_catalog": round(selected.max_similarity, 6),
                "thermodynamic_proxy": round(selected.thermodynamic_proxy, 6),
                "uncertainty_proxy": round(selected.uncertainty_proxy, 6),
                "oxidation_valid": bool(selected.oxidation_valid),
                "n_electrons_max": round(selected.n_electrons_max, 6),
                "C_theoretical_mAh_per_g": (
                    round(float(selected.c_theoretical), 6)
                    if selected.c_theoretical is not None
                    else None
                ),
                "uncertainty_reject_threshold": self._UNCERTAINTY_REJECT_THRESHOLD,
                "similarity_reject_threshold": self._NOVELTY_REJECT_SIMILARITY,
            },
        }
