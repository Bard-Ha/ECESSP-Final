from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import csv
import math
import sys

from design.physics_chemistry import parse_formula
from materials.chemistry_engine import (
    PolyanionLibrary,
    StructureClassifier,
    InsertionFilter,
    AlkaliValidator,
)


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "batteries_parsed.csv"
)

TRANSITION_SUBSTITUTIONS: Dict[str, tuple[str, ...]] = {
    "Mn": ("Fe", "Co", "Ni"),
    "Fe": ("Mn", "Co", "Ni"),
    "Co": ("Ni", "Mn", "Fe"),
    "Ni": ("Co", "Mn", "Fe"),
    "V": ("Ti", "Cr", "Mn"),
    "Ti": ("V", "Mn"),
    "Cr": ("Mn", "Fe"),
}


@dataclass
class MaterialCandidate:
    candidate_id: str
    framework_formula: str
    working_ion: str
    material_id: Optional[str] = None
    battery_type: str = "insertion"
    source_mode: str = "composition_perturbation"
    valid: bool = True
    valid_reasons: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "framework_formula": self.framework_formula,
            "working_ion": self.working_ion,
            "material_id": self.material_id,
            "battery_type": self.battery_type,
            "source_mode": self.source_mode,
            "valid": self.valid,
            "valid_reasons": list(self.valid_reasons),
            "metadata": dict(self.metadata),
        }


class MaterialGenerator:
    """
    Stage-1 material generator for insertion hosts.

    This implementation is deterministic and dataset-driven:
    - Mode 1 (latent optimization): represented by top-ranked host reuse.
    - Mode 2 (composition perturbation): represented by nearby-host retrieval.
    - Mode 3 (CIF-conditioned refinement): represented by optional seed host.
    """

    TARGET_FIELDS = (
        "average_voltage",
        "capacity_grav",
        "energy_grav",
        "max_delta_volume",
        "stability_charge",
        "stability_discharge",
    )

    def __init__(self, dataset_path: Path | None = None):
        self._dataset_path = dataset_path or DEFAULT_DATASET_PATH
        self._rows_cache: list[dict[str, str]] | None = None
        self._polyanion_library = PolyanionLibrary()
        self._structure_classifier = StructureClassifier()
        self._insertion_filter = InsertionFilter()
        self._alkali_validator = AlkaliValidator()

    @staticmethod
    def _format_coeff(value: float) -> str:
        rounded = round(float(value), 6)
        if abs(rounded - round(rounded)) < 1e-8:
            rounded = int(round(rounded))
            return "" if rounded == 1 else str(rounded)
        return f"{rounded:.6f}".rstrip("0").rstrip(".")

    @classmethod
    def _composition_to_formula(cls, composition: Dict[str, float]) -> str:
        ordered = sorted(
            [(el, float(amount)) for el, amount in composition.items() if float(amount) > 0.0],
            key=lambda kv: kv[0],
        )
        return "".join(f"{el}{cls._format_coeff(amount)}" for el, amount in ordered)

    @classmethod
    def _propose_formula_variants(
        cls,
        formula: str,
        *,
        working_ion: str,
        max_variants: int = 3,
    ) -> list[str]:
        variants: list[str] = []
        if not formula:
            return variants
        try:
            composition = parse_formula(formula)
        except Exception:
            return variants

        ion = str(working_ion or "").strip()
        non_oxygen_elements = [el for el in composition.keys() if el not in {"O", ion}]

        # Variant 1: transition-metal substitution (composition-preserving).
        for el in non_oxygen_elements:
            candidates = TRANSITION_SUBSTITUTIONS.get(el, ())
            if not candidates:
                continue
            for sub in candidates:
                trial = dict(composition)
                amt = float(trial.pop(el, 0.0))
                if amt <= 0.0:
                    continue
                trial[sub] = float(trial.get(sub, 0.0) + amt)
                f = cls._composition_to_formula(trial)
                if f and f != formula and f not in variants:
                    variants.append(f)
                if len(variants) >= max_variants:
                    return variants
            if len(variants) >= max_variants:
                return variants

        # Variant 2: slight stoichiometric perturbation on first non-ion element.
        for el in non_oxygen_elements:
            base = float(composition.get(el, 0.0))
            if base <= 0.0:
                continue
            delta = 0.25 if base >= 1.0 else 0.10
            for sign in (1.0, -1.0):
                trial = dict(composition)
                new_amt = max(0.05, base + sign * delta)
                trial[el] = new_amt
                f = cls._composition_to_formula(trial)
                if f and f != formula and f not in variants:
                    variants.append(f)
                if len(variants) >= max_variants:
                    return variants
            if len(variants) >= max_variants:
                return variants

        return variants

    def _load_rows(self) -> list[dict[str, str]]:
        if self._rows_cache is not None:
            return self._rows_cache

        rows: list[dict[str, str]] = []
        if not self._dataset_path.exists():
            self._rows_cache = rows
            return rows

        # Some curated CSV exports contain wide embedded fields that exceed
        # Python's default CSV field size; lift the parser cap defensively.
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                break
            except OverflowError:
                limit = int(limit / 10)

        with self._dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
        self._rows_cache = rows
        return rows

    @staticmethod
    def _safe_float(raw: Any, default: float | None = None) -> float | None:
        if raw in (None, ""):
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value):
            return default
        return value

    def _objective_distance(
        self,
        row: dict[str, str],
        target_property_vector: Dict[str, float],
    ) -> float:
        distances: list[float] = []
        for field in self.TARGET_FIELDS:
            target = self._safe_float(target_property_vector.get(field))
            if target is None:
                continue
            observed = self._safe_float(row.get(field))
            if observed is None:
                continue
            scale = max(abs(target), 1.0)
            distances.append(abs(observed - target) / scale)
        if not distances:
            return 1.0
        return float(sum(distances) / len(distances))

    def _role_adjusted_distance(
        self,
        row: dict[str, str],
        target_property_vector: Dict[str, float],
        *,
        role_condition: str | None,
    ) -> float:
        base = float(self._objective_distance(row, target_property_vector))
        role = str(role_condition or "").strip().lower()
        voltage = self._safe_float(row.get("average_voltage"), default=None)
        if voltage is None:
            return base
        if role == "anode":
            # Prefer low-potential hosts for anode-conditioned generation.
            return base + max(0.0, float(voltage) - 1.2) / 2.5
        if role == "cathode":
            # Prefer high-potential hosts for cathode-conditioned generation.
            return base + max(0.0, 2.0 - float(voltage)) / 2.0
        return base

    @staticmethod
    def _mode_for_rank(rank: int, has_seed: bool) -> str:
        if has_seed and rank == 0:
            return "cif_conditioned_refinement"
        cycle = rank % 3
        if cycle == 0:
            return "latent_optimization"
        if cycle == 1:
            return "composition_perturbation"
        return "cif_conditioned_refinement"

    @staticmethod
    def _normalize_formula(value: str | None) -> str:
        return str(value or "").strip()

    def _iter_filtered_rows(self, working_ion: str) -> Iterable[dict[str, str]]:
        ion = (working_ion or "").strip()
        rows = self._load_rows()
        for row in rows:
            if (row.get("battery_type") or "").strip().lower() != "insertion":
                continue
            row_ion = (row.get("working_ion") or "").strip()
            if ion and row_ion and row_ion != ion:
                continue
            formula = self._normalize_formula(row.get("framework_formula") or row.get("battery_formula"))
            if not formula:
                continue
            structure = self._structure_classifier.classify_formula(
                formula,
                working_ion=ion or row_ion,
            )
            if not structure.valid:
                continue
            insertion = self._insertion_filter.evaluate_formula(
                formula,
                structure_family=structure.family,
            )
            if not insertion.valid:
                continue
            alkali = self._alkali_validator.validate_formula(formula, ion or row_ion)
            if not alkali.valid:
                continue
            polyanion = self._polyanion_library.validate_formula(formula)
            if not polyanion.valid:
                continue
            yield row

    def _family_priority(self, *, target_property_vector: Dict[str, float], working_ion: str) -> list[str]:
        ion = str(working_ion or "Li").strip()
        voltage = self._safe_float(target_property_vector.get("average_voltage"), default=3.5)
        if ion == "Na":
            base = ["NASICON", "Prussian Blue", "Polyanion framework", "Layered oxide", "Spinel", "Olivine"]
        elif ion in {"Mg", "Ca", "Zn"}:
            base = ["Spinel", "Polyanion framework", "NASICON", "Prussian Blue", "Layered oxide", "Olivine"]
        else:
            base = ["Olivine", "Layered oxide", "Spinel", "Polyanion framework", "NASICON", "Prussian Blue"]
        if voltage >= 3.9:
            high_v = ["Layered oxide", "Spinel", "Olivine", "Polyanion framework", "NASICON", "Prussian Blue"]
            return high_v
        return base

    def _row_family(self, row: dict[str, str], *, working_ion: str) -> str | None:
        formula = self._normalize_formula(row.get("framework_formula") or row.get("battery_formula"))
        if not formula:
            return None
        rep = self._structure_classifier.classify_formula(formula, working_ion=working_ion or None)
        return rep.family if rep.valid else None

    def generate_candidates(
        self,
        *,
        working_ion: str,
        target_property_vector: Dict[str, float],
        optional_seed_structure: Optional[str] = None,
        candidate_pool_size: int = 150,
        interpolation_enabled: bool = True,
        extrapolation_enabled: bool = True,
        role_condition: Optional[str] = None,
    ) -> List[MaterialCandidate]:
        pool_size = max(1, int(candidate_pool_size))
        interpolation_enabled = bool(interpolation_enabled)
        extrapolation_enabled = bool(extrapolation_enabled)
        if not interpolation_enabled and not extrapolation_enabled:
            interpolation_enabled = True
        filtered_rows = list(self._iter_filtered_rows(working_ion))

        scored_rows = sorted(
            filtered_rows,
            key=lambda row: self._role_adjusted_distance(
                row,
                target_property_vector,
                role_condition=role_condition,
            ),
        )
        family_priority = self._family_priority(
            target_property_vector=target_property_vector,
            working_ion=working_ion,
        )
        by_family: dict[str, list[dict[str, str]]] = {k: [] for k in family_priority}
        others: list[dict[str, str]] = []
        for row in scored_rows:
            fam = self._row_family(row, working_ion=working_ion or str(row.get("working_ion") or ""))
            if fam and fam in by_family:
                by_family[fam].append(row)
            else:
                others.append(row)
        prioritized_rows: list[dict[str, str]] = []
        max_bucket = max((len(v) for v in by_family.values()), default=0)
        for i in range(max_bucket):
            for fam in family_priority:
                bucket = by_family.get(fam, [])
                if i < len(bucket):
                    prioritized_rows.append(bucket[i])
        prioritized_rows.extend(others)

        output: list[MaterialCandidate] = []
        seen_formula: set[str] = set()

        seed_formula = self._normalize_formula(optional_seed_structure)
        if seed_formula and interpolation_enabled:
            output.append(
                MaterialCandidate(
                    candidate_id="seed_000",
                    material_id=None,
                    framework_formula=seed_formula,
                    working_ion=working_ion or "Li",
                    source_mode="cif_conditioned_refinement",
                    metadata={"seeded": True},
                )
            )
            seen_formula.add(seed_formula.lower())

        for rank, row in enumerate(prioritized_rows):
            if len(output) >= pool_size:
                break

            formula = self._normalize_formula(row.get("framework_formula") or row.get("battery_formula"))
            signature = formula.lower()
            if not formula or signature in seen_formula:
                continue
            if interpolation_enabled:
                seen_formula.add(signature)
                output.append(
                    MaterialCandidate(
                        candidate_id=f"host_{len(output):03d}",
                        material_id=(row.get("battery_id") or None),
                        framework_formula=formula,
                        working_ion=working_ion or (row.get("working_ion") or "Li"),
                        source_mode=self._mode_for_rank(rank, bool(seed_formula)),
                        metadata={
                            "dataset_row_id": row.get("battery_id"),
                            "host_structure": row.get("host_structure"),
                            "reference_voltage": self._safe_float(row.get("average_voltage")),
                            "reference_capacity_grav": self._safe_float(row.get("capacity_grav")),
                            "reference_energy_grav": self._safe_float(row.get("energy_grav")),
                            "reference_stability_charge": self._safe_float(row.get("stability_charge")),
                            "reference_stability_discharge": self._safe_float(row.get("stability_discharge")),
                            "role_condition": str(role_condition or ""),
                        },
                    )
                )
                if len(output) >= pool_size:
                    break

            if extrapolation_enabled:
                variants = self._propose_formula_variants(
                    formula,
                    working_ion=working_ion or (row.get("working_ion") or "Li"),
                    max_variants=3,
                )
                for vi, variant_formula in enumerate(variants):
                    if len(output) >= pool_size:
                        break
                    variant_signature = variant_formula.lower()
                    if not variant_formula or variant_signature in seen_formula:
                        continue
                    seen_formula.add(variant_signature)
                    output.append(
                        MaterialCandidate(
                            candidate_id=f"var_{len(output):03d}",
                            material_id=None,
                            framework_formula=variant_formula,
                            working_ion=working_ion or (row.get("working_ion") or "Li"),
                            source_mode="latent_variation",
                            metadata={
                                "variant_of_formula": formula,
                            "variant_of_dataset_row_id": row.get("battery_id"),
                            "variant_index": int(vi),
                            "reference_voltage": self._safe_float(row.get("average_voltage")),
                            "reference_capacity_grav": self._safe_float(row.get("capacity_grav")),
                            "reference_energy_grav": self._safe_float(row.get("energy_grav")),
                            "role_condition": str(role_condition or ""),
                        },
                    )
                )

        return output[:pool_size]
