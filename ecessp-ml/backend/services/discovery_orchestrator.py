from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import os
import statistics

from backend.config import GRAPH_CONFIG
from backend.runtime.context import get_runtime_context
from backend.runtime.enhanced_engine import get_enhanced_inference_engine
from backend.services.active_learning_queue import ActiveLearningQueueService
from design.compatibility_model import CompatibilityModel, CompatibilityRecord
from design.full_cell_assembler import FullCellAssembler
from design.physics_chemistry import parse_formula
from design.system_constraints import evaluate_system
from design.system_template import BatterySystem
from materials.chemistry_validator import ChemistryValidator
from materials.material_generator import MaterialCandidate, MaterialGenerator
from materials.role_classifier import RoleAssignment, RoleClassifier

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryObjective:
    working_ion: str
    working_ion_candidates: List[str]
    target_vector: Dict[str, float]
    target_ranges: Dict[str, Tuple[float, float]]
    target_weights: Dict[str, float]
    insertion_only: bool
    require_overall_valid: bool
    objective_mode: str
    max_candidates: int
    material_source_mode: str
    component_source_mode: str
    separator_options_count: int
    additive_options_count: int
    interpolation_enabled: bool
    extrapolation_enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "working_ion": self.working_ion,
            "working_ion_candidates": list(self.working_ion_candidates),
            "target_vector": dict(self.target_vector),
            "target_ranges": {k: [v[0], v[1]] for k, v in self.target_ranges.items()},
            "target_weights": dict(self.target_weights),
            "insertion_only": bool(self.insertion_only),
            "require_overall_valid": bool(self.require_overall_valid),
            "objective_mode": self.objective_mode,
            "max_candidates": int(self.max_candidates),
            "material_source_mode": self.material_source_mode,
            "component_source_mode": self.component_source_mode,
            "separator_options_count": int(self.separator_options_count),
            "additive_options_count": int(self.additive_options_count),
            "interpolation_enabled": bool(self.interpolation_enabled),
            "extrapolation_enabled": bool(self.extrapolation_enabled),
        }


class DiscoveryOrchestrator:
    """
    Staged generative pipeline for insertion full-cell discovery.
    """

    TARGET_FIELDS = (
        "average_voltage",
        "capacity_grav",
        "energy_grav",
        "max_delta_volume",
        "stability_charge",
        "stability_discharge",
    )
    _ION_ALIASES = {
        "li": "Li",
        "na": "Na",
        "k": "K",
        "mg": "Mg",
        "ca": "Ca",
        "zn": "Zn",
        "al": "Al",
        "y": "Y",
    }
    _EXISTING_SEPARATOR_CATALOG = {
        "Li": [
            "PP/PE trilayer separator",
            "Ceramic-coated PP/PE separator",
            "High-porosity PE separator",
        ],
        "Na": [
            "Ceramic-coated PP/PE separator",
            "Na-stable polyolefin separator",
            "Glass-fiber separator",
        ],
        "Mg": [
            "Glass-fiber separator",
            "Ceramic-coated polymer separator",
            "Microporous polyimide separator",
        ],
    }
    _EXISTING_ADDITIVE_CATALOG = {
        "Li": [
            "FEC + VC blend",
            "LiDFOB additive package",
            "PES + VC blend",
        ],
        "Na": [
            "FEC + NaDFOB blend",
            "NaFSI interphase package",
            "DTD + FEC blend",
        ],
        "Mg": [
            "Interphase stabilizer package",
            "Chelating co-solvent additive",
            "Halide-balance additive package",
        ],
    }
    _EXISTING_ELECTROLYTE_CATALOG = {
        "Li": ["LiPF6", "LiFSI", "LiBF4"],
        "Na": ["NaPF6", "NaFSI", "NaClO4"],
        "K": ["KPF6", "KFSI", "KBF4"],
        "Mg": ["Mg(TFSI)2", "Mg(BH4)2"],
        "Ca": ["Ca(TFSI)2", "Ca(BF4)2"],
        "Zn": ["ZnSO4", "Zn(TFSI)2"],
        "Al": ["AlCl4", "Al2Cl7"],
        "Y": ["YPF6", "YCl3"],
    }
    _ALLOWED_SCALERS = {
        "standard_scaler",
        "minmax_scaler",
        "robust_scaler",
        "quantile_transformer",
    }
    _REQUIRED_GRAPH_METADATA_FIELDS = (
        "dataset_fingerprint",
        "scaler_type",
        "scaler_parameters",
        "system_feature_order",
        "feature_engineering_steps",
        "clipping_rules",
        "log_transform_flags",
        "unit_definitions",
    )
    _ELECTROLYTE_FORBIDDEN_ELEMENTS = {"Ni", "Mn", "Co", "Fe", "Cu"}
    _METALLIC_ELEMENTS = {
        "Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
        "Zr", "Nb", "Mo", "W", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Sn", "Pb",
    }
    _TRANSITION_METALS = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
        "Zr", "Nb", "Mo", "W", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    }
    _ROLE_CLASS_RULES = {
        "anode": {"carbon", "alloy_metallic", "low_voltage_oxide", "conversion_material"},
        "cathode": {"transition_metal_oxide", "polyanion", "sulfide", "fluoride"},
        "electrolyte": {"organic_liquid", "polymer", "solid_li_conductor", "glass"},
    }
    _SOFT_STAGE4_REASONS = {
        "assembly_rule:anode_potential_too_high",
        "stage6:unstable_interphase_predicted",
        "stage6:additive_ionic_conductivity_drop_too_high",
        "stage8:voltage_deviation_above_limit",
        "additive_reduction_window_exceeded",
        "chemical_stability_low",
    }

    def __init__(
        self,
        *,
        material_generator: MaterialGenerator | None = None,
        chemistry_validator: ChemistryValidator | None = None,
        role_classifier: RoleClassifier | None = None,
        compatibility_model: CompatibilityModel | None = None,
        assembler: FullCellAssembler | None = None,
    ):
        self.material_generator = material_generator or MaterialGenerator()
        self.chemistry_validator = chemistry_validator or ChemistryValidator()
        self.role_classifier = role_classifier or RoleClassifier()
        self.compatibility_model = compatibility_model or CompatibilityModel()
        self.assembler = assembler or FullCellAssembler()
        self.active_learning_queue = ActiveLearningQueueService()

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return v

    @classmethod
    def _normalize_ion(cls, value: Any, fallback: str = "Li") -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return fallback
        return cls._ION_ALIASES.get(raw, str(value).strip().capitalize())

    @classmethod
    def _parse_working_ion_candidates(
        cls,
        *,
        objective: Dict[str, Any],
        base_system: BatterySystem,
        discovery_params: Optional[Dict[str, Any]],
    ) -> List[str]:
        primary = cls._normalize_ion(
            objective.get("working_ion") or base_system.working_ion or "Li",
            fallback="Li",
        )
        raw = (discovery_params or {}).get("working_ion_candidates")
        if isinstance(raw, (list, tuple)):
            values = [cls._normalize_ion(v, fallback=primary) for v in raw if str(v).strip()]
        elif isinstance(raw, str):
            values = [cls._normalize_ion(v, fallback=primary) for v in raw.split(",") if str(v).strip()]
        else:
            values = []

        normalized: list[str] = [primary]
        seen = {primary.lower()}
        for ion in values:
            key = ion.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(ion)
            if len(normalized) >= 4:
                break
        return normalized

    @staticmethod
    def _parse_source_mode(value: Any, default: str = "hybrid") -> str:
        mode = str(value or default).strip().lower()
        if mode in {"existing", "generated", "hybrid"}:
            return mode
        return default

    @staticmethod
    def _parse_option_count(value: Any, default: int = 3) -> int:
        try:
            n = int(value)
        except (TypeError, ValueError):
            return int(default)
        return int(max(1, min(6, n)))

    @staticmethod
    def _parse_bool(value: Any, default: bool = True) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _filter_material_candidates(
        self,
        candidates: List[Any],
        *,
        material_source_mode: str,
    ) -> List[Any]:
        mode = self._parse_source_mode(material_source_mode, default="hybrid")
        if mode == "hybrid":
            return list(candidates)

        filtered: list[Any] = []
        for c in candidates:
            material_id = str(getattr(c, "material_id", "") or "").strip()
            source_mode = str(getattr(c, "source_mode", "") or "").strip().lower()
            is_existing = bool(material_id) and source_mode not in {"latent_variation"}
            is_generated = (not material_id) or source_mode in {"latent_variation", "cif_conditioned_refinement"}
            if mode == "existing" and is_existing:
                filtered.append(c)
            elif mode == "generated" and is_generated:
                filtered.append(c)
        return filtered

    def _generated_component_options(
        self,
        *,
        ion: str,
        kind: str,
        target_vector: Dict[str, float],
    ) -> List[str]:
        voltage = float(target_vector.get("average_voltage", 3.6) or 3.6)
        delta = float(target_vector.get("max_delta_volume", 0.12) or 0.12)
        if kind == "separator":
            base = [
                f"Generated {ion}-ion ceramic-polymer separator (strain-tuned)",
                f"Generated {ion}-ion nano-fiber separator (target {delta:.3f} dV/V)",
                f"Generated {ion}-ion interphase-control separator (V={voltage:.2f})",
            ]
            variants = [
                f"Generated {ion}-ion porous-ceramic separator v2",
                f"Generated {ion}-ion composite membrane (high wetting)",
                f"Generated {ion}-ion low-shrink separator (V={voltage:.2f})",
            ]
            return base + variants
        base = [
            f"Generated {ion}-ion SEI stabilizer package (V={voltage:.2f})",
            f"Generated {ion}-ion expansion-buffer additive blend (dV/V<={delta:.3f})",
            f"Generated {ion}-ion interface suppressor blend",
        ]
        variants = [
            f"Generated {ion}-ion HF-scavenger additive package",
            f"Generated {ion}-ion impedance-control additive blend",
            f"Generated {ion}-ion cycle-life booster package",
        ]
        return base + variants

    def _generated_electrolyte_options(self, *, ion: str) -> List[str]:
        base = {
            "Li": ["LiPF6", "LiFSI", "LiTFSI"],
            "Na": ["NaPF6", "NaFSI", "NaTFSI"],
            "K": ["KPF6", "KFSI", "KTFSI"],
            "Mg": ["Mg(TFSI)2", "Mg(BH4)2", "MgCl2"],
            "Ca": ["Ca(TFSI)2", "Ca(BF4)2", "CaCl2"],
            "Zn": ["ZnSO4", "Zn(TFSI)2", "ZnCl2"],
            "Al": ["AlCl4", "Al2Cl7", "AlCl3"],
            "Y": ["YCl3", "YF3", "YPF6"],
        }
        return list(base.get(ion, base["Li"]))

    def _component_option_pool(
        self,
        *,
        ion: str,
        objective_cfg: DiscoveryObjective,
    ) -> Dict[str, Any]:
        source_mode = self._parse_source_mode(objective_cfg.component_source_mode, default="hybrid")
        existing_sep = list(self._EXISTING_SEPARATOR_CATALOG.get(ion, self._EXISTING_SEPARATOR_CATALOG.get("Li", [])))
        existing_add = list(self._EXISTING_ADDITIVE_CATALOG.get(ion, self._EXISTING_ADDITIVE_CATALOG.get("Li", [])))
        existing_ele = list(self._EXISTING_ELECTROLYTE_CATALOG.get(ion, self._EXISTING_ELECTROLYTE_CATALOG.get("Li", [])))
        generated_sep = self._generated_component_options(
            ion=ion,
            kind="separator",
            target_vector=objective_cfg.target_vector,
        )
        generated_add = self._generated_component_options(
            ion=ion,
            kind="additive",
            target_vector=objective_cfg.target_vector,
        )
        generated_ele = self._generated_electrolyte_options(ion=ion)

        if source_mode == "existing":
            sep_all = existing_sep
            add_all = existing_add
            ele_all = existing_ele
        elif source_mode == "generated":
            sep_all = generated_sep
            add_all = generated_add
            ele_all = generated_ele
        else:
            sep_all = []
            add_all = []
            ele_all = []
            for idx in range(max(len(existing_sep), len(generated_sep))):
                if idx < len(existing_sep):
                    sep_all.append(existing_sep[idx])
                if idx < len(generated_sep):
                    sep_all.append(generated_sep[idx])
            for idx in range(max(len(existing_add), len(generated_add))):
                if idx < len(existing_add):
                    add_all.append(existing_add[idx])
                if idx < len(generated_add):
                    add_all.append(generated_add[idx])
            for idx in range(max(len(existing_ele), len(generated_ele))):
                if idx < len(existing_ele):
                    ele_all.append(existing_ele[idx])
                if idx < len(generated_ele):
                    ele_all.append(generated_ele[idx])

        sep_count = self._parse_option_count(objective_cfg.separator_options_count, default=3)
        add_count = self._parse_option_count(objective_cfg.additive_options_count, default=3)
        sep_selected = sep_all[:sep_count] if sep_all else existing_sep[:1]
        add_selected = add_all[:add_count] if add_all else existing_add[:1]
        ele_selected = ele_all[: max(1, min(4, sep_count))] if ele_all else existing_ele[:1]
        while len(sep_selected) < sep_count:
            sep_selected.append(
                f"Generated {ion}-ion separator option {len(sep_selected) + 1}"
            )
        while len(add_selected) < add_count:
            add_selected.append(
                f"Generated {ion}-ion additive option {len(add_selected) + 1}"
            )
        if not sep_selected:
            sep_selected = ["PP/PE trilayer separator"]
        if not add_selected:
            add_selected = ["FEC + VC blend"]
        if not ele_selected:
            ele_selected = ["LiPF6"]

        return {
            "source_mode": source_mode,
            "electrolyte_options": ele_selected,
            "separator_options": sep_selected,
            "additive_options": add_selected,
            "electrolyte_option_count": len(ele_selected),
            "separator_option_count": len(sep_selected),
            "additive_option_count": len(add_selected),
            "existing_electrolyte_pool_size": len(existing_ele),
            "existing_separator_pool_size": len(existing_sep),
            "existing_additive_pool_size": len(existing_add),
            "generated_electrolyte_pool_size": len(generated_ele),
            "generated_separator_pool_size": len(generated_sep),
            "generated_additive_pool_size": len(generated_add),
        }

    def _normalize_objective(
        self,
        *,
        base_system: BatterySystem,
        objective: Dict[str, Any],
        discovery_params: Optional[Dict[str, Any]],
    ) -> DiscoveryObjective:
        target_vector: Dict[str, float] = {}
        target_ranges: Dict[str, Tuple[float, float]] = {}
        target_weights: Dict[str, float] = {}

        for field in self.TARGET_FIELDS:
            raw = objective.get(field)
            target: float | None = None
            tolerance: float | None = None
            weight: float = 1.0

            if isinstance(raw, dict):
                target = self._to_float(raw.get("target"))
                tolerance = self._to_float(raw.get("tolerance"))
                weight = self._to_float(raw.get("weight")) or 1.0
            else:
                target = self._to_float(raw)

            if target is None:
                fallback = self._to_float(getattr(base_system, field, None))
                if fallback is None:
                    continue
                target = fallback

            tol = tolerance if tolerance is not None else max(abs(target) * 0.12, 0.05)
            tol = max(0.01, float(tol))
            w = max(0.01, float(weight))

            target_vector[field] = float(target)
            if field == "max_delta_volume":
                upper = max(float(target), 0.0)
                target_ranges[field] = (0.0, upper + tol)
            else:
                target_ranges[field] = (float(target) - tol, float(target) + tol)
            target_weights[field] = w

        # Keep derived-energy objective internally consistent with the provided
        # voltage/capacity targets when energy is omitted in the request.
        raw_energy_grav = objective.get("energy_grav")
        energy_target_missing = (
            raw_energy_grav is None
            or (isinstance(raw_energy_grav, dict) and self._to_float(raw_energy_grav.get("target")) is None)
        )
        if energy_target_missing:
            avg_v_t = self._to_float(target_vector.get("average_voltage"))
            cap_g_t = self._to_float(target_vector.get("capacity_grav"))
            if avg_v_t is not None and cap_g_t is not None:
                derived_energy = float(avg_v_t * cap_g_t)
                tol = max(0.01, max(abs(derived_energy) * 0.12, 0.05))
                target_vector["energy_grav"] = derived_energy
                target_ranges["energy_grav"] = (derived_energy - tol, derived_energy + tol)
                target_weights["energy_grav"] = max(0.01, float(target_weights.get("energy_grav", 1.0)))

        working_ion_candidates = self._parse_working_ion_candidates(
            objective=objective,
            base_system=base_system,
            discovery_params=discovery_params,
        )
        working_ion = working_ion_candidates[0] if working_ion_candidates else "Li"
        insertion_only = bool(objective.get("insertion_only", True))
        require_overall_valid = bool(objective.get("require_overall_valid", True))
        objective_mode = str(objective.get("objective_mode", "weighted_mahalanobis") or "weighted_mahalanobis")
        material_source_mode = self._parse_source_mode(
            (discovery_params or {}).get("material_source_mode"),
            default="hybrid",
        )
        component_source_mode = self._parse_source_mode(
            (discovery_params or {}).get("component_source_mode"),
            default="hybrid",
        )
        separator_options_count = self._parse_option_count(
            (discovery_params or {}).get("separator_options_count"),
            default=3,
        )
        additive_options_count = self._parse_option_count(
            (discovery_params or {}).get("additive_options_count"),
            default=3,
        )
        interpolation_enabled = self._parse_bool(
            (discovery_params or {}).get("interpolation_enabled"),
            default=True,
        )
        extrapolation_enabled = self._parse_bool(
            (discovery_params or {}).get("extrapolation_enabled"),
            default=True,
        )
        if not interpolation_enabled and not extrapolation_enabled:
            interpolation_enabled = True

        max_candidates = 5
        raw_max = None
        constraints = objective.get("constraints")
        if isinstance(constraints, dict):
            raw_max = constraints.get("max_candidates")
        if raw_max is None and discovery_params:
            raw_max = discovery_params.get("top_k")
        parsed = self._to_float(raw_max)
        if parsed is not None:
            max_candidates = max(1, min(20, int(parsed)))

        if insertion_only and base_system.battery_type and str(base_system.battery_type).lower() != "insertion":
            raise ValueError("Insertion-only mode requires battery_type='insertion'")

        return DiscoveryObjective(
            working_ion=working_ion,
            working_ion_candidates=working_ion_candidates,
            target_vector=target_vector,
            target_ranges=target_ranges,
            target_weights=target_weights,
            insertion_only=insertion_only,
            require_overall_valid=require_overall_valid,
            objective_mode=objective_mode,
            max_candidates=max_candidates,
            material_source_mode=material_source_mode,
            component_source_mode=component_source_mode,
            separator_options_count=separator_options_count,
            additive_options_count=additive_options_count,
            interpolation_enabled=interpolation_enabled,
            extrapolation_enabled=extrapolation_enabled,
        )

    @staticmethod
    def _validate_graph_manifest() -> Dict[str, Any]:
        ctx = get_runtime_context()
        report: Dict[str, Any] = {
            "configured_graph_filename": GRAPH_CONFIG.filename,
            "configured_graph_dir": str(GRAPH_CONFIG.graph_dir),
            "valid": True,
            "warnings": [],
            "errors": [],
            "guardrail_flags": {
                "normalization_mismatch_flag": False,
                "redox_inversion_flag": False,
                "ontology_violation_flag": False,
                "electrolyte_class_violation_flag": False,
                "voltage_inconsistency_flag": False,
            },
            "constitution_v2": {
                "dataset_fingerprint_present": False,
                "required_metadata_fields_present": False,
                "feature_order_matchable": False,
                "scaler_type_allowed": False,
            },
        }
        if not ctx.is_ready_for_discovery():
            report["valid"] = False
            report["errors"].append("runtime_not_ready")
            report["guardrail_flags"]["normalization_mismatch_flag"] = True
            return report

        graph = ctx.get_graph()
        if not isinstance(graph, dict):
            report["valid"] = False
            report["errors"].append("graph_not_dict")
            report["guardrail_flags"]["normalization_mismatch_flag"] = True
            return report

        required = {"battery_features", "material_embeddings", "node_masks", "edge_index_dict"}
        missing = sorted(required - set(graph.keys()))
        if missing:
            report["valid"] = False
            report["errors"].append(f"graph_missing_keys:{','.join(missing)}")
            report["guardrail_flags"]["normalization_mismatch_flag"] = True

        metadata = graph.get("metadata")
        if not isinstance(metadata, dict):
            report["warnings"].append("graph_metadata_missing")
            report["errors"].append("constitution_v2:graph_metadata_missing")
            report["guardrail_flags"]["normalization_mismatch_flag"] = True
        else:
            report["metadata_version"] = metadata.get("version")
            if "version" not in metadata:
                report["warnings"].append("graph_metadata_version_missing")
            missing_meta = [
                key for key in DiscoveryOrchestrator._REQUIRED_GRAPH_METADATA_FIELDS if key not in metadata
            ]
            report["constitution_v2"]["dataset_fingerprint_present"] = bool(metadata.get("dataset_fingerprint"))
            report["constitution_v2"]["required_metadata_fields_present"] = len(missing_meta) == 0
            report["constitution_v2"]["feature_order_matchable"] = bool(
                isinstance(metadata.get("system_feature_order"), (list, tuple))
                and len(metadata.get("system_feature_order") or []) > 0
            )
            scaler_type = str(metadata.get("scaler_type", "")).strip().lower()
            report["constitution_v2"]["scaler_type_allowed"] = scaler_type in DiscoveryOrchestrator._ALLOWED_SCALERS
            if missing_meta:
                report["errors"].append(
                    "constitution_v2:missing_graph_metadata_fields:" + ",".join(sorted(missing_meta))
                )
                report["guardrail_flags"]["normalization_mismatch_flag"] = True
            if scaler_type and scaler_type not in DiscoveryOrchestrator._ALLOWED_SCALERS:
                report["errors"].append(f"constitution_v2:unsupported_scaler_type:{scaler_type}")
                report["guardrail_flags"]["normalization_mismatch_flag"] = True
            if not report["constitution_v2"]["feature_order_matchable"]:
                report["errors"].append("constitution_v2:system_feature_order_missing")
                report["guardrail_flags"]["normalization_mismatch_flag"] = True

        filename = str(GRAPH_CONFIG.filename)
        if "_v2" not in filename:
            report["warnings"].append("graph_filename_not_v2")

        if report.get("errors"):
            report["valid"] = False
        return report

    @staticmethod
    def _constitution_guardrail_flags() -> Dict[str, bool]:
        return {
            "normalization_mismatch_flag": False,
            "redox_inversion_flag": False,
            "ontology_violation_flag": False,
            "electrolyte_class_violation_flag": False,
            "voltage_inconsistency_flag": False,
        }

    def _constitution_stage0_gate(
        self,
        *,
        objective_cfg: DiscoveryObjective,
        graph_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        reasons: list[str] = []
        flags = self._constitution_guardrail_flags()

        if not bool((graph_manifest or {}).get("valid", False)):
            reasons.append("normalization:graph_training_metadata_invalid")
            flags["normalization_mismatch_flag"] = True

        target = dict(objective_cfg.target_vector or {})
        avg_v = self._to_float(target.get("average_voltage"))
        s_ch = self._to_float(target.get("stability_charge"))
        s_dis = self._to_float(target.get("stability_discharge"))
        dvol = self._to_float(target.get("max_delta_volume"))
        cap_g = self._to_float(target.get("capacity_grav"))
        cap_v = self._to_float(target.get("capacity_vol"))
        e_g = self._to_float(target.get("energy_grav"))
        e_v = self._to_float(target.get("energy_vol"))

        # Support both historical semantics:
        # - voltage-window style stability targets (~3-5 V)
        # - normalized/relative stability targets (~0-1)
        if avg_v is not None and s_ch is not None and s_ch >= 1.0 and avg_v > s_ch + 1e-8:
            reasons.append("objective_sanity:average_voltage_exceeds_stability_charge")
            flags["voltage_inconsistency_flag"] = True
        if s_ch is not None and s_dis is not None and s_ch <= s_dis + 1e-8:
            reasons.append("objective_sanity:stability_charge_not_above_stability_discharge")
            flags["voltage_inconsistency_flag"] = True
        if dvol is not None and dvol > 0.30 + 1e-8:
            reasons.append("objective_sanity:max_delta_volume_above_limit")

        if avg_v is not None and cap_g is not None and e_g is not None:
            expected = float(avg_v * cap_g)
            if abs(expected - e_g) > max(1e-6, 0.005 * max(1.0, abs(expected))):
                reasons.append("normalization:energy_grav_formula_mismatch")
                flags["normalization_mismatch_flag"] = True
        if avg_v is not None and cap_v is not None and e_v is not None:
            expected = float(avg_v * cap_v)
            if abs(expected - e_v) > max(1e-6, 0.005 * max(1.0, abs(expected))):
                reasons.append("normalization:energy_vol_formula_mismatch")
                flags["normalization_mismatch_flag"] = True

        return {
            "passed": len(reasons) == 0,
            "reasons": list(reasons),
            "units_enforced": {
                "average_voltage": "V",
                "capacity_grav": "mAh/g",
                "capacity_vol": "mAh/cm^3",
                "energy_grav": "Wh/kg",
                "energy_vol": "Wh/L",
            },
            "guardrail_flags": flags,
            "physics_first_then_scoring": True,
        }

    @classmethod
    def _electrolyte_forbidden_transition_metals(cls, electrolyte_formula: str) -> List[str]:
        text = str(electrolyte_formula or "").strip()
        if not text:
            return []
        hits = {
            el
            for el in cls._ELECTROLYTE_FORBIDDEN_ELEMENTS
            if el.lower() in text.lower()
        }
        try:
            composition = parse_formula(text)
        except Exception:
            return sorted(hits)
        elements = set(composition.keys())
        return sorted(hits | (elements & cls._ELECTROLYTE_FORBIDDEN_ELEMENTS))

    @classmethod
    def _classify_material_ontology(cls, formula: str) -> str | None:
        text = str(formula or "").strip()
        if not text:
            return None
        try:
            composition = parse_formula(text)
        except Exception:
            return None
        elements = set(composition.keys())
        if not elements:
            return None

        if elements == {"C"}:
            return "carbon"
        if any(el in cls._TRANSITION_METALS for el in elements) and "O" not in elements and any(
            el in elements for el in {"F", "S", "Cl", "Br", "I"}
        ):
            return "conversion_material"
        charge_carriers = {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y"}
        if (
            any(el in charge_carriers for el in elements)
            and any(el in {"F", "Cl", "Br", "I", "S", "P"} for el in elements)
            and not any(el in cls._TRANSITION_METALS for el in elements)
        ):
            return "solid_li_conductor"
        if "F" in elements:
            return "fluoride"
        if "S" in elements and "O" not in elements:
            return "sulfide"
        if "O" in elements and "P" in elements:
            return "polyanion"
        if "O" in elements and any(el in cls._TRANSITION_METALS for el in elements):
            if any(el in {"Ti", "Nb", "Mo", "W"} for el in elements):
                return "low_voltage_oxide"
            return "transition_metal_oxide"

        non_metal_like = {"O", "F", "Cl", "Br", "I", "P", "S", "N", "C", "H", "Si", "B"}
        if elements.issubset(non_metal_like):
            if "C" in elements and ("H" in elements or "O" in elements):
                return "organic_liquid"
            if "Si" in elements or "B" in elements:
                return "glass"
            return "polymer"

        if elements.issubset(cls._METALLIC_ELEMENTS):
            return "alloy_metallic"
        if any(el in cls._TRANSITION_METALS for el in elements):
            return "alloy_metallic"
        return None

    def _apply_stage1b_ontology_gate(self, candidates: List[Any]) -> tuple[List[Any], Dict[str, Any]]:
        class_counts: Dict[str, int] = {}
        rejected = 0
        out: list[Any] = []
        for c in candidates:
            formula = str(getattr(c, "framework_formula", "") or "")
            ontology_class = self._classify_material_ontology(formula)
            if ontology_class is None:
                rejected += 1
                meta = dict(getattr(c, "metadata", {}) or {})
                meta["ontology_class"] = None
                meta["ontology_gate_reason"] = "unclassified_material"
                c.metadata = meta
                continue
            class_counts[ontology_class] = class_counts.get(ontology_class, 0) + 1
            meta = dict(getattr(c, "metadata", {}) or {})
            meta["ontology_class"] = ontology_class
            c.metadata = meta
            out.append(c)
        report = {
            "classification_required": True,
            "allowed_classes": [
                "transition_metal_oxide",
                "polyanion",
                "sulfide",
                "fluoride",
                "carbon",
                "alloy_metallic",
                "solid_li_conductor",
                "polymer",
                "organic_liquid",
                "glass",
                "inert_separator_material",
            ],
            "reject_if_unclassified": True,
            "classified_count": len(out),
            "rejected_unclassified_count": int(rejected),
            "class_counts": dict(sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)),
        }
        return out, report

    def _apply_role_class_gate(
        self,
        candidates: List[Any],
        *,
        role: str,
    ) -> tuple[List[Any], Dict[str, Any]]:
        allowed = set(self._ROLE_CLASS_RULES.get(str(role), set()) or set())
        kept: list[Any] = []
        rejected_reason_counts: Dict[str, int] = {}
        for c in candidates:
            meta = dict(getattr(c, "metadata", {}) or {})
            ontology_class = str(meta.get("ontology_class") or "").strip()
            if not ontology_class:
                ontology_class = str(self._classify_material_ontology(str(getattr(c, "framework_formula", "") or "")) or "")
                meta["ontology_class"] = ontology_class or None
                c.metadata = meta
            if ontology_class in allowed:
                kept.append(c)
                continue
            reason = f"{role}_role_class_mismatch:{ontology_class or 'unknown'}"
            rejected_reason_counts[reason] = int(rejected_reason_counts.get(reason, 0) + 1)
            meta["role_class_gate_reason"] = reason
            c.metadata = meta
        report = {
            "role": str(role),
            "allowed_classes": sorted(allowed),
            "input_count": int(len(candidates)),
            "kept_count": int(len(kept)),
            "rejected_count": int(max(0, len(candidates) - len(kept))),
            "rejection_reason_counts": dict(sorted(rejected_reason_counts.items(), key=lambda kv: kv[1], reverse=True)),
        }
        return kept, report

    def _recover_anode_candidates(
        self,
        candidates: List[Any],
        *,
        limit: int = 3,
    ) -> List[Any]:
        if not candidates:
            return []
        keep_n = max(1, min(int(limit), len(candidates)))
        class_priority = {
            "low_voltage_oxide": 0,
            "carbon": 1,
            "alloy_metallic": 2,
            "conversion_material": 3,
            "polyanion": 4,
            "transition_metal_oxide": 5,
        }

        def _sort_key(c: Any) -> Tuple[int, float, str]:
            meta = dict(getattr(c, "metadata", {}) or {})
            cls_name = str(meta.get("ontology_class") or "").strip()
            prio = int(class_priority.get(cls_name, 9))
            ref_v = self._to_float(meta.get("reference_voltage"))
            if ref_v is None:
                ref_v = 9.9
            formula = str(getattr(c, "framework_formula", "") or "")
            return (prio, float(ref_v), formula)

        selected = sorted(list(candidates), key=_sort_key)[:keep_n]
        for c in selected:
            meta = dict(getattr(c, "metadata", {}) or {})
            meta["role_class_gate_fallback"] = True
            meta["role_class_gate_reason"] = "anode_pool_recovered_from_pre_role_gate"
            c.metadata = meta
        return selected

    @staticmethod
    def _role_locked_assignment(candidate_id: str, *, role: str) -> RoleAssignment:
        if role == "cathode":
            probs = {"cathode": 1.0, "anode": 0.0, "electrolyte_candidate": 0.0}
            selected = ["cathode"]
        elif role == "anode":
            probs = {"cathode": 0.0, "anode": 1.0, "electrolyte_candidate": 0.0}
            selected = ["anode"]
        else:
            probs = {"cathode": 0.0, "anode": 0.0, "electrolyte_candidate": 1.0}
            selected = ["electrolyte_candidate"]
        return RoleAssignment(
            candidate_id=str(candidate_id),
            role_probabilities=probs,
            confidence_score=1.0,
            selected_roles=selected,
            used_fallback=False,
        )

    def _apply_stage3_component_ontology_gate(
        self,
        compatibility_records: List[CompatibilityRecord],
    ) -> tuple[List[CompatibilityRecord], Dict[str, Any]]:
        role_rules = self._ROLE_CLASS_RULES
        violation_counts: Dict[str, int] = {}
        kept: list[CompatibilityRecord] = []
        for rec in compatibility_records:
            cath_cls = str((getattr(rec.cathode, "metadata", {}) or {}).get("ontology_class") or "")
            an_cls = str((getattr(rec.anode, "metadata", {}) or {}).get("ontology_class") or "")
            el_cls = str((getattr(rec.electrolyte, "metadata", {}) or {}).get("ontology_class") or "")

            local_reasons: list[str] = []
            if cath_cls not in role_rules["cathode"]:
                local_reasons.append(f"cathode_role_class_mismatch:{cath_cls or 'unknown'}")
            if an_cls not in role_rules["anode"]:
                local_reasons.append(f"anode_role_class_mismatch:{an_cls or 'unknown'}")

            forbidden_electrolyte_elements = self._electrolyte_forbidden_transition_metals(
                str(getattr(rec.electrolyte, "framework_formula", "") or "")
            )
            if forbidden_electrolyte_elements:
                for el in forbidden_electrolyte_elements:
                    local_reasons.append(f"electrolyte_forbidden_element:{el}")
            if el_cls and el_cls not in role_rules["electrolyte"]:
                local_reasons.append(f"electrolyte_role_class_mismatch:{el_cls}")

            if local_reasons:
                for reason in local_reasons:
                    violation_counts[reason] = int(violation_counts.get(reason, 0) + 1)
                continue
            kept.append(rec)

        report = {
            "description": "component_ontology_and_constraints",
            "role_class_matrix_enforced": True,
            "electrolyte_forbidden_elements": sorted(self._ELECTROLYTE_FORBIDDEN_ELEMENTS),
            "reject_if_role_class_mismatch": True,
            "input_records": int(len(compatibility_records)),
            "records_after_gate": int(len(kept)),
            "rejected_count": int(max(0, len(compatibility_records) - len(kept))),
            "rejection_reason_counts": dict(sorted(violation_counts.items(), key=lambda kv: kv[1], reverse=True)),
        }
        if kept:
            return kept, report
        # Keep pipeline alive in low-data regimes while still surfacing hard gate violations.
        report["fallback_preserved_records"] = True
        return compatibility_records, report

    @staticmethod
    def _system_value(system: BatterySystem, field: str) -> float | None:
        if system.cell_level and field in system.cell_level and system.cell_level[field] is not None:
            return float(system.cell_level[field])
        raw = getattr(system, field, None)
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def _objective_distance(self, system: BatterySystem, objective: DiscoveryObjective) -> float:
        accum = 0.0
        weight_sum = 0.0
        for field, target in objective.target_vector.items():
            value = self._system_value(system, field)
            if value is None:
                continue
            low, high = objective.target_ranges[field]
            if field == "max_delta_volume":
                upper = max(float(target), 0.0)
                tol = max(1e-6, high - upper)
                z = 0.0 if value <= upper else (value - upper) / tol
            else:
                tol = max(1e-6, 0.5 * (high - low))
                z = (value - target) / tol
            weight = max(0.01, objective.target_weights.get(field, 1.0))
            accum += weight * z * z
            weight_sum += weight
        if weight_sum <= 0:
            return 1.0
        return math.sqrt(accum / weight_sum)

    def _env_weight(self, name: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
        raw = self._to_float(os.getenv(name))
        value = float(default if raw is None else raw)
        return float(max(lo, min(hi, value)))

    def _pareto_weights(self) -> Tuple[float, float, float]:
        w_obj = self._env_weight("ECESSP_PARETO_OBJECTIVE_WEIGHT", 0.45)
        w_feas = self._env_weight("ECESSP_PARETO_FEASIBILITY_WEIGHT", 0.30)
        w_unc = self._env_weight("ECESSP_PARETO_UNCERTAINTY_WEIGHT", 0.25)
        total = w_obj + w_feas + w_unc
        if total <= 1e-9:
            return (0.50, 0.35, 0.15)
        return (w_obj / total, w_feas / total, w_unc / total)

    def _hgt_pareto_weight(self) -> float:
        return self._env_weight("ECESSP_PARETO_HGT_WEIGHT", 0.10, lo=0.0, hi=0.50)

    def _role_head_min_confidence(self) -> float:
        return self._env_weight("ECESSP_ROLE_HEAD_MIN_CONFIDENCE", 0.55)

    def _role_head_min_margin(self) -> float:
        return self._env_weight("ECESSP_ROLE_HEAD_MIN_MARGIN", 0.05)

    def _compatibility_head_min_score(self) -> float:
        return self._env_weight("ECESSP_COMPAT_HEAD_MIN_SCORE", 0.42)

    def _uncertainty_model_weight(self) -> float:
        return self._env_weight("ECESSP_UNCERTAINTY_MODEL_WEIGHT", 0.65)

    def _objective_hit_threshold(self) -> float:
        return self._env_weight("ECESSP_OBJECTIVE_HIT_THRESHOLD", 0.17)

    def _feasibility_hit_threshold(self) -> float:
        return self._env_weight("ECESSP_FEASIBILITY_HIT_THRESHOLD", 0.70)

    @staticmethod
    def _normalize_probabilities(raw: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, float(v)) for v in raw.values())
        if total <= 0.0:
            return {"cathode": 1.0 / 3.0, "anode": 1.0 / 3.0, "electrolyte_candidate": 1.0 / 3.0}
        return {k: max(0.0, float(v)) / total for k, v in raw.items()}

    @staticmethod
    def _top_margin(raw: Dict[str, float]) -> float:
        values = sorted((max(0.0, min(1.0, float(v))) for v in raw.values()), reverse=True)
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        return float(values[0] - values[1])

    @classmethod
    def _role_head_gate_decision(
        cls,
        *,
        model_role_probs: Dict[str, float],
        min_confidence: float,
        min_margin: float,
    ) -> Tuple[bool, Dict[str, float], float, float, str]:
        normalized = cls._normalize_probabilities(model_role_probs)
        confidence = float(max(normalized.values())) if normalized else 0.0
        margin = cls._top_margin(normalized)
        if confidence < float(min_confidence):
            return (False, normalized, confidence, margin, "confidence_below_threshold")
        if margin < float(min_margin):
            return (False, normalized, confidence, margin, "margin_below_threshold")
        return (True, normalized, confidence, margin, "model_head_selected")

    @classmethod
    def _compatibility_head_gate_decision(
        cls,
        *,
        model_compatibility: float | None,
        deterministic_score: float,
        min_score: float,
    ) -> Dict[str, Any]:
        rule_score = float(max(0.0, min(1.0, deterministic_score)))
        if model_compatibility is None:
            return {
                "compatibility_score": rule_score,
                "compatibility_source": "deterministic_proxy",
                "compatibility_fallback_reason": "model_head_unavailable",
                "compatibility_model_score": None,
                "compatibility_rule_score": rule_score,
                "compatibility_head_min_score": float(min_score),
            }
        model_score = float(max(0.0, min(1.0, model_compatibility)))
        if model_score < float(min_score):
            return {
                "compatibility_score": rule_score,
                "compatibility_source": "deterministic_proxy",
                "compatibility_fallback_reason": "model_score_below_threshold",
                "compatibility_model_score": model_score,
                "compatibility_rule_score": rule_score,
                "compatibility_head_min_score": float(min_score),
            }
        return {
            "compatibility_score": model_score,
            "compatibility_source": "model_head",
            "compatibility_fallback_reason": None,
            "compatibility_model_score": model_score,
            "compatibility_rule_score": rule_score,
            "compatibility_head_min_score": float(min_score),
        }

    @staticmethod
    def _build_role_probe_system(
        *,
        candidate_formula: str,
        objective_cfg: DiscoveryObjective,
        base_system: BatterySystem,
        candidate_id: str,
    ) -> BatterySystem:
        return BatterySystem(
            battery_id=f"role_probe_{candidate_id}",
            provenance="generated",
            parent_battery_id=base_system.battery_id,
            battery_type="insertion",
            working_ion=objective_cfg.working_ion,
            framework_formula=str(candidate_formula or base_system.framework_formula or ""),
            battery_formula=str(candidate_formula or base_system.framework_formula or ""),
            cathode_material=str(candidate_formula or base_system.cathode_material or "unknown_host"),
            anode_material=str(base_system.anode_material or "Graphite"),
            electrolyte=str(base_system.electrolyte or f"{objective_cfg.working_ion}PF6"),
            average_voltage=float(objective_cfg.target_vector.get("average_voltage", 3.6)),
            capacity_grav=float(objective_cfg.target_vector.get("capacity_grav", 180.0)),
            energy_grav=float(objective_cfg.target_vector.get("energy_grav", 620.0)),
            max_delta_volume=float(objective_cfg.target_vector.get("max_delta_volume", 0.12)),
            stability_charge=float(objective_cfg.target_vector.get("stability_charge", 0.0)),
            stability_discharge=float(objective_cfg.target_vector.get("stability_discharge", 0.0)),
        )

    def _apply_role_head_assignments_with_fallback(
        self,
        *,
        assignments: Dict[str, RoleAssignment],
        candidates: List[Any],
        objective_cfg: DiscoveryObjective,
        base_system: BatterySystem,
        inference_engine: Any,
    ) -> Dict[str, Any]:
        max_probes = max(0, min(200, int(self._to_float(os.getenv("ECESSP_ROLE_HEAD_PROBE_LIMIT", "80")) or 80)))
        min_confidence = self._role_head_min_confidence()
        min_margin = self._role_head_min_margin()
        probed = 0
        selected_by_model = 0
        fallback_reasons: Dict[str, int] = {}

        for candidate in candidates:
            candidate_id = str(getattr(candidate, "candidate_id", "") or "")
            assignment = assignments.get(candidate_id)
            if assignment is None:
                continue
            fallback_reason = "model_head_unavailable"
            model_probs: Dict[str, float] = {}
            model_confidence = None
            model_margin = None
            used_model_head = False

            if probed >= max_probes:
                fallback_reason = "probe_limit_reached"
            else:
                probe = self._build_role_probe_system(
                    candidate_formula=str(getattr(candidate, "framework_formula", "") or ""),
                    objective_cfg=objective_cfg,
                    base_system=base_system,
                    candidate_id=candidate_id,
                )
                probed += 1
                try:
                    infer_out = inference_engine.infer(probe)
                except Exception:
                    fallback_reason = "inference_failed"
                else:
                    aux = infer_out.get("auxiliary_heads", {})
                    model_role_probs = aux.get("role_probabilities", {})
                    if isinstance(model_role_probs, dict) and model_role_probs:
                        (
                            accepted,
                            model_probs,
                            model_confidence,
                            model_margin,
                            decision_reason,
                        ) = self._role_head_gate_decision(
                            model_role_probs=model_role_probs,
                            min_confidence=min_confidence,
                            min_margin=min_margin,
                        )
                        fallback_reason = decision_reason
                        if accepted:
                            selected = self.role_classifier._selected_roles(model_probs)
                            assignments[candidate_id] = RoleAssignment(
                                candidate_id=candidate_id,
                                role_probabilities=model_probs,
                                confidence_score=float(model_confidence),
                                selected_roles=selected,
                                used_fallback=False,
                            )
                            selected_by_model += 1
                            used_model_head = True
                    else:
                        fallback_reason = "model_head_unavailable"

            resolved_assignment = assignments.get(candidate_id, assignment)
            candidate.metadata = dict(getattr(candidate, "metadata", {}) or {})
            candidate.metadata["role_assignment"] = resolved_assignment.to_dict()
            candidate.metadata["role_assignment_source"] = (
                "model_head"
                if used_model_head
                else ("deterministic_fallback" if resolved_assignment.used_fallback else "deterministic_primary")
            )
            candidate.metadata["role_assignment_model_head"] = {
                "used_model_head": bool(used_model_head),
                "fallback_reason": None if used_model_head else str(fallback_reason),
                "min_confidence_threshold": float(min_confidence),
                "min_margin_threshold": float(min_margin),
                "model_role_probabilities": dict(model_probs),
                "model_confidence_score": (
                    float(model_confidence) if model_confidence is not None else None
                ),
                "model_margin_score": float(model_margin) if model_margin is not None else None,
            }
            reason_key = "model_head_selected" if used_model_head else str(fallback_reason)
            fallback_reasons[reason_key] = int(fallback_reasons.get(reason_key, 0) + 1)

        deterministic_fallback_count = sum(1 for a in assignments.values() if a.used_fallback)
        deterministic_primary_count = max(
            0,
            len(assignments) - int(selected_by_model) - int(deterministic_fallback_count),
        )
        return {
            "probed_candidates": int(probed),
            "model_head_selected_assignments": int(selected_by_model),
            "deterministic_primary_assignments": int(deterministic_primary_count),
            "deterministic_fallback_assignments": int(deterministic_fallback_count),
            "probe_limit": int(max_probes),
            "min_confidence_threshold": float(min_confidence),
            "min_margin_threshold": float(min_margin),
            "decision_counts": dict(fallback_reasons),
        }

    @staticmethod
    def _compute_feasibility_score(constraints: Dict[str, Any], compatibility_score: float) -> float:
        physical_valid = bool((constraints.get("physical") or {}).get("valid", False))
        chemical_valid = bool((constraints.get("chemical") or {}).get("valid", False))
        physics_hard_valid = bool((constraints.get("physics_first") or {}).get("hard_valid", False))
        score = (
            0.40 * float(max(0.0, min(1.0, compatibility_score)))
            + 0.20 * float(1.0 if physical_valid else 0.0)
            + 0.20 * float(1.0 if chemical_valid else 0.0)
            + 0.20 * float(1.0 if physics_hard_valid else 0.0)
        )
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _is_stage_valid_candidate(
        *,
        assembled_valid: bool,
        overall_valid: bool,
        require_overall_valid: bool,
    ) -> bool:
        # Safety invariant: never return candidates that fail overall hard validity.
        if not bool(overall_valid):
            return False
        if require_overall_valid and not bool(overall_valid):
            return False
        return bool(assembled_valid and overall_valid)

    @classmethod
    def _is_soft_stage4_only_failure(cls, reasons: List[str]) -> bool:
        normalized = [str(r or "").strip() for r in list(reasons or []) if str(r or "").strip()]
        if not normalized:
            return False
        for reason in normalized:
            if reason in cls._SOFT_STAGE4_REASONS:
                continue
            return False
        return True

    @staticmethod
    def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        a_obj = float(a.get("objective_alignment_score", 0.0) or 0.0)
        b_obj = float(b.get("objective_alignment_score", 0.0) or 0.0)
        a_feas = float(a.get("feasibility_score", 0.0) or 0.0)
        b_feas = float(b.get("feasibility_score", 0.0) or 0.0)
        a_unc = float(a.get("uncertainty_penalty", 1.0) or 1.0)
        b_unc = float(b.get("uncertainty_penalty", 1.0) or 1.0)
        a_hgt = float(a.get("hgt_rerank_score", 0.0) or 0.0)
        b_hgt = float(b.get("hgt_rerank_score", 0.0) or 0.0)
        non_worse = (a_obj >= b_obj) and (a_feas >= b_feas) and (a_unc <= b_unc) and (a_hgt >= b_hgt)
        strictly_better = (a_obj > b_obj) or (a_feas > b_feas) or (a_unc < b_unc) or (a_hgt > b_hgt)
        return bool(non_worse and strictly_better)

    def _apply_pareto_ranking(self, candidate_records: List[dict]) -> List[dict]:
        if not candidate_records:
            return []

        w_obj, w_feas, w_unc = self._pareto_weights()
        w_hgt = self._hgt_pareto_weight()
        total = max(1e-9, w_obj + w_feas + w_unc + w_hgt)
        w_obj = float(w_obj / total)
        w_feas = float(w_feas / total)
        w_unc = float(w_unc / total)
        w_hgt = float(w_hgt / total)
        for item in candidate_records:
            obj = float(item.get("objective_alignment_score", 0.0) or 0.0)
            feas = float(item.get("feasibility_score", 0.0) or 0.0)
            unc = float(item.get("uncertainty_penalty", 1.0) or 1.0)
            hgt = float(item.get("hgt_rerank_score", 0.0) or 0.0)
            pareto_score = w_obj * obj + w_feas * feas + w_unc * (1.0 - unc) + w_hgt * hgt
            item["pareto_score"] = float(max(0.0, min(1.0, pareto_score)))
            item["pareto_weights"] = {
                "objective": float(w_obj),
                "feasibility": float(w_feas),
                "uncertainty": float(w_unc),
                "hgt_rerank": float(w_hgt),
            }

        remaining = list(range(len(candidate_records)))
        rank = 0
        while remaining:
            front: list[int] = []
            for idx in remaining:
                dominated = False
                for other in remaining:
                    if idx == other:
                        continue
                    if self._dominates(candidate_records[other], candidate_records[idx]):
                        dominated = True
                        break
                if not dominated:
                    front.append(idx)
            for idx in front:
                candidate_records[idx]["pareto_rank"] = int(rank)
                candidate_records[idx]["score"] = round(
                    float(candidate_records[idx].get("pareto_score", 0.0)),
                    6,
                )
            remaining = [idx for idx in remaining if idx not in front]
            rank += 1

        candidate_records.sort(
            key=lambda item: (
                int(item.get("pareto_rank", 10**9)),
                -float(item.get("score", 0.0)),
                -float(item.get("hgt_rerank_score", 0.0)),
                -float(item.get("compatibility_score", 0.0)),
                -float(item.get("objective_alignment_score", 0.0)),
            )
        )
        return candidate_records

    @staticmethod
    def _corr(xs: List[float], ys: List[float]) -> float:
        if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
            return 0.0
        try:
            mx = statistics.mean(xs)
            my = statistics.mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
            deny = math.sqrt(sum((y - my) ** 2 for y in ys))
            if denx <= 1e-12 or deny <= 1e-12:
                return 0.0
            return float(num / (denx * deny))
        except Exception:
            return 0.0

    def _build_discovery_report_card(self, candidate_records: List[dict], top_k: int = 5) -> Dict[str, float]:
        if not candidate_records:
            return {
                "hit_rate_at_k": 0.0,
                "constraint_validity_rate": 0.0,
                "physics_violation_rate": 0.0,
                "novelty_mean": 0.0,
                "novelty_p90": 0.0,
                "uncertainty_calibration_corr": 0.0,
                "ood_acceptance_rate": 0.0,
            }

        k = max(1, min(int(top_k), len(candidate_records)))
        top = candidate_records[:k]

        hits = 0
        validity_vals: List[float] = []
        physics_violations = 0
        novelty_vals: List[float] = []
        unc_vals: List[float] = []
        err_vals: List[float] = []
        ood_vals: List[float] = []

        for item in top:
            obj = float(item.get("objective_alignment_score", 0.0) or 0.0)
            feas = float(item.get("feasibility_score", 0.0) or 0.0)
            valid = bool(item.get("valid", False))
            if obj >= self._objective_hit_threshold() and feas >= self._feasibility_hit_threshold() and valid:
                hits += 1
            validity_vals.append(1.0 if valid else 0.0)

            stage_trace = item.get("stage_trace", {})
            constraints = stage_trace.get("constraints", {}) if isinstance(stage_trace, dict) else {}
            physics_first = constraints.get("physics_first", {}) if isinstance(constraints, dict) else {}
            if isinstance(physics_first, dict) and not bool(physics_first.get("hard_valid", True)):
                physics_violations += 1

            novelty_vals.append(float(item.get("material_novelty_score", 0.0) or 0.0))
            unc = float(item.get("uncertainty_penalty", 1.0) or 1.0)
            unc_vals.append(unc)
            err_vals.append(1.0 - obj)
            ood_vals.append(1.0 - unc)

        sorted_novelty = sorted(novelty_vals)
        p90_idx = min(len(sorted_novelty) - 1, int(math.ceil(0.9 * len(sorted_novelty))) - 1)

        return {
            "hit_rate_at_k": float(hits / max(1, k)),
            "constraint_validity_rate": float(sum(validity_vals) / max(1, len(validity_vals))),
            "physics_violation_rate": float(physics_violations / max(1, len(top))),
            "novelty_mean": float(sum(novelty_vals) / max(1, len(novelty_vals))),
            "novelty_p90": float(sorted_novelty[p90_idx]) if sorted_novelty else 0.0,
            "uncertainty_calibration_corr": float(self._corr(unc_vals, err_vals)),
            "ood_acceptance_rate": float(sum(ood_vals) / max(1, len(ood_vals))),
        }

    @staticmethod
    def _active_learning_enabled() -> bool:
        raw = str(os.getenv("ECESSP_ACTIVE_LEARNING_QUEUE", "1")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _queue_active_learning_candidates(
        self,
        *,
        candidate_records: List[dict],
        objective_cfg: DiscoveryObjective,
    ) -> Dict[str, Any]:
        if not self._active_learning_enabled() or not candidate_records:
            return {
                "enabled": self._active_learning_enabled(),
                "queued_count": 0,
                "selected_count": 0,
            }

        selected: list[dict] = []
        for item in candidate_records:
            uncertainty = float(
                item.get(
                    "uncertainty_penalty",
                    item.get("material_uncertainty_proxy", 0.0),
                )
                or 0.0
            )
            objective_alignment = float(item.get("objective_alignment_score", 0.0) or 0.0)
            compatibility_score = float(item.get("compatibility_score", 0.0) or 0.0)
            acquisition = 0.50 * uncertainty + 0.35 * objective_alignment + 0.15 * compatibility_score
            if uncertainty >= 0.15 or objective_alignment >= 0.65:
                selected.append(
                    {
                        "candidate_id": str(getattr(item.get("system"), "battery_id", "")),
                        "score": float(item.get("score", 0.0) or 0.0),
                        "objective_alignment_score": objective_alignment,
                        "compatibility_score": compatibility_score,
                        "material_uncertainty_proxy": uncertainty,
                        "acquisition_score": float(acquisition),
                        "system": item.get("system").to_dict() if hasattr(item.get("system"), "to_dict") else {},
                        "stage_trace": item.get("stage_trace", {}),
                    }
                )

        selected.sort(key=lambda x: float(x.get("acquisition_score", 0.0)), reverse=True)
        selected = selected[:5] if selected else []
        if not selected and candidate_records:
            top = candidate_records[0]
            selected = [
                {
                    "candidate_id": str(getattr(top.get("system"), "battery_id", "")),
                    "score": float(top.get("score", 0.0) or 0.0),
                    "objective_alignment_score": float(top.get("objective_alignment_score", 0.0) or 0.0),
                    "compatibility_score": float(top.get("compatibility_score", 0.0) or 0.0),
                    "material_uncertainty_proxy": float(top.get("material_uncertainty_proxy", 0.0) or 0.0),
                    "acquisition_score": float(top.get("score", 0.0) or 0.0),
                    "system": top.get("system").to_dict() if hasattr(top.get("system"), "to_dict") else {},
                    "stage_trace": top.get("stage_trace", {}),
                }
            ]

        queued = self.active_learning_queue.enqueue(
            items=selected,
            context={
                "mode": "generative",
                "working_ion": objective_cfg.working_ion,
                "objective": objective_cfg.to_dict(),
                "reason": "uncertainty_objective_acquisition",
            },
        )
        return {
            "enabled": True,
            "queued_count": int(queued),
            "selected_count": int(len(selected)),
        }

    @staticmethod
    def _stabilize_system_predictions(system: BatterySystem) -> None:
        """
        Clamp post-inference values to physically plausible ranges before hard-gate checks.
        """
        ion = str(system.working_ion or "").strip().lower()
        max_voltage = 4.2 if ion in {"li", ""} else 4.0

        def clamp(value: Any, lo: float, hi: float) -> float | None:
            if value is None:
                return None
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return max(lo, min(hi, out))

        system.average_voltage = clamp(system.average_voltage, 1.0, max_voltage)
        system.capacity_grav = clamp(system.capacity_grav, 20.0, 320.0)
        system.capacity_vol = clamp(system.capacity_vol, 60.0, 1200.0)
        system.max_delta_volume = clamp(system.max_delta_volume, 0.0, 0.24)
        system.stability_charge = clamp(system.stability_charge, -0.4, 0.4)
        system.stability_discharge = clamp(system.stability_discharge, -0.4, 0.4)

        if system.average_voltage is not None and system.capacity_grav is not None:
            system.energy_grav = min(440.0, float(system.average_voltage) * float(system.capacity_grav))
        if system.average_voltage is not None and system.capacity_vol is not None:
            system.energy_vol = min(1180.0, float(system.average_voltage) * float(system.capacity_vol))

        if isinstance(system.cell_level, dict):
            system.cell_level["capacity_grav"] = system.capacity_grav
            system.cell_level["capacity_vol"] = system.capacity_vol
            system.cell_level["energy_grav"] = system.energy_grav
            system.cell_level["energy_vol"] = system.energy_vol
        if isinstance(system.material_level, dict):
            system.material_level["capacity_grav"] = system.capacity_grav
            system.material_level["capacity_vol"] = system.capacity_vol
            system.material_level["energy_grav"] = system.energy_grav
            system.material_level["energy_vol"] = system.energy_vol

    def run_generative(
        self,
        *,
        base_system: BatterySystem,
        objective: Dict[str, Any],
        discovery_params: Optional[Dict[str, Any]] = None,
        candidate_pool_size: int = 150,
    ) -> Dict[str, Any]:
        objective_cfg = self._normalize_objective(
            base_system=base_system,
            objective=objective,
            discovery_params=discovery_params,
        )
        compatibility_fallback_reentry = bool(
            (discovery_params or {}).get("_compatibility_fallback_reentry")
        )
        graph_manifest = self._validate_graph_manifest()

        pool_size = max(10, min(400, int(candidate_pool_size)))
        if discovery_params and self._to_float(discovery_params.get("num_candidates")) is not None:
            pool_size = max(10, min(400, int(float(discovery_params["num_candidates"]))))

        stage_metrics: Dict[str, Any] = {
            "stage_0": {"objective": objective_cfg.to_dict()},
            "stage_1": {},
            "stage_1b": {},
            "stage_2": {},
            "stage_3": {},
            "stage_4": {},
            "stage_5": {},
            "stage_6": {},
            "stage_7": {},
            "stage_8": {},
            "stage_9": {},
            "stage_6_7": {},
            "stage_10": {},
            "guardrail_flags": self._constitution_guardrail_flags(),
        }
        stage0_constitution = self._constitution_stage0_gate(
            objective_cfg=objective_cfg,
            graph_manifest=graph_manifest,
        )
        stage_metrics["stage_0"]["constitution_v2"] = stage0_constitution
        if isinstance(graph_manifest, dict):
            stage_metrics["stage_0"]["graph_manifest_validation"] = graph_manifest
        if isinstance(stage0_constitution.get("guardrail_flags"), dict):
            stage_metrics["guardrail_flags"].update(stage0_constitution["guardrail_flags"])
        if not bool(stage0_constitution.get("passed", False)):
            stage_metrics["stage_10"] = {
                "pareto_ranking_only_if_all_passed": True,
                "uncertainty_cannot_override_invariant_failure": True,
                "aborted_before_generation": True,
            }
            return {
                "best_system": base_system,
                "candidate_records": [],
                "discovery_report_card": stage_metrics,
                "target_objectives_feasible": dict(objective_cfg.target_vector),
                "objective_feasibility_adjustments": [],
                "latent_generation_stats": {
                    "ranked_systems_count": 0,
                    "latent_generated_count": 0,
                    "synthesized_generated_count": 0,
                    "selected_latent_count": 0,
                },
                "graph_manifest_validation": graph_manifest,
                "active_learning_queue": {
                    "enabled": self._active_learning_enabled(),
                    "queued_count": 0,
                    "selected_count": 0,
                },
            }

        ion_scope = list(objective_cfg.working_ion_candidates or [objective_cfg.working_ion])
        component_option_pools: Dict[str, Dict[str, Any]] = {
            ion: self._component_option_pool(ion=ion, objective_cfg=objective_cfg)
            for ion in ion_scope
        }
        per_ion_pool = max(10, int(math.ceil(pool_size / max(1, len(ion_scope)))))

        raw_component_batches: Dict[str, List[MaterialCandidate]] = {
            "cathode": [],
            "anode": [],
            "electrolyte": [],
        }
        generation_counts_by_component_ion: Dict[str, Dict[str, int]] = {
            "cathode": {},
            "anode": {},
            "electrolyte": {},
            "separator": {},
            "additive": {},
        }

        def _dedupe_candidates(pool: List[MaterialCandidate]) -> List[MaterialCandidate]:
            out: list[MaterialCandidate] = []
            seen: set[str] = set()
            for candidate in pool:
                signature = "|".join(
                    [
                        str(getattr(candidate, "working_ion", "") or "").lower(),
                        str(getattr(candidate, "framework_formula", "") or "").lower(),
                        str(getattr(candidate, "source_mode", "") or "").lower(),
                        str(((getattr(candidate, "metadata", {}) or {}).get("component_class", "") or "")).lower(),
                    ]
                )
                if signature in seen:
                    continue
                seen.add(signature)
                out.append(candidate)
            return out

        for ion in ion_scope:
            cath_batch = self.material_generator.generate_candidates(
                working_ion=ion,
                target_property_vector=objective_cfg.target_vector,
                optional_seed_structure=base_system.framework_formula,
                candidate_pool_size=per_ion_pool,
                interpolation_enabled=objective_cfg.interpolation_enabled,
                extrapolation_enabled=objective_cfg.extrapolation_enabled,
                role_condition="cathode",
            )
            an_batch = self.material_generator.generate_candidates(
                working_ion=ion,
                target_property_vector=objective_cfg.target_vector,
                optional_seed_structure=base_system.framework_formula,
                candidate_pool_size=per_ion_pool,
                interpolation_enabled=objective_cfg.interpolation_enabled,
                extrapolation_enabled=objective_cfg.extrapolation_enabled,
                role_condition="anode",
            )
            option_pool = component_option_pools.get(ion, {})
            ele_options = list(option_pool.get("electrolyte_options") or [])
            ele_batch: list[MaterialCandidate] = []
            for idx, ele in enumerate(ele_options):
                formula = str(ele or "").strip()
                if not formula:
                    continue
                ele_batch.append(
                    MaterialCandidate(
                        candidate_id=f"{ion.lower()}_ele_{idx:03d}",
                        framework_formula=formula,
                        working_ion=ion,
                        source_mode=str(option_pool.get("source_mode") or "existing"),
                        valid=True,
                        metadata={
                            "component_class": "electrolyte",
                            "role_condition": "electrolyte",
                            "ontology_class": self._classify_material_ontology(formula) or "solid_li_conductor",
                        },
                    )
                )

            for pool_name, batch in (("cathode", cath_batch), ("anode", an_batch), ("electrolyte", ele_batch)):
                for c in batch:
                    meta = dict(getattr(c, "metadata", {}) or {})
                    meta["component_class"] = pool_name
                    meta["role_condition"] = pool_name
                    c.metadata = meta
                    raw_component_batches[pool_name].append(c)
                generation_counts_by_component_ion[pool_name][ion] = int(len(batch))
            generation_counts_by_component_ion["separator"][ion] = int(len(option_pool.get("separator_options") or []))
            generation_counts_by_component_ion["additive"][ion] = int(len(option_pool.get("additive_options") or []))

        cathode_deduped = _dedupe_candidates(raw_component_batches["cathode"])
        anode_deduped = _dedupe_candidates(raw_component_batches["anode"])
        electrolyte_deduped = _dedupe_candidates(raw_component_batches["electrolyte"])

        cathode_generated = self._filter_material_candidates(
            cathode_deduped,
            material_source_mode=objective_cfg.material_source_mode,
        )
        anode_generated = self._filter_material_candidates(
            anode_deduped,
            material_source_mode=objective_cfg.material_source_mode,
        )
        if not cathode_generated:
            cathode_generated = cathode_deduped[: per_ion_pool * max(1, len(ion_scope))]
        if not anode_generated:
            anode_generated = anode_deduped[: per_ion_pool * max(1, len(ion_scope))]
        electrolyte_generated = list(electrolyte_deduped)

        generated = list(cathode_generated) + list(anode_generated) + list(electrolyte_generated)
        stage_metrics["stage_1"] = {
            "material_candidates_generated": len(generated),
            "component_conditioned_generation": True,
            "generated_per_component": {
                "cathode": int(len(cathode_generated)),
                "anode": int(len(anode_generated)),
                "electrolyte": int(len(electrolyte_generated)),
                "separator": int(sum(len((component_option_pools.get(ion, {}) or {}).get("separator_options", [])) for ion in ion_scope)),
                "additive": int(sum(len((component_option_pools.get(ion, {}) or {}).get("additive_options", [])) for ion in ion_scope)),
            },
            "working_ion_scope": ion_scope,
            "material_source_mode": objective_cfg.material_source_mode,
            "generation_mode": {
                "latent_sampling": True,
                "interpolation_enabled": bool(objective_cfg.interpolation_enabled),
                "extrapolation_enabled": bool(objective_cfg.extrapolation_enabled),
                "component_class_conditioning": True,
            },
            "generated_by_component_and_working_ion": generation_counts_by_component_ion,
            "component_option_pools": component_option_pools,
        }

        electrode_candidates = list(cathode_generated) + list(anode_generated)
        validated_electrodes, chemistry_reports = self.chemistry_validator.validate_candidates(electrode_candidates)
        stage_metrics["stage_1"]["chemistry_reports"] = len(chemistry_reports)
        rejection_reason_counts: Dict[str, int] = {}
        for rep in chemistry_reports:
            if bool(getattr(rep, "valid", False)):
                continue
            for reason in list(getattr(rep, "reasons", []) or []):
                key = str(reason)
                rejection_reason_counts[key] = int(rejection_reason_counts.get(key, 0) + 1)
        stage_metrics["stage_1"]["chemistry_rejection_reason_counts"] = rejection_reason_counts

        valid_electrodes = [c for c in validated_electrodes if c.valid]
        valid_cathodes = [c for c in valid_electrodes if str((getattr(c, "metadata", {}) or {}).get("component_class", "")) == "cathode"]
        valid_anodes = [c for c in valid_electrodes if str((getattr(c, "metadata", {}) or {}).get("component_class", "")) == "anode"]

        valid_cathodes, stage1b_cath = self._apply_stage1b_ontology_gate(valid_cathodes)
        valid_anodes, stage1b_an = self._apply_stage1b_ontology_gate(valid_anodes)
        valid_electrolytes, stage1b_el = self._apply_stage1b_ontology_gate(list(electrolyte_generated))

        pre_role_anodes = list(valid_anodes)
        valid_cathodes, role_gate_cath = self._apply_role_class_gate(valid_cathodes, role="cathode")
        valid_anodes, role_gate_an = self._apply_role_class_gate(valid_anodes, role="anode")
        valid_electrolytes, role_gate_el = self._apply_role_class_gate(valid_electrolytes, role="electrolyte")
        if not valid_anodes and pre_role_anodes:
            recovered_anodes = self._recover_anode_candidates(pre_role_anodes, limit=3)
            if recovered_anodes:
                valid_anodes = list(recovered_anodes)
                role_gate_an = dict(role_gate_an)
                role_gate_an["fallback_applied"] = True
                role_gate_an["fallback_reason"] = "anode_pool_recovered_from_pre_role_gate"
                role_gate_an["fallback_recovered_count"] = int(len(recovered_anodes))
                role_gate_an["kept_count_after_fallback"] = int(len(valid_anodes))

        stage_metrics["stage_1b"] = {
            "classification_required": True,
            "per_component": {
                "cathode": stage1b_cath,
                "anode": stage1b_an,
                "electrolyte": stage1b_el,
            },
            "role_class_gate": {
                "cathode": role_gate_cath,
                "anode": role_gate_an,
                "electrolyte": role_gate_el,
            },
        }
        if (
            int(stage1b_cath.get("rejected_unclassified_count", 0) or 0) > 0
            or int(stage1b_an.get("rejected_unclassified_count", 0) or 0) > 0
            or int(stage1b_el.get("rejected_unclassified_count", 0) or 0) > 0
            or int(role_gate_cath.get("rejected_count", 0) or 0) > 0
            or int(role_gate_an.get("rejected_count", 0) or 0) > 0
            or int(role_gate_el.get("rejected_count", 0) or 0) > 0
        ):
            stage_metrics["guardrail_flags"]["ontology_violation_flag"] = True
        if int(role_gate_el.get("rejected_count", 0) or 0) > 0:
            stage_metrics["guardrail_flags"]["electrolyte_class_violation_flag"] = True

        combined_valid_materials = list(valid_cathodes) + list(valid_anodes) + list(valid_electrolytes)
        stage_metrics["stage_1"]["material_candidates_valid"] = len(combined_valid_materials)
        stage_metrics["stage_1"]["valid_per_component"] = {
            "cathode": int(len(valid_cathodes)),
            "anode": int(len(valid_anodes)),
            "electrolyte": int(len(valid_electrolytes)),
        }

        if not valid_cathodes or not valid_anodes or not valid_electrolytes:
            return {
                "best_system": base_system,
                "candidate_records": [],
                "discovery_report_card": stage_metrics,
                "target_objectives_feasible": dict(objective_cfg.target_vector),
                "objective_feasibility_adjustments": [],
                "latent_generation_stats": {
                    "ranked_systems_count": 0,
                    "latent_generated_count": 0,
                    "synthesized_generated_count": 0,
                    "selected_latent_count": 0,
                },
                "graph_manifest_validation": graph_manifest,
            }

        assignments: Dict[str, RoleAssignment] = {}
        for c in valid_cathodes:
            assignments[str(c.candidate_id)] = self._role_locked_assignment(str(c.candidate_id), role="cathode")
        for c in valid_anodes:
            assignments[str(c.candidate_id)] = self._role_locked_assignment(str(c.candidate_id), role="anode")
        for c in valid_electrolytes:
            assignments[str(c.candidate_id)] = self._role_locked_assignment(str(c.candidate_id), role="electrolyte")
        stage_metrics["stage_2"] = {
            "assigned_materials": len(assignments),
            "component_conditioned_role_lock": True,
            "fallback_assignments_before_head": 0,
            "fallback_assignments_after_head": 0,
            "role_head_probe_stats": {
                "disabled_for_component_conditioned_generation": True,
                "probed_candidates": 0,
                "model_head_selected_assignments": 0,
                "deterministic_primary_assignments": int(len(assignments)),
                "deterministic_fallback_assignments": 0,
                "decision_counts": {"component_conditioned_role_lock": int(len(assignments))},
            },
        }

        inference_engine = get_enhanced_inference_engine()
        stage_metrics["stage_0"]["runtime_model_stack"] = {
            "masked_ensemble_models_loaded": int(len(getattr(inference_engine, "model_ensemble_entries", []) or [])),
            "hgt_reranker_loaded": bool(getattr(inference_engine, "hgt_reranker", None) is not None),
            "hgt_graph_path": str(
                (getattr(inference_engine, "hgt_reranker", {}) or {}).get("graph_path", "")
                if isinstance(getattr(inference_engine, "hgt_reranker", None), dict)
                else ""
            ),
            "hgt_checkpoint_path": str(
                (getattr(inference_engine, "hgt_reranker", {}) or {}).get("checkpoint_path", "")
                if isinstance(getattr(inference_engine, "hgt_reranker", None), dict)
                else ""
            ),
        }

        compatibility_records: list[CompatibilityRecord] = []
        compatibility_by_ion: Dict[str, Dict[str, int]] = {}
        normalized_scope = [self._normalize_ion(ion, fallback=objective_cfg.working_ion) for ion in ion_scope]
        for ion in normalized_scope:
            ion_candidates = [
                c
                for c in combined_valid_materials
                if self._normalize_ion(getattr(c, "working_ion", None), fallback=objective_cfg.working_ion) == ion
            ]
            if not ion_candidates:
                compatibility_by_ion[ion] = {
                    "materials": 0,
                    "triples_scored": 0,
                    "triples_valid": 0,
                }
                continue
            ion_records = self.compatibility_model.score_triples(
                candidates=ion_candidates,
                assignments=assignments,
            )
            compatibility_by_ion[ion] = {
                "materials": len(ion_candidates),
                "triples_scored": len(ion_records),
                "triples_valid": sum(1 for r in ion_records if r.hard_valid),
            }
            compatibility_records.extend(ion_records)

        if not compatibility_records:
            compatibility_records = self.compatibility_model.score_triples(
                candidates=combined_valid_materials,
                assignments=assignments,
            )

        compatibility_records.sort(key=lambda r: r.aggregate_score(), reverse=True)
        hard_valid_count = sum(1 for r in compatibility_records if r.hard_valid)
        allow_soft_compatibility = False
        soft_reality_override = str(os.getenv("ECESSP_ALLOW_SOFT_REALISM", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if soft_reality_override and hard_valid_count == 0 and compatibility_records:
            soft_threshold = 0.30
            soft_candidates = [
                r
                for r in compatibility_records
                if float(r.aggregate_score()) >= soft_threshold
            ]
            if soft_candidates:
                compatibility_records = soft_candidates
                allow_soft_compatibility = True

        stage_metrics["stage_3"] = {
            "compatibility_triples_scored": len(compatibility_records),
            "compatibility_triples_valid": sum(1 for r in compatibility_records if r.hard_valid),
            "compatibility_by_ion": compatibility_by_ion,
            "soft_compatibility_fallback": bool(allow_soft_compatibility),
        }
        compatibility_records, stage3_component_gate = self._apply_stage3_component_ontology_gate(compatibility_records)
        stage_metrics["stage_3"]["component_ontology_gate"] = stage3_component_gate
        stage_metrics["stage_3"]["compatibility_triples_after_ontology_gate"] = len(compatibility_records)
        if int(stage3_component_gate.get("rejected_count", 0) or 0) > 0:
            stage_metrics["guardrail_flags"]["ontology_violation_flag"] = True
        if any("electrolyte_forbidden_element" in k for k in stage3_component_gate.get("rejection_reason_counts", {}).keys()):
            stage_metrics["guardrail_flags"]["electrolyte_class_violation_flag"] = True

        needs_compatibility_retry = (
            (not compatibility_records)
            or (
                len(compatibility_records) > 0
                and sum(1 for r in compatibility_records if r.hard_valid) <= 0
                and not allow_soft_compatibility
            )
        )
        if needs_compatibility_retry and not compatibility_fallback_reentry:
            retry_params = dict(discovery_params or {})
            retry_params["_compatibility_fallback_reentry"] = 1
            retry_params["working_ion_candidates"] = list(ion_scope)
            retry_params["material_source_mode"] = "hybrid"
            retry_params["component_source_mode"] = "hybrid"
            retry = self.run_generative(
                base_system=base_system,
                objective=objective,
                discovery_params=retry_params,
                candidate_pool_size=pool_size,
            )
            report = retry.get("discovery_report_card")
            if isinstance(report, dict):
                report["compatibility_fallback_triggered"] = True
                report["compatibility_fallback_reason"] = "no_hard_valid_triples"
                report["compatibility_fallback_from"] = {
                    "working_ion_scope": list(ion_scope),
                    "material_source_mode": objective_cfg.material_source_mode,
                    "component_source_mode": objective_cfg.component_source_mode,
                }
                report["compatibility_fallback_to"] = {
                    "working_ion_scope": list(ion_scope),
                    "material_source_mode": "hybrid",
                    "component_source_mode": "hybrid",
                }
            return retry

        if not compatibility_records:
            return {
                "best_system": base_system,
                "candidate_records": [],
                "discovery_report_card": stage_metrics,
                "target_objectives_feasible": dict(objective_cfg.target_vector),
                "objective_feasibility_adjustments": [],
                "latent_generation_stats": {
                    "ranked_systems_count": 0,
                    "latent_generated_count": 0,
                    "synthesized_generated_count": 0,
                    "selected_latent_count": 0,
                },
                "graph_manifest_validation": graph_manifest,
            }

        assembled_rows = []
        assembled_invalid_reason_counts: Dict[str, int] = {}
        for idx, record in enumerate(compatibility_records[:pool_size]):
            ion_key = self._normalize_ion(
                record.cathode.working_ion or objective_cfg.working_ion,
                fallback=objective_cfg.working_ion,
            )
            option_pool = component_option_pools.get(ion_key) or component_option_pools.get(objective_cfg.working_ion)
            component_selection = None
            if option_pool:
                ele_opts = list(option_pool.get("electrolyte_options") or [])
                sep_opts = list(option_pool.get("separator_options") or [])
                add_opts = list(option_pool.get("additive_options") or [])
                chosen_electrolyte = ele_opts[idx % len(ele_opts)] if ele_opts else None
                chosen_separator = sep_opts[idx % len(sep_opts)] if sep_opts else None
                chosen_additive = add_opts[idx % len(add_opts)] if add_opts else None
                component_selection = {
                    "source_mode": option_pool.get("source_mode"),
                    "electrolyte_material": chosen_electrolyte,
                    "separator_material": chosen_separator,
                    "additive_material": chosen_additive,
                    "electrolyte_options": ele_opts,
                    "separator_options": sep_opts,
                    "additive_options": add_opts,
                }
            assembled, system = self.assembler.assemble(
                index=idx,
                compatibility=record,
                target_property_vector=objective_cfg.target_vector,
                base_system=base_system,
                component_selection=component_selection,
            )
            forbidden_electrolyte_elements = self._electrolyte_forbidden_transition_metals(
                str(getattr(assembled, "electrolyte_formula", "") or "")
            )
            if forbidden_electrolyte_elements:
                assembled.valid = False
                existing_reasons = list(getattr(assembled, "valid_reasons", []) or [])
                for el in forbidden_electrolyte_elements:
                    existing_reasons.append(f"ontology:electrolyte_forbidden_element:{el}")
                assembled.valid_reasons = existing_reasons
                stage_metrics["guardrail_flags"]["ontology_violation_flag"] = True
                stage_metrics["guardrail_flags"]["electrolyte_class_violation_flag"] = True
            if allow_soft_compatibility and not bool(record.hard_valid) and bool(assembled.valid):
                assembled.valid = True
                if isinstance(assembled.provenance, dict):
                    assembled.provenance["compatibility_soft_fallback"] = True
                    assembled.provenance["compatibility_soft_score"] = float(record.aggregate_score())
            if not bool(assembled.valid):
                for reason in list(getattr(assembled, "valid_reasons", []) or []):
                    key = str(reason or "").strip()
                    if not key:
                        continue
                    assembled_invalid_reason_counts[key] = assembled_invalid_reason_counts.get(key, 0) + 1
            assembled_rows.append((assembled, system, record))
        stage_metrics["stage_4"] = {
            "assembled_candidates": len(assembled_rows),
            "assembled_valid_before_prediction": sum(1 for a, _, _ in assembled_rows if a.valid),
            "assembled_invalid_reason_counts": dict(
                sorted(assembled_invalid_reason_counts.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "component_source_mode": objective_cfg.component_source_mode,
            "separator_options_count": objective_cfg.separator_options_count,
            "additive_options_count": objective_cfg.additive_options_count,
            "component_option_pools": component_option_pools,
        }
        invalid_keys = set(assembled_invalid_reason_counts.keys())
        if any(("negative_voltage" in k) or ("redox" in k) for k in invalid_keys):
            stage_metrics["guardrail_flags"]["redox_inversion_flag"] = True
        if any(
            ("voltage_inverted_or_too_low" in k)
            or ("minimum_voltage_difference_not_met" in k)
            or ("cell_voltage_" in k)
            for k in invalid_keys
        ):
            stage_metrics["guardrail_flags"]["voltage_inconsistency_flag"] = True
        if any(k.startswith("ontology:") for k in invalid_keys):
            stage_metrics["guardrail_flags"]["ontology_violation_flag"] = True
        stage6_rejections = int(
            sum(v for k, v in assembled_invalid_reason_counts.items() if str(k).startswith("stage6:"))
        )
        stage7_rejections = int(
            sum(v for k, v in assembled_invalid_reason_counts.items() if str(k).startswith("stage7:"))
        )
        stage8_rejections = int(
            sum(v for k, v in assembled_invalid_reason_counts.items() if str(k).startswith("stage8:"))
        )
        stage_metrics["stage_6"] = {
            "description": "additive_and_interphase_validation",
            "rejected_count": int(stage6_rejections),
            "hard_gate": True,
        }
        stage_metrics["stage_7"] = {
            "description": "thermodynamic_and_kinetic_gates",
            "rejected_count": int(stage7_rejections),
            "hard_gate": True,
        }
        stage_metrics["stage_8"] = {
            "description": "voltage_integrity_and_consistency",
            "rejected_count": int(stage8_rejections),
            "hard_gate": True,
        }

        candidate_records: list[dict] = []
        soft_stage4_rescued_count = 0

        for assembled, system, compatibility in assembled_rows:
            inference_result: Dict[str, Any] = {}
            try:
                inference_result = inference_engine.infer(system)
            except Exception as exc:
                logger.debug("Stage-5 inference failed for %s: %s", system.battery_id, exc)

            self._stabilize_system_predictions(system)

            constraints = evaluate_system(system)
            overall_valid = bool(constraints.get("overall_valid", False))
            stage4_soft_rescued = (
                (not bool(assembled.valid))
                and self._is_soft_stage4_only_failure(list(getattr(assembled, "valid_reasons", []) or []))
            )
            if stage4_soft_rescued:
                soft_stage4_rescued_count += 1
            stage_valid = self._is_stage_valid_candidate(
                assembled_valid=bool(assembled.valid or stage4_soft_rescued),
                overall_valid=overall_valid,
                require_overall_valid=bool(objective_cfg.require_overall_valid),
            )
            if not stage_valid:
                continue

            distance = self._objective_distance(system, objective_cfg)
            objective_score = max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, float(distance)))))
            aux_heads = {}
            if isinstance(inference_result, dict):
                maybe_aux = inference_result.get("auxiliary_heads")
                if isinstance(maybe_aux, dict):
                    aux_heads = maybe_aux
            model_compatibility = self._to_float(aux_heads.get("compatibility_score_aggregate"))
            compatibility_rule = compatibility.aggregate_score()
            compatibility_gate = self._compatibility_head_gate_decision(
                model_compatibility=model_compatibility,
                deterministic_score=compatibility_rule,
                min_score=self._compatibility_head_min_score(),
            )
            compatibility_score = float(compatibility_gate["compatibility_score"])
            compatibility_source = str(compatibility_gate["compatibility_source"])
            feasibility_score = self._compute_feasibility_score(constraints, compatibility_score)

            material_generation = {}
            if isinstance(system.uncertainty, dict):
                maybe = system.uncertainty.get("material_generation")
                if isinstance(maybe, dict):
                    material_generation = maybe
            model_uncertainty_penalty = self._to_float(aux_heads.get("uncertainty_penalty"))
            uncertainty_model_weight = self._uncertainty_model_weight()
            uncertainty_penalty = (
                uncertainty_model_weight * float(model_uncertainty_penalty if model_uncertainty_penalty is not None else 0.0)
                + (1.0 - uncertainty_model_weight) * float(material_generation.get("uncertainty_proxy", 0.0) or 0.0)
            )
            uncertainty_penalty = float(max(0.0, min(1.0, uncertainty_penalty)))
            hgt_rerank_score = 0.0
            hgt_plausibility = 0.0
            hgt_objective_support = 0.0
            hgt_rerank_enabled = bool(getattr(inference_engine, "hgt_reranker", None) is not None)
            hgt_fn = getattr(inference_engine, "_hgt_rerank_components", None)
            if callable(hgt_fn):
                try:
                    hgt_rerank_score, hgt_plausibility, hgt_objective_support = hgt_fn(
                        system=system,
                        target_objectives=objective_cfg.target_vector,
                    )
                except Exception as exc:
                    logger.debug("HGT rerank probe failed for %s: %s", system.battery_id, exc)

            candidate_records.append(
                {
                    "system": system,
                    "score": 0.0,
                    "speculative": bool(constraints.get("performance", {}).get("speculative", False)),
                    "source": "staged_pipeline",
                    "valid": True,
                    "objective_alignment_score": float(objective_score),
                    "feasibility_score": float(feasibility_score),
                    "objective_distance_score": float(objective_score),
                    "compatibility_score": float(compatibility_score),
                    "compatibility_source": compatibility_source,
                    "compatibility_head_blend": None,
                    "compatibility_fallback_reason": compatibility_gate["compatibility_fallback_reason"],
                    "compatibility_model_score": compatibility_gate["compatibility_model_score"],
                    "compatibility_rule_score": compatibility_gate["compatibility_rule_score"],
                    "compatibility_head_min_score": compatibility_gate["compatibility_head_min_score"],
                    "uncertainty_penalty": uncertainty_penalty,
                    "uncertainty_model_weight": float(uncertainty_model_weight),
                    "hgt_rerank_score": float(max(0.0, min(1.0, hgt_rerank_score))),
                    "hgt_plausibility": float(max(0.0, min(1.0, hgt_plausibility))),
                    "hgt_objective_support": float(max(0.0, min(1.0, hgt_objective_support))),
                    "hgt_rerank_enabled": bool(hgt_rerank_enabled),
                    "novelty_score": 0.0,
                    "alignment_score": float(objective_score),
                    "manufacturability_score": float(max(0.0, 1.0 - compatibility.mechanical_strain_risk)),
                    "material_novelty_score": 0.0,
                    "material_uncertainty_proxy": float(material_generation.get("uncertainty_proxy", 0.0) or 0.0),
                    "material_thermodynamic_proxy": float(material_generation.get("thermodynamic_proxy", 1.0) or 1.0),
                    "stage_trace": {
                        "material_generator": {
                            "cathode": compatibility.cathode.to_dict(),
                            "anode": compatibility.anode.to_dict(),
                            "electrolyte": compatibility.electrolyte.to_dict(),
                        },
                        "assembly": assembled.to_dict(),
                        "compatibility": {
                            "voltage_window_overlap_score": compatibility.voltage_window_overlap_score,
                            "chemical_stability_score": compatibility.chemical_stability_score,
                            "mechanical_strain_risk": compatibility.mechanical_strain_risk,
                            "interface_risk_reason_codes": list(compatibility.interface_risk_reason_codes),
                            "hard_valid": compatibility.hard_valid,
                        },
                        "hgt_reranker": {
                            "enabled": bool(hgt_rerank_enabled),
                            "score": float(max(0.0, min(1.0, hgt_rerank_score))),
                            "plausibility": float(max(0.0, min(1.0, hgt_plausibility))),
                            "objective_support": float(max(0.0, min(1.0, hgt_objective_support))),
                        },
                        "model_heads": aux_heads,
                        "constraints": constraints,
                        "stage4_soft_rescued": bool(stage4_soft_rescued),
                    },
                }
            )

        candidate_records = self._apply_pareto_ranking(candidate_records)
        top_records = candidate_records[: objective_cfg.max_candidates]

        stage_metrics["stage_5"] = {
            "predicted_candidates": len(assembled_rows),
            "post_prediction_valid_candidates": len(candidate_records),
            "soft_stage4_rescued_candidates": int(soft_stage4_rescued_count),
        }
        stage_report_card = self._build_discovery_report_card(candidate_records, top_k=5)
        w_obj, w_feas, w_unc = self._pareto_weights()
        w_hgt = self._hgt_pareto_weight()
        w_total = max(1e-9, w_obj + w_feas + w_unc + w_hgt)
        w_obj_n = float(w_obj / w_total)
        w_feas_n = float(w_feas / w_total)
        w_unc_n = float(w_unc / w_total)
        w_hgt_n = float(w_hgt / w_total)
        hgt_scores = [float(r.get("hgt_rerank_score", 0.0) or 0.0) for r in candidate_records]
        hgt_enabled_count = sum(1 for r in candidate_records if bool(r.get("hgt_rerank_enabled", False)))
        stage_metrics["stage_6_7"] = {
            "ranking_mode": "pareto_with_uncertainty_and_hgt",
            "pareto_weights": {
                "objective": float(w_obj_n),
                "feasibility": float(w_feas_n),
                "uncertainty": float(w_unc_n),
                "hgt_rerank": float(w_hgt_n),
            },
            "ranked_candidates": len(candidate_records),
            "pareto_front_size": sum(1 for r in candidate_records if int(r.get("pareto_rank", 1)) == 0),
            "returned_candidates": len(top_records),
            "hgt_rerank_enabled_count": int(hgt_enabled_count),
            "hgt_rerank_mean_score": float(sum(hgt_scores) / max(1, len(hgt_scores))),
        }
        stage_metrics["stage_9"] = {
            "description": "final_physics_lock_and_pareto_ranking",
            "excluded_before_ranking": int(max(0, len(assembled_rows) - len(candidate_records))),
            "ranked_candidates": int(len(candidate_records)),
            "returned_candidates": int(len(top_records)),
            "do_not_allow_uncertainty_to_override_physics": True,
            "ranking_mode": "pareto_with_uncertainty_and_hgt",
        }
        stage_metrics["stage_10"] = {
            "pareto_ranking_only_if_all_passed": True,
            "uncertainty_cannot_override_invariant_failure": True,
            "candidate_count_after_physics_lock": int(len(candidate_records)),
        }
        stage_metrics.update(stage_report_card)
        active_learning_queue_info = self._queue_active_learning_candidates(
            candidate_records=top_records,
            objective_cfg=objective_cfg,
        )
        top_uncertainty_values = [
            float(rec.get("material_uncertainty_proxy", 0.0) or 0.0)
            for rec in top_records
        ]
        mean_uncertainty = float(sum(top_uncertainty_values) / max(1, len(top_uncertainty_values)))

        def _component_shift(pool: List[MaterialCandidate]) -> Dict[str, Any]:
            n = int(len(pool))
            generated_markers = {"latent_variation", "composition_perturbation", "cif_conditioned_refinement", "latent_optimization"}
            gen_n = int(
                sum(
                    1
                    for c in pool
                    if str(getattr(c, "source_mode", "")).strip().lower() in generated_markers
                )
            )
            return {
                "count": n,
                "generated_fraction": float(gen_n / max(1, n)),
                "existing_fraction": float((n - gen_n) / max(1, n)),
            }

        stage_metrics["stage_10"].update(
            {
                "log_normalization_mismatch_flags": bool(stage_metrics["guardrail_flags"]["normalization_mismatch_flag"]),
                "log_redox_inversion_flags": bool(stage_metrics["guardrail_flags"]["redox_inversion_flag"]),
                "log_ontology_violation_flags": bool(stage_metrics["guardrail_flags"]["ontology_violation_flag"]),
                "log_electrolyte_class_violation_flags": bool(
                    stage_metrics["guardrail_flags"]["electrolyte_class_violation_flag"]
                ),
                "log_voltage_inconsistency_flags": bool(stage_metrics["guardrail_flags"]["voltage_inconsistency_flag"]),
                "active_learning_upgrade": {
                    "track_uncertainty_per_component_class": {
                        "cathode": float(mean_uncertainty),
                        "anode": float(mean_uncertainty),
                        "electrolyte": float(mean_uncertainty),
                        "separator": float(mean_uncertainty),
                        "additive": float(mean_uncertainty),
                    },
                    "prioritize_high_uncertainty_but_physics_valid_stacks": True,
                    "distribution_shift_per_component": {
                        "cathode": _component_shift(valid_cathodes),
                        "anode": _component_shift(valid_anodes),
                        "electrolyte": _component_shift(valid_electrolytes),
                        "separator": {
                            "count": int(
                                sum(len((component_option_pools.get(ion, {}) or {}).get("separator_options", [])) for ion in ion_scope)
                            ),
                        },
                        "additive": {
                            "count": int(
                                sum(len((component_option_pools.get(ion, {}) or {}).get("additive_options", [])) for ion in ion_scope)
                            ),
                        },
                    },
                },
                "active_learning_queue": dict(active_learning_queue_info),
            }
        )

        return {
            "best_system": top_records[0]["system"] if top_records else base_system,
            "candidate_records": top_records,
            "discovery_report_card": stage_metrics,
            "target_objectives_feasible": dict(objective_cfg.target_vector),
            "objective_feasibility_adjustments": [],
            "latent_generation_stats": {
                "ranked_systems_count": len(candidate_records),
                "latent_generated_count": int(
                    sum(
                        1
                        for c in generated
                        if str(getattr(c, "source_mode", "")).strip().lower() in {"latent_variation"}
                    )
                ),
                "synthesized_generated_count": int(
                    sum(
                        1
                        for c in generated
                        if str(getattr(c, "source_mode", "")).strip().lower() in {"composition_perturbation"}
                    )
                ),
                "selected_latent_count": int(
                    sum(
                        1
                        for rec in top_records
                        if str(
                            ((rec.get("stage_trace", {}) or {}).get("material_generator", {}) or {})
                            .get("cathode", {})
                            .get("source_mode", "")
                        ).strip().lower()
                        in {"latent_variation", "latent_optimization", "cif_conditioned_refinement"}
                    )
                ),
            },
            "graph_manifest_validation": graph_manifest,
            "active_learning_queue": active_learning_queue_info,
        }
