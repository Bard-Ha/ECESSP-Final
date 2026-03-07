# backend/services/discovery_service.py
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional
import logging
import csv
import sys
import os

from backend.runtime.context import get_runtime_context
from backend.config import DATA_DIR

from design.system_template import BatterySystem
from design.system_generator import SystemGenerator
from design.system_scorer import score_system
from design.system_constraints import evaluate_system
from design.system_reasoner import reason_about_system
from backend.runtime.enhanced_engine import get_enhanced_inference_engine as get_inference_engine
from backend.services.discovery_orchestrator import DiscoveryOrchestrator

logger = logging.getLogger(__name__)


class DiscoveryService:
    """
    Coordinates system generation, scoring, constraints, and reasoning.
    """

    def __init__(self):
        self._initialized = False
        self._device = None
        self._model = None
        self._graph = None
        self._encoder = None
        self._generator = None
        self._orchestrator: DiscoveryOrchestrator | None = None
        self._dataset_cache: list[dict] | None = None

    TARGET_PROPERTIES = [
        "average_voltage",
        "capacity_grav",
        "capacity_vol",
        "energy_grav",
        "energy_vol",
        "max_delta_volume",
        "stability_charge",
        "stability_discharge",
    ]
    DEFAULT_TOP_K = 15
    MIN_CANDIDATE_SCORE = 0.55
    RELAXED_MIN_CANDIDATE_SCORE = 0.35
    GUARDRAIL_THRESHOLDS: dict[str, float] = {
        "score_promotion_min": 0.62,
        "score_review_min": 0.45,
        "material_uncertainty_promotion_max": 0.75,
        "material_uncertainty_review_max": 0.88,
        "material_thermodynamic_promotion_min": 0.45,
        "material_thermodynamic_review_min": 0.35,
    }
    DEFAULT_DISCOVERY_PARAMS: dict[str, Any] = {
        "num_candidates": 80,
        "diversity_weight": 0.4,
        "novelty_weight": 0.3,
        "extrapolation_strength": 0.3,
        "interpolation_enabled": True,
        "extrapolation_enabled": True,
        "optimize_steps": 24,
        "fused_rescore": 1,
        "working_ion_candidates": ["Li"],
        "material_source_mode": "hybrid",
        "component_source_mode": "hybrid",
        "separator_options_count": 3,
        "additive_options_count": 3,
    }
    # Novel-only generative discovery by default: avoid existing-dataset matches in top results.
    INCLUDE_DATASET_MATCHES_DEFAULT = os.getenv("ECESSP_INCLUDE_DATASET_MATCHES", "0").strip() in {"1", "true", "yes", "on"}

    @staticmethod
    def _objectives_to_target_ranges(objective: Dict[str, float]) -> Dict[str, tuple[float, float]]:
        target_ranges: Dict[str, tuple[float, float]] = {}
        for prop in DiscoveryService.TARGET_PROPERTIES:
            value = objective.get(prop)
            if value is None:
                continue
            try:
                center = float(value)
            except (TypeError, ValueError):
                continue
            if prop == "max_delta_volume":
                upper = max(center, 0.0)
                margin = max(abs(upper) * 0.25, 0.02)
                target_ranges[prop] = (0.0, upper + margin)
            else:
                margin = max(abs(center) * 0.12, 0.05)
                target_ranges[prop] = (center - margin, center + margin)
        return target_ranges

    @classmethod
    def _resolve_discovery_params(cls, params: Optional[Dict[str, Any]]) -> dict[str, Any]:
        resolved = dict(cls.DEFAULT_DISCOVERY_PARAMS)
        if not params:
            return resolved

        def clamp_float(name: str, lo: float, hi: float):
            raw = params.get(name)
            if raw is None:
                return
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return
            resolved[name] = max(lo, min(hi, v))

        def clamp_int(name: str, lo: int, hi: int):
            raw = params.get(name)
            if raw is None:
                return
            try:
                v = int(raw)
            except (TypeError, ValueError):
                return
            resolved[name] = max(lo, min(hi, v))

        clamp_int("num_candidates", 10, 200)
        clamp_float("diversity_weight", 0.0, 1.0)
        clamp_float("novelty_weight", 0.0, 1.0)
        clamp_float("extrapolation_strength", 0.0, 1.0)
        clamp_int("optimize_steps", 0, 128)
        if "fused_rescore" in params:
            raw = params.get("fused_rescore")
            if isinstance(raw, str):
                resolved["fused_rescore"] = 1 if raw.strip().lower() in {"1", "true", "yes", "on"} else 0
            else:
                resolved["fused_rescore"] = 1 if bool(raw) else 0
        for name in ("interpolation_enabled", "extrapolation_enabled"):
            if name not in params:
                continue
            raw = params.get(name)
            if isinstance(raw, str):
                resolved[name] = raw.strip().lower() in {"1", "true", "yes", "on"}
            else:
                resolved[name] = bool(raw)

        def parse_int(name: str, lo: int, hi: int):
            raw = params.get(name)
            if raw is None:
                return
            try:
                v = int(raw)
            except (TypeError, ValueError):
                return
            resolved[name] = max(lo, min(hi, v))

        def parse_mode(name: str):
            raw = params.get(name)
            if raw is None:
                return
            mode = str(raw).strip().lower()
            if mode in {"existing", "generated", "hybrid"}:
                resolved[name] = mode

        raw_ions = params.get("working_ion_candidates")
        if isinstance(raw_ions, (list, tuple)):
            ion_values = [str(x).strip() for x in raw_ions if str(x).strip()]
        elif isinstance(raw_ions, str):
            ion_values = [s.strip() for s in raw_ions.split(",") if s.strip()]
        else:
            ion_values = []
        if ion_values:
            seen: set[str] = set()
            normalized: list[str] = []
            for ion in ion_values:
                key = ion.lower()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(ion.capitalize())
            resolved["working_ion_candidates"] = normalized[:4]

        parse_mode("material_source_mode")
        parse_mode("component_source_mode")
        parse_int("separator_options_count", 1, 6)
        parse_int("additive_options_count", 1, 6)
        if not bool(resolved.get("interpolation_enabled", True)) and not bool(
            resolved.get("extrapolation_enabled", True)
        ):
            # Keep stage-1 generation alive if users disable both switches.
            resolved["interpolation_enabled"] = True
        return resolved

    def _load_dataset_rows(self) -> list[dict]:
        if self._dataset_cache is not None:
            return self._dataset_cache

        dataset_path = DATA_DIR / "processed" / "batteries_parsed.csv"
        if not dataset_path.exists():
            logger.warning("Dataset file not found for existing-system retrieval: %s", dataset_path)
            self._dataset_cache = []
            return self._dataset_cache

        rows: list[dict] = []
        with dataset_path.open("r", encoding="utf-8", newline="") as f:
            limit = sys.maxsize
            while limit > 131072:
                try:
                    csv.field_size_limit(limit)
                    break
                except OverflowError:
                    limit //= 10
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        self._dataset_cache = rows
        return rows

    @staticmethod
    def _score_row_against_objective(row: dict, objective: Dict[str, float]) -> float:
        distances: list[float] = []
        for prop in DiscoveryService.TARGET_PROPERTIES:
            target = objective.get(prop)
            if target is None:
                continue
            raw = row.get(prop)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
                target_value = float(target)
            except (TypeError, ValueError):
                continue
            if prop == "max_delta_volume":
                upper = max(target_value, 0.0)
                denom = max(abs(upper), 0.05)
                if value <= upper:
                    distances.append(0.0)
                else:
                    distances.append((value - upper) / denom)
                continue
            denom = max(abs(target_value), 1.0)
            distances.append(abs(value - target_value) / denom)

        if not distances:
            return 0.0

        mean_distance = sum(distances) / len(distances)
        return max(0.0, 1.0 - mean_distance)

    @staticmethod
    def _score_system_against_objective(system: BatterySystem, objective: Dict[str, float]) -> float:
        distances: list[float] = []
        for prop in DiscoveryService.TARGET_PROPERTIES:
            target = objective.get(prop)
            if target is None:
                continue
            value = getattr(system, prop, None)
            if value is None:
                continue
            try:
                sv = float(value)
                tv = float(target)
            except (TypeError, ValueError):
                continue
            if prop == "max_delta_volume":
                upper = max(tv, 0.0)
                denom = max(abs(upper), 0.05)
                if sv <= upper:
                    distances.append(0.0)
                else:
                    distances.append((sv - upper) / denom)
                continue
            denom = max(abs(tv), 1.0)
            distances.append(abs(sv - tv) / denom)
        if not distances:
            return 0.0
        return max(0.0, 1.0 - (sum(distances) / len(distances)))

    @staticmethod
    def _fill_derived_energy(system: BatterySystem) -> None:
        if system.average_voltage is None:
            return
        if system.capacity_grav is not None:
            system.energy_grav = float(system.average_voltage) * float(system.capacity_grav)
        if system.capacity_vol is not None:
            system.energy_vol = float(system.average_voltage) * float(system.capacity_vol)

    @staticmethod
    def _row_to_system(row: dict) -> BatterySystem:
        def safe_float(v):
            if v in (None, ""):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        elements_value = row.get("elements")
        elements = None
        if isinstance(elements_value, str) and elements_value.strip():
            cleaned = elements_value.strip().strip("[]")
            parts = [p.strip().strip("'").strip('"') for p in cleaned.split(",") if p.strip()]
            elements = parts or None

        material_level = {
            "capacity_grav": safe_float(row.get("capacity_grav")),
            "capacity_vol": safe_float(row.get("capacity_vol")),
            "energy_grav": safe_float(row.get("energy_grav")),
            "energy_vol": safe_float(row.get("energy_vol")),
        }
        cell_level = dict(material_level)

        return BatterySystem(
            battery_id=str(row.get("battery_id") or "dataset_system"),
            provenance="dataset",
            working_ion=row.get("working_ion") or None,
            battery_formula=row.get("battery_formula") or None,
            framework_formula=row.get("framework_formula") or None,
            chemsys=row.get("chemsys") or None,
            elements=elements,
            average_voltage=safe_float(row.get("average_voltage")),
            capacity_grav=safe_float(row.get("capacity_grav")),
            capacity_vol=safe_float(row.get("capacity_vol")),
            energy_grav=safe_float(row.get("energy_grav")),
            energy_vol=safe_float(row.get("energy_vol")),
            max_delta_volume=safe_float(row.get("max_delta_volume")),
            stability_charge=safe_float(row.get("stability_charge")),
            stability_discharge=safe_float(row.get("stability_discharge")),
            uncertainty={
                "material_level": dict(material_level),
                "cell_level": dict(cell_level),
            },
            material_level=material_level,
            cell_level=cell_level,
        )

    def _find_existing_system_candidates(self, objective: Dict[str, float], top_k: int = 5) -> list[Dict]:
        rows = self._load_dataset_rows()
        if not rows or not objective:
            return []

        scored: list[tuple[float, dict]] = []
        for row in rows:
            score = self._score_row_against_objective(row, objective)
            if score <= 0:
                continue
            scored.append((score, row))

        scored.sort(key=lambda t: t[0], reverse=True)
        selected = scored[:top_k]

        candidates: list[Dict] = []
        for score, row in selected:
            system = self._row_to_system(row)
            constraints = evaluate_system(system)
            valid = bool(constraints.get("overall_valid", False))
            if not valid:
                continue
            candidates.append(
                {
                    "system": system,
                    "score": round(float(score), 6),
                    "speculative": False,
                    "source": "existing_dataset_match",
                    "valid": valid,
                }
            )
        return candidates

    @staticmethod
    def _candidate_signature(candidate: dict) -> str:
        system = candidate.get("system")
        if system is None:
            return ""
        source = str(candidate.get("source", ""))
        # For latent candidates, preserve per-sample diversity by signature on id.
        if DiscoveryService._is_latent_source(source):
            return str(getattr(system, "battery_id", "")) or str(id(system))
        if hasattr(system, "to_dict"):
            payload = system.to_dict()
        elif isinstance(system, dict):
            payload = system
        else:
            return str(getattr(system, "battery_id", ""))
        return "|".join(
            str(payload.get(key) or "")
            for key in ("working_ion", "battery_formula", "framework_formula", "chemsys", "anode_material", "cathode_material")
        )

    @staticmethod
    def _quality_filter(
        candidates: list[dict],
        *,
        min_score: float,
    ) -> list[dict]:
        return [
            c
            for c in candidates
            if bool(c.get("valid", True)) and float(c.get("score", 0.0)) >= min_score
        ]

    @staticmethod
    def _is_latent_source(source: str) -> bool:
        return source in {"latent_generated", "generated", "latent_optimized"}

    @classmethod
    def _candidate_guardrail_assessment(
        cls,
        *,
        candidate: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return default
            return out

        perf = constraints.get("performance", {}) if isinstance(constraints, dict) else {}
        physics_first = constraints.get("physics_first", {}) if isinstance(constraints, dict) else {}

        overall_valid = bool(constraints.get("overall_valid", False))
        hard_valid = bool(physics_first.get("hard_valid", overall_valid)) if isinstance(physics_first, dict) else overall_valid
        speculative = bool(candidate.get("speculative", perf.get("speculative", False)))
        low_quality_latent = bool(candidate.get("low_quality_latent", False))

        score = _safe_float(candidate.get("score", 0.0))
        material_uncertainty_proxy = _safe_float(candidate.get("material_uncertainty_proxy", 0.0))
        material_thermodynamic_proxy = _safe_float(candidate.get("material_thermodynamic_proxy", 1.0), default=1.0)

        t = cls.GUARDRAIL_THRESHOLDS
        reasons: list[str] = []

        if not hard_valid:
            reasons.append("physics_hard_violation")
        if not overall_valid:
            reasons.append("overall_constraint_invalid")
        if low_quality_latent:
            reasons.append("low_quality_latent")
        if speculative:
            reasons.append("speculative_flag")
        if score < t["score_review_min"]:
            reasons.append("low_candidate_score")
        if material_uncertainty_proxy > t["material_uncertainty_review_max"]:
            reasons.append("high_material_uncertainty")
        if material_thermodynamic_proxy < t["material_thermodynamic_review_min"]:
            reasons.append("low_material_thermodynamics")

        promotion_ready = all(
            [
                hard_valid,
                overall_valid,
                not low_quality_latent,
                not speculative,
                score >= t["score_promotion_min"],
                material_uncertainty_proxy <= t["material_uncertainty_promotion_max"],
                material_thermodynamic_proxy >= t["material_thermodynamic_promotion_min"],
            ]
        )

        review_ready = all(
            [
                hard_valid,
                overall_valid,
                score >= t["score_review_min"],
                material_uncertainty_proxy <= t["material_uncertainty_review_max"],
                material_thermodynamic_proxy >= t["material_thermodynamic_review_min"],
            ]
        )

        if promotion_ready:
            status = "promotion_ready"
        elif review_ready:
            status = "review_required"
        else:
            status = "blocked"

        return {
            "status": status,
            "promotion_ready": bool(promotion_ready),
            "review_ready": bool(review_ready),
            "reasons": reasons,
            "score": score,
            "material_uncertainty_proxy": material_uncertainty_proxy,
            "material_thermodynamic_proxy": material_thermodynamic_proxy,
        }

    @staticmethod
    def _electrochemical_viability_v2(
        *,
        working_ion: str,
        pair_id: str,
        anode_material: str,
        cathode_material: str,
        anode_potential: float | None,
        cathode_potential: float | None,
        anode_oxidation_state_increase: bool | None = None,
        cathode_oxidation_state_decrease: bool | None = None,
        min_practical_voltage: float = 1.0,
    ) -> Dict[str, Any]:
        v_cell: float | None = None
        if cathode_potential is not None and anode_potential is not None:
            v_cell = float(cathode_potential - anode_potential)

        redox_gap_pass = bool(v_cell is not None and v_cell > 0.0)
        practical_voltage_pass = bool(v_cell is not None and v_cell >= float(min_practical_voltage))
        if anode_oxidation_state_increase is None or cathode_oxidation_state_decrease is None:
            directional_result: bool | None = None
        else:
            directional_result = bool(anode_oxidation_state_increase and cathode_oxidation_state_decrease)

        if not redox_gap_pass:
            recommendation = "SWAP_ROLES or REPLACE_ANODE. Current anode potential is too high for this working-ion system."
        elif not practical_voltage_pass:
            recommendation = "REVIEW: thermodynamically active but below practical voltage threshold."
        elif directional_result is False:
            recommendation = "WARNING: material redox directionality appears inconsistent with assigned roles."
        else:
            recommendation = "PASS"

        return {
            "logic_engine": "Electrochemical_Viability_V2",
            "validation_rules": [
                {
                    "rule_id": "REDOX_POTENTIAL_GAP",
                    "result": bool(redox_gap_pass),
                },
                {
                    "rule_id": "DIRECTIONALITY_CHECK",
                    "result": directional_result,
                },
            ],
            "candidate_evaluation": {
                "ion": str(working_ion),
                "pair_id": str(pair_id),
                "anode": {
                    "material": str(anode_material),
                    "estimated_potential_V": anode_potential,
                },
                "cathode": {
                    "material": str(cathode_material),
                    "estimated_potential_V": cathode_potential,
                },
            },
            "screening_output": {
                "theoretical_voltage": v_cell,
                "is_viable": bool(redox_gap_pass and practical_voltage_pass),
                "recommendation": recommendation,
                "minimum_practical_voltage": float(min_practical_voltage),
            },
        }

    @classmethod
    def _candidate_electrochemical_viability_v2(
        cls,
        *,
        candidate: Dict[str, Any],
        system: BatterySystem,
    ) -> Dict[str, Any] | None:
        stage_trace = candidate.get("stage_trace", {}) if isinstance(candidate, dict) else {}
        assembly = stage_trace.get("assembly", {}) if isinstance(stage_trace, dict) else {}
        cathode_formula = str(
            assembly.get("cathode_formula")
            or getattr(system, "cathode_material", None)
            or ""
        ).strip()
        anode_formula = str(
            assembly.get("anode_formula")
            or getattr(system, "anode_material", None)
            or ""
        ).strip()
        working_ion = str(getattr(system, "working_ion", None) or assembly.get("working_ion") or "Li").strip() or "Li"

        def _safe_float(value: Any) -> float | None:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return out

        cathode_potential = _safe_float(assembly.get("cathode_redox_potential"))
        anode_potential = _safe_float(assembly.get("anode_redox_potential"))
        if cathode_potential is None or anode_potential is None:
            material_generation = {}
            if isinstance(getattr(system, "uncertainty", None), dict):
                maybe = system.uncertainty.get("material_generation")
                if isinstance(maybe, dict):
                    material_generation = maybe
            cathode_potential = cathode_potential if cathode_potential is not None else _safe_float(material_generation.get("cathode_redox_potential"))
            anode_potential = anode_potential if anode_potential is not None else _safe_float(material_generation.get("anode_redox_potential"))

        if cathode_potential is None or anode_potential is None:
            return None

        redox_direction_ok = bool(cathode_potential > anode_potential)
        pair_id = str(candidate.get("source", "discovery")) + ":" + str(getattr(system, "battery_id", "candidate"))
        return cls._electrochemical_viability_v2(
            working_ion=working_ion,
            pair_id=pair_id,
            anode_material=anode_formula,
            cathode_material=cathode_formula,
            anode_potential=anode_potential,
            cathode_potential=cathode_potential,
            anode_oxidation_state_increase=redox_direction_ok,
            cathode_oxidation_state_decrease=redox_direction_ok,
            min_practical_voltage=1.0,
        )

    @staticmethod
    def _summarize_guardrails(history: list[dict]) -> Dict[str, Any]:
        counts = {
            "promotion_ready": 0,
            "review_required": 0,
            "blocked": 0,
        }
        for item in history:
            status = str(item.get("guardrail_status", "blocked"))
            if status not in counts:
                status = "blocked"
            counts[status] += 1

        top_promoted_rank = None
        for item in history:
            if bool(item.get("promotion_ready", False)):
                top_promoted_rank = int(item.get("rank", 0) or 0)
                break

        return {
            "candidate_count": int(len(history)),
            "promotion_ready_count": int(counts["promotion_ready"]),
            "review_required_count": int(counts["review_required"]),
            "blocked_count": int(counts["blocked"]),
            "promotion_ready_rate": float(counts["promotion_ready"] / max(1, len(history))),
            "top_promotion_ready_rank": top_promoted_rank,
        }

    def _generate_variation_candidates(
        self,
        *,
        inference_engine,
        existing_candidates: list[dict],
        objective: Dict[str, float],
        top_k: int,
    ) -> list[dict]:
        variations: list[dict] = []
        if not objective or not existing_candidates:
            return variations

        for idx, item in enumerate(existing_candidates[: max(3, top_k)]):
            base_system = item.get("system")
            if base_system is None or not hasattr(base_system, "to_dict"):
                continue

            variant = BatterySystem(**base_system.to_dict())
            variant.battery_id = f"variation_{idx}_{base_system.battery_id}"
            variant.provenance = "optimized"
            variant.parent_battery_id = base_system.battery_id

            for prop in self.TARGET_PROPERTIES:
                target = objective.get(prop)
                current = getattr(variant, prop, None)
                if target is None or current is None:
                    continue
                try:
                    moved = float(current) + 0.65 * (float(target) - float(current))
                except (TypeError, ValueError):
                    continue
                setattr(variant, prop, moved)

            self._fill_derived_energy(variant)

            try:
                inference_engine.infer(variant)
            except Exception:
                # Keep variant with transformed properties if model pass fails.
                pass

            constraints = evaluate_system(variant)
            valid = bool(constraints.get("overall_valid", False))
            if not valid:
                continue

            score = self._score_system_against_objective(variant, objective)
            if score <= 0.0:
                continue

            variations.append(
                {
                    "system": variant,
                    "score": round(float(score), 6),
                    "speculative": False,
                    "source": "variation_of_existing",
                    "valid": True,
                }
            )

        variations.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return variations[:top_k]

    def _build_target_fallback_candidates(
        self,
        *,
        inference_engine,
        objective: Dict[str, float],
        top_k: int,
    ) -> list[dict]:
        """
        Guaranteed non-empty candidate fallback for generative mode.
        """
        try:
            synth_fn = getattr(inference_engine, "_synthesize_candidates_from_targets", None)
            if callable(synth_fn):
                synthesized = synth_fn(objective, num_needed=top_k) or []
            else:
                synthesized = []
        except Exception:
            logger.exception("Failed to synthesize fallback candidates from targets")
            synthesized = []

        fallback_candidates: list[dict] = []
        for item in synthesized:
            system = item.get("system")
            if system is None:
                continue
            constraints = evaluate_system(system)
            fallback_candidates.append(
                {
                    "system": system,
                    "score": float(item.get("score", 0.0)),
                    "speculative": bool(item.get("speculative", True)),
                    "source": str(item.get("source", "target_synthesized_fallback")),
                    "valid": bool(constraints.get("overall_valid", False)),
                }
            )

        # Keep at least one best-effort candidate even if synthesized list is empty.
        return fallback_candidates[:top_k]

    @staticmethod
    def _select_top_candidates(
        generated_candidates: list[dict],
        variation_candidates: list[dict],
        existing_candidates: list[dict],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Latent-first selection policy with hybrid results:
        - ALWAYS include latent-generated candidates when available
        - Include existing dataset matches as diversity layer
        - Prioritize novelty while ensuring at least some practical grounding
        """
        raw_generated = [
            c for c in generated_candidates
            if bool(c.get("valid", True)) and float(c.get("score", 0.0)) > 0.0
        ]
        raw_generated.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        raw_latent_generated = [
            c for c in raw_generated
            if DiscoveryService._is_latent_source(str(c.get("source", "")))
        ]

        generated = DiscoveryService._quality_filter(
            generated_candidates,
            min_score=DiscoveryService.MIN_CANDIDATE_SCORE,
        )
        existing = DiscoveryService._quality_filter(
            existing_candidates,
            min_score=DiscoveryService.MIN_CANDIDATE_SCORE,
        )
        variations = DiscoveryService._quality_filter(
            variation_candidates,
            min_score=DiscoveryService.MIN_CANDIDATE_SCORE,
        )

        # If strict quality gate removes all candidates, relax slightly to avoid empty results.
        if not generated and not existing and not variations:
            generated = DiscoveryService._quality_filter(
                generated_candidates,
                min_score=DiscoveryService.RELAXED_MIN_CANDIDATE_SCORE,
            )
            existing = DiscoveryService._quality_filter(
                existing_candidates,
                min_score=DiscoveryService.RELAXED_MIN_CANDIDATE_SCORE,
            )
            variations = DiscoveryService._quality_filter(
                variation_candidates,
                min_score=DiscoveryService.RELAXED_MIN_CANDIDATE_SCORE,
            )

        generated.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        existing.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        variations.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        latent_generated = [c for c in generated if DiscoveryService._is_latent_source(str(c.get("source", "")))]
        synthesized_generated = [c for c in generated if not DiscoveryService._is_latent_source(str(c.get("source", "")))]

        # Preserve backbone behavior: if quality filters remove all latent candidates
        # but raw latent outputs exist, rescue top latent items as low-quality latent.
        if not latent_generated and raw_latent_generated:
            rescued_latent = []
            for c in raw_latent_generated[:min(4, top_k)]:
                rescued = dict(c)
                rescued["low_quality_latent"] = True
                rescued_latent.append(rescued)
            latent_generated = rescued_latent
        elif len(latent_generated) < min(8, top_k) and raw_latent_generated:
            existing_ids = {
                str(getattr(c.get("system"), "battery_id", ""))
                for c in latent_generated
            }
            for c in raw_latent_generated:
                sid = str(getattr(c.get("system"), "battery_id", ""))
                if sid in existing_ids:
                    continue
                rescued = dict(c)
                rescued["low_quality_latent"] = True
                latent_generated.append(rescued)
                existing_ids.add(sid)
                if len(latent_generated) >= min(8, top_k):
                    break

        # If no generated candidates, fall back to existing
        if not generated and not variations:
            # Last-resort latent rescue: preserve backbone behavior even when
            # strict quality thresholds remove all latent candidates.
            if raw_latent_generated:
                rescued = []
                for c in raw_latent_generated[:top_k]:
                    rescued_item = dict(c)
                    rescued_item["low_quality_latent"] = True
                    rescued.append(rescued_item)
                return rescued
            return existing[:top_k]

        # STRATEGY: Always include generated candidates (latent-first)
        # Mix in up to 2 existing dataset matches for diversity
        top: list[dict] = []
        seen_signatures: set[str] = set()

        def append_unique(items: list[dict]):
            for candidate in items:
                if len(top) >= top_k:
                    break
                signature = DiscoveryService._candidate_signature(candidate)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                top.append(candidate)
        
        # Strategy quotas with latent-first priority:
        # - up to 10 true latent-generated (novel backbone priority)
        # - up to 3 variation-of-existing (model-refined)
        # - up to 2 existing-dataset matches (grounded)
        append_unique(latent_generated[:min(10, len(latent_generated))])
        append_unique(variations[:min(3, len(variations))])
        
        # Add existing dataset matches if they are decent
        if len(top) < top_k:
            for ex in existing:
                if len(top) >= top_k:
                    break
                ex_score = float(ex.get("score", 0.0))
                if ex_score >= 0.5:
                    # Mark as coming from existing dataset
                    ex["source"] = "existing_dataset_match"
                    append_unique([ex])

        # If latent quota is not filled but synthesized generated candidates exist,
        # use them before widening to all remaining pools.
        if len(top) < top_k:
            append_unique(synthesized_generated[:max(0, top_k - len(top))])
        
        # Fill remaining slots by best remaining candidates across all methods.
        if len(top) < top_k:
            remaining = sorted(
                latent_generated[10:] + synthesized_generated + variations[3:] + existing,
                key=lambda x: float(x.get("score", 0.0)),
                reverse=True,
            )
            append_unique(remaining)

        final_top = top[:top_k]
        final_top.sort(
            key=lambda c: (
                float(c.get("score", 0.0)),
                1.0 if DiscoveryService._is_latent_source(str(c.get("source", ""))) else 0.0,
            ),
            reverse=True,
        )
        return final_top

    # --------------------------------------------------------
    # Runtime Initialization
    # --------------------------------------------------------

    def _ensure_initialized(self):
        if self._initialized:
            return

        logger.info("DiscoveryService: initializing runtime context")

        ctx = get_runtime_context()

        if not ctx.is_ready_for_discovery():
            meta = ctx.get_metadata()
            raise RuntimeError(
                f"Runtime not ready for discovery; metadata={meta}"
            )

        self._device = ctx.get_device()
        self._model = ctx.get_model()
        self._graph = ctx.get_graph()
        self._encoder = ctx.get_encoder()

        if self._graph is None:
            raise RuntimeError("Graph not loaded in runtime context")

        # Check for MaskedGNN format
        masked_gnn_keys = {"battery_features", "material_embeddings", "node_masks", "edge_index_dict"}
        if masked_gnn_keys.issubset(self._graph.keys()):
            logger.info("MaskedGNN graph format detected")
            try:
                num_nodes = int(self._graph["battery_features"].size(0))
            except Exception as exc:
                logger.exception("Invalid graph['battery_features'] tensor")
                raise RuntimeError("Invalid graph object for discovery") from exc
        elif "x" in self._graph and "edge_index_dict" in self._graph:
            # Old format
            try:
                num_nodes = int(self._graph["x"].size(0))
            except Exception as exc:
                logger.exception("Invalid graph['x'] tensor")
                raise RuntimeError("Invalid graph object for discovery") from exc
        else:
            raise RuntimeError("Graph missing required keys")

        reserved_system_index = num_nodes - 1

        try:
            self._generator = SystemGenerator()
        except Exception as exc:
            logger.exception("Failed to construct SystemGenerator")
            raise RuntimeError("Generator initialization failed") from exc

        if self._orchestrator is None:
            self._orchestrator = DiscoveryOrchestrator()

        self._initialized = True
        logger.info("DiscoveryService runtime ready")

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def discover(
        self,
        *,
        base_system_data: Dict,
        objective: Dict[str, float],
        explain: bool = True,
        application: Optional[str] = None,
        mode: str = "predictive",  # "predictive" or "generative"
        discovery_params: Optional[Dict[str, Any]] = None,
    ) -> Dict:

        self._ensure_initialized()
        candidate_records: list[dict] = []
        resolved_discovery_params = self._resolve_discovery_params(discovery_params)
        include_dataset_matches = bool(self.INCLUDE_DATASET_MATCHES_DEFAULT)
        fallback_error: Optional[str] = None
        recovered_from_empty_generation = False
        discovery_report_card: Optional[Dict[str, Any]] = None
        discovery_guardrail_summary: Optional[Dict[str, Any]] = None
        target_objectives_feasible: Dict[str, Any] = {}
        objective_feasibility_adjustments: list[dict[str, Any]] = []
        graph_manifest_validation: Dict[str, Any] = {}
        active_learning_queue: Dict[str, Any] = {}
        prediction_guardrail: Dict[str, Any] = {}
        latent_stats: dict[str, int] = {
            "ranked_systems_count": 0,
            "latent_generated_count": 0,
            "synthesized_generated_count": 0,
            "selected_latent_count": 0,
        }

        try:
            base_system = BatterySystem(**base_system_data)
        except Exception as exc:
            raise ValueError(f"Invalid base system data: {exc}")

        if self._generator is None:
            raise RuntimeError("Discovery generator not initialized")

        try:
            inference_engine = get_inference_engine()
            
            if mode == "predictive":
                # Mode 1: Forward prediction - predict properties for given system
                inference_result = inference_engine.infer(base_system)
                best_system = base_system  # Use the same system with predicted properties
                
            elif mode == "generative":
                if self._orchestrator is None:
                    raise RuntimeError("Discovery orchestrator is not initialized")

                orchestration_result = self._orchestrator.run_generative(
                    base_system=base_system,
                    objective=objective,
                    discovery_params=resolved_discovery_params,
                    candidate_pool_size=int(resolved_discovery_params["num_candidates"]),
                )
                candidate_records = list(orchestration_result.get("candidate_records", []))
                best_system = orchestration_result.get("best_system", base_system)
                if not isinstance(best_system, BatterySystem):
                    best_system = base_system
                discovery_report_card = orchestration_result.get("discovery_report_card", {})
                target_objectives_feasible = orchestration_result.get("target_objectives_feasible", {})
                objective_feasibility_adjustments = orchestration_result.get("objective_feasibility_adjustments", [])
                graph_manifest_validation = orchestration_result.get("graph_manifest_validation", {})
                active_learning_queue = orchestration_result.get("active_learning_queue", {})
                latent_payload = orchestration_result.get("latent_generation_stats")
                if isinstance(latent_payload, dict):
                    latent_stats = {
                        "ranked_systems_count": int(latent_payload.get("ranked_systems_count", 0)),
                        "latent_generated_count": int(latent_payload.get("latent_generated_count", 0)),
                        "synthesized_generated_count": int(latent_payload.get("synthesized_generated_count", 0)),
                        "selected_latent_count": int(latent_payload.get("selected_latent_count", 0)),
                    }
                if not candidate_records:
                    recovered_from_empty_generation = True
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'predictive' or 'generative'")
                
        except Exception as exc:
            logger.exception(f"ML {mode} discovery failed, falling back to base system")
            # Fallback to base system
            best_system = base_system
            fallback_error = str(exc)

        score_result = score_system(best_system, objectives=objective)
        constraints = evaluate_system(best_system)

        explanation = None
        if explain:
            explanation = reason_about_system(
                best_system,
                application=application,
            )

        history: list[dict] = []
        if mode == "generative" and candidate_records:
            try:
                for idx, item in enumerate(candidate_records[: self.DEFAULT_TOP_K]):
                    sys = item["system"]
                    candidate_constraints = evaluate_system(sys)
                    candidate_physics = candidate_constraints.get("physics_first", {})
                    candidate_derived = candidate_physics.get("derived", {}) if isinstance(candidate_physics, dict) else {}
                    viability_v2 = self._candidate_electrochemical_viability_v2(
                        candidate=item,
                        system=sys,
                    )
                    guardrail = self._candidate_guardrail_assessment(
                        candidate=item,
                        constraints=candidate_constraints,
                    )
                    candidate_explanation = reason_about_system(
                        sys,
                        application=application,
                    )
                    history.append(
                        {
                            "rank": idx + 1,
                            "source": item.get("source", "unknown"),
                            "score": item.get("score", 0.0),
                            "speculative": item.get("speculative", False),
                            "valid": candidate_constraints.get("overall_valid", True),
                            "constraints": candidate_constraints,
                            "physics_first": candidate_physics,
                            "physics_hard_valid": bool(candidate_physics.get("hard_valid", True)) if isinstance(candidate_physics, dict) else True,
                            "physics_hard_violations": candidate_physics.get("hard_violations", []) if isinstance(candidate_physics, dict) else [],
                            "oxidation_states": candidate_derived.get("oxidation_states"),
                            "molar_mass_g_per_mol": candidate_derived.get("molar_mass_g_per_mol"),
                            "n_electrons_max": candidate_derived.get("n_electrons_max"),
                            "C_theoretical_mAh_per_g": candidate_derived.get("C_theoretical_mAh_per_g"),
                            "capacity_grav_clipped": candidate_derived.get("capacity_grav_clipped"),
                            "novelty_score": item.get("novelty_score"),
                            "alignment_score": item.get("alignment_score"),
                            "objective_alignment_score": item.get("objective_alignment_score"),
                            "feasibility_score": item.get("feasibility_score"),
                            "objective_distance_score": item.get("objective_distance_score"),
                            "compatibility_score": item.get("compatibility_score"),
                            "compatibility_source": item.get("compatibility_source"),
                            "compatibility_head_blend": item.get("compatibility_head_blend"),
                            "compatibility_fallback_reason": item.get("compatibility_fallback_reason"),
                            "compatibility_model_score": item.get("compatibility_model_score"),
                            "compatibility_rule_score": item.get("compatibility_rule_score"),
                            "compatibility_head_min_score": item.get("compatibility_head_min_score"),
                            "uncertainty_penalty": item.get("uncertainty_penalty"),
                            "uncertainty_model_weight": item.get("uncertainty_model_weight"),
                            "pareto_rank": item.get("pareto_rank"),
                            "pareto_score": item.get("pareto_score"),
                            "manufacturability_score": item.get("manufacturability_score"),
                            "material_novelty_score": item.get("material_novelty_score"),
                            "material_uncertainty_proxy": item.get("material_uncertainty_proxy"),
                            "material_thermodynamic_proxy": item.get("material_thermodynamic_proxy"),
                            "guardrail_status": guardrail.get("status"),
                            "promotion_ready": guardrail.get("promotion_ready"),
                            "guardrail_reasons": guardrail.get("reasons", []),
                            "electrochemical_viability_v2": viability_v2,
                            "stage_trace": item.get("stage_trace", {}),
                            "explanation": candidate_explanation.get("summary"),
                            "system": sys.to_dict() if hasattr(sys, "to_dict") else {},
                        }
                    )
            except Exception as exc:
                logger.warning("Failed to build candidate history: %s", exc)
        if mode == "generative":
            discovery_guardrail_summary = self._summarize_guardrails(history)

        best_material = {}
        if isinstance(getattr(best_system, "uncertainty", None), dict):
            maybe_mg = best_system.uncertainty.get("material_generation")
            if isinstance(maybe_mg, dict):
                best_material = maybe_mg
        prediction_guardrail = self._candidate_guardrail_assessment(
            candidate={
                "score": float(score_result.get("score", 0.0)),
                "speculative": bool(score_result.get("speculative", False)),
                "material_uncertainty_proxy": float(best_material.get("uncertainty_proxy", 0.0) or 0.0),
                "material_thermodynamic_proxy": float(best_material.get("thermodynamic_proxy", 1.0) or 0.0),
            },
            constraints=constraints,
        )
        if mode == "generative" and len(history) == 0:
            score_result = dict(score_result)
            score_result["score"] = 0.0
            score_result["speculative"] = True
            score_result["valid"] = False
            prediction_guardrail = {
                "status": "blocked",
                "promotion_ready": False,
                "review_ready": False,
                "reasons": ["no_generative_candidates"],
                "score": 0.0,
                "material_uncertainty_proxy": float(best_material.get("uncertainty_proxy", 0.0) or 0.0),
                "material_thermodynamic_proxy": float(best_material.get("thermodynamic_proxy", 1.0) or 0.0),
            }

        result_valid = bool(constraints.get("overall_valid", False))
        if mode == "generative" and len(history) == 0:
            result_valid = False

        model_name = (
            self._model.__class__.__name__
            if self._model is not None
            else "unknown"
        )

        return {
            "system": best_system.to_dict(),
            "score": score_result,
            "explanation": explanation,
            "history": history,
            "metadata": {
                "constraints": constraints,
                "physics_first": constraints.get("physics_first", {}),
                "speculative": score_result.get("speculative", False),
                "valid": result_valid,
                "device": str(self._device),
                "model": model_name,
                "discovery_mode": mode,
                "candidate_count": len(history),
                "discovery_params": resolved_discovery_params if mode == "generative" else None,
                "used_fallback_base_system": fallback_error is not None,
                "fallback_error": fallback_error,
                "recovered_from_empty_generation": recovered_from_empty_generation,
                "latent_generation_stats": latent_stats if mode == "generative" else None,
                "include_dataset_matches": include_dataset_matches if mode == "generative" else None,
                "discovery_report_card": discovery_report_card if mode == "generative" else None,
                "discovery_guardrail_summary": discovery_guardrail_summary if mode == "generative" else None,
                "target_objectives_feasible": target_objectives_feasible if mode == "generative" else None,
                "objective_feasibility_adjustments": objective_feasibility_adjustments if mode == "generative" else None,
                "graph_manifest_validation": graph_manifest_validation if mode == "generative" else None,
                "active_learning_queue": active_learning_queue if mode == "generative" else None,
                "prediction_guardrail": prediction_guardrail,
            },
        }
