# backend/api/routes.py
# ============================================================
# API Routes — Battery System Discovery
# ============================================================

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import Any
import logging
import csv
import hashlib
import os
import re
from pathlib import Path
import threading

from .schemas import (
    DiscoveryRequest,
    CifDiscoveryRequest,
    DiscoveryResponse,
    PredictionRequest,
    PredictionResponse,
)
from ..runtime.context import get_runtime_context
from ..config import MODEL_CONFIG, GRAPH_CONFIG, DATA_DIR
from design.physics_chemistry import parse_formula
from design.electrolyte_stability_model import (
    evaluate_electrolyte_stability,
    sei_requirement_flag,
    thermal_risk_flag,
)
from materials.chemistry_engine import (
    StrictOxidationSolver,
    PolyanionLibrary,
    StructureClassifier,
    InsertionFilter,
    AlkaliValidator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Battery Discovery"],
)

_MATERIALS_LOCK = threading.Lock()
_MATERIALS_CACHE: list[dict[str, str]] | None = None
_STRICT_OX = StrictOxidationSolver()
_POLYANION = PolyanionLibrary()
_STRUCTURE = StructureClassifier()
_INSERTION = InsertionFilter()
_ALKALI = AlkaliValidator()


def _materials_catalog_path() -> Path:
    return DATA_DIR / "processed" / "material_catalog.csv"


def _load_materials_catalog() -> list[dict[str, str]]:
    global _MATERIALS_CACHE
    if _MATERIALS_CACHE is not None:
        return _MATERIALS_CACHE

    with _MATERIALS_LOCK:
        if _MATERIALS_CACHE is not None:
            return _MATERIALS_CACHE

        catalog_path = _materials_catalog_path()
        if not catalog_path.exists():
            raise FileNotFoundError(f"Materials catalog not found: {catalog_path}")

        items: list[dict[str, str]] = []
        with catalog_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "material_id" not in (reader.fieldnames or []):
                raise RuntimeError("Materials catalog missing 'material_id' column")

            for row in reader:
                material_id = (row.get("material_id") or "").strip()
                if not material_id:
                    continue
                name = (row.get("name") or "").strip()
                formula = (row.get("formula") or "").strip()
                display_name = name or (f"{formula} ({material_id})" if formula else material_id)
                items.append(
                    {
                        "material_id": material_id,
                        "name": display_name,
                        "formula": formula,
                    }
                )

        _MATERIALS_CACHE = items
        return _MATERIALS_CACHE

# ============================================================
# Dependency Providers
# ============================================================

def get_discovery_service():
    from backend.runtime.context import get_runtime_context
    ctx = get_runtime_context()
    
    if ctx.is_ready_for_discovery():
        from backend.services.discovery_service import DiscoveryService
        return DiscoveryService()
    else:
        raise RuntimeError("ML runtime not available. Ensure PyTorch is installed and a valid model checkpoint exists.")


def get_cif_service():
    from backend.services.cif_service import CifService
    return CifService()


# ============================================================
# Helpers
# ============================================================

def _ensure_runtime_ready():
    ctx = get_runtime_context()
    if not ctx.is_ready_for_discovery():
        detail = (
            "ML runtime not available. Ensure PyTorch is installed "
            "and a valid model checkpoint exists."
        )
        meta = ctx.get_metadata()
        if meta:
            detail = f"{detail} Runtime metadata: {meta}"

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


# ============================================================
# Routes
# ============================================================

@router.post(
    "/discover",
    response_model=DiscoveryResponse,
    status_code=status.HTTP_200_OK,
)
def discover_system(
    payload: DiscoveryRequest,
    discovery_service = Depends(get_discovery_service),
):
    # Only check runtime for real discovery service, not mock
    from backend.runtime.context import get_runtime_context
    ctx = get_runtime_context()
    if ctx.is_ready_for_discovery():
        _ensure_runtime_ready()

    try:
        return discovery_service.discover(
            base_system_data=payload.system.model_dump(),
            objective=payload.objective.objectives,
            explain=payload.explain,
            application=payload.application,
            mode=payload.mode,
            discovery_params=(
                payload.discovery_params.model_dump(exclude_none=True)
                if payload.discovery_params is not None
                else None
            ),
        )

    except ValueError as exc:
        logger.warning("Bad discovery request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception("Discovery failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Discovery failed",
        )


@router.post(
    "/discover-from-cif",
    response_model=DiscoveryResponse,
    status_code=status.HTTP_200_OK,
)
def discover_from_cif(
    payload: CifDiscoveryRequest,
    discovery_service = Depends(get_discovery_service),
    cif_service = Depends(get_cif_service),
):
    _ensure_runtime_ready()

    try:
        system = cif_service.system_from_cif(payload.cif_text)
        logger.info("CIF converted to system")
    except ValueError as exc:
        logger.warning("Invalid CIF input: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("CIF processing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CIF processing failed",
        )

    try:
        return discovery_service.discover(
            base_system_data=system,
            objective=payload.objective.objectives,
            explain=True,
            application=payload.application,
            mode="generative",
        )

    except Exception as exc:
        logger.exception("Discovery from CIF failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Discovery failed",
        )


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    ctx = get_runtime_context()
    return {
        "status": "ok",
        "runtime_ready": ctx.is_ready_for_discovery(),
        "service": "ecessp-ml",
    }


@router.get("/materials", status_code=status.HTTP_200_OK)
def list_materials(
    query: str = Query("", description="Case-insensitive substring filter across material_id, name, and formula"),
    limit: int = Query(300, ge=1, le=300),
    offset: int = Query(0, ge=0),
):
    """
    List materials sourced from processed atomic embeddings.
    """
    try:
        materials = _load_materials_catalog()
    except Exception as exc:
        logger.exception("Failed to load materials catalog: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load materials catalog",
        )

    q = query.strip().lower()
    if q:
        filtered = [
            m for m in materials
            if (
                q in m["material_id"].lower()
                or q in m.get("name", "").lower()
                or q in m.get("formula", "").lower()
            )
        ]
    else:
        filtered = materials

    total = len(filtered)
    page = filtered[offset : offset + limit]

    return {
        "items": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "source": str(_materials_catalog_path().name),
    }


@router.get("/runtime-diagnostics", status_code=status.HTTP_200_OK)
def runtime_diagnostics():
    """
    Startup/runtime observability endpoint with explicit readiness reasons.
    """
    ctx = get_runtime_context()
    metadata = ctx.get_metadata()

    checkpoint_path = str(MODEL_CONFIG.checkpoint_path)
    graph_path = str(GRAPH_CONFIG.graph_dir / GRAPH_CONFIG.filename)
    checkpoint_exists = MODEL_CONFIG.checkpoint_path.exists()
    graph_exists = (GRAPH_CONFIG.graph_dir / GRAPH_CONFIG.filename).exists()
    runtime_ready = ctx.is_ready_for_discovery()

    reasons = []
    if not checkpoint_exists:
        reasons.append(f"checkpoint_missing:{checkpoint_path}")
    if not graph_exists:
        reasons.append(f"graph_missing:{graph_path}")
    if metadata.get("errors"):
        reasons.extend([f"runtime_error:{err}" for err in metadata["errors"]])
    if not runtime_ready and not reasons:
        reasons.append("runtime_not_initialized")

    return {
        "status": "ok",
        "service": "ecessp-ml",
        "runtime_ready": runtime_ready,
        "ready_reason": "ready" if runtime_ready else "not_ready",
        "not_ready_reasons": reasons,
        "paths": {
            "checkpoint": checkpoint_path,
            "checkpoint_exists": checkpoint_exists,
            "graph": graph_path,
            "graph_exists": graph_exists,
        },
        "metadata": metadata,
    }


def _resolve_component_formula(component_value: str | None) -> str:
    """
    Resolve component IDs to chemical formulas using the materials catalog.
    If no match is found, return the raw component string.
    """
    raw = str(component_value or "").strip()
    if not raw:
        return raw
    try:
        catalog = _load_materials_catalog()
    except Exception:
        return raw

    for item in catalog:
        material_id = str(item.get("material_id") or "").strip()
        if material_id != raw:
            continue
        formula = str(item.get("formula") or "").strip()
        return formula or raw
    return raw


def _estimate_component_potential(formula: str, role: str, working_ion: str) -> float | None:
    text = str(formula or "").strip()
    if not text:
        return None
    try:
        comp = parse_formula(text)
    except Exception:
        return None

    if role == "anode" and "C" in comp and "O" not in comp:
        return 0.2

    cath_map = {"Mn": 3.8, "Fe": 3.4, "Co": 3.9, "Ni": 4.0, "V": 3.3, "Cr": 3.2, "Ti": 2.2, "Cu": 3.6}
    an_map = {"Mn": 1.2, "Fe": 0.9, "Co": 1.1, "Ni": 1.0, "V": 1.3, "Cr": 1.0, "Ti": 0.8, "Cu": 1.2, "C": 0.2, "Si": 0.25, "Sn": 0.45}
    table = cath_map if role == "cathode" else an_map

    num = 0.0
    den = 0.0
    for el, coeff in comp.items():
        if el in table:
            num += float(coeff) * float(table[el])
            den += float(coeff)
    if den <= 0.0:
        base = 3.3 if role == "cathode" else 1.0
    else:
        base = num / den

    ion = str(working_ion or "Li").strip()
    if ion == "Na":
        base -= 0.2
    elif ion == "Mg":
        base -= 0.35
    elif ion == "K":
        base -= 0.25
    return float(base)


def _electrochemical_viability_v2(
    *,
    working_ion: str,
    pair_id: str,
    anode_material: str,
    cathode_material: str,
    anode_potential: float | None,
    cathode_potential: float | None,
    anode_state_change: str | None = None,
    cathode_state_change: str | None = None,
    anode_oxidation_state_increase: bool | None = None,
    cathode_oxidation_state_decrease: bool | None = None,
    min_practical_voltage: float = 1.0,
) -> dict[str, Any]:
    v_cell: float | None = None
    if cathode_potential is not None and anode_potential is not None:
        v_cell = float(cathode_potential - anode_potential)

    redox_gap_pass = bool(v_cell is not None and v_cell > 0.0)
    practical_voltage_pass = bool(v_cell is not None and v_cell >= float(min_practical_voltage))

    directional_result: bool | None
    if anode_oxidation_state_increase is None or cathode_oxidation_state_decrease is None:
        directional_result = None
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
        "definitions": {
            "Anode": "Site of Oxidation (electron loss) during discharge. Must have LOWER potential.",
            "Cathode": "Site of Reduction (electron gain) during discharge. Must have HIGHER potential.",
        },
        "validation_rules": [
            {
                "rule_id": "REDOX_POTENTIAL_GAP",
                "logic": "E_cathode - E_anode > 0",
                "unit": "Volts (V) vs Li/Li+",
                "failure_action": "REJECT",
                "reason": "The cell is thermodynamically inactive; electrons will not flow spontaneously.",
                "result": bool(redox_gap_pass),
            },
            {
                "rule_id": "DIRECTIONALITY_CHECK",
                "check": "Anode_Oxidation_State_Increase == True && Cathode_Oxidation_State_Decrease == True",
                "context": "During Discharge",
                "failure_action": "WARNING",
                "reason": "Material redox centers are incompatible with assigned roles.",
                "result": directional_result,
            },
        ],
        "candidate_evaluation": {
            "ion": str(working_ion),
            "pair_id": str(pair_id),
            "anode": {
                "material": str(anode_material),
                "role": "Oxidation",
                "estimated_potential_V": anode_potential,
                "state_change": anode_state_change,
            },
            "cathode": {
                "material": str(cathode_material),
                "role": "Reduction",
                "estimated_potential_V": cathode_potential,
                "state_change": cathode_state_change,
            },
        },
        "screening_output": {
            "theoretical_voltage": v_cell,
            "is_viable": bool(redox_gap_pass and practical_voltage_pass),
            "recommendation": recommendation,
            "minimum_practical_voltage": float(min_practical_voltage),
        },
    }


def _predictive_chemistry_gate(
    *,
    working_ion: str,
    cathode_formula: str,
    anode_formula: str,
    electrolyte_formula: str,
    dual_ion_mode: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"working_ion": working_ion}

    def _validate_formula(role: str, formula: str) -> dict[str, Any]:
        role_report: dict[str, Any] = {"formula": formula}
        strict = _STRICT_OX.solve_formula(formula)
        role_report["strict_oxidation"] = {
            "valid": bool(strict.valid),
            "unique_solution": bool(strict.unique_solution),
            "oxidation_states": dict(strict.oxidation_states),
            "c_theoretical_mAh_g": strict.c_theoretical_mAh_g,
            "reasons": list(strict.reasons),
        }
        if not strict.valid:
            reasons.append(f"{role}:strict_oxidation_unsolved")

        poly = _POLYANION.validate_formula(formula)
        role_report["polyanion"] = {
            "valid": bool(poly.valid),
            "recognized_units": list(poly.recognized_units),
            "reasons": list(poly.reasons),
        }
        if not poly.valid:
            reasons.extend([f"{role}:{r}" for r in poly.reasons])

        structure = _STRUCTURE.classify_formula(formula, working_ion=working_ion)
        role_report["structure"] = {
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
            reasons.extend([f"{role}:{r}" for r in structure.reasons])

        insertion = _INSERTION.evaluate_formula(formula, structure_family=structure.family)
        role_report["insertion_filter"] = {
            "valid": bool(insertion.valid),
            "insertion_probability": float(insertion.insertion_probability),
            "reasons": list(insertion.reasons),
        }
        if not insertion.valid:
            reasons.extend([f"{role}:{r}" for r in insertion.reasons])

        alk = _ALKALI.validate_formula(formula, working_ion, dual_ion_mode=bool(dual_ion_mode))
        role_report["alkali"] = {
            "valid": bool(alk.valid),
            "alkali_elements_present": list(alk.alkali_elements_present),
            "reasons": list(alk.reasons),
        }
        if not alk.valid:
            reasons.extend([f"{role}:{r}" for r in alk.reasons])
        return role_report

    cath = _validate_formula("cathode", cathode_formula)
    an = _validate_formula("anode", anode_formula)
    details["cathode"] = cath
    details["anode"] = an

    electrolyte = _ALKALI.validate_formula(
        electrolyte_formula,
        working_ion,
        dual_ion_mode=bool(dual_ion_mode),
    )
    details["electrolyte"] = {
        "formula": electrolyte_formula,
        "alkali": {
            "valid": bool(electrolyte.valid),
            "alkali_elements_present": list(electrolyte.alkali_elements_present),
            "reasons": list(electrolyte.reasons),
        },
    }
    if not electrolyte.valid:
        reasons.extend([f"electrolyte:{r}" for r in electrolyte.reasons])

    cath_v = _estimate_component_potential(cathode_formula, "cathode", working_ion)
    an_v = _estimate_component_potential(anode_formula, "anode", working_ion)
    v_cell = None
    if cath_v is not None and an_v is not None:
        v_cell = float(cath_v - an_v)
        if v_cell <= 1.0:
            reasons.append("electrochemistry:voltage_inverted_or_too_low")
    redox_direction_ok = None
    if cath_v is not None and an_v is not None:
        redox_direction_ok = bool(cath_v > an_v)
    electro_v2 = _electrochemical_viability_v2(
        working_ion=working_ion,
        pair_id=f"{working_ion}:{cathode_formula}|{anode_formula}",
        anode_material=anode_formula,
        cathode_material=cathode_formula,
        anode_potential=(float(an_v) if an_v is not None else None),
        cathode_potential=(float(cath_v) if cath_v is not None else None),
        anode_oxidation_state_increase=redox_direction_ok,
        cathode_oxidation_state_decrease=redox_direction_ok,
        min_practical_voltage=1.0,
    )
    v2_rules = {
        str(item.get("rule_id")): item
        for item in electro_v2.get("validation_rules", [])
        if isinstance(item, dict)
    }
    if not bool(v2_rules.get("REDOX_POTENTIAL_GAP", {}).get("result", True)):
        reasons.append("electrochemistry:redox_potential_gap_failed")
    if v2_rules.get("DIRECTIONALITY_CHECK", {}).get("result", True) is False:
        warnings.append("electrochemistry:directionality_inconsistent")
    electrolyte_eval = evaluate_electrolyte_stability(
        electrolyte=electrolyte_formula,
        working_ion=working_ion,
        cathode_potential=float(cath_v or 0.0),
        anode_potential=float(an_v or 0.0),
    )
    if not bool(electrolyte_eval.valid):
        reasons.append("electrochemistry:electrolyte_stability_window_violation")
    cath_ox_states = cath.get("strict_oxidation", {}).get("oxidation_states", {})
    max_cath_ox = max((float(v) for v in cath_ox_states.values()), default=0.0)
    sei_expected = sei_requirement_flag(
        anode_potential=float(an_v or 0.0),
        working_ion=working_ion,
    )
    thermal_risk = thermal_risk_flag(
        cathode_potential=float(cath_v or 0.0),
        max_oxidation_state=max_cath_ox,
    )
    details["electrochemistry"] = {
        "cathode_potential": cath_v,
        "anode_potential": an_v,
        "v_cell": v_cell,
        "minimum_v_cell_required": 1.0,
        "electrolyte_window": {
            "reduction_limit": float(electrolyte_eval.window.reduction_limit),
            "oxidation_limit": float(electrolyte_eval.window.oxidation_limit),
            "source": electrolyte_eval.window.source,
            "reduction_ok": bool(electrolyte_eval.reduction_ok),
            "oxidation_ok": bool(electrolyte_eval.oxidation_ok),
            "valid": bool(electrolyte_eval.valid),
        },
        "interface_flags": {
            "sei_expected": bool(sei_expected),
            "thermal_risk": bool(thermal_risk),
        },
        "viability_v2": electro_v2,
    }

    c_cath = cath.get("strict_oxidation", {}).get("c_theoretical_mAh_g")
    c_an = an.get("strict_oxidation", {}).get("c_theoretical_mAh_g")
    np_ratio = None
    if isinstance(c_cath, (int, float)) and isinstance(c_an, (int, float)) and float(c_cath) > 0:
        np_ratio = float(c_an) / max(float(c_cath), 1e-6)
        if not (1.05 <= np_ratio <= 1.2):
            reasons.append("electrochemistry:np_ratio_out_of_bounds")
    details["balancing"] = {
        "c_theoretical_cathode": c_cath,
        "c_theoretical_anode": c_an,
        "np_ratio": np_ratio,
        "np_ratio_target_range": [1.05, 1.2],
    }

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "warnings": warnings,
        "details": details,
    }


def _components_to_feature_hint(components: dict[str, str]) -> dict[str, float]:
    """
    Create deterministic feature hints from selected components.
    This ensures /predict output changes with selected materials.
    """
    joined = "|".join(f"{k}:{components.get(k,'')}" for k in sorted(components.keys()))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    seed = int(digest[:12], 16)

    # Stable normalized [0, 1)
    u1 = ((seed >> 0) & 0xFFFF) / 65535.0
    u2 = ((seed >> 16) & 0xFFFF) / 65535.0
    u3 = ((seed >> 32) & 0xFFFF) / 65535.0
    u4 = ((seed >> 48) & 0xFFFF) / 65535.0

    # Reasonable electrochemical ranges for seed system hints
    return {
        "average_voltage": 2.5 + 2.0 * u1,       # 2.5 - 4.5 V
        "capacity_grav": 80.0 + 220.0 * u2,      # 80 - 300 mAh/g
        "capacity_vol": 250.0 + 950.0 * u3,      # 250 - 1200
        "energy_grav": 180.0 + 520.0 * u4,       # 180 - 700 Wh/kg
        "energy_vol": 600.0 + 2200.0 * u2,       # 600 - 2800 Wh/L
        "stability_charge": -0.05 + 0.20 * u3,   # -0.05 - 0.15
        "stability_discharge": -0.05 + 0.20 * u4,# -0.05 - 0.15
    }


def _infer_working_ion(components: dict[str, str]) -> str:
    text = " ".join(str(v or "") for v in components.values()).lower()
    patterns = (
        ("Na", (r"\bsodium\b", r"\bna[\w\(\)\+\-]*\b")),
        ("Mg", (r"\bmagnesium\b", r"\bmg[\w\(\)\+\-]*\b")),
        ("K", (r"\bpotassium\b", r"\bk[\w\(\)\+\-]*\b")),
        ("Ca", (r"\bcalcium\b", r"\bca[\w\(\)\+\-]*\b")),
        ("Zn", (r"\bzinc\b", r"\bzn[\w\(\)\+\-]*\b")),
        ("Al", (r"\baluminum\b", r"\baluminium\b", r"\bal[\w\(\)\+\-]*\b")),
        ("Y", (r"\byttrium\b",)),
        ("Li", (r"\blithium\b", r"\bli[\w\(\)\+\-]*\b")),
    )
    for ion, ion_patterns in patterns:
        if any(re.search(pattern, text) for pattern in ion_patterns):
            return ion
    return "Li"


def _prediction_confidence(
    *,
    system_score: float,
    valid: bool,
    speculative: bool,
    uncertainty_penalty: float,
    guardrail_status: str,
) -> float:
    base = 0.20 + 0.55 * system_score + 0.15 * (1.0 if valid else 0.0) + 0.10 * (1.0 - uncertainty_penalty)
    if speculative:
        base *= 0.90
    if not valid:
        base *= 0.75
    if guardrail_status == "reject":
        base *= 0.80
    return max(0.05, min(0.99, float(base)))


def _resolve_prediction_uncertainty_policy(
    *,
    uncertainty_penalty: float,
    guardrail_status: str,
    chemistry_valid: bool,
    feature_hint: dict[str, float],
) -> dict[str, Any]:
    mode = str(os.getenv("ECESSP_PREDICTION_UNCERTAINTY_MODE", "reject")).strip().lower()
    if mode not in {"reject", "fallback", "explain"}:
        mode = "reject"

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    medium_threshold = _env_float("ECESSP_PREDICTION_UNCERTAINTY_MEDIUM", 0.20)
    high_threshold = max(
        medium_threshold + 1e-6,
        _env_float("ECESSP_PREDICTION_UNCERTAINTY_HIGH", 0.35),
    )
    reject_on_guardrail = str(os.getenv("ECESSP_REJECT_ON_GUARDRAIL_REJECT", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if uncertainty_penalty >= high_threshold:
        severity = "high"
    elif uncertainty_penalty >= medium_threshold:
        severity = "medium"
    else:
        severity = "low"

    reject_reasons: list[str] = []
    if not chemistry_valid:
        reject_reasons.append("chemistry_gate_invalid")
    if reject_on_guardrail and guardrail_status == "reject":
        reject_reasons.append("guardrail_reject")
    if mode == "reject" and severity == "high":
        reject_reasons.append("high_uncertainty")

    if reject_reasons:
        action = "reject"
    elif mode == "fallback" and severity == "high":
        action = "fallback"
    elif mode == "explain" and severity in {"medium", "high"}:
        action = "explain"
    elif mode == "reject" and severity == "medium":
        action = "explain"
    else:
        action = "accept"

    fallback_properties = {
        "average_voltage": float(feature_hint.get("average_voltage", 0.0)),
        "capacity_grav": float(feature_hint.get("capacity_grav", 0.0)),
        "capacity_vol": float(feature_hint.get("capacity_vol", 0.0)),
        "energy_grav": float(feature_hint.get("energy_grav", 0.0)),
        "energy_vol": float(feature_hint.get("energy_vol", 0.0)),
        "max_delta_volume": 0.0,
        "stability_charge": float(feature_hint.get("stability_charge", 0.0)),
        "stability_discharge": float(feature_hint.get("stability_discharge", 0.0)),
    }

    return {
        "mode": mode,
        "action": action,
        "severity": severity,
        "thresholds": {
            "medium": float(medium_threshold),
            "high": float(high_threshold),
        },
        "reject_on_guardrail": bool(reject_on_guardrail),
        "reasons": reject_reasons,
        "fallback_properties": fallback_properties,
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_system(
    payload: PredictionRequest,
    discovery_service = Depends(get_discovery_service),
):
    _ensure_runtime_ready()

    try:
        feature_hint = _components_to_feature_hint(payload.components)
        joined = "|".join(f"{k}:{payload.components.get(k,'')}" for k in sorted(payload.components.keys()))
        stable_id = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:10]
        working_ion = _infer_working_ion(payload.components)
        cathode_formula = _resolve_component_formula(payload.components.get("cathode"))
        anode_formula = _resolve_component_formula(payload.components.get("anode"))
        electrolyte_formula = _resolve_component_formula(payload.components.get("electrolyte"))

        # Create base system hint from selected materials so predictions are component-dependent.
        base_system_data = {
            'battery_id': f"predicted_{stable_id}",
            'provenance': 'prediction',
            'battery_type': 'insertion',
            'working_ion': working_ion,
            'cathode_material': cathode_formula or payload.components.get('cathode'),
            'anode_material': anode_formula or payload.components.get('anode'),
            'electrolyte': electrolyte_formula or payload.components.get('electrolyte'),
            'separator_material': payload.components.get('separator'),
            'additive_material': payload.components.get('additives', payload.components.get('additive')),
            'framework_formula': cathode_formula or payload.components.get('cathode', ''),
            'average_voltage': feature_hint['average_voltage'],
            'capacity_grav': feature_hint['capacity_grav'],
            'capacity_vol': feature_hint['capacity_vol'],
            'energy_grav': feature_hint['energy_grav'],
            'energy_vol': feature_hint['energy_vol'],
            'stability_charge': feature_hint['stability_charge'],
            'stability_discharge': feature_hint['stability_discharge'],
        }
        
        # Use predictive mode to get properties
        result = discovery_service.discover(
            base_system_data=base_system_data,
            objective={},  # No specific objectives for prediction
            explain=False,
            application=None,
            mode='predictive'
        )
        
        # Extract predicted properties from the result
        predicted_properties = result['system']
        metadata = result.get("metadata", {}) or {}
        score_payload = result.get("score", {}) or {}
        model_heads = (
            (predicted_properties.get("uncertainty") or {}).get("model_heads", {})
            if isinstance(predicted_properties.get("uncertainty"), dict)
            else {}
        )
        uncertainty_penalty = float(model_heads.get("uncertainty_penalty", 0.15) or 0.15)
        compatibility_score = model_heads.get("compatibility_score_aggregate")
        role_probabilities = model_heads.get("role_probabilities", {})
        valid = bool(metadata.get("valid", False))
        speculative = bool(score_payload.get("speculative", False))
        chemistry_gate = _predictive_chemistry_gate(
            working_ion=working_ion,
            cathode_formula=cathode_formula or str(payload.components.get("cathode") or ""),
            anode_formula=anode_formula or str(payload.components.get("anode") or ""),
            electrolyte_formula=electrolyte_formula or str(payload.components.get("electrolyte") or ""),
        )
        chemistry_valid = bool(chemistry_gate.get("valid", False))
        if not chemistry_valid:
            valid = False
        raw_objective_score = float(score_payload.get("score", 0.0) or 0.0)
        base_score = raw_objective_score if raw_objective_score > 0.0 else (0.65 if valid else 0.35)
        predictive_score = max(0.0, min(1.0, float(base_score) * (1.0 - float(uncertainty_penalty))))
        guardrail = metadata.get("prediction_guardrail", {}) if isinstance(metadata, dict) else {}
        guardrail_status = str(guardrail.get("status", "unknown"))
        if not chemistry_valid:
            guardrail_status = "reject"
            predictive_score = max(0.05, min(0.65, predictive_score * 0.65))
        confidence_score = _prediction_confidence(
            system_score=predictive_score,
            valid=valid,
            speculative=speculative,
            uncertainty_penalty=uncertainty_penalty,
            guardrail_status=guardrail_status,
        )
        def _num(value: object, default: float = 0.0) -> float:
            try:
                v = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return float(default)
            return float(v)
        # Filter to only the target properties
        target_properties = {
            'average_voltage': _num(predicted_properties.get('average_voltage')),
            'capacity_grav': _num(predicted_properties.get('capacity_grav')),
            'capacity_vol': _num(predicted_properties.get('capacity_vol')),
            'energy_grav': _num(predicted_properties.get('energy_grav')),
            'energy_vol': _num(predicted_properties.get('energy_vol')),
            'max_delta_volume': _num(predicted_properties.get('max_delta_volume')),
            'stability_charge': _num(predicted_properties.get('stability_charge')),
            'stability_discharge': _num(predicted_properties.get('stability_discharge'))
        }
        uncertainty_policy = _resolve_prediction_uncertainty_policy(
            uncertainty_penalty=uncertainty_penalty,
            guardrail_status=guardrail_status,
            chemistry_valid=chemistry_valid,
            feature_hint=feature_hint,
        )
        policy_action = str(uncertainty_policy.get("action", "accept"))
        if policy_action == "fallback":
            target_properties = dict(uncertainty_policy["fallback_properties"])
            predictive_score = min(float(predictive_score), 0.45)
            confidence_score = min(float(confidence_score), 0.55)
        elif policy_action == "explain":
            predictive_score = min(float(predictive_score), 0.70)
            confidence_score = min(float(confidence_score), 0.70)
        elif policy_action == "reject":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "Prediction rejected by uncertainty policy",
                    "policy": uncertainty_policy,
                },
            )
        
        return PredictionResponse(
            system_name=(
                f"C:{payload.components.get('cathode', 'Unknown')} | "
                f"A:{payload.components.get('anode', 'Unknown')} | "
                f"E:{payload.components.get('electrolyte', 'Unknown')} | "
                f"S:{payload.components.get('separator', 'Unknown')} | "
                f"Ad:{payload.components.get('additives', payload.components.get('additive', 'Unknown'))}"
            ),
            predicted_properties=target_properties,
            confidence_score=confidence_score,
            score=predictive_score,
            diagnostics={
                "valid": valid,
                "speculative": speculative,
                "uncertainty_penalty": uncertainty_penalty,
                "compatibility_score": compatibility_score,
                "role_probabilities": role_probabilities if isinstance(role_probabilities, dict) else {},
                "guardrail_status": guardrail_status,
                "raw_objective_score": raw_objective_score,
                "chemistry_gate_valid": chemistry_valid,
                "chemistry_gate_reasons": list(chemistry_gate.get("reasons", [])),
                "chemistry_gate_details": chemistry_gate.get("details", {}),
                "uncertainty_policy": uncertainty_policy,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        )
