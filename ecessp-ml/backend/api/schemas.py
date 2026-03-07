# backend/api/schemas.py
# ============================================================
# API Schemas (Canonical Contract Layer)
# ============================================================
# Purpose:
#   - Define strict request/response contracts
#   - Protect backend & ML internals
#   - Enable frontend + external clients safely
#
# NO BUSINESS LOGIC
# NO ML OBJECTS
# ============================================================

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# Base Configuration
# ============================================================

class APISchema(BaseModel):
    """
    Base schema with locked behavior.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    )


# ============================================================
# -------------------- INPUT SCHEMAS -------------------------
# ============================================================

class MaterialInput(APISchema):
    """
    Raw material-level input (CIF-derived or manual).
    """
    material_id: Optional[str] = Field(
        None, description="External or internal material identifier"
    )
    formula: Optional[str] = Field(
        None, description="Chemical formula (e.g. LiFePO4)"
    )
    elements: Optional[List[str]] = Field(
        None, description="List of chemical elements"
    )
    cif_text: Optional[str] = Field(
        None, description="Raw CIF content (string)"
    )


class ObjectiveSchema(APISchema):
    """
    Optimization objective definition.
    """
    objectives: Dict[str, float] = Field(
        ...,
        description="Property targets (e.g. {'energy_grav': 450.0, 'max_delta_volume': 0.12})",
        example={
            "energy_grav": 450.0,
            "max_delta_volume": 0.12,
        },
    )


class BatterySystemSchema(APISchema):
    """
    Minimal system input schema.
    """
    battery_id: Optional[str] = None

    # Electrochemical
    average_voltage: Optional[float] = None
    capacity_grav: Optional[float] = None
    capacity_vol: Optional[float] = None
    energy_grav: Optional[float] = None
    energy_vol: Optional[float] = None

    # Mechanical / Stability
    max_delta_volume: Optional[float] = None
    stability_charge: Optional[float] = None
    stability_discharge: Optional[float] = None

    # Composition
    elements: Optional[List[str]] = None
    working_ion: Optional[str] = None


class DiscoveryParamsSchema(APISchema):
    """
    Optional advanced controls for generative discovery.
    """
    num_candidates: Optional[int] = Field(None, ge=10, le=200)
    diversity_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    novelty_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    extrapolation_strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    interpolation_enabled: Optional[bool] = Field(
        None,
        description="Enable interpolation-mode latent generation.",
    )
    extrapolation_enabled: Optional[bool] = Field(
        None,
        description="Enable extrapolation-mode latent generation.",
    )
    optimize_steps: Optional[int] = Field(None, ge=0, le=128)
    working_ion_candidates: Optional[List[str]] = Field(
        None,
        description="Optional multi-ion search scope (e.g. ['Li','Na','Mg'])",
    )
    material_source_mode: Optional[str] = Field(
        None,
        description="Host-material source mode: 'existing', 'generated', or 'hybrid'",
    )
    component_source_mode: Optional[str] = Field(
        None,
        description="Separator/additive source mode: 'existing', 'generated', or 'hybrid'",
    )
    separator_options_count: Optional[int] = Field(None, ge=1, le=6)
    additive_options_count: Optional[int] = Field(None, ge=1, le=6)


class DiscoveryRequest(APISchema):
    """
    Full discovery request payload.
    """
    system: BatterySystemSchema = Field(..., description="Seed system to optimize from")
    objective: ObjectiveSchema = Field(..., description="Optimization targets")
    application: Optional[str] = Field(None, description="Target application context")
    explain: bool = Field(True, description="Generate reasoning")
    mode: str = Field("generative", description="Discovery mode: 'predictive' or 'generative'")
    discovery_params: Optional[DiscoveryParamsSchema] = Field(
        None,
        description="Optional advanced controls for latent-space discovery",
    )


class SystemGenerationRequest(DiscoveryRequest):
    """
    Canonical alias for DiscoveryRequest.

    Preserved for backward compatibility with earlier
    API versions and frontend integrations.
    """
    pass


class CifDiscoveryRequest(APISchema):
    """
    Request to discover systems starting from a CIF file.
    """
    cif_text: str = Field(..., description="Raw CIF file content")
    objective: ObjectiveSchema = Field(..., description="Optimization targets")
    application: Optional[str] = Field(None, description="Target application context")


class ScoringRequest(APISchema):
    """
    Multi-objective scoring request.
    """
    objectives: Dict[str, float] = Field(
        ..., description="Objective weights {property: weight}"
    )
    top_k: Optional[int] = Field(
        None, ge=1, description="Return only top-k systems"
    )


class ExplanationRequest(APISchema):
    """
    Explain a specific battery system.
    """
    battery_id: str = Field(..., description="Battery system identifier")
    application: Optional[str] = Field(
        None, description="Target application profile"
    )


class ChatRequest(APISchema):
    """
    Conversational request tied to battery systems.
    """
    message: str = Field(..., description="User query")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Optional structured context"
    )
    contextSystemId: Optional[int] = Field(
        None, description="ID of system being discussed"
    )


class PredictionRequest(APISchema):
    """
    Prediction request for material combinations.
    """
    components: Dict[str, str] = Field(
        ..., 
        description="Material selections by component type",
        example={
            "cathode": "LiFePO4",
            "anode": "Graphite", 
            "electrolyte": "LiPF6",
            "separator": "PE",
            "additive": "VC"
        }
    )


class PredictionResponse(APISchema):
    """
    Prediction response for material combinations.
    """
    system_name: str = Field(..., description="Generated system name")
    predicted_properties: Dict[str, float] = Field(
        ..., 
        description="Predicted battery properties",
        example={
            "average_voltage": 3.7,
            "capacity_grav": 150.0,
            "capacity_vol": 400.0,
            "energy_grav": 550.0,
            "energy_vol": 1500.0,
            "max_delta_volume": 0.12,
            "stability_charge": 0.85,
            "stability_discharge": 0.88
        }
    )
    confidence_score: float = Field(..., description="Prediction confidence")
    score: Optional[float] = Field(None, description="Objective-agnostic internal system score")
    diagnostics: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional runtime diagnostics: validity/speculative/uncertainty/guardrails",
    )


# ============================================================
# -------------------- OUTPUT SCHEMAS ------------------------
# ============================================================

class ConstraintReport(APISchema):
    valid: bool
    violations: List[str]
    score_penalty: float = 0.0
    speculative: bool = False


class ConstraintSummary(APISchema):
    overall_valid: bool
    physical: ConstraintReport
    chemical: ConstraintReport
    performance: ConstraintReport


class SystemProperties(APISchema):
    """
    Numerical properties only (frontend-safe).
    """
    average_voltage: Optional[float] = None
    capacity_grav: Optional[float] = None
    capacity_vol: Optional[float] = None
    energy_grav: Optional[float] = None
    energy_vol: Optional[float] = None
    max_delta_volume: Optional[float] = None
    stability_charge: Optional[float] = None
    stability_discharge: Optional[float] = None
    material_level: Optional[Dict[str, Optional[float]]] = None
    cell_level: Optional[Dict[str, Optional[float]]] = None


class BatterySystemResponse(APISchema):
    """
    Canonical system output.
    """
    battery_id: str
    working_ion: Optional[str] = None
    elements: List[str] = Field(default_factory=list)

    properties: SystemProperties
    constraints: Optional[ConstraintSummary] = None

    score: Optional[float] = None
    speculative: bool = False


class DiscoveryResponse(APISchema):
    """
    Structured discovery response.
    """
    system: Dict[str, Any]
    score: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RankedSystemsResponse(APISchema):
    """
    Result of discovery / optimization.
    """
    systems: List[BatterySystemResponse]
    objectives: Dict[str, float]


class ExplanationResponse(APISchema):
    """
    Human-readable explanation.
    """
    battery_id: str
    application: Optional[str]

    valid: bool
    speculative: bool

    summary: str
    strengths: List[str]
    weaknesses: List[str]
    tradeoffs: List[str]
    constraint_notes: List[str]

    material_roles: Dict[str, Any]


class ChatResponse(APISchema):
    """
    Conversational response.
    """
    response: str
    references: Optional[List[str]] = None
    relatedSystems: Optional[List[Any]] = None
