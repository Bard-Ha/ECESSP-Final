# backend/api/__init__.py
# ============================================================
# API Package
# ============================================================
# Purpose:
#   - Define the public API boundary
#   - Expose request/response schemas
#   - Prevent leakage of internal logic
#
# This package is SAFE to import at startup.
# ============================================================

from .schemas import (
    # --------------------
    # Input schemas
    # --------------------
    MaterialInput,
    BatterySystemSchema,
    ObjectiveSchema,
    DiscoveryRequest,
    SystemGenerationRequest,  # backward compatibility
    CifDiscoveryRequest,
    ScoringRequest,
    ExplanationRequest,
    ChatRequest,

    # --------------------
    # Output schemas
    # --------------------
    BatterySystemResponse,
    RankedSystemsResponse,
    DiscoveryResponse,
    ExplanationResponse,
    ChatResponse,

    # --------------------
    # Shared / constraints
    # --------------------
    ConstraintReport,
    ConstraintSummary,
    SystemProperties,
)

__all__ = [
    # Input
    "MaterialInput",
    "BatterySystemSchema",
    "ObjectiveSchema",
    "DiscoveryRequest",
    "SystemGenerationRequest",
    "CifDiscoveryRequest",
    "ScoringRequest",
    "ExplanationRequest",
    "ChatRequest",

    # Output
    "BatterySystemResponse",
    "RankedSystemsResponse",
    "DiscoveryResponse",
    "ExplanationResponse",
    "ChatResponse",

    # Shared
    "ConstraintReport",
    "ConstraintSummary",
    "SystemProperties",
]
