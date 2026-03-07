# backend/runtime/__init__.py
# ============================================================
# Runtime Subsystem Public Interface
# ============================================================
# This package defines the ONLY approved runtime access points.
#
# Exposed:
#   - get_runtime_context → shared model/graph/encoder state
#   - get_inference_engine → deterministic execution engine
#
# Hidden:
#   - internal initialization logic
#   - device resolution
#   - loader orchestration
#
# Importing this package MUST NOT trigger heavy loads.
# ============================================================

from .context import get_runtime_context
from .enhanced_engine import get_enhanced_inference_engine as get_inference_engine

__all__ = [
    "get_runtime_context",
    "get_inference_engine",
]
