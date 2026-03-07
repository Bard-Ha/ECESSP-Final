# backend/__init__.py
# ============================================================
# Backend Package Boundary
# ============================================================
# Purpose:
#   - Define backend as a cohesive runtime + service layer
#   - Control what is importable from "backend"
#   - Prevent accidental coupling with design / models
#
# The backend:
#   - Owns runtime initialization
#   - Exposes services
#   - Never exposes loaders or internals directly
# ============================================================

from .runtime.context import get_runtime_context

# Avoid importing service modules at package import time because some
# services import heavy ML libraries (torch, numpy) which may not be
# available in lightweight runtime environments. Import services
# lazily from their modules when needed.

__all__ = [
    "get_runtime_context",
]
