# backend/services/__init__.py
# ============================================================
# Backend Service Layer
# ============================================================
# Purpose:
#   - Expose the canonical service interfaces
#   - Define the public backend API surface
#   - Prevent accidental internal imports
#
# Only services listed in __all__ are considered stable.
# ============================================================

"""backend.services
Public service names only — avoid importing submodules at package import time.

This file intentionally does NOT import service modules to prevent heavy
initialization (e.g., torch/model loading) when code does `import backend.services`.
Import service implementations directly from their modules instead, or use
lazy providers from `backend.api.routes`.
"""

__all__ = [
    "DiscoveryService",
    "CifService",
    "ExplainService",
    "ChatService",
]
