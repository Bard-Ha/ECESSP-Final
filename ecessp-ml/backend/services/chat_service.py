# backend/services/chat_service.py
# ============================================================
# Chat Orchestration Service (FINAL · PRODUCTION)
# ============================================================
# Purpose:
#   - Act as the natural-language interface layer
#   - Interpret user intent
#   - Route requests to appropriate backend services
#   - Aggregate structured + explainable responses
#
# This service:
#   - DOES NOT perform ML inference directly
#   - DOES NOT parse CIF or optimize systems itself
#   - ONLY orchestrates existing services
# ============================================================

from __future__ import annotations

from typing import Dict, Any, Optional
import uuid
import datetime

from design.system_template import BatterySystem

from backend.services.discovery_service import DiscoveryService
from backend.services.cif_service import CifService
from backend.services.explain_service import ExplainService
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Supported Chat Intents
# ============================================================

SUPPORTED_INTENTS = {
    "discover": "Run battery system discovery / optimization",
    "explain": "Explain an existing battery system",
    "from_cif": "Create a battery system from CIF input",
    "health": "Backend health & capability check",
}


# ============================================================
# Chat Service
# ============================================================

class ChatService:
    """
    High-level conversational orchestration layer.

    This service:
    - Interprets user intent
    - Routes requests deterministically
    - Returns structured, frontend-friendly responses

    It is intentionally conservative:
    - No free-form hallucination
    - No scientific claims without data
    """

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------
    def __init__(self):
        self.discovery_service = DiscoveryService()
        self.cif_service = CifService()
        self.explain_service = ExplainService()

        self.service_name = "Chat Orchestration Service"
        self.version = "1.0.0"

    # ========================================================
    # Public Chat Entry Point
    # ========================================================

    def handle(
        self,
        intent: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Primary chat entry point.

        Parameters
        ----------
        intent : str
            One of SUPPORTED_INTENTS
        payload : dict
            Intent-specific structured input

        Returns
        -------
        dict
            Structured response for frontend / API
        """

        request_id = f"chat-{uuid.uuid4().hex[:10]}"
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        if intent not in SUPPORTED_INTENTS:
            return self._error_response(
                request_id,
                f"Unsupported intent '{intent}'. "
                f"Supported intents: {list(SUPPORTED_INTENTS)}",
            )

        try:
            if intent == "health":
                result = self._health_check()

            elif intent == "from_cif":
                result = self._handle_from_cif(payload)

            elif intent == "discover":
                result = self._handle_discover(payload)

            elif intent == "explain":
                result = self._handle_explain(payload)

            else:
                raise RuntimeError("Unhandled intent")

        except Exception as exc:
            return self._error_response(
                request_id,
                str(exc),
            )

        return {
            "request_id": request_id,
            "timestamp": timestamp,
            "intent": intent,
            "result": result,
        }

    # ========================================================
    # Intent Handlers
    # ========================================================

    def _handle_from_cif(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        CIF → BatterySystem flow.
        """

        cif_text = payload.get("cif_text")
        metadata = payload.get("metadata", {})

        if not cif_text:
            raise ValueError("Missing 'cif_text' in payload")
        system_dict = self.cif_service.system_from_cif(
            cif_text=cif_text,
            metadata=metadata,
        )

        # `system_from_cif` returns an API-safe dict already.
        return {
            "system": system_dict,
            "provenance": system_dict.get("provenance"),
        }

    def _handle_discover(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discovery / optimization flow.
        """

        base_system_dict = payload.get("base_system")
        objective = payload.get("objective")
        application = payload.get("application")

        if not base_system_dict:
            raise ValueError("Missing 'base_system' in payload")

        if not objective:
            raise ValueError("Missing 'objective' in payload")

        # DiscoveryService.discover expects a dict (base_system_data)
        try:
            return self.discovery_service.discover(
                base_system_data=base_system_dict,
                objective=objective,
                explain=True,
                application=application,
            )
        except Exception:
            logger.exception("Discovery failed for payload")
            raise

    def _handle_explain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explanation-only flow.
        """

        system_dict = payload.get("system")
        application = payload.get("application")

        if not system_dict:
            raise ValueError("Missing 'system' in payload")

        system = BatterySystem.from_dict(system_dict)

        # ExplainService exposes `explain_system`
        try:
            explanation = self.explain_service.explain_system(
                system=system,
                application=application,
            )
        except Exception:
            logger.exception("ExplainService failed")
            raise

        return explanation

    # ========================================================
    # Health & Capability Reporting
    # ========================================================

    def _health_check(self) -> Dict[str, Any]:
        """
        Backend readiness & capability report.
        """

        return {
            "service": self.service_name,
            "version": self.version,
            "status": "healthy",
            "capabilities": SUPPORTED_INTENTS,
            "deterministic": True,
            "ml_execution": "isolated (via runtime context)",
        }

    # ========================================================
    # Error Handling
    # ========================================================

    def _error_response(
        self,
        request_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Standardized error response.
        """

        return {
            "request_id": request_id,
            "error": True,
            "message": message,
        }
