# backend/services/cif_service.py
# ============================================================
# CIF Ingestion & System Translation Service (API-SAFE)
# ============================================================

from __future__ import annotations

from typing import Dict, Any
import uuid
import datetime

from materials.cif_parser import parse_cif_text
from design.system_template import BatterySystem
import logging

logger = logging.getLogger(__name__)


class CifService:
    """
    Canonical CIF → BatterySystem ingestion service.

    This service:
    - Accepts CIF input
    - Uses BatterySystem internally
    - Returns API-safe dict output
    - Preserves uncertainty & provenance
    """

    def __init__(self):
        self.service_name = "CIF Ingestion Service"
        self.version = "2.0.0"

    # ========================================================
    # Public API
    # ========================================================

    def system_from_cif(
        self,
        cif_text: str,
        *,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Convert CIF text into a partial BatterySystem (API-safe dict).

        This is the FIRST step in the reasoning pipeline.
        """

        if not isinstance(cif_text, str) or not cif_text.strip():
            raise ValueError("cif_text must be a non-empty string")

        # ----------------------------------------------------
        # Step 1: CIF → BatterySystem (internal only)
        # ----------------------------------------------------
        try:
            logger.info("Parsing CIF text (len=%s)", len(cif_text))
            system = parse_cif_text(cif_text)
            logger.info("CIF parsed to BatterySystem: %s", getattr(system, 'battery_id', None))
        except Exception as exc:
            logger.exception("CIF parsing failed")
            raise ValueError(f"CIF parsing failed: {exc}") from exc

        if not isinstance(system, BatterySystem):
            raise RuntimeError(
                "parse_cif_text did not return BatterySystem"
            )

        # ----------------------------------------------------
        # Step 2: Identity
        # ----------------------------------------------------
        if not system.battery_id:
            system.battery_id = f"cif-{uuid.uuid4().hex[:12]}"

        # ----------------------------------------------------
        # Step 3: Provenance (internal annotation)
        # ----------------------------------------------------
        system.provenance = {
            "source": "cif_upload",
            "service": self.service_name,
            "version": self.version,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "confidence": "heuristic",
            "pipeline": [
                "parse_cif_text",
                "structure → heuristics → BatterySystem",
            ],
            "user_metadata": metadata or {},
        }

        # ----------------------------------------------------
        # Step 4: API-safe serialization
        # ----------------------------------------------------
        try:
            return system.to_dict()
        except Exception as exc:
            logger.exception("Failed to serialize BatterySystem to dict")
            raise RuntimeError("Failed to serialize parsed BatterySystem") from exc

    # ========================================================
    # Introspection
    # ========================================================

    def describe(self) -> Dict[str, Any]:
        return {
            "service": self.service_name,
            "version": self.version,
            "role": "ingestion",
            "output": "BatterySystem (partial, dict)",
            "deterministic": True,
        }
