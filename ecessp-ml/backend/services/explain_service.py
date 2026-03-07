# backend/services/explain_service.py
# ============================================================
# Explainability Service (PRODUCTION)
# ============================================================
# Purpose:
#   - Generate structured, human-readable explanations
#   - Bind system reasoning + constraints
#   - Serve frontend, API, and chat layers
#
# This service performs NO inference.
# ============================================================

from __future__ import annotations

from typing import Dict, Optional

from design.system_template import BatterySystem
from design.system_reasoner import reason_about_system
from design.system_constraints import evaluate_system


# ============================================================
# Explain Service
# ============================================================

class ExplainService:
    """
    High-level explainability interface.

    This service:
    - Explains why a system is good/bad
    - Surfaces trade-offs and risks
    - Labels speculative regimes
    - Adds application context
    """

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def explain_system(
        self,
        system: BatterySystem,
        application: Optional[str] = None,
    ) -> Dict:
        """
        Generate a structured explanation for a battery system.

        Parameters
        ----------
        system : BatterySystem
            Fully populated system object
        application : str, optional
            Application profile (e.g. "EV", "Grid", "Aerospace")

        Returns
        -------
        dict
            JSON-serializable explanation
        """

        if not isinstance(system, BatterySystem):
            raise TypeError(
                "ExplainService expects a BatterySystem instance"
            )

        # ----------------------------------------------------
        # Constraint evaluation (ground truth)
        # ----------------------------------------------------
        constraints = evaluate_system(system)

        # ----------------------------------------------------
        # Core reasoning
        # ----------------------------------------------------
        explanation = reason_about_system(
            system=system,
            application=application,
        )

        # ----------------------------------------------------
        # Attach constraint metadata
        # ----------------------------------------------------
        explanation["constraints"] = constraints

        explanation["confidence"] = self._estimate_confidence(
            constraints=constraints
        )

        return explanation

    # --------------------------------------------------------
    # Confidence Estimation (Heuristic)
    # --------------------------------------------------------
    def _estimate_confidence(self, constraints: Dict) -> float:
        """
        Estimate explanation reliability based on constraint status.

        Returns
        -------
        float in [0, 1]
        """

        confidence = 1.0

        # Hard invalid → zero confidence
        if not constraints["overall_valid"]:
            return 0.0

        # Soft penalties reduce confidence
        perf = constraints["performance"]

        confidence -= perf.get("score_penalty", 0.0)

        if perf.get("speculative", False):
            confidence -= 0.2

        return round(max(0.0, min(confidence, 1.0)), 3)
