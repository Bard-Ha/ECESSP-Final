# ============================================================
# Battery System Reasoner (ECESSP  TARGET-DRIVEN)
# ============================================================

from typing import Dict, List, Tuple, Optional, Any

from .system_template import BatterySystem
from .system_constraints import evaluate_system


# ============================================================
# Canonical Target Properties (LOCKED)
# ============================================================

TARGET_PROPERTIES = {
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "max_delta_volume",
    "stability_charge",
    "stability_discharge",
}


# ============================================================
# SystemReasoner
# ============================================================

class SystemReasoner:
    """
    Authoritative reasoning layer for ECESSP.

    Responsibilities:
     Enforce physical + chemical feasibility (discovery)
     Enforce user-defined target ranges (discovery)
     Produce schema-compatible explanations (prediction)

    This class:
     NEVER generates systems
     NEVER runs ML
     NEVER guesses objectives
    """

    # --------------------------------------------------------
    # DISCOVERY: hard feasibility filtering
    # --------------------------------------------------------
    def filter_generated_systems(
        self,
        systems: List[BatterySystem],
        target_ranges: Dict[str, Tuple[float, float]],
    ) -> Tuple[List[BatterySystem], List[Dict[str, Any]]]:
        """
        Filters GENERATED systems based on:
          1) Physical / chemical constraints
          2) User-defined target property ranges

        Returns:
           feasible BatterySystem objects
           rejected system records (traceable, frontend-safe)
        """

        feasible: List[BatterySystem] = []
        rejected: List[Dict[str, Any]] = []

        def cell_or_system_value(sys: BatterySystem, field: str) -> float | None:
            if sys.cell_level and field in sys.cell_level and sys.cell_level[field] is not None:
                return float(sys.cell_level[field])
            if sys.uncertainty and isinstance(sys.uncertainty, dict):
                cell_level = sys.uncertainty.get("cell_level")
                if isinstance(cell_level, dict) and cell_level.get(field) is not None:
                    return float(cell_level[field])
            raw = getattr(sys, field, None)
            return float(raw) if raw is not None else None

        for system in systems:
            reasons: List[str] = []

            # -----------------------------
            # 1. Physical / chemical checks
            # -----------------------------
            constraints = evaluate_system(system)

            if not constraints["overall_valid"]:
                reasons.extend(constraints["physical"]["violations"])
                reasons.extend(constraints["chemical"]["violations"])

            # -----------------------------
            # 2. Target range enforcement
            # -----------------------------
            for prop, (low, high) in target_ranges.items():
                if prop not in TARGET_PROPERTIES:
                    continue

                if prop in {"capacity_grav", "capacity_vol", "energy_grav", "energy_vol"}:
                    value = cell_or_system_value(system, prop)
                else:
                    value = getattr(system, prop, None)

                if value is None:
                    reasons.append(f"{prop} missing prediction")
                elif not (low <= value <= high):
                    reasons.append(
                        f"{prop}={value:.3f} outside [{low}, {high}]"
                    )

            # Electrolyte voltage window check (simple heuristic).
            if system.electrolyte and system.average_voltage is not None:
                electrolyte = str(system.electrolyte).lower()
                if "carbonate" in electrolyte or "lp30" in electrolyte:
                    if system.average_voltage > 4.3:
                        reasons.append("electrolyte voltage window exceeded for carbonate")

            # -----------------------------
            # 3. Accept / reject
            # -----------------------------
            if reasons:
                rejected.append({
                    "battery_id": system.battery_id,
                    "reasons": reasons,
                })
            else:
                feasible.append(system)

        return feasible, rejected

    # --------------------------------------------------------
    # PREDICTION: explanation (schema-compatible)
    # --------------------------------------------------------
    def explain_predicted_system(
        self,
        system: BatterySystem,
        application: Optional[str] = None,
        material_roles: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a schema-compatible ExplanationResponse
        for ONE predicted BatterySystem.

        This method:
         Is non-blocking
         Does NOT affect feasibility
         Does NOT expose ML internals
        """

        constraints = evaluate_system(system)

        speculative = bool(
            constraints.get("performance", {}).get("speculative", False)
        )

        strengths: List[str] = []
        weaknesses: List[str] = []
        tradeoffs: List[str] = []
        constraint_notes: List[str] = []

        # -----------------------------
        # Strengths / weaknesses
        # -----------------------------
        for prop in TARGET_PROPERTIES:
            value = getattr(system, prop, None)
            if value is not None:
                strengths.append(f"{prop}: {value:.3f}")

        if not constraints["overall_valid"]:
            weaknesses.append("System violates one or more feasibility constraints")

        # -----------------------------
        # Constraint notes (explicit)
        # -----------------------------
        for domain in ["physical", "chemical", "performance"]:
            report = constraints.get(domain, {})
            for violation in report.get("violations", []):
                constraint_notes.append(f"{domain}: {violation}")

        # -----------------------------
        # Trade-offs (lightweight, non-ML)
        # -----------------------------
        if system.energy_grav and system.stability_charge:
            tradeoffs.append(
                "Energy density may trade off against charge stability"
            )

        # -----------------------------
        # Summary (human-readable)
        # -----------------------------
        summary_parts = []
        if constraints["overall_valid"]:
            summary_parts.append("System is physically and chemically feasible.")
        else:
            summary_parts.append("System exhibits feasibility violations.")

        perf_violations = constraints.get("performance", {}).get("violations", [])
        if constraints["overall_valid"] and perf_violations:
            summary_parts.append(
                "However, it includes soft realism warnings (optimistic chemistry/performance regime)."
            )

        if speculative:
            summary_parts.append(
                "Predictions involve extrapolation beyond dominant training regimes."
            )

        summary = " ".join(summary_parts)

        return {
            "battery_id": system.battery_id,
            "application": application,
            "valid": constraints["overall_valid"],
            "speculative": speculative,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "tradeoffs": tradeoffs,
            "constraint_notes": constraint_notes,
            "material_roles": material_roles or {},
        }


# ============================================================
# Legacy compatibility: reason_about_system function
# ============================================================

def reason_about_system(
    system: BatterySystem,
    application: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Legacy compatibility wrapper for SystemReasoner.explain_predicted_system().
    
    Provides a simple interface for reasoning about a single system.
    """
    reasoner = SystemReasoner()
    return reasoner.explain_predicted_system(system, application=application)

