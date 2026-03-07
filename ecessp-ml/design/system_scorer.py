# ============================================================
# Battery System Scorer (ECESSP  TARGET-DRIVEN)
# ============================================================
# Purpose:
#   - Rank FEASIBLE battery systems ONLY
#   - Score by distance-to-target ranges
#   - Preserve speculative labels
#   - NEVER reject systems
#
# Notes:
#   - Output is INTERNAL ranking structure
#   - Conversion to API schemas happens elsewhere
# ============================================================

from typing import Dict, List, Tuple, Any
import math

from .system_template import BatterySystem
from .system_constraints import evaluate_system, SYSTEM_LIMITS


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
# Utility: distance-to-target scoring
# ============================================================

def target_score(
    value: float,
    low: float,
    high: float,
) -> float:
    """
    Compute distance-to-target score in [0, 1].

    Semantics:
      * 1.0 -> exactly at center of target range
      * 0.0 -> on boundary or outside
    """

    if value is None:
        return 0.0

    half_width = 0.5 * (high - low)
    if half_width <= 0:
        return 0.0

    center = 0.5 * (low + high)
    distance = abs(value - center) / half_width

    return max(0.0, 1.0 - distance)


# ============================================================
# SystemScorer
# ============================================================

class SystemScorer:
    """
    Scores and ranks FEASIBLE BatterySystem objects
    against user-defined target ranges.

    Guarantees:
      * Never mutates feasibility
      * Ranking-only penalties (e.g. speculative)
      * Hard-rejects invalid systems (constraints fail)
    """

    def score(
        self,
        systems: List[BatterySystem],
        target_ranges: Dict[str, Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        """
        Rank systems by aggregate distance-to-target score.

        Returns:
          Internal ranking records with:
            - system (BatterySystem)
            - score (float)
            - speculative (bool)
            - property_scores (per-target breakdown)
        """

        ranked: List[Dict[str, Any]] = []

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
            constraints = evaluate_system(system)
            if not constraints.get("overall_valid", False):
                # Hard reject invalid systems.
                continue
            speculative = bool(
                constraints.get("performance", {}).get("speculative", False)
            )
            penalty_multiplier = 1.0

            # Additional hard realism penalties (ranking-only)
            energy_grav = cell_or_system_value(system, "energy_grav")
            energy_vol = cell_or_system_value(system, "energy_vol")
            if energy_grav is not None and energy_grav > SYSTEM_LIMITS["max_energy_grav"]:
                penalty_multiplier *= 0.2
            if energy_vol is not None and energy_vol > SYSTEM_LIMITS["max_energy_vol"]:
                penalty_multiplier *= 0.2

            property_scores: Dict[str, Dict[str, Any]] = {}
            total_score = 0.0
            n_active = 0

            for prop, (low, high) in target_ranges.items():
                if prop not in TARGET_PROPERTIES:
                    continue

                if prop in {"capacity_grav", "capacity_vol", "energy_grav", "energy_vol"}:
                    value = cell_or_system_value(system, prop)
                else:
                    value = getattr(system, prop, None)
                if prop == "max_delta_volume":
                    if value is None:
                        s = 0.0
                    else:
                        upper = max(float(high), 0.0)
                        if float(value) <= upper:
                            s = 1.0
                        else:
                            denom = max(abs(upper), 0.05)
                            s = max(0.0, 1.0 - ((float(value) - upper) / denom))
                else:
                    s = target_score(value, low, high)

                property_scores[prop] = {
                    "value": value,
                    "target": (low, high),
                    "score": round(s, 4),
                }

                total_score += s
                n_active += 1

            # Normalize by number of active targets
            final_score = total_score / max(1, n_active)

            # Ranking-only speculative penalty
            if speculative:
                final_score *= 0.85

            final_score *= penalty_multiplier

            ranked.append({
                "system": system,
                "score": round(final_score, 6),
                "speculative": speculative,
                "property_scores": property_scores,
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked


# ============================================================
# Legacy compatibility: score_system function
# ============================================================

def score_system(
    system: BatterySystem,
    objectives: Dict[str, float],
) -> Dict[str, Any]:
    """
    Legacy compatibility wrapper for SystemScorer.score().
    
    Converts objectives dict to target ranges and returns a single system score.
    """
    scorer = SystemScorer()
    constraints = evaluate_system(system)
    is_valid = bool(constraints.get("overall_valid", False))
    
    # Convert objectives to target ranges (simple conversion for compatibility)
    # In a real implementation, objectives would define target ranges
    target_ranges = {}
    for prop, weight in objectives.items():
        if prop in TARGET_PROPERTIES:
            current_raw = getattr(system, prop, None)
            try:
                current_value = float(current_raw) if current_raw is not None else None
            except (TypeError, ValueError):
                current_value = None

            try:
                requested_target = float(weight)
            except (TypeError, ValueError):
                requested_target = None

            center = current_value
            if center is None or center <= 0:
                center = requested_target
            if center is None or center <= 0:
                continue

            # Keep legacy behavior (tight range) while supporting missing fields.
            if prop == "max_delta_volume":
                upper = max(center, 0.0)
                margin = max(abs(upper) * 0.25, 0.02)
                target_ranges[prop] = (0.0, upper + margin)
            else:
                margin = max(abs(center) * 0.1, 0.05)
                target_ranges[prop] = (center - margin, center + margin)
    
    if not target_ranges:
        # If no valid objectives, return a default score
        return {
            "score": 0.0,
            "speculative": False,
            "valid": is_valid,
        }
    
    # Score the system
    results = scorer.score([system], target_ranges)
    if results:
        result = results[0]
        return {
            "score": result["score"],
            "speculative": result["speculative"],
            "valid": is_valid,
        }
    
    return {
        "score": 0.0,
        "speculative": False,
        "valid": is_valid,
    }

