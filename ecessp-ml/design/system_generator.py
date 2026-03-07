# ============================================================
# Battery System Generator (ECESSP – ROLE-AWARE, NON-HALLUCINATORY)
# ============================================================
# Purpose:
#   - Generate NEW battery systems
#   - Use inferred material roles
#   - Explore combinatorial system space
#   - Produce schema-valid systems ONLY
#
# Rules:
#   - NO objectives
#   - NO scoring
#   - NO filtering
#   - NO ML calls
#   - Generator proposes, reasoner filters
# ============================================================

from __future__ import annotations

from typing import Dict, List
import itertools
import uuid
import random

from .system_template import BatterySystem


# ============================================================
# Generator Configuration
# ============================================================

GENERATOR_CONFIG = {
    "max_systems": 64,            # hard cap per request
    "max_variants_per_role": 3,   # limit combinatorics
}


# ============================================================
# SystemGenerator
# ============================================================

class SystemGenerator:
    """
    Role-aware combinatorial generator.

    This generator operates ONLY in system-definition space.
    It does NOT predict properties and does NOT enforce constraints.
    """

    # --------------------------------------------------------
    # Public API (used by engine)
    # --------------------------------------------------------
    def generate(
        self,
        role_predictions: Dict,
    ) -> List[BatterySystem]:
        """
        Generate NEW candidate battery systems based on
        inferred material roles.
        """

        role_to_materials = self._group_by_role(role_predictions)

        if not role_to_materials:
            return []

        systems: List[BatterySystem] = []

        role_keys = sorted(role_to_materials.keys())

        material_options = [
            role_to_materials[role][:GENERATOR_CONFIG["max_variants_per_role"]]
            for role in role_keys
            if role_to_materials.get(role)
        ]

        if not material_options:
            return []

        for combination in itertools.product(*material_options):
            system = self._build_system(
                role_keys=role_keys,
                materials=combination,
            )
            systems.append(system)

            if len(systems) >= GENERATOR_CONFIG["max_systems"]:
                break

        return systems

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------
    def _group_by_role(self, role_predictions: Dict) -> Dict[str, List[Dict]]:
        """
        Group candidate materials by inferred role.
        """

        grouped: Dict[str, List[Dict]] = {}

        for item in role_predictions.get("materials", []):
            role = item.get("role")
            if not role:
                continue

            grouped.setdefault(role, []).append(item)

        # Shuffle to avoid deterministic bias
        for mats in grouped.values():
            random.shuffle(mats)

        return grouped

    def _build_system(
        self,
        role_keys: List[str],
        materials: List[Dict],
    ) -> BatterySystem:
        """
        Construct a BatterySystem from a role-material assignment.
        """

        system = BatterySystem(
            battery_id=str(uuid.uuid4()),
            provenance="generated",
        )

        attached: List[Dict] = []

        for role, material in zip(role_keys, materials):
            attached.append({
                "role": role,
                "component": material.get("component"),
                "predefined_material": material.get("predefined_material"),
                "elements": material.get("elements"),
                "composition": material.get("composition"),
                "molecule": material.get("molecule"),
                "free_text": material.get("free_text"),
            })

        system.attached_materials = attached
        return system