from __future__ import annotations

import pytest

from backend.services.discovery_service import DiscoveryService


class _OneCandidateOrchestrator:
    def run_generative(self, *, base_system, objective, discovery_params, candidate_pool_size):
        candidate = {
            "system": base_system,
            "score": 0.71,
            "speculative": False,
            "source": "staged_pipeline",
            "material_uncertainty_proxy": 0.1,
            "material_thermodynamic_proxy": 0.8,
            "stage_trace": {
                "assembly": {
                    "working_ion": "Li",
                    "cathode_formula": "MoP2O7",
                    "anode_formula": "Mn5(FeO6)2",
                    "cathode_redox_potential": 2.6,
                    "anode_redox_potential": 3.2,
                    "full_cell_voltage": -0.6,
                }
            },
        }
        return {
            "best_system": base_system,
            "candidate_records": [candidate],
            "discovery_report_card": {"stage_1": {"material_candidates_generated": 1, "material_candidates_valid": 1}},
            "target_objectives_feasible": dict(objective),
            "objective_feasibility_adjustments": [],
            "graph_manifest_validation": {},
            "active_learning_queue": {},
            "latent_generation_stats": {
                "ranked_systems_count": 1,
                "latent_generated_count": 0,
                "synthesized_generated_count": 1,
                "selected_latent_count": 0,
            },
        }


def test_discovery_history_includes_viability_v2_for_candidate(monkeypatch) -> None:
    service = DiscoveryService()
    monkeypatch.setattr(service, "_ensure_initialized", lambda: None)
    service._generator = object()
    service._orchestrator = _OneCandidateOrchestrator()
    service._device = "cpu"

    result = service.discover(
        base_system_data={
            "battery_id": "viability_v2_seed",
            "working_ion": "Li",
            "cathode_material": "MoP2O7",
            "anode_material": "Mn5(FeO6)2",
            "average_voltage": 3.4,
            "capacity_grav": 150.0,
            "capacity_vol": 500.0,
            "energy_grav": 510.0,
            "energy_vol": 1700.0,
            "max_delta_volume": 0.10,
            "stability_charge": 4.0,
            "stability_discharge": 3.6,
        },
        objective={"average_voltage": 3.7, "capacity_grav": 180.0, "max_delta_volume": 0.1},
        explain=False,
        mode="generative",
        discovery_params={"num_candidates": 20},
    )

    history = list(result.get("history") or [])
    assert len(history) >= 1
    v2 = history[0].get("electrochemical_viability_v2", {})
    screening = v2.get("screening_output", {})
    assert float(screening.get("theoretical_voltage", 0.0)) == pytest.approx(-0.6)
    assert screening.get("is_viable") is False
    assert "SWAP_ROLES" in str(screening.get("recommendation", ""))
