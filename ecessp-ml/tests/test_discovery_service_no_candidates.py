from __future__ import annotations

from backend.services.discovery_service import DiscoveryService


class _EmptyOrchestrator:
    def run_generative(self, *, base_system, objective, discovery_params, candidate_pool_size):
        return {
            "best_system": base_system,
            "candidate_records": [],
            "discovery_report_card": {"stage_1": {"material_candidates_generated": 12, "material_candidates_valid": 0}},
            "target_objectives_feasible": dict(objective),
            "objective_feasibility_adjustments": [],
            "graph_manifest_validation": {},
            "active_learning_queue": {},
            "latent_generation_stats": {
                "ranked_systems_count": 0,
                "latent_generated_count": 0,
                "synthesized_generated_count": 0,
                "selected_latent_count": 0,
            },
        }


def test_generative_mode_without_candidates_is_not_promotion_ready(monkeypatch) -> None:
    service = DiscoveryService()
    monkeypatch.setattr(service, "_ensure_initialized", lambda: None)
    service._generator = object()
    service._orchestrator = _EmptyOrchestrator()
    service._device = "cpu"

    result = service.discover(
        base_system_data={
            "battery_id": "empty_gen_seed",
            "working_ion": "Li",
            "average_voltage": 3.5,
            "capacity_grav": 150.0,
            "capacity_vol": 500.0,
            "energy_grav": 525.0,
            "energy_vol": 1750.0,
            "max_delta_volume": 0.10,
            "stability_charge": 4.0,
            "stability_discharge": 3.6,
        },
        objective={"average_voltage": 3.7, "capacity_grav": 180.0, "max_delta_volume": 0.1},
        explain=False,
        mode="generative",
        discovery_params={"num_candidates": 20},
    )

    assert result.get("history") == []
    assert result.get("score", {}).get("valid") is False
    assert float(result.get("score", {}).get("score", 1.0)) == 0.0
    metadata = result.get("metadata", {})
    assert metadata.get("valid") is False
    guardrail = metadata.get("prediction_guardrail", {})
    assert guardrail.get("status") == "blocked"
    assert "no_generative_candidates" in list(guardrail.get("reasons", []))
