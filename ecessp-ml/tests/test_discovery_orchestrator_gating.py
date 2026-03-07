from dataclasses import dataclass

from backend.services.discovery_orchestrator import DiscoveryOrchestrator
from design.compatibility_model import CompatibilityRecord
from design.system_template import BatterySystem
from materials.material_generator import MaterialCandidate
from materials.role_classifier import RoleAssignment


def test_role_head_gate_accepts_confident_prediction() -> None:
    accepted, normalized, confidence, margin, reason = DiscoveryOrchestrator._role_head_gate_decision(
        model_role_probs={"cathode": 0.90, "anode": 0.06, "electrolyte_candidate": 0.04},
        min_confidence=0.55,
        min_margin=0.05,
    )
    assert accepted is True
    assert reason == "model_head_selected"
    assert confidence >= 0.55
    assert margin >= 0.05
    assert abs(sum(normalized.values()) - 1.0) < 1e-9


def test_role_head_gate_rejects_low_confidence() -> None:
    accepted, _, confidence, _, reason = DiscoveryOrchestrator._role_head_gate_decision(
        model_role_probs={"cathode": 0.34, "anode": 0.33, "electrolyte_candidate": 0.33},
        min_confidence=0.55,
        min_margin=0.05,
    )
    assert accepted is False
    assert reason == "confidence_below_threshold"
    assert confidence < 0.55


def test_role_head_gate_rejects_low_margin() -> None:
    accepted, _, confidence, margin, reason = DiscoveryOrchestrator._role_head_gate_decision(
        model_role_probs={"cathode": 0.52, "anode": 0.48, "electrolyte_candidate": 0.0},
        min_confidence=0.50,
        min_margin=0.08,
    )
    assert accepted is False
    assert reason == "margin_below_threshold"
    assert confidence >= 0.50
    assert margin < 0.08


def test_compatibility_gate_uses_deterministic_when_head_missing() -> None:
    out = DiscoveryOrchestrator._compatibility_head_gate_decision(
        model_compatibility=None,
        deterministic_score=0.73,
        min_score=0.42,
    )
    assert out["compatibility_source"] == "deterministic_proxy"
    assert out["compatibility_fallback_reason"] == "model_head_unavailable"
    assert out["compatibility_score"] == out["compatibility_rule_score"]


def test_compatibility_gate_uses_deterministic_when_model_below_threshold() -> None:
    out = DiscoveryOrchestrator._compatibility_head_gate_decision(
        model_compatibility=0.25,
        deterministic_score=0.62,
        min_score=0.42,
    )
    assert out["compatibility_source"] == "deterministic_proxy"
    assert out["compatibility_fallback_reason"] == "model_score_below_threshold"
    assert out["compatibility_score"] == out["compatibility_rule_score"]
    assert out["compatibility_model_score"] == 0.25


def test_compatibility_gate_uses_model_when_threshold_passes() -> None:
    out = DiscoveryOrchestrator._compatibility_head_gate_decision(
        model_compatibility=0.67,
        deterministic_score=0.50,
        min_score=0.42,
    )
    assert out["compatibility_source"] == "model_head"
    assert out["compatibility_fallback_reason"] is None
    assert out["compatibility_score"] == 0.67


def test_validity_invariant_blocks_overall_invalid_candidates() -> None:
    assert (
        DiscoveryOrchestrator._is_stage_valid_candidate(
            assembled_valid=True,
            overall_valid=False,
            require_overall_valid=False,
        )
        is False
    )
    assert (
        DiscoveryOrchestrator._is_stage_valid_candidate(
            assembled_valid=False,
            overall_valid=True,
            require_overall_valid=True,
        )
        is False
    )
    assert (
        DiscoveryOrchestrator._is_stage_valid_candidate(
            assembled_valid=True,
            overall_valid=True,
            require_overall_valid=True,
        )
        is True
    )


def test_soft_stage4_allowlist_includes_anode_potential_rule() -> None:
    assert DiscoveryOrchestrator._is_soft_stage4_only_failure(
        ["assembly_rule:anode_potential_too_high", "stage6:unstable_interphase_predicted"]
    ) is True


@dataclass
class _DummyChemistryReport:
    candidate_id: str
    valid: bool
    reasons: list[str]


class _DummyMaterialGenerator:
    def generate_candidates(
        self,
        *,
        working_ion,
        target_property_vector,
        optional_seed_structure,
        candidate_pool_size,
        interpolation_enabled=True,
        extrapolation_enabled=True,
        role_condition=None,
    ):
        role = str(role_condition or "").strip().lower()
        if role == "anode":
            formula = "C6"
        elif role == "electrolyte":
            formula = f"{working_ion}PF6"
        else:
            formula = f"{working_ion}FePO4"
        return [
            MaterialCandidate(
                candidate_id=f"{working_ion.lower()}_{role or 'host'}_1",
                framework_formula=formula,
                working_ion=working_ion,
                source_mode="latent_variation",
                valid=True,
            )
        ]


class _DummyChemistryValidator:
    def validate_candidates(self, candidates):
        return list(candidates), [
            _DummyChemistryReport(
                candidate_id=str(c.candidate_id),
                valid=True,
                reasons=[],
            )
            for c in candidates
        ]


class _DummyRoleClassifier:
    def classify_candidates(self, candidates, *, target_property_vector):
        assignments = {}
        for c in candidates:
            assignments[c.candidate_id] = RoleAssignment(
                candidate_id=c.candidate_id,
                role_probabilities={"cathode": 0.7, "anode": 0.2, "electrolyte_candidate": 0.1},
                confidence_score=0.7,
                selected_roles=["cathode"],
                used_fallback=False,
            )
        return assignments


class _DummyCompatibilityModel:
    def score_triples(self, *, candidates, assignments):
        return []


def test_compatibility_retry_preserves_requested_ion_scope(monkeypatch) -> None:
    orchestrator = DiscoveryOrchestrator(
        material_generator=_DummyMaterialGenerator(),
        chemistry_validator=_DummyChemistryValidator(),
        role_classifier=_DummyRoleClassifier(),
        compatibility_model=_DummyCompatibilityModel(),
    )
    monkeypatch.setattr(
        "backend.services.discovery_orchestrator.get_enhanced_inference_engine",
        lambda: object(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_role_head_assignments_with_fallback",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "_validate_graph_manifest",
        lambda: {"valid": True, "warnings": [], "errors": []},
    )

    requested_scope = ["Li", "Na", "Mg"]
    captured: dict[str, list[str]] = {}
    original_run = DiscoveryOrchestrator.run_generative

    def _wrapped_run(self, *, base_system, objective, discovery_params=None, candidate_pool_size=150):
        params = dict(discovery_params or {})
        if params.get("_compatibility_fallback_reentry"):
            captured["scope"] = list(params.get("working_ion_candidates") or [])
            return {
                "best_system": base_system,
                "candidate_records": [],
                "discovery_report_card": {
                    "stage_0": {"objective": {"working_ion_candidates": list(captured["scope"])}}
                },
            }
        return original_run(
            self,
            base_system=base_system,
            objective=objective,
            discovery_params=params,
            candidate_pool_size=candidate_pool_size,
        )

    monkeypatch.setattr(DiscoveryOrchestrator, "run_generative", _wrapped_run)

    base_system = BatterySystem(
        battery_id="scope_retry_test",
        working_ion="Li",
        average_voltage=3.7,
        capacity_grav=120.0,
        energy_grav=420.0,
        max_delta_volume=0.15,
        stability_charge=4.1,
        stability_discharge=3.6,
    )
    result = orchestrator.run_generative(
        base_system=base_system,
        objective={
            "average_voltage": 3.7,
            "capacity_grav": 120.0,
            "energy_grav": 444.0,
            "max_delta_volume": 0.15,
            "stability_charge": 4.1,
            "stability_discharge": 3.6,
        },
        discovery_params={
            "working_ion_candidates": list(requested_scope),
            "material_source_mode": "generated",
            "component_source_mode": "generated",
            "num_candidates": 30,
        },
        candidate_pool_size=30,
    )

    assert captured.get("scope") == requested_scope
    stage0_scope = (
        ((result.get("discovery_report_card") or {}).get("stage_0") or {})
        .get("objective", {})
        .get("working_ion_candidates")
    )
    assert stage0_scope == requested_scope


def test_pareto_ranking_uses_hgt_rerank_signal(monkeypatch) -> None:
    orchestrator = DiscoveryOrchestrator()
    monkeypatch.setenv("ECESSP_PARETO_HGT_WEIGHT", "0.30")
    records = [
        {
            "objective_alignment_score": 0.70,
            "feasibility_score": 0.80,
            "uncertainty_penalty": 0.20,
            "hgt_rerank_score": 0.90,
            "compatibility_score": 0.75,
        },
        {
            "objective_alignment_score": 0.70,
            "feasibility_score": 0.80,
            "uncertainty_penalty": 0.20,
            "hgt_rerank_score": 0.10,
            "compatibility_score": 0.75,
        },
    ]
    ranked = orchestrator._apply_pareto_ranking(records)
    assert len(ranked) == 2
    assert float(ranked[0]["hgt_rerank_score"]) >= float(ranked[1]["hgt_rerank_score"])
    weights = ranked[0].get("pareto_weights", {})
    assert "hgt_rerank" in weights


def test_constitution_stage0_gate_rejects_objective_sanity_violation() -> None:
    orchestrator = DiscoveryOrchestrator()
    base_system = BatterySystem(
        battery_id="constitution_sanity_test",
        working_ion="Li",
        average_voltage=3.7,
        capacity_grav=120.0,
        energy_grav=444.0,
        max_delta_volume=0.12,
        stability_charge=4.1,
        stability_discharge=3.6,
    )
    objective_cfg = orchestrator._normalize_objective(
        base_system=base_system,
        objective={
            "average_voltage": 4.2,
            "capacity_grav": 120.0,
            "capacity_vol": 220.0,
            "energy_grav": 504.0,
            "energy_vol": 924.0,
            "max_delta_volume": 0.12,
            "stability_charge": 4.0,
            "stability_discharge": 4.1,
        },
        discovery_params={"working_ion_candidates": ["Li"]},
    )
    gate = orchestrator._constitution_stage0_gate(
        objective_cfg=objective_cfg,
        graph_manifest={"valid": True},
    )
    assert gate["passed"] is False
    assert "objective_sanity:average_voltage_exceeds_stability_charge" in gate["reasons"]
    assert "objective_sanity:stability_charge_not_above_stability_discharge" in gate["reasons"]
    assert gate["guardrail_flags"]["voltage_inconsistency_flag"] is True


def test_constitution_stage0_gate_rejects_derived_energy_mismatch() -> None:
    orchestrator = DiscoveryOrchestrator()
    base_system = BatterySystem(
        battery_id="constitution_energy_test",
        working_ion="Li",
        average_voltage=3.7,
        capacity_grav=120.0,
        energy_grav=444.0,
        max_delta_volume=0.12,
        stability_charge=4.1,
        stability_discharge=3.6,
    )
    objective_cfg = orchestrator._normalize_objective(
        base_system=base_system,
        objective={
            "average_voltage": 3.8,
            "capacity_grav": 100.0,
            "capacity_vol": 200.0,
            "energy_grav": 450.0,
            "energy_vol": 760.0,
            "max_delta_volume": 0.10,
            "stability_charge": 4.1,
            "stability_discharge": 3.6,
        },
        discovery_params={"working_ion_candidates": ["Li"]},
    )
    gate = orchestrator._constitution_stage0_gate(
        objective_cfg=objective_cfg,
        graph_manifest={"valid": True},
    )
    assert gate["passed"] is False
    assert "normalization:energy_grav_formula_mismatch" in gate["reasons"]
    assert gate["guardrail_flags"]["normalization_mismatch_flag"] is True


def test_electrolyte_forbidden_transition_metals_detected() -> None:
    hits = DiscoveryOrchestrator._electrolyte_forbidden_transition_metals("NiSO4")
    assert hits == ["Ni"]


def test_run_generative_aborts_on_constitution_stage0_failure(monkeypatch) -> None:
    orchestrator = DiscoveryOrchestrator(
        material_generator=_DummyMaterialGenerator(),
        chemistry_validator=_DummyChemistryValidator(),
        role_classifier=_DummyRoleClassifier(),
        compatibility_model=_DummyCompatibilityModel(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_validate_graph_manifest",
        lambda: {"valid": True, "warnings": [], "errors": []},
    )
    base_system = BatterySystem(
        battery_id="constitution_abort_test",
        working_ion="Li",
        average_voltage=3.7,
        capacity_grav=120.0,
        energy_grav=444.0,
        max_delta_volume=0.12,
        stability_charge=4.1,
        stability_discharge=3.6,
    )
    result = orchestrator.run_generative(
        base_system=base_system,
        objective={
            "average_voltage": 4.2,
            "capacity_grav": 120.0,
            "energy_grav": 504.0,
            "max_delta_volume": 0.12,
            "stability_charge": 4.0,
            "stability_discharge": 4.1,
        },
        discovery_params={"working_ion_candidates": ["Li"], "num_candidates": 30},
        candidate_pool_size=30,
    )
    assert result.get("candidate_records") == []
    stage0 = ((result.get("discovery_report_card") or {}).get("stage_0") or {})
    constitution = stage0.get("constitution_v2", {})
    assert constitution.get("passed") is False


def test_material_ontology_classifier_detects_polyanion() -> None:
    assert DiscoveryOrchestrator._classify_material_ontology("LiFePO4") == "polyanion"


def test_material_ontology_classifier_classifies_li_salt_as_electrolyte() -> None:
    assert DiscoveryOrchestrator._classify_material_ontology("LiPF6") == "solid_li_conductor"


def test_stage1b_ontology_gate_rejects_unclassified() -> None:
    orchestrator = DiscoveryOrchestrator()
    candidates = [
        MaterialCandidate(candidate_id="known_1", framework_formula="LiFePO4", working_ion="Li", valid=True),
        MaterialCandidate(candidate_id="bad_1", framework_formula="???", working_ion="Li", valid=True),
    ]
    passed, report = orchestrator._apply_stage1b_ontology_gate(candidates)
    assert len(passed) == 1
    assert int(report.get("rejected_unclassified_count", 0)) == 1


def test_recover_anode_candidates_prefers_low_voltage_profiles() -> None:
    orchestrator = DiscoveryOrchestrator()
    a1 = MaterialCandidate(candidate_id="a1", framework_formula="NaFePO4", working_ion="Na", valid=True)
    a1.metadata = {"ontology_class": "transition_metal_oxide", "reference_voltage": 1.8}
    a2 = MaterialCandidate(candidate_id="a2", framework_formula="NaTi2(PO4)3", working_ion="Na", valid=True)
    a2.metadata = {"ontology_class": "low_voltage_oxide", "reference_voltage": 1.4}
    a3 = MaterialCandidate(candidate_id="a3", framework_formula="Na3V2(PO4)3", working_ion="Na", valid=True)
    a3.metadata = {"ontology_class": "polyanion", "reference_voltage": 1.6}
    recovered = orchestrator._recover_anode_candidates([a1, a2, a3], limit=2)
    assert len(recovered) == 2
    assert recovered[0].candidate_id == "a2"
    assert bool((recovered[0].metadata or {}).get("role_class_gate_fallback")) is True


def test_normalize_objective_forces_at_least_one_generation_mode() -> None:
    orchestrator = DiscoveryOrchestrator()
    base_system = BatterySystem(
        battery_id="mode_guard_test",
        working_ion="Li",
        average_voltage=3.7,
        capacity_grav=120.0,
        energy_grav=444.0,
        max_delta_volume=0.12,
        stability_charge=4.1,
        stability_discharge=3.6,
    )
    objective_cfg = orchestrator._normalize_objective(
        base_system=base_system,
        objective={"average_voltage": 3.7, "capacity_grav": 120.0, "energy_grav": 444.0},
        discovery_params={"interpolation_enabled": False, "extrapolation_enabled": False},
    )
    assert objective_cfg.interpolation_enabled is True
    assert objective_cfg.extrapolation_enabled is False


def test_stage3_component_ontology_gate_rejects_role_mismatch() -> None:
    orchestrator = DiscoveryOrchestrator()
    cath = MaterialCandidate(candidate_id="c1", framework_formula="LiFePO4", working_ion="Li", valid=True)
    an = MaterialCandidate(candidate_id="a1", framework_formula="LiFePO4", working_ion="Li", valid=True)
    el = MaterialCandidate(candidate_id="e1", framework_formula="LiPF6", working_ion="Li", valid=True)
    cath.metadata = {"ontology_class": "polyanion"}
    an.metadata = {"ontology_class": "polyanion"}
    el.metadata = {"ontology_class": "solid_li_conductor"}
    record = CompatibilityRecord(
        cathode=cath,
        anode=an,
        electrolyte=el,
        voltage_window_overlap_score=0.8,
        chemical_stability_score=0.8,
        mechanical_strain_risk=0.2,
        interface_risk_reason_codes=[],
        hard_valid=True,
    )
    kept, report = orchestrator._apply_stage3_component_ontology_gate([record])
    assert kept == [record]
    assert int(report.get("rejected_count", 0)) == 1
    assert bool(report.get("fallback_preserved_records")) is True
    assert "anode_role_class_mismatch:polyanion" in (report.get("rejection_reason_counts") or {})
