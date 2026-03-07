from backend.api.routes import _infer_working_ion
from design.compatibility_model import CompatibilityRecord
from design.full_cell_assembler import FullCellAssembler
from design.system_template import BatterySystem
from materials.material_generator import MaterialCandidate


def _candidate(candidate_id: str, ion: str, formula: str) -> MaterialCandidate:
    return MaterialCandidate(
        candidate_id=candidate_id,
        framework_formula=formula,
        working_ion=ion,
    )


def _compatibility_record(ion: str, electrolyte_formula: str = "LiPF6") -> CompatibilityRecord:
    return CompatibilityRecord(
        cathode=_candidate("c1", ion, "LiFePO4"),
        anode=_candidate("a1", ion, "C6"),
        electrolyte=_candidate("e1", ion, electrolyte_formula),
        voltage_window_overlap_score=0.8,
        chemical_stability_score=0.7,
        mechanical_strain_risk=0.2,
        interface_risk_reason_codes=[],
        hard_valid=True,
    )


def _compatibility_record_with_roles(
    ion: str,
    *,
    cathode_formula: str,
    anode_formula: str,
    electrolyte_formula: str = "LiPF6",
) -> CompatibilityRecord:
    return CompatibilityRecord(
        cathode=_candidate("c_custom", ion, cathode_formula),
        anode=_candidate("a_custom", ion, anode_formula),
        electrolyte=_candidate("e_custom", ion, electrolyte_formula),
        voltage_window_overlap_score=0.8,
        chemical_stability_score=0.7,
        mechanical_strain_risk=0.2,
        interface_risk_reason_codes=[],
        hard_valid=True,
    )


def test_assembler_sets_separator_and_additive_for_li() -> None:
    assembler = FullCellAssembler()
    base = BatterySystem(battery_id="base", battery_type="insertion", working_ion="Li")
    assembled, system = assembler.assemble(
        index=0,
        compatibility=_compatibility_record("Li"),
        target_property_vector={"average_voltage": 3.7, "capacity_grav": 180.0, "max_delta_volume": 0.12},
        base_system=base,
    )
    assert system.separator_material is not None
    assert system.additive_material is not None
    assert assembled.separator_material == system.separator_material
    assert assembled.additive_material == system.additive_material


def test_assembler_uses_ion_default_electrolyte_when_empty() -> None:
    assembler = FullCellAssembler()
    base = BatterySystem(battery_id="base_na", battery_type="insertion", working_ion="Na")
    _, system = assembler.assemble(
        index=1,
        compatibility=_compatibility_record("Na", electrolyte_formula=""),
        target_property_vector={"average_voltage": 3.0, "capacity_grav": 140.0, "max_delta_volume": 0.10},
        base_system=base,
    )
    assert system.electrolyte == "NaPF6 in EC/DEC"


def test_prediction_working_ion_inference_supports_extended_ions() -> None:
    assert _infer_working_ion({"electrolyte": "ZnSO4 aqueous"}) == "Zn"
    assert _infer_working_ion({"electrolyte": "calcium TFSI"}) == "Ca"
    assert _infer_working_ion({"electrolyte": "aluminum chloride ionic liquid"}) == "Al"


def test_assembler_rejects_mixed_alkali_electrolyte() -> None:
    assembler = FullCellAssembler()
    base = BatterySystem(battery_id="base_na_mixed", battery_type="insertion", working_ion="Na")
    assembled, _ = assembler.assemble(
        index=2,
        compatibility=_compatibility_record("Na", electrolyte_formula="LiPF6"),
        target_property_vector={"average_voltage": 3.0, "capacity_grav": 140.0, "max_delta_volume": 0.10},
        base_system=base,
    )
    assert assembled.valid is False
    assert any("electrolyte:mixed_alkali_inconsistency" in reason for reason in assembled.valid_reasons)


def test_assembler_applies_component_selection_from_orchestrator() -> None:
    assembler = FullCellAssembler()
    base = BatterySystem(battery_id="base_component_selection", battery_type="insertion", working_ion="Na")
    assembled, system = assembler.assemble(
        index=3,
        compatibility=_compatibility_record("Na", electrolyte_formula="NaPF6"),
        target_property_vector={"average_voltage": 3.2, "capacity_grav": 150.0, "max_delta_volume": 0.10},
        base_system=base,
        component_selection={
            "source_mode": "existing",
            "separator_material": "Glass-fiber separator",
            "additive_material": "FEC + NaDFOB blend",
        },
    )
    assert system.separator_material == "Glass-fiber separator"
    assert system.additive_material == "FEC + NaDFOB blend"
    assert assembled.separator_material == "Glass-fiber separator"
    assert assembled.additive_material == "FEC + NaDFOB blend"
    component_meta = (assembled.provenance or {}).get("component_selection", {})
    assert component_meta.get("source_mode") == "existing"


def test_component_compatibility_rejects_multivalent_on_polyolefin_separator() -> None:
    assembler = FullCellAssembler()
    valid, reasons, diag = assembler._evaluate_component_compatibility(
        working_ion="Mg",
        cathode_family="Spinel",
        anode_family="Spinel",
        cathode_potential=3.8,
        anode_potential=0.8,
        full_cell_voltage=3.0,
        electrolyte_formula="Mg(TFSI)2 in glyme solvent",
        separator_material="PP/PE trilayer separator",
        additive_material="Interphase stabilizer package",
    )
    assert valid is False
    assert "separator_ion_not_supported" in reasons
    assert diag["separator_profile"]["name"] == "polyolefin"


def test_component_compatibility_rejects_additive_ion_mismatch() -> None:
    assembler = FullCellAssembler()
    valid, reasons, _ = assembler._evaluate_component_compatibility(
        working_ion="Na",
        cathode_family="Layered oxide",
        anode_family="Polyanion framework",
        cathode_potential=3.9,
        anode_potential=0.9,
        full_cell_voltage=3.0,
        electrolyte_formula="NaPF6 in EC/DEC",
        separator_material="Ceramic-coated PP/PE separator",
        additive_material="LiDFOB additive package",
    )
    assert valid is False
    assert "additive_ion_not_supported" in reasons


def test_battery_assembly_rules_reject_inverted_redox_and_window_failure() -> None:
    assembler = FullCellAssembler()
    valid, reasons, diagnostics = assembler._evaluate_battery_assembly_rules(
        working_ion="Li",
        cathode_formula="MoP2O7",
        anode_formula="Mn5(FeO6)2",
        electrolyte_formula="1M LiPF6 in EC/EMC",
        separator_material="PP/PE trilayer separator",
        additive_material="FEC + VC blend",
        cathode_potential=1.6,
        anode_potential=3.2,
        full_cell_voltage=-0.6,
        delta_volume_ratio=0.05,
        electrolyte_window_valid=False,
    )
    assert valid is False
    assert "anode_potential_too_high" in reasons
    assert "cathode_potential_too_low" in reasons
    assert "cell_voltage_negative" in reasons
    assert "cell_voltage_below_minimum" in reasons
    assert "electrolyte_window_not_covering_voltage" in reasons
    assert diagnostics["hard_valid"] is False


def test_battery_assembly_rules_use_ion_specific_anode_potential_limit_for_na() -> None:
    assembler = FullCellAssembler()
    valid, reasons, diagnostics = assembler._evaluate_battery_assembly_rules(
        working_ion="Na",
        cathode_formula="NaFePO4",
        anode_formula="NaTi2(PO4)3",
        electrolyte_formula="NaPF6 in EC/DEC",
        separator_material="Ceramic-coated PP/PE separator",
        additive_material="FEC + NaDFOB blend",
        cathode_potential=3.6,
        anode_potential=1.4,
        full_cell_voltage=2.2,
        delta_volume_ratio=0.08,
        electrolyte_window_valid=True,
    )
    assert "anode_potential_too_high" not in reasons
    assert float(diagnostics["electrode_role_rules"]["anode_potential_max_v"]) == 1.5
    assert valid is True


def test_assembler_emits_battery_assembly_rule_diagnostics() -> None:
    assembler = FullCellAssembler()
    base = BatterySystem(battery_id="base_rules_diag", battery_type="insertion", working_ion="Li")
    assembled, _ = assembler.assemble(
        index=4,
        compatibility=_compatibility_record("Li", electrolyte_formula="LiPF6"),
        target_property_vector={"average_voltage": 3.7, "capacity_grav": 180.0, "max_delta_volume": 0.12},
        base_system=base,
    )
    chemistry_checks = (assembled.provenance or {}).get("chemistry_checks", {})
    rules_diag = chemistry_checks.get("battery_assembly_rules", {})
    material_generation = (assembled.uncertainty or {}).get("material_generation", {})

    assert isinstance(rules_diag, dict)
    assert "hard_valid" in rules_diag
    assert "battery_assembly_rule_valid" in material_generation


def test_role_assignment_engine_locks_roles_by_redox_order() -> None:
    assembler = FullCellAssembler()
    base = BatterySystem(battery_id="base_role_lock", battery_type="insertion", working_ion="Li")
    assembled, system = assembler.assemble(
        index=5,
        compatibility=_compatibility_record_with_roles(
            "Li",
            cathode_formula="C6",
            anode_formula="LiFePO4",
            electrolyte_formula="LiPF6",
        ),
        target_property_vector={"average_voltage": 3.7, "capacity_grav": 180.0, "max_delta_volume": 0.12},
        base_system=base,
    )

    assert assembled.anode_formula == "C6"
    assert assembled.cathode_formula == "LiFePO4"
    assert system.anode_material == "C6"
    assert system.cathode_material == "LiFePO4"

    role_engine = (assembled.provenance or {}).get("role_assignment_engine", {})
    assigned_roles = role_engine.get("assigned_roles", {})
    assert assigned_roles.get("swapped_from_input_labels") is True
    assert role_engine.get("override_allowed") is False


def test_role_assignment_engine_rejects_low_gap_and_high_overlap() -> None:
    assembler = FullCellAssembler()
    valid, reasons, diagnostics = assembler._evaluate_role_assignment_engine(
        anode_potential=1.1,
        cathode_potential=2.0,
    )
    assert valid is False
    assert "minimum_voltage_difference_not_met" in reasons
    assert "overlap_percentage_above_threshold" in reasons
    assert diagnostics["hard_valid"] is False


def test_stage7_stack_physics_gate_rejects_high_barrier_transport() -> None:
    assembler = FullCellAssembler()
    valid, reasons, diagnostics = assembler._evaluate_stack_physics_gates(
        working_ion="K",
        full_cell_voltage=2.2,
        delta_volume_ratio=0.10,
        cathode_diagnostics={"structure": {"diffusion_dimensionality": "1D"}},
        anode_diagnostics={"structure": {"diffusion_dimensionality": "1D"}},
        mechanical_strain_risk=1.0,
    )
    assert valid is False
    assert "ion_diffusion_barrier_above_limit" in reasons or "ion_diffusion_coefficient_below_limit" in reasons
    assert diagnostics["hard_valid"] is False


def test_stage6_additive_interphase_gate_rejects_high_gas_risk() -> None:
    assembler = FullCellAssembler()
    valid, reasons, diagnostics = assembler._evaluate_additive_interphase_validation(
        additive_material="Generated halide-balance additive package",
        anode_potential=0.3,
        cathode_potential=4.3,
        electrolyte_reduction_limit=0.9,
        electrolyte_oxidation_limit=4.6,
        component_diag={"additive_profile": {"matched": False, "min_voltage": None, "max_voltage": None}},
        sei_expected=False,
        thermal_risk=True,
    )
    assert valid is False
    assert "additive_gas_generation_risk_high" in reasons
    assert "unstable_interphase_predicted" in reasons
    assert diagnostics["hard_valid"] is False


def test_component_compatibility_rejects_redox_active_electrolyte() -> None:
    assembler = FullCellAssembler()
    valid, reasons, diagnostics = assembler._evaluate_component_compatibility(
        working_ion="Li",
        cathode_family="Layered oxide",
        anode_family="carbon",
        cathode_potential=3.9,
        anode_potential=0.2,
        full_cell_voltage=3.7,
        electrolyte_formula="LiNiO2",
        separator_material="PP/PE trilayer separator",
        additive_material="FEC + VC blend",
    )
    assert valid is False
    assert "electrolyte_not_electronically_insulating" in reasons
    assert diagnostics["electrolyte_constraints"]["hard_valid"] is False
