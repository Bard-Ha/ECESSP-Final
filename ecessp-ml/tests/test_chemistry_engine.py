from __future__ import annotations

from materials.chemistry_engine import (
    StrictOxidationSolver,
    PolyanionLibrary,
    StructureClassifier,
    InsertionFilter,
    AlkaliValidator,
)
from materials.chemistry_validator import ChemistryValidator
from materials.material_generator import MaterialCandidate, MaterialGenerator
from design.compatibility_model import CompatibilityRecord
from design.full_cell_assembler import FullCellAssembler
from design.system_template import BatterySystem


def _candidate(cid: str, formula: str, ion: str = "Li") -> MaterialCandidate:
    return MaterialCandidate(
        candidate_id=cid,
        framework_formula=formula,
        working_ion=ion,
        source_mode="test",
    )


def test_strict_oxidation_rejects_ambiguous_assignments() -> None:
    solver = StrictOxidationSolver()
    rep = solver.solve_formula("CuMnO2")
    assert rep.valid is False
    assert "ambiguous_oxidation_assignments" in rep.reasons


def test_polyanion_library_rejects_unknown_cluster() -> None:
    rep = PolyanionLibrary.validate_formula("LiP2O5")
    assert rep.valid is False
    assert "unknown_polyanion_cluster" in rep.reasons


def test_structure_classifier_flags_unknown_non_insertion_like_formula() -> None:
    rep = StructureClassifier.classify_formula("LiNiF2")
    assert rep.valid is False
    assert rep.family is None


def test_structure_classifier_uses_prototype_registry_for_olivine() -> None:
    rep = StructureClassifier.classify_formula("LiFePO4", working_ion="Li")
    assert rep.valid is True
    assert rep.family in {"Olivine", "Polyanion framework"}
    assert isinstance(rep.prototype_name, str) and len(rep.prototype_name) > 0


def test_insertion_filter_rejects_alloy_host() -> None:
    rep = InsertionFilter.evaluate_formula("SnSi")
    assert rep.valid is False
    assert "alloy_type_host_rejected" in rep.reasons
    assert rep.insertion_probability < 0.7


def test_insertion_filter_rejects_simple_binary_oxide() -> None:
    rep = InsertionFilter.evaluate_formula("Fe2O3", structure_family="Layered oxide")
    assert rep.valid is False
    assert "simple_binary_oxide_no_open_framework" in rep.reasons
    assert rep.insertion_probability < 0.7


def test_alkali_validator_rejects_mixed_alkali_for_na() -> None:
    rep = AlkaliValidator.validate_formula("LiFePO4", "Na")
    assert rep.valid is False
    assert "mixed_alkali_inconsistency" in rep.reasons


def test_chemistry_validator_rejects_mixed_alkali_candidate() -> None:
    validator = ChemistryValidator()
    candidates, reports = validator.validate_candidates([_candidate("c1", "LiFePO4", "Na")])
    assert len(candidates) == 1
    assert len(reports) == 1
    assert candidates[0].valid is False
    assert any("mixed_alkali" in reason for reason in candidates[0].valid_reasons)


def test_chemistry_validator_rejects_carbonate_electrode() -> None:
    validator = ChemistryValidator()
    candidate = _candidate("c_carbonate", "Fe2(CO3)3", "Li")
    candidate.metadata = {"role_condition": "cathode"}
    candidates, _ = validator.validate_candidates([candidate])
    assert candidates[0].valid is False
    assert "carbonate_electrode_forbidden" in candidates[0].valid_reasons
    assert "gas_evolution_proxy:co2_risk" in candidates[0].valid_reasons


def test_chemistry_validator_requires_redox_active_metal_for_electrode() -> None:
    validator = ChemistryValidator()
    candidate = _candidate("c_no_redox", "Li3PO4", "Li")
    candidate.metadata = {"role_condition": "cathode"}
    candidates, _ = validator.validate_candidates([candidate])
    assert candidates[0].valid is False
    assert "missing_redox_active_metal" in candidates[0].valid_reasons


def test_chemistry_validator_enforces_role_voltage_window() -> None:
    validator = ChemistryValidator()
    candidate = _candidate("c_vwindow", "LiFePO4", "Li")
    candidate.metadata = {"role_condition": "cathode", "reference_voltage": 1.1}
    candidates, _ = validator.validate_candidates([candidate])
    assert candidates[0].valid is False
    assert "role_voltage_window_violation" in candidates[0].valid_reasons


def test_chemistry_validator_uses_ion_specific_voltage_window_for_na() -> None:
    validator = ChemistryValidator()
    candidate = _candidate("c_na_window", "NaFePO4", "Na")
    candidate.metadata = {"role_condition": "anode", "reference_voltage": 1.6}
    candidates, _ = validator.validate_candidates([candidate])
    assert "role_voltage_window_violation" not in candidates[0].valid_reasons


def test_chemistry_validator_uses_ion_specific_voltage_window_for_mg() -> None:
    validator = ChemistryValidator()
    candidate = _candidate("c_mg_window", "MgV2O5", "Mg")
    candidate.metadata = {"role_condition": "cathode", "reference_voltage": 3.6}
    candidates, _ = validator.validate_candidates([candidate])
    assert "role_voltage_window_violation" not in candidates[0].valid_reasons


def test_full_cell_assembler_applies_chemistry_and_volume_rejection() -> None:
    assembler = FullCellAssembler()
    base = BatterySystem(battery_id="base", battery_type="insertion", working_ion="Na")

    rec = CompatibilityRecord(
        cathode=_candidate("cath", "LiFePO4", ion="Na"),
        anode=_candidate("an", "NaTi2(PO4)3", ion="Na"),
        electrolyte=_candidate("el", "Na3V2(PO4)3", ion="Na"),
        voltage_window_overlap_score=0.8,
        chemical_stability_score=0.8,
        mechanical_strain_risk=1.0,
        interface_risk_reason_codes=[],
        hard_valid=True,
    )
    assembled, system = assembler.assemble(
        index=0,
        compatibility=rec,
        target_property_vector={
            "average_voltage": 3.5,
            "capacity_grav": 160.0,
            "max_delta_volume": 0.20,
            "stability_charge": 3.8,
            "stability_discharge": 3.3,
        },
        base_system=base,
    )

    assert assembled.valid is False
    assert any("mixed_alkali_inconsistency" in reason for reason in assembled.valid_reasons)
    assert "volume_expansion_exceeds_threshold" in assembled.valid_reasons
    assert assembled.full_cell_voltage > 1.0
    assert system.average_voltage > 1.0


def test_material_generator_family_priority_uses_safe_default_voltage() -> None:
    gen = MaterialGenerator()
    out = gen._family_priority(target_property_vector={}, working_ion="Li")
    assert isinstance(out, list)
    assert len(out) > 0
