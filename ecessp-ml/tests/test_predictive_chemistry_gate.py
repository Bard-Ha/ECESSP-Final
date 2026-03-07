from __future__ import annotations

import pytest

from backend.api.routes import _predictive_chemistry_gate, _resolve_component_formula


def test_predictive_gate_rejects_mixed_alkali_pair() -> None:
    gate = _predictive_chemistry_gate(
        working_ion="Na",
        cathode_formula="LiFePO4",
        anode_formula="NaTi2(PO4)3",
        electrolyte_formula="NaPF6",
    )
    assert gate["valid"] is False
    reasons = list(gate.get("reasons", []))
    assert any("mixed_alkali_inconsistency" in r for r in reasons)


def test_predictive_gate_accepts_balanced_li_pair() -> None:
    gate = _predictive_chemistry_gate(
        working_ion="Li",
        cathode_formula="Li3V2(PO4)3",
        anode_formula="Li4Ti5O12",
        electrolyte_formula="LiPF6",
    )
    assert gate["valid"] is True
    electro = gate.get("details", {}).get("electrochemistry", {})
    assert float(electro.get("v_cell", 0.0) or 0.0) > 1.0


def test_predictive_gate_rejects_electrolyte_window_violation() -> None:
    gate = _predictive_chemistry_gate(
        working_ion="Zn",
        cathode_formula="ZnMn2O4",
        anode_formula="ZnFe2O4",
        electrolyte_formula="ZnSO4 aqueous",
    )
    assert gate["valid"] is False
    reasons = list(gate.get("reasons", []))
    assert any("electrolyte_stability_window_violation" in r for r in reasons)


def test_resolve_component_formula_falls_back_to_raw_for_unknown_ids() -> None:
    raw = "unknown_component_123"
    assert _resolve_component_formula(raw) == raw


def test_predictive_gate_rejects_negative_redox_potential_gap(monkeypatch) -> None:
    def _mock_estimate(formula: str, role: str, working_ion: str) -> float:
        if role == "cathode":
            return 2.6
        return 3.2

    monkeypatch.setattr("backend.api.routes._estimate_component_potential", _mock_estimate)

    gate = _predictive_chemistry_gate(
        working_ion="Li",
        cathode_formula="MoP2O7",
        anode_formula="Mn5(FeO6)2",
        electrolyte_formula="LiPF6",
    )
    assert gate["valid"] is False
    reasons = list(gate.get("reasons", []))
    assert "electrochemistry:redox_potential_gap_failed" in reasons
    electro = (((gate.get("details") or {}).get("electrochemistry") or {}).get("viability_v2") or {})
    screening = electro.get("screening_output", {})
    assert float(screening.get("theoretical_voltage", 0.0)) == pytest.approx(-0.6)
    assert screening.get("is_viable") is False
    assert "SWAP_ROLES" in str(screening.get("recommendation", ""))
