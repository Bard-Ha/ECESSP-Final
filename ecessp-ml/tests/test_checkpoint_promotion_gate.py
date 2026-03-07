from __future__ import annotations

from argparse import Namespace

from tools.checkpoint_promotion_gate import _extract_metrics, _gate_decision, _selection_score


def _summary_with_metrics(
    *,
    iid_overall_r2: float,
    ood_overall_r2: float,
    iid_capacity_r2: float,
    ood_capacity_r2: float,
    iid_capacity_violation_rate: float,
    ood_capacity_violation_rate: float,
    iid_voltage_violation_rate: float,
    ood_voltage_violation_rate: float,
) -> dict:
    return {
        "evaluation_metrics": {
            "iid": {
                "raw_space": {
                    "overall_micro": {"r2": iid_overall_r2},
                    "capacity_grav": {"r2": iid_capacity_r2},
                    "average_voltage": {"r2": 0.9},
                }
            },
            "ood": {
                "raw_space": {
                    "overall_micro": {"r2": ood_overall_r2},
                    "capacity_grav": {"r2": ood_capacity_r2},
                    "average_voltage": {"r2": 0.9},
                }
            },
        },
        "physics_violation_statistics": {
            "iid": {
                "capacity_violation_rate": iid_capacity_violation_rate,
                "voltage_above_4p4_rate": iid_voltage_violation_rate,
            },
            "ood": {
                "capacity_violation_rate": ood_capacity_violation_rate,
                "voltage_above_4p4_rate": ood_voltage_violation_rate,
            },
        },
    }


def _args() -> Namespace:
    return Namespace(
        min_iid_r2=0.45,
        min_ood_r2=0.45,
        min_iid_capacity_r2=0.40,
        min_ood_capacity_r2=0.40,
        max_capacity_violation_rate=0.02,
        max_voltage_violation_rate=0.02,
    )


def test_gate_decision_accepts_good_summary() -> None:
    summary = _summary_with_metrics(
        iid_overall_r2=0.70,
        ood_overall_r2=0.68,
        iid_capacity_r2=0.55,
        ood_capacity_r2=0.50,
        iid_capacity_violation_rate=0.0,
        ood_capacity_violation_rate=0.0,
        iid_voltage_violation_rate=0.0,
        ood_voltage_violation_rate=0.0,
    )
    metrics = _extract_metrics(summary)
    passed, reasons = _gate_decision(metrics, _args())
    assert passed is True
    assert reasons == []


def test_gate_decision_rejects_low_ood_and_high_violation() -> None:
    summary = _summary_with_metrics(
        iid_overall_r2=0.70,
        ood_overall_r2=0.30,
        iid_capacity_r2=0.55,
        ood_capacity_r2=0.20,
        iid_capacity_violation_rate=0.01,
        ood_capacity_violation_rate=0.10,
        iid_voltage_violation_rate=0.0,
        ood_voltage_violation_rate=0.03,
    )
    metrics = _extract_metrics(summary)
    passed, reasons = _gate_decision(metrics, _args())
    assert passed is False
    assert any("ood_overall_r2" in reason for reason in reasons)
    assert any("ood_capacity_violation_rate" in reason for reason in reasons)


def test_selection_score_penalizes_violations() -> None:
    clean = _extract_metrics(
        _summary_with_metrics(
            iid_overall_r2=0.65,
            ood_overall_r2=0.60,
            iid_capacity_r2=0.50,
            ood_capacity_r2=0.48,
            iid_capacity_violation_rate=0.0,
            ood_capacity_violation_rate=0.0,
            iid_voltage_violation_rate=0.0,
            ood_voltage_violation_rate=0.0,
        )
    )
    dirty = _extract_metrics(
        _summary_with_metrics(
            iid_overall_r2=0.65,
            ood_overall_r2=0.60,
            iid_capacity_r2=0.50,
            ood_capacity_r2=0.48,
            iid_capacity_violation_rate=0.05,
            ood_capacity_violation_rate=0.05,
            iid_voltage_violation_rate=0.03,
            ood_voltage_violation_rate=0.03,
        )
    )
    assert _selection_score(clean) > _selection_score(dirty)
