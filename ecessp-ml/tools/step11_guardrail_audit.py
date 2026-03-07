#!/usr/bin/env python3
"""
Step 11: Unified prediction + discovery guardrail audit.

This script reads the latest report artifacts and emits a single readiness report
that can be used as a deployment gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _check_metric(
    checks: List[Dict[str, Any]],
    *,
    name: str,
    value: float,
    threshold: float,
    comparator: str,
) -> None:
    if comparator == ">=":
        passed = value >= threshold
    elif comparator == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported comparator: {comparator}")

    checks.append(
        {
            "name": name,
            "value": float(value),
            "threshold": float(threshold),
            "comparator": comparator,
            "passed": bool(passed),
        }
    )


def _evaluate_prediction(training_summary: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    iid_raw = training_summary.get("evaluation_metrics", {}).get("iid", {}).get("raw_space", {})
    ood_raw = training_summary.get("evaluation_metrics", {}).get("ood", {}).get("raw_space", {})
    phys = training_summary.get("physics_violation_statistics", {})
    iid_phys = phys.get("iid", {}) if isinstance(phys, dict) else {}
    ood_phys = phys.get("ood", {}) if isinstance(phys, dict) else {}

    _check_metric(
        checks,
        name="iid_overall_r2",
        value=_safe_float(iid_raw.get("overall_micro", {}).get("r2"), -1.0),
        threshold=0.25,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="ood_overall_r2",
        value=_safe_float(ood_raw.get("overall_micro", {}).get("r2"), -1.0),
        threshold=0.25,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="iid_voltage_r2",
        value=_safe_float(iid_raw.get("average_voltage", {}).get("r2"), -1.0),
        threshold=0.75,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="iid_capacity_vol_r2",
        value=_safe_float(iid_raw.get("capacity_vol", {}).get("r2"), -1.0),
        threshold=0.70,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="iid_capacity_grav_r2",
        value=_safe_float(iid_raw.get("capacity_grav", {}).get("r2"), -1.0),
        threshold=0.00,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="iid_capacity_violation_rate",
        value=_safe_float(iid_phys.get("capacity_violation_rate"), 1.0),
        threshold=0.0,
        comparator="<=",
    )
    _check_metric(
        checks,
        name="ood_capacity_violation_rate",
        value=_safe_float(ood_phys.get("capacity_violation_rate"), 1.0),
        threshold=0.0,
        comparator="<=",
    )
    _check_metric(
        checks,
        name="iid_voltage_above_4p4_rate",
        value=_safe_float(iid_phys.get("voltage_above_4p4_rate"), 1.0),
        threshold=0.0,
        comparator="<=",
    )
    _check_metric(
        checks,
        name="ood_voltage_above_4p4_rate",
        value=_safe_float(ood_phys.get("voltage_above_4p4_rate"), 1.0),
        threshold=0.0,
        comparator="<=",
    )

    failed = [c for c in checks if not c["passed"]]
    return {
        "passed": len(failed) == 0,
        "checks": checks,
        "failed_checks": failed,
    }


def _evaluate_discovery(discovery_validation: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    agg = discovery_validation.get("aggregate_report_card", {})

    _check_metric(
        checks,
        name="hit_rate_at_k",
        value=_safe_float(agg.get("hit_rate_at_k"), 0.0),
        threshold=0.65,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="constraint_validity_rate",
        value=_safe_float(agg.get("constraint_validity_rate"), 0.0),
        threshold=0.70,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="physics_violation_rate",
        value=_safe_float(agg.get("physics_violation_rate"), 1.0),
        threshold=0.05,
        comparator="<=",
    )
    _check_metric(
        checks,
        name="novelty_mean",
        value=_safe_float(agg.get("novelty_mean"), 0.0),
        threshold=0.10,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="ood_acceptance_rate",
        value=_safe_float(agg.get("ood_acceptance_rate"), 0.0),
        threshold=0.10,
        comparator=">=",
    )
    _check_metric(
        checks,
        name="counterfactual_pass_rate",
        value=_safe_float(discovery_validation.get("counterfactual_pass_rate"), 0.0),
        threshold=0.80,
        comparator=">=",
    )

    failed = [c for c in checks if not c["passed"]]
    return {
        "passed": len(failed) == 0,
        "checks": checks,
        "failed_checks": failed,
    }


def _build_recommendations(pred_failed: List[Dict[str, Any]], disc_failed: List[Dict[str, Any]]) -> List[str]:
    recs: List[str] = []

    failed_names = {str(item.get("name")) for item in pred_failed + disc_failed}
    if "iid_capacity_grav_r2" in failed_names:
        recs.append("Reweight/transform gravimetric capacity target (log transform + robust loss + ion-conditioned head).")
    if "novelty_mean" in failed_names or "ood_acceptance_rate" in failed_names:
        recs.append("Reduce objective-attraction strength or add Pareto ranking to restore novelty under constraints.")
    if "counterfactual_pass_rate" in failed_names:
        recs.append("Increase counterfactual consistency regularization and rerun directional validation.")
    if "hit_rate_at_k" in failed_names:
        recs.append("Tune latent optimization steps/diversity weight to improve top-k target hit rate.")
    if "constraint_validity_rate" in failed_names or "physics_violation_rate" in failed_names:
        recs.append("Tighten hard-gate projection and material validity thresholds before ranking.")

    if not recs:
        recs.append("All guardrails passed; proceed to deployment with monitoring enabled.")
    return recs


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11 prediction+discovery guardrail audit")
    parser.add_argument("--training-summary", default="reports/training_summary.json")
    parser.add_argument("--discovery-validation", default="reports/discovery_validation_step9.json")
    parser.add_argument("--out", default="reports/step11_guardrail_report.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    training_summary_path = (root / args.training_summary).resolve()
    discovery_validation_path = (root / args.discovery_validation).resolve()
    out_path = (root / args.out).resolve()

    report: Dict[str, Any] = {
        "step": 11,
        "inputs": {
            "training_summary": str(training_summary_path),
            "discovery_validation": str(discovery_validation_path),
        },
    }

    pred_eval: Dict[str, Any]
    if training_summary_path.exists():
        pred_eval = _evaluate_prediction(_load_json(training_summary_path))
    else:
        pred_eval = {
            "passed": False,
            "checks": [],
            "failed_checks": [{"name": "training_summary_missing", "path": str(training_summary_path)}],
        }

    disc_eval: Dict[str, Any]
    if discovery_validation_path.exists():
        disc_eval = _evaluate_discovery(_load_json(discovery_validation_path))
    else:
        disc_eval = {
            "passed": False,
            "checks": [],
            "failed_checks": [{"name": "discovery_validation_missing", "path": str(discovery_validation_path)}],
        }

    report["prediction_pipeline"] = pred_eval
    report["discovery_pipeline"] = disc_eval
    report["overall_passed"] = bool(pred_eval.get("passed", False) and disc_eval.get("passed", False))
    report["recommendations"] = _build_recommendations(
        pred_failed=pred_eval.get("failed_checks", []),
        disc_failed=disc_eval.get("failed_checks", []),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_path)
    print(f"overall_passed={report['overall_passed']}")


if __name__ == "__main__":
    main()
