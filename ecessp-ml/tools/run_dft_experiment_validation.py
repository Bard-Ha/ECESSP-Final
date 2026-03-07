#!/usr/bin/env python
"""
Compute DFT/experiment validation metrics and confidence index for ECESSP.

Example:
  python tools/run_dft_experiment_validation.py \
    --predictions-csv reports/validation_protocol/templates/predictions_manifest_template.csv \
    --dft-csv reports/validation_protocol/templates/dft_results_template.csv \
    --experiment-csv reports/validation_protocol/templates/experiment_results_template.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "reports" / "validation_protocol" / "dft_experiment_validation_protocol_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "validation_protocol" / "dft_experiment_validation_report.json"


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return float(out)


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {str(k or "").strip(): str(v or "").strip() for k, v in raw.items()}
            if any(v for v in row.values()):
                rows.append(row)
    return rows


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    mean_x = sum(xs[:n]) / float(n)
    mean_y = sum(ys[:n]) / float(n)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs[:n], ys[:n]):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    if den <= 1e-12:
        return None
    return float(num / den)


def _mae(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(abs(v) for v in values) / float(len(values)))


def _score_high_good(value: Optional[float], target_min: float) -> float:
    if value is None or target_min <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(target_min)))


def _score_low_good(value: Optional[float], target_max: float) -> float:
    if value is None or target_max <= 0.0:
        return 0.0
    if value <= target_max:
        return 1.0
    return max(0.0, min(1.0, float(target_max) / float(value)))


def _tier(confidence_index: float, tiers: Dict[str, float]) -> str:
    high = float(tiers.get("high", 85.0))
    medium = float(tiers.get("medium", 70.0))
    if confidence_index >= high:
        return "high"
    if confidence_index >= medium:
        return "medium"
    return "low"


def _eval_dft(
    predictions_by_id: Dict[str, Dict[str, str]],
    dft_rows: List[Dict[str, str]],
    gate_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[float], List[float]]:
    records: List[Dict[str, Any]] = []
    skipped_missing_prediction = 0
    abs_voltage_errors: List[float] = []
    uncertainty_values: List[float] = []

    for row in dft_rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        pred = predictions_by_id.get(candidate_id)
        if pred is None:
            skipped_missing_prediction += 1
            continue

        pred_voltage = _to_float(pred.get("predicted_voltage_v"))
        pred_uncertainty = _to_float(pred.get("predicted_uncertainty_penalty"))
        dft_voltage = _to_float(row.get("dft_voltage_v"))
        dft_volume = _to_float(row.get("dft_volume_change_ratio"))
        dft_decomp = _to_float(row.get("dft_decomposition_energy_ev_atom"))
        dft_barrier = _to_float(row.get("dft_diffusion_barrier_ev"))
        dft_converged = _to_bool(row.get("dft_converged"))
        insertion_stable = _to_bool(row.get("dft_insertion_stable"))

        reasons: List[str] = []
        abs_err = None
        if pred_voltage is not None and dft_voltage is not None:
            abs_err = abs(float(pred_voltage) - float(dft_voltage))
            abs_voltage_errors.append(float(abs_err))
            if pred_uncertainty is not None:
                uncertainty_values.append(float(pred_uncertainty))

        if bool(gate_cfg.get("require_converged", True)) and dft_converged is not True:
            reasons.append("dft_not_converged")
        if bool(gate_cfg.get("require_insertion_stable", True)) and insertion_stable is not True:
            reasons.append("dft_not_insertion_stable")

        max_v_err = _to_float(gate_cfg.get("max_abs_voltage_error_v"))
        if max_v_err is not None and abs_err is not None and abs_err > max_v_err:
            reasons.append("voltage_error_exceeds_threshold")

        max_volume = _to_float(gate_cfg.get("max_volume_change_ratio"))
        if max_volume is not None and dft_volume is not None and dft_volume > max_volume:
            reasons.append("volume_change_exceeds_threshold")

        max_decomp = _to_float(gate_cfg.get("max_decomposition_energy_ev_atom"))
        if max_decomp is not None and dft_decomp is not None and dft_decomp > max_decomp:
            reasons.append("decomposition_energy_exceeds_threshold")

        max_barrier = _to_float(gate_cfg.get("max_diffusion_barrier_ev"))
        if max_barrier is not None and dft_barrier is not None and dft_barrier > max_barrier:
            reasons.append("diffusion_barrier_exceeds_threshold")

        records.append(
            {
                "candidate_id": candidate_id,
                "pass": len(reasons) == 0,
                "reasons": reasons,
                "abs_voltage_error_v": abs_err,
                "prediction": {
                    "predicted_voltage_v": pred_voltage,
                    "predicted_uncertainty_penalty": pred_uncertainty,
                },
                "dft": {
                    "dft_converged": dft_converged,
                    "dft_insertion_stable": insertion_stable,
                    "dft_voltage_v": dft_voltage,
                    "dft_volume_change_ratio": dft_volume,
                    "dft_decomposition_energy_ev_atom": dft_decomp,
                    "dft_diffusion_barrier_ev": dft_barrier,
                },
            }
        )

    attempted = len(records)
    passed = sum(1 for r in records if bool(r.get("pass")))
    out = {
        "attempted": attempted,
        "passed": passed,
        "pass_rate": float(passed / max(1, attempted)),
        "voltage_mae_v": _mae(abs_voltage_errors),
        "records": records,
        "skipped_missing_prediction": skipped_missing_prediction,
    }
    return out, uncertainty_values, abs_voltage_errors


def _eval_experiment(
    predictions_by_id: Dict[str, Dict[str, str]],
    exp_rows: List[Dict[str, str]],
    gate_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[float], List[float]]:
    records: List[Dict[str, Any]] = []
    skipped_missing_prediction = 0
    abs_voltage_errors: List[float] = []
    uncertainty_values: List[float] = []

    for row in exp_rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        pred = predictions_by_id.get(candidate_id)
        if pred is None:
            skipped_missing_prediction += 1
            continue

        pred_voltage = _to_float(pred.get("predicted_voltage_v"))
        pred_capacity = _to_float(pred.get("predicted_capacity_mah_g"))
        pred_uncertainty = _to_float(pred.get("predicted_uncertainty_penalty"))

        exp_voltage = _to_float(row.get("exp_voltage_v"))
        exp_capacity = _to_float(row.get("exp_capacity_mah_g"))
        exp_retention = _to_float(row.get("exp_capacity_retention_50cyc_pct"))
        exp_ce = _to_float(row.get("exp_initial_coulombic_efficiency_pct"))
        exp_completed = _to_bool(row.get("experiment_completed"))
        exp_reversible = _to_bool(row.get("exp_insertion_reversible"))

        reasons: List[str] = []
        abs_err = None
        if pred_voltage is not None and exp_voltage is not None:
            abs_err = abs(float(pred_voltage) - float(exp_voltage))
            abs_voltage_errors.append(float(abs_err))
            if pred_uncertainty is not None:
                uncertainty_values.append(float(pred_uncertainty))

        if exp_completed is not True:
            reasons.append("experiment_not_completed")
        if bool(gate_cfg.get("require_reversible_insertion", True)) and exp_reversible is not True:
            reasons.append("experiment_not_reversible")

        max_v_err = _to_float(gate_cfg.get("max_abs_voltage_error_v"))
        if max_v_err is not None and abs_err is not None and abs_err > max_v_err:
            reasons.append("voltage_error_exceeds_threshold")

        min_frac = _to_float(gate_cfg.get("min_capacity_fraction_vs_prediction"))
        if min_frac is not None and pred_capacity is not None and pred_capacity > 0 and exp_capacity is not None:
            if float(exp_capacity) / float(pred_capacity) < min_frac:
                reasons.append("capacity_fraction_below_threshold")

        min_retention = _to_float(gate_cfg.get("min_capacity_retention_50cyc_pct"))
        if min_retention is not None and exp_retention is not None and exp_retention < min_retention:
            reasons.append("capacity_retention_below_threshold")

        min_ce = _to_float(gate_cfg.get("min_initial_coulombic_efficiency_pct"))
        if min_ce is not None and exp_ce is not None and exp_ce < min_ce:
            reasons.append("initial_ce_below_threshold")

        records.append(
            {
                "candidate_id": candidate_id,
                "pass": len(reasons) == 0,
                "reasons": reasons,
                "abs_voltage_error_v": abs_err,
                "prediction": {
                    "predicted_voltage_v": pred_voltage,
                    "predicted_capacity_mah_g": pred_capacity,
                    "predicted_uncertainty_penalty": pred_uncertainty,
                },
                "experiment": {
                    "experiment_completed": exp_completed,
                    "exp_insertion_reversible": exp_reversible,
                    "exp_voltage_v": exp_voltage,
                    "exp_capacity_mah_g": exp_capacity,
                    "exp_capacity_retention_50cyc_pct": exp_retention,
                    "exp_initial_coulombic_efficiency_pct": exp_ce,
                },
            }
        )

    attempted = len(records)
    passed = sum(1 for r in records if bool(r.get("pass")))
    out = {
        "attempted": attempted,
        "passed": passed,
        "pass_rate": float(passed / max(1, attempted)),
        "voltage_mae_v": _mae(abs_voltage_errors),
        "records": records,
        "skipped_missing_prediction": skipped_missing_prediction,
    }
    return out, uncertainty_values, abs_voltage_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DFT/experiment validation protocol.")
    parser.add_argument("--protocol-json", default=str(DEFAULT_PROTOCOL), help="Validation protocol JSON path.")
    parser.add_argument("--predictions-csv", required=True, help="Prediction manifest CSV.")
    parser.add_argument("--dft-csv", required=True, help="DFT results CSV.")
    parser.add_argument("--experiment-csv", required=True, help="Experiment results CSV.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT), help="Output report JSON path.")
    args = parser.parse_args()

    protocol_path = Path(args.protocol_json).resolve()
    predictions_path = Path(args.predictions_csv).resolve()
    dft_path = Path(args.dft_csv).resolve()
    exp_path = Path(args.experiment_csv).resolve()
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pred_rows = _load_csv(predictions_path)
    dft_rows = _load_csv(dft_path)
    exp_rows = _load_csv(exp_path)
    predictions_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in pred_rows
        if str(row.get("candidate_id") or "").strip()
    }

    dft_eval, dft_unc, dft_abs_err = _eval_dft(
        predictions_by_id=predictions_by_id,
        dft_rows=dft_rows,
        gate_cfg=dict(protocol.get("stage_gates", {}).get("dft", {})),
    )
    exp_eval, exp_unc, exp_abs_err = _eval_experiment(
        predictions_by_id=predictions_by_id,
        exp_rows=exp_rows,
        gate_cfg=dict(protocol.get("stage_gates", {}).get("experiment", {})),
    )

    dft_corr = _pearson(dft_unc, dft_abs_err)
    exp_corr = _pearson(exp_unc, exp_abs_err)
    combined_unc = dft_unc + exp_unc
    combined_abs_err = dft_abs_err + exp_abs_err
    combined_corr = _pearson(combined_unc, combined_abs_err)

    accept_cfg = dict(protocol.get("program_acceptance", {}))
    min_samples = dict(accept_cfg.get("minimum_sample_sizes", {}))
    min_pass = dict(accept_cfg.get("minimum_pass_rates", {}))
    max_err = dict(accept_cfg.get("maximum_error_metrics", {}))
    min_cal = dict(accept_cfg.get("minimum_calibration", {}))

    gates: Dict[str, Dict[str, Any]] = {}
    gates["min_dft_sample_size"] = {
        "passed": int(dft_eval["attempted"]) >= int(min_samples.get("dft_candidates", 0)),
        "value": int(dft_eval["attempted"]),
        "target_min": int(min_samples.get("dft_candidates", 0)),
    }
    gates["min_experiment_sample_size"] = {
        "passed": int(exp_eval["attempted"]) >= int(min_samples.get("experiment_candidates", 0)),
        "value": int(exp_eval["attempted"]),
        "target_min": int(min_samples.get("experiment_candidates", 0)),
    }
    gates["min_dft_pass_rate"] = {
        "passed": float(dft_eval["pass_rate"]) >= float(min_pass.get("dft", 0.0)),
        "value": float(dft_eval["pass_rate"]),
        "target_min": float(min_pass.get("dft", 0.0)),
    }
    gates["min_experiment_pass_rate"] = {
        "passed": float(exp_eval["pass_rate"]) >= float(min_pass.get("experiment", 0.0)),
        "value": float(exp_eval["pass_rate"]),
        "target_min": float(min_pass.get("experiment", 0.0)),
    }
    dft_mae = _to_float(dft_eval.get("voltage_mae_v"))
    exp_mae = _to_float(exp_eval.get("voltage_mae_v"))
    gates["max_dft_voltage_mae"] = {
        "passed": (dft_mae is not None) and (dft_mae <= float(max_err.get("dft_voltage_mae_v", 1e9))),
        "value": dft_mae,
        "target_max": float(max_err.get("dft_voltage_mae_v", 1e9)),
    }
    gates["max_experiment_voltage_mae"] = {
        "passed": (exp_mae is not None) and (exp_mae <= float(max_err.get("experiment_voltage_mae_v", 1e9))),
        "value": exp_mae,
        "target_max": float(max_err.get("experiment_voltage_mae_v", 1e9)),
    }
    min_corr = float(min_cal.get("uncertainty_vs_abs_voltage_error_pearson_r", 0.0))
    gates["min_uncertainty_calibration_corr"] = {
        "passed": (combined_corr is not None) and (combined_corr >= min_corr),
        "value": combined_corr,
        "target_min": min_corr,
        "details": "Pearson r between predicted uncertainty penalty and observed absolute voltage error.",
    }

    critical_ok = all(bool(v.get("passed")) for v in gates.values())

    ci_cfg = dict(protocol.get("confidence_index", {}))
    ci_weights = dict(ci_cfg.get("weights", {}))
    w_dft = float(ci_weights.get("dft_pass_rate", 0.35))
    w_exp = float(ci_weights.get("experiment_pass_rate", 0.35))
    w_dft_mae = float(ci_weights.get("dft_voltage_mae", 0.15))
    w_exp_mae = float(ci_weights.get("experiment_voltage_mae", 0.10))
    w_cal = float(ci_weights.get("uncertainty_calibration", 0.05))

    dft_pass_score = _score_high_good(_to_float(dft_eval.get("pass_rate")), float(min_pass.get("dft", 1.0)))
    exp_pass_score = _score_high_good(_to_float(exp_eval.get("pass_rate")), float(min_pass.get("experiment", 1.0)))
    dft_mae_score = _score_low_good(dft_mae, float(max_err.get("dft_voltage_mae_v", 1.0)))
    exp_mae_score = _score_low_good(exp_mae, float(max_err.get("experiment_voltage_mae_v", 1.0)))
    cal_score = _score_high_good(combined_corr, min_corr if min_corr > 0.0 else 1.0)

    confidence_index = 100.0 * (
        w_dft * dft_pass_score
        + w_exp * exp_pass_score
        + w_dft_mae * dft_mae_score
        + w_exp_mae * exp_mae_score
        + w_cal * cal_score
    )
    confidence_index = max(0.0, min(100.0, confidence_index))
    confidence_tier = _tier(confidence_index, dict(ci_cfg.get("tiers", {})))
    ready_for_scale_up = bool(critical_ok and confidence_tier in {"high", "medium"})

    report = {
        "report_type": "ecessp_dft_experiment_validation_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "version": protocol.get("version"),
        },
        "inputs": {
            "predictions_csv": str(predictions_path),
            "dft_csv": str(dft_path),
            "experiment_csv": str(exp_path),
            "prediction_count": len(predictions_by_id),
            "dft_rows": len(dft_rows),
            "experiment_rows": len(exp_rows),
        },
        "stage_metrics": {
            "dft": dft_eval,
            "experiment": exp_eval,
        },
        "calibration": {
            "dft_uncertainty_vs_abs_voltage_error_pearson_r": dft_corr,
            "experiment_uncertainty_vs_abs_voltage_error_pearson_r": exp_corr,
            "combined_uncertainty_vs_abs_voltage_error_pearson_r": combined_corr,
            "n_pairs_dft": min(len(dft_unc), len(dft_abs_err)),
            "n_pairs_experiment": min(len(exp_unc), len(exp_abs_err)),
        },
        "acceptance_gates": gates,
        "confidence": {
            "confidence_index_0_to_100": round(float(confidence_index), 3),
            "tier": confidence_tier,
            "component_scores": {
                "dft_pass_rate_score": round(float(dft_pass_score), 4),
                "experiment_pass_rate_score": round(float(exp_pass_score), 4),
                "dft_voltage_mae_score": round(float(dft_mae_score), 4),
                "experiment_voltage_mae_score": round(float(exp_mae_score), 4),
                "uncertainty_calibration_score": round(float(cal_score), 4),
            },
        },
        "decision": {
            "ready_for_scale_up": ready_for_scale_up,
            "critical_gates_passed": critical_ok,
            "recommended_mode": "production_candidate_screening" if ready_for_scale_up else "exploratory_active_learning_only",
        },
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] validation report written to: {output_path}")
    print(
        f"[summary] ready_for_scale_up={ready_for_scale_up} "
        f"confidence_index={report['confidence']['confidence_index_0_to_100']} "
        f"tier={confidence_tier}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
