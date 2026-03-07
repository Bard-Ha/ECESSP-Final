#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_checkpoint(path_like: str, root: Path) -> Optional[Path]:
    raw = str(path_like or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (root / p).resolve()
    if p.exists() and p.is_file():
        return p
    return None


def _extract_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    iid = summary.get("evaluation_metrics", {}).get("iid", {}).get("raw_space", {})
    ood = summary.get("evaluation_metrics", {}).get("ood", {}).get("raw_space", {})
    phys = summary.get("physics_violation_statistics", {})
    iid_phys = phys.get("iid", {}) if isinstance(phys, dict) else {}
    ood_phys = phys.get("ood", {}) if isinstance(phys, dict) else {}
    return {
        "iid_overall_r2": _safe_float((iid.get("overall_micro") or {}).get("r2"), -1.0),
        "ood_overall_r2": _safe_float((ood.get("overall_micro") or {}).get("r2"), -1.0),
        "iid_capacity_r2": _safe_float((iid.get("capacity_grav") or {}).get("r2"), -1.0),
        "ood_capacity_r2": _safe_float((ood.get("capacity_grav") or {}).get("r2"), -1.0),
        "iid_voltage_r2": _safe_float((iid.get("average_voltage") or {}).get("r2"), -1.0),
        "ood_voltage_r2": _safe_float((ood.get("average_voltage") or {}).get("r2"), -1.0),
        "iid_capacity_violation_rate": _safe_float(iid_phys.get("capacity_violation_rate"), 1.0),
        "ood_capacity_violation_rate": _safe_float(ood_phys.get("capacity_violation_rate"), 1.0),
        "iid_voltage_violation_rate": _safe_float(iid_phys.get("voltage_above_4p4_rate"), 1.0),
        "ood_voltage_violation_rate": _safe_float(ood_phys.get("voltage_above_4p4_rate"), 1.0),
    }


def _selection_score(metrics: Dict[str, float]) -> float:
    total_violation = (
        metrics["iid_capacity_violation_rate"]
        + metrics["ood_capacity_violation_rate"]
        + metrics["iid_voltage_violation_rate"]
        + metrics["ood_voltage_violation_rate"]
    )
    return float(
        0.30 * metrics["iid_overall_r2"]
        + 0.30 * metrics["ood_overall_r2"]
        + 0.15 * metrics["iid_capacity_r2"]
        + 0.15 * metrics["ood_capacity_r2"]
        + 0.05 * metrics["iid_voltage_r2"]
        + 0.05 * metrics["ood_voltage_r2"]
        - 3.0 * total_violation
    )


def _gate_decision(metrics: Dict[str, float], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if metrics["iid_overall_r2"] < float(args.min_iid_r2):
        reasons.append(f"iid_overall_r2<{args.min_iid_r2}")
    if metrics["ood_overall_r2"] < float(args.min_ood_r2):
        reasons.append(f"ood_overall_r2<{args.min_ood_r2}")
    if metrics["iid_capacity_r2"] < float(args.min_iid_capacity_r2):
        reasons.append(f"iid_capacity_r2<{args.min_iid_capacity_r2}")
    if metrics["ood_capacity_r2"] < float(args.min_ood_capacity_r2):
        reasons.append(f"ood_capacity_r2<{args.min_ood_capacity_r2}")
    if metrics["iid_capacity_violation_rate"] > float(args.max_capacity_violation_rate):
        reasons.append(f"iid_capacity_violation_rate>{args.max_capacity_violation_rate}")
    if metrics["ood_capacity_violation_rate"] > float(args.max_capacity_violation_rate):
        reasons.append(f"ood_capacity_violation_rate>{args.max_capacity_violation_rate}")
    if metrics["iid_voltage_violation_rate"] > float(args.max_voltage_violation_rate):
        reasons.append(f"iid_voltage_violation_rate>{args.max_voltage_violation_rate}")
    if metrics["ood_voltage_violation_rate"] > float(args.max_voltage_violation_rate):
        reasons.append(f"ood_voltage_violation_rate>{args.max_voltage_violation_rate}")
    return (len(reasons) == 0), reasons


def _latest_checkpoint(models_dir: Path, pattern: str) -> Optional[Path]:
    candidates = sorted(models_dir.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _load_manifest_models(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = _load_json(path)
    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in models:
        if isinstance(item, dict):
            out.append(item)
    return out


def _resolve_hgt_checkpoint(
    *,
    root: Path,
    models_dir: Path,
    hgt_summary_path: Path,
    final_manifest_models: List[Dict[str, Any]],
    enable_hgt: bool,
) -> Optional[Path]:
    if not enable_hgt:
        return None

    if hgt_summary_path.exists():
        hgt_summary = _load_json(hgt_summary_path)
        p = _resolve_checkpoint(str(hgt_summary.get("best_checkpoint", "")), root)
        if p is not None:
            return p

    for item in final_manifest_models:
        family = str(item.get("family", "")).strip().lower()
        interaction = str(item.get("material_interaction", "")).strip().lower()
        name = str(item.get("name", "")).strip().lower()
        if family == "hetero_hgt" or interaction == "hgt" or name == "hetero_hgt":
            p = _resolve_checkpoint(str(item.get("checkpoint_path", "")), root)
            if p is not None:
                return p

    return _latest_checkpoint(models_dir, "hgt_best_*.pt")


def _prune_checkpoints(models_dir: Path, keep_paths: set[Path]) -> List[str]:
    removed: List[str] = []
    for p in sorted(models_dir.glob("*.pt")):
        rp = p.resolve()
        if rp in keep_paths:
            continue
        p.unlink(missing_ok=True)
        removed.append(str(rp))
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote one runtime checkpoint, regenerate clean summary, and prune stale checkpoints.")
    ap.add_argument("--training-summary", default="reports/training_summary.json")
    ap.add_argument("--hgt-summary", default="reports/training_summary_hgt.json")
    ap.add_argument("--models-dir", default="reports/models")
    ap.add_argument("--three-model-manifest", default="reports/three_model_ensemble_manifest.json")
    ap.add_argument("--final-manifest", default="reports/final_family_ensemble_manifest.json")
    ap.add_argument("--promoted-summary-out", default="reports/training_summary.promoted.json")
    ap.add_argument("--report-out", default="reports/checkpoint_promotion_report.json")
    ap.add_argument("--promote-in-place", action="store_true")
    ap.add_argument("--prune-checkpoints", action="store_true")
    ap.add_argument("--disable-hgt", action="store_true")

    ap.add_argument("--min-iid-r2", type=float, default=0.45)
    ap.add_argument("--min-ood-r2", type=float, default=0.45)
    ap.add_argument("--min-iid-capacity-r2", type=float, default=0.40)
    ap.add_argument("--min-ood-capacity-r2", type=float, default=0.40)
    ap.add_argument("--max-capacity-violation-rate", type=float, default=0.02)
    ap.add_argument("--max-voltage-violation-rate", type=float, default=0.02)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary_path = (root / args.training_summary).resolve()
    hgt_summary_path = (root / args.hgt_summary).resolve()
    models_dir = (root / args.models_dir).resolve()
    three_manifest_path = (root / args.three_model_manifest).resolve()
    final_manifest_path = (root / args.final_manifest).resolve()
    promoted_summary_path = (root / args.promoted_summary_out).resolve()
    report_out_path = (root / args.report_out).resolve()

    if not summary_path.exists():
        raise FileNotFoundError(f"training summary missing: {summary_path}")
    if not models_dir.exists():
        raise FileNotFoundError(f"models directory missing: {models_dir}")

    summary = _load_json(summary_path)
    metrics = _extract_metrics(summary)
    gate_passed, gate_reasons = _gate_decision(metrics, args)
    score = _selection_score(metrics)

    selected_main = _resolve_checkpoint(str(summary.get("best_checkpoint", "")), root)
    if selected_main is None:
        selected_main = _latest_checkpoint(models_dir, "ensemble_0_best_*.pt")
    if selected_main is None:
        selected_main = _latest_checkpoint(models_dir, "best_model_layer_norm_*.pt")
    if selected_main is None:
        raise RuntimeError("No suitable masked checkpoint found for promotion.")

    final_manifest_models = _load_manifest_models(final_manifest_path)
    selected_hgt = _resolve_hgt_checkpoint(
        root=root,
        models_dir=models_dir,
        hgt_summary_path=hgt_summary_path,
        final_manifest_models=final_manifest_models,
        enable_hgt=(not bool(args.disable_hgt)),
    )

    promoted_summary = dict(summary)
    promoted_summary["best_checkpoint"] = str(selected_main.resolve())
    promoted_summary["checkpoint_promotion"] = {
        "timestamp_epoch_sec": int(time.time()),
        "source_tool": "tools/checkpoint_promotion_gate.py",
        "gate_passed": bool(gate_passed),
        "gate_failure_reasons": list(gate_reasons),
        "selection_score": float(score),
        "selected_main_checkpoint": str(selected_main.resolve()),
        "selected_hgt_checkpoint": str(selected_hgt.resolve()) if selected_hgt is not None else None,
        "thresholds": {
            "min_iid_r2": float(args.min_iid_r2),
            "min_ood_r2": float(args.min_ood_r2),
            "min_iid_capacity_r2": float(args.min_iid_capacity_r2),
            "min_ood_capacity_r2": float(args.min_ood_capacity_r2),
            "max_capacity_violation_rate": float(args.max_capacity_violation_rate),
            "max_voltage_violation_rate": float(args.max_voltage_violation_rate),
        },
        "metrics": dict(metrics),
    }
    _save_json(promoted_summary_path, promoted_summary)

    one_model_manifest = {
        "version": "v2",
        "created_at_epoch_sec": int(time.time()),
        "source": "tools/checkpoint_promotion_gate.py",
        "models": [
            {
                "name": "masked_runtime_promoted",
                "checkpoint_path": str(selected_main.resolve()),
                "weight": 1.0,
                "material_interaction": str(
                    (summary.get("model_architecture") or {}).get("material_interaction", "masked_gnn")
                ),
                "selection_score": float(score),
                "summary_path": str(promoted_summary_path),
                "summary_timestamp": str(summary.get("timestamp", "")),
            }
        ],
        "primary_model": "masked_runtime_promoted",
    }
    _save_json(three_manifest_path, one_model_manifest)

    final_models: List[Dict[str, Any]] = [
        {
            "name": "masked_runtime_promoted",
            "family": "masked_gnn",
            "checkpoint_path": str(selected_main.resolve()),
            "score": float(score),
            "source": str(promoted_summary_path),
            "summary_path": str(promoted_summary_path),
            "material_interaction": str(
                (summary.get("model_architecture") or {}).get("material_interaction", "masked_gnn")
            ),
            "weight": 0.80 if selected_hgt is not None else 1.0,
        }
    ]
    if selected_hgt is not None:
        final_models.append(
            {
                "name": "hetero_hgt",
                "family": "hetero_hgt",
                "checkpoint_path": str(selected_hgt.resolve()),
                "score": float(max(0.0, score - 0.02)),
                "source": str(hgt_summary_path if hgt_summary_path.exists() else final_manifest_path),
                "summary_path": str(hgt_summary_path if hgt_summary_path.exists() else ""),
                "material_interaction": "hgt",
                "weight": 0.20,
            }
        )

    final_manifest = {
        "version": "v2",
        "created_at_epoch_sec": int(time.time()),
        "source": "tools/checkpoint_promotion_gate.py",
        "models": final_models,
        "primary_model": "masked_runtime_promoted",
        "note": (
            "Promotion-gated runtime manifest. Masked model is primary for prediction; "
            "hetero_hgt is optional discovery reranker."
        ),
    }
    _save_json(final_manifest_path, final_manifest)

    in_place_path = summary_path if args.promote_in_place else None
    if in_place_path is not None:
        _save_json(in_place_path, promoted_summary)

    keep = {selected_main.resolve()}
    if selected_hgt is not None:
        keep.add(selected_hgt.resolve())
    removed = _prune_checkpoints(models_dir, keep) if args.prune_checkpoints else []

    report = {
        "timestamp_epoch_sec": int(time.time()),
        "gate_passed": bool(gate_passed),
        "gate_failure_reasons": list(gate_reasons),
        "selection_score": float(score),
        "selected_main_checkpoint": str(selected_main.resolve()),
        "selected_hgt_checkpoint": str(selected_hgt.resolve()) if selected_hgt is not None else None,
        "promoted_summary_path": str(promoted_summary_path),
        "summary_in_place_updated": bool(args.promote_in_place),
        "three_model_manifest_path": str(three_manifest_path),
        "final_manifest_path": str(final_manifest_path),
        "pruned": {
            "enabled": bool(args.prune_checkpoints),
            "removed_count": int(len(removed)),
            "removed_paths": removed,
            "kept_paths": sorted(str(p) for p in keep),
        },
    }
    _save_json(report_out_path, report)

    print(f"[promotion] gate_passed={gate_passed} score={score:.6f}")
    if gate_reasons:
        print("[promotion] gate_reasons=" + ",".join(gate_reasons))
    print(f"[promotion] main={selected_main.resolve()}")
    if selected_hgt is not None:
        print(f"[promotion] hgt={selected_hgt.resolve()}")
    print(f"[promotion] promoted_summary={promoted_summary_path}")
    print(f"[promotion] report={report_out_path}")
    if removed:
        print(f"[promotion] pruned={len(removed)} checkpoints")


if __name__ == "__main__":
    main()
