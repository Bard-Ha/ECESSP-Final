#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _score_from_summary(summary: Dict[str, Any]) -> float:
    iid = summary.get("evaluation_metrics", {}).get("iid", {}).get("raw_space", {})
    ood = summary.get("evaluation_metrics", {}).get("ood", {}).get("raw_space", {})
    phys = summary.get("physics_violation_statistics", {})
    iid_phys = phys.get("iid", {}) if isinstance(phys, dict) else {}
    ood_phys = phys.get("ood", {}) if isinstance(phys, dict) else {}

    iid_overall = _safe_float((iid.get("overall_micro") or {}).get("r2"), -1.0)
    ood_overall = _safe_float((ood.get("overall_micro") or {}).get("r2"), -1.0)
    iid_cap = _safe_float((iid.get("capacity_grav") or {}).get("r2"), -1.0)
    ood_cap = _safe_float((ood.get("capacity_grav") or {}).get("r2"), -1.0)
    iid_vol = _safe_float((iid.get("average_voltage") or {}).get("r2"), -1.0)
    ood_vol = _safe_float((ood.get("average_voltage") or {}).get("r2"), -1.0)
    violation = (
        _safe_float(iid_phys.get("capacity_violation_rate"), 0.0)
        + _safe_float(iid_phys.get("voltage_above_4p4_rate"), 0.0)
        + _safe_float(ood_phys.get("capacity_violation_rate"), 0.0)
        + _safe_float(ood_phys.get("voltage_above_4p4_rate"), 0.0)
    )

    return float(
        0.32 * iid_overall
        + 0.32 * ood_overall
        + 0.16 * iid_cap
        + 0.16 * ood_cap
        + 0.02 * iid_vol
        + 0.02 * ood_vol
        - 3.0 * violation
    )


def _softmax(vals: List[float], temp: float = 0.5) -> List[float]:
    if not vals:
        return []
    t = max(1e-6, float(temp))
    vmax = max(vals)
    ex = [math.exp((v - vmax) / t) for v in vals]
    z = sum(ex)
    if z <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [float(v / z) for v in ex]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse masked ensemble + HGT into one weighted family manifest")
    ap.add_argument("--masked-manifest", default="reports/three_model_ensemble_manifest.json")
    ap.add_argument("--hgt-summary", default="reports/training_summary_hgt.json")
    ap.add_argument("--out-manifest", default="reports/final_family_ensemble_manifest.json")
    ap.add_argument("--out-report", default="reports/final_family_ensemble_report.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    masked_manifest_path = (root / args.masked_manifest).resolve()
    hgt_summary_path = (root / args.hgt_summary).resolve()

    if not masked_manifest_path.exists():
        raise FileNotFoundError(f"masked manifest missing: {masked_manifest_path}")
    masked_manifest = _load_json(masked_manifest_path)
    masked_models = list(masked_manifest.get("models", []))

    entries: List[Dict[str, Any]] = []
    for m in masked_models:
        score = _safe_float(m.get("selection_score"), 0.0)
        entries.append(
            {
                "name": str(m.get("name")),
                "family": "masked_gnn",
                "checkpoint_path": str(m.get("checkpoint_path")),
                "score": float(score),
                "source": str(masked_manifest_path),
                "summary_path": str(m.get("summary_path", "")),
                "material_interaction": str(m.get("material_interaction", "")),
            }
        )

    if hgt_summary_path.exists():
        hgt_summary = _load_json(hgt_summary_path)
        entries.append(
            {
                "name": "hetero_hgt",
                "family": "hetero_hgt",
                "checkpoint_path": str(hgt_summary.get("best_checkpoint", "")),
                "score": _score_from_summary(hgt_summary),
                "source": str(hgt_summary_path),
                "summary_path": str(hgt_summary_path),
                "material_interaction": "hgt",
            }
        )

    ranked = sorted(entries, key=lambda x: float(x.get("score", -1e9)), reverse=True)
    weights = _softmax([float(x["score"]) for x in ranked], temp=0.5)
    for i, item in enumerate(ranked):
        item["weight"] = float(weights[i]) if i < len(weights) else 0.0

    out_manifest = {
        "version": "v1",
        "created_at_epoch_sec": int(time.time()),
        "source": "tools/fuse_family_ensemble.py",
        "models": ranked,
        "primary_model": ranked[0]["name"] if ranked else None,
        "note": (
            "Runtime prediction executes masked_gnn ensemble directly; "
            "hetero_hgt is loaded as a discovery reranker (plausibility/objective-support prior)."
        ),
    }

    out_report = {
        "ranked": ranked,
        "masked_manifest": str(masked_manifest_path),
        "hgt_summary": str(hgt_summary_path) if hgt_summary_path.exists() else None,
    }

    out_manifest_path = (root / args.out_manifest).resolve()
    out_report_path = (root / args.out_report).resolve()
    _save_json(out_manifest_path, out_manifest)
    _save_json(out_report_path, out_report)

    print(f"[done] manifest: {out_manifest_path}")
    print(f"[done] report: {out_report_path}")
    if ranked:
        print("[weights] " + ", ".join(f"{x['name']}={x['weight']:.3f}" for x in ranked))


if __name__ == "__main__":
    main()
