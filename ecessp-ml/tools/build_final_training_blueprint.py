#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _ion_counts(parsed_path: Path) -> Dict[str, int]:
    try:
        csv.field_size_limit(10**9)
    except Exception:
        pass
    ctr: Counter[str] = Counter()
    with parsed_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ion = str(row.get("working_ion") or "").strip()
            if ion:
                ctr[ion] += 1
    return {k: int(v) for k, v in ctr.most_common()}


def _resolve_runtime_checkpoint(root: Path) -> str:
    # Reuse backend resolver for exact runtime path.
    import sys

    sys.path.insert(0, str(root))
    from backend.config import MODEL_CONFIG  # type: ignore

    return str(MODEL_CONFIG.checkpoint_path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parsed = root / "data" / "processed" / "batteries_parsed_curated.csv"
    ml = root / "data" / "processed" / "batteries_ml_curated.csv"
    final_manifest_path = root / "reports" / "final_family_ensemble_manifest.json"
    three_manifest_path = root / "reports" / "three_model_ensemble_manifest.json"
    train_summary_path = root / "reports" / "training_summary.json"
    out_path = root / "reports" / "final_training_blueprint.json"

    final_manifest = _load_json(final_manifest_path) if final_manifest_path.exists() else {}
    three_manifest = _load_json(three_manifest_path) if three_manifest_path.exists() else {}
    train_summary = _load_json(train_summary_path) if train_summary_path.exists() else {}

    ion_counts = _ion_counts(parsed) if parsed.exists() else {}

    output = {
        "report_type": "final_training_blueprint",
        "as_of_date": "2026-02-24",
        "data_sources": {
            "parsed_curated": str(parsed),
            "ml_curated": str(ml),
            "row_count_parsed_curated": int(sum(ion_counts.values())) if ion_counts else None,
            "working_ion_distribution": ion_counts,
        },
        "final_target_policy": {
            "regression_targets_model_output_7": [
                "average_voltage",
                "capacity_grav",
                "capacity_vol",
                "energy_grav",
                "energy_vol",
                "stability_charge",
                "stability_discharge",
            ],
            "objective_and_constraint_targets": [
                "max_delta_volume"
            ],
            "auxiliary_supervision_heads": {
                "role_head": [
                    "cathode",
                    "anode",
                    "electrolyte_candidate"
                ],
                "compatibility_head": [
                    "voltage_window_overlap_score",
                    "chemical_stability_score",
                    "mechanical_strain_risk"
                ],
                "uncertainty_head": "heteroscedastic_log_variance_for_7_targets"
            },
            "rationale": "Keep 7-output checkpoint compatibility while supervising max_delta_volume through compatibility/constraints stack."
        },
        "feature_stratification": {
            "input_features": {
                "system_node_features_7": [
                    "average_voltage_norm",
                    "capacity_grav_norm",
                    "capacity_vol_norm",
                    "energy_grav_norm",
                    "energy_vol_norm",
                    "stability_charge_norm",
                    "stability_discharge_norm"
                ],
                "graph_context_features_7": "Neighbor-aggregated system features in graph space",
                "material_node_embeddings_64d": "atomic_embeddings_curated + graph material embeddings",
                "node_masks": "component presence mask",
                "formula_derived_features": [
                    "theoretical_capacity_proxy",
                    "max_electron_transfer_proxy",
                    "working_ion_code",
                    "composition_vector"
                ]
            },
            "output_features": {
                "primary_regression": "7 canonical properties",
                "auxiliary_classification": "role probabilities",
                "auxiliary_regression": "compatibility scores",
                "uncertainty": "per-target log variance"
            }
        },
        "runtime_checkpoint_status": {
            "training_summary_best_checkpoint": str(train_summary.get("best_checkpoint", "")),
            "three_model_ensemble_primary": str(three_manifest.get("primary_model", "")),
            "final_family_primary": str(final_manifest.get("primary_model", "")),
            "runtime_resolved_checkpoint": _resolve_runtime_checkpoint(root),
            "final_family_note": str(final_manifest.get("note", "")),
        },
        "recommended_fast_final_retrain": {
            "masked_trio_command": "python tools/train_three_model_ensemble.py --base-config reports/training_config_final_fast.json --epochs 72 --patience 12 --ensemble-size 1 --allow-dev-shortcuts",
            "hgt_command": "python train_hetero_hgt.py --epochs 80 --patience 16",
            "fusion_command": "python tools/fuse_family_ensemble.py",
            "target_artifact": "reports/final_family_ensemble_manifest.json"
        }
    }

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[ok] blueprint -> {out_path}")


if __name__ == "__main__":
    main()
