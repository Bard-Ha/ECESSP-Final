#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
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


def _selection_score(summary: Dict[str, Any]) -> float:
    iid = summary.get("evaluation_metrics", {}).get("iid", {}).get("raw_space", {})
    ood = summary.get("evaluation_metrics", {}).get("ood", {}).get("raw_space", {})
    phys = summary.get("physics_violation_statistics", {})
    iid_phys = phys.get("iid", {}) if isinstance(phys, dict) else {}
    ood_phys = phys.get("ood", {}) if isinstance(phys, dict) else {}

    iid_overall_r2 = _safe_float((iid.get("overall_micro") or {}).get("r2"), -1.0)
    ood_overall_r2 = _safe_float((ood.get("overall_micro") or {}).get("r2"), -1.0)
    iid_cap_r2 = _safe_float((iid.get("capacity_grav") or {}).get("r2"), -1.0)
    ood_cap_r2 = _safe_float((ood.get("capacity_grav") or {}).get("r2"), -1.0)
    iid_volt_r2 = _safe_float((iid.get("average_voltage") or {}).get("r2"), -1.0)
    ood_volt_r2 = _safe_float((ood.get("average_voltage") or {}).get("r2"), -1.0)
    iid_cvol_r2 = _safe_float((iid.get("capacity_vol") or {}).get("r2"), -1.0)
    ood_cvol_r2 = _safe_float((ood.get("capacity_vol") or {}).get("r2"), -1.0)

    violation = 0.0
    violation += _safe_float(iid_phys.get("capacity_violation_rate"), 0.0)
    violation += _safe_float(iid_phys.get("voltage_above_4p4_rate"), 0.0)
    violation += _safe_float(ood_phys.get("capacity_violation_rate"), 0.0)
    violation += _safe_float(ood_phys.get("voltage_above_4p4_rate"), 0.0)

    return float(
        0.28 * iid_overall_r2
        + 0.28 * ood_overall_r2
        + 0.18 * iid_cap_r2
        + 0.18 * ood_cap_r2
        + 0.04 * iid_volt_r2
        + 0.02 * ood_volt_r2
        + 0.01 * iid_cvol_r2
        + 0.01 * ood_cvol_r2
        - 3.0 * violation
    )


def _softmax_weights(scores: List[float], temperature: float = 0.5) -> List[float]:
    if not scores:
        return []
    t = max(1e-6, float(temperature))
    smax = max(scores)
    exps = [math.exp((s - smax) / t) for s in scores]
    z = sum(exps)
    if z <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [float(v / z) for v in exps]


def run_trial(
    root: Path,
    cfg_path: Path,
    *,
    ensemble_size: int,
    feature_dropout: float,
    feature_noise_std: float,
    allow_dev_shortcuts: bool = False,
    use_amp: bool = False,
) -> Dict[str, Any]:
    cmd = [
        "python",
        "train_physics_first.py",
        "--config",
        str(cfg_path.relative_to(root)),
        "--ensemble-size",
        str(int(max(1, ensemble_size))),
        "--dataset-mode",
        "auto",
        "--feature-dropout",
        str(feature_dropout),
        "--feature-noise-std",
        str(feature_noise_std),
    ]
    if allow_dev_shortcuts:
        cmd.append("--allow-dev-shortcuts")
    if use_amp:
        cmd.append("--amp")

    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    dt = time.time() - t0

    out: Dict[str, Any] = {
        "returncode": int(p.returncode),
        "runtime_sec": float(dt),
        "stdout_tail": "\n".join(p.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(p.stderr.splitlines()[-20:]),
    }
    if p.returncode != 0:
        return out

    summary_path = root / "reports" / "training_summary.json"
    if not summary_path.exists():
        out["returncode"] = 1
        out["stderr_tail"] = (out.get("stderr_tail", "") + "\ntraining_summary.json missing after trial").strip()
        return out

    summary = load_json(summary_path)
    out["summary"] = summary
    out["selection_score"] = _selection_score(summary)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a 3-model ensemble: transformer + gatv2 + mpnn")
    ap.add_argument("--base-config", default="reports/training_config.json")
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--allow-dev-shortcuts", action="store_true")
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--out-manifest", default="reports/three_model_ensemble_manifest.json")
    ap.add_argument("--out-results", default="reports/three_model_ensemble_results.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base = load_json((root / args.base_config).resolve())
    trials_dir = root / "reports" / "three_model_ensemble_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Dict[str, Any]] = [
        {
            "name": "tr_ctx1_l2_h128",
            "material_interaction": "transformer",
            "enable_graph_context": True,
            "num_gnn_layers": 2,
            "hidden_dim": 128,
            "dropout": 0.15,
            "learning_rate": 8e-4,
            "weight_decay": 5e-5,
            "feature_dropout": 0.10,
            "feature_noise_std": 0.02,
        },
        {
            "name": "gatv2_ctx1_l3_h128",
            "material_interaction": "gatv2",
            "enable_graph_context": True,
            "num_gnn_layers": 3,
            "hidden_dim": 128,
            "dropout": 0.20,
            "learning_rate": 7e-4,
            "weight_decay": 1e-4,
            "feature_dropout": 0.08,
            "feature_noise_std": 0.015,
        },
        {
            "name": "mpnn_ctx1_l3_h128",
            "material_interaction": "mpnn",
            "enable_graph_context": True,
            "num_gnn_layers": 3,
            "hidden_dim": 128,
            "dropout": 0.20,
            "learning_rate": 7e-4,
            "weight_decay": 1e-4,
            "feature_dropout": 0.08,
            "feature_noise_std": 0.015,
        },
    ]

    all_results: List[Dict[str, Any]] = []
    for cand in candidates:
        cfg = copy.deepcopy(base)
        cfg["epochs"] = int(args.epochs)
        cfg["early_stopping_patience"] = int(args.patience)
        cfg["lr_patience"] = min(int(cfg.get("lr_patience", 10)), int(args.patience))
        cfg["material_interaction"] = cand["material_interaction"]
        cfg["enable_graph_context"] = bool(cand["enable_graph_context"])
        cfg["num_gnn_layers"] = int(cand["num_gnn_layers"])
        cfg["hidden_dim"] = int(cand["hidden_dim"])
        cfg["dropout"] = float(cand["dropout"])
        cfg["learning_rate"] = float(cand["learning_rate"])
        cfg["weight_decay"] = float(cand["weight_decay"])

        cfg_path = trials_dir / f"{cand['name']}.json"
        save_json(cfg_path, cfg)

        print(f"[trial] {cand['name']}")
        trial_res = run_trial(
            root=root,
            cfg_path=cfg_path,
            ensemble_size=int(args.ensemble_size),
            feature_dropout=float(cand["feature_dropout"]),
            feature_noise_std=float(cand["feature_noise_std"]),
            allow_dev_shortcuts=bool(args.allow_dev_shortcuts),
            use_amp=bool(args.use_amp),
        )
        trial_res["candidate"] = cand

        if trial_res.get("returncode", 1) == 0:
            summary = trial_res.get("summary", {})
            summary_copy_path = trials_dir / f"{cand['name']}_training_summary.json"
            save_json(summary_copy_path, summary)
            trial_res["summary_path"] = str(summary_copy_path.resolve())

        all_results.append(trial_res)
        print(f"[trial] rc={trial_res.get('returncode')} score={trial_res.get('selection_score')}")

    successful = [r for r in all_results if int(r.get("returncode", 1)) == 0]
    ranked = sorted(successful, key=lambda r: float(r.get("selection_score", -1e9)), reverse=True)

    if ranked:
        weights = _softmax_weights([float(r["selection_score"]) for r in ranked], temperature=0.5)
    else:
        weights = []

    manifest_models: List[Dict[str, Any]] = []
    for i, r in enumerate(ranked):
        summary = r.get("summary", {})
        ckpt = str(summary.get("best_checkpoint", "")).strip()
        if not ckpt:
            continue
        manifest_models.append(
            {
                "name": str((r.get("candidate") or {}).get("name", f"model_{i}")),
                "checkpoint_path": ckpt,
                "weight": float(weights[i]) if i < len(weights) else 0.0,
                "material_interaction": str((r.get("candidate") or {}).get("material_interaction", "")),
                "selection_score": float(r.get("selection_score", 0.0)),
                "summary_path": str(r.get("summary_path", "")),
                "summary_timestamp": str(summary.get("timestamp", "")),
            }
        )

    # Preserve at most 3 models in manifest.
    manifest_models = manifest_models[:3]
    wsum = sum(float(m.get("weight", 0.0)) for m in manifest_models)
    if wsum > 0:
        for m in manifest_models:
            m["weight"] = float(m["weight"] / wsum)

    manifest = {
        "version": "v1",
        "created_at_epoch_sec": int(time.time()),
        "source": "tools/train_three_model_ensemble.py",
        "base_config": str((root / args.base_config).resolve()),
        "models": manifest_models,
        "primary_model": manifest_models[0]["name"] if manifest_models else None,
    }

    results_payload = {
        "search_epochs": int(args.epochs),
        "ensemble_size": int(args.ensemble_size),
        "candidates_tested": len(candidates),
        "successful_trials": len(successful),
        "ranked": ranked,
        "all_results": all_results,
        "manifest_preview": manifest,
    }

    out_results = (root / args.out_results).resolve()
    out_manifest = (root / args.out_manifest).resolve()
    save_json(out_results, results_payload)
    save_json(out_manifest, manifest)

    print(f"[done] results: {out_results}")
    print(f"[done] manifest: {out_manifest}")
    if manifest_models:
        print(
            "[ensemble] "
            + ", ".join(
                f"{m['name']}@{m['weight']:.3f}"
                for m in manifest_models
            )
        )


if __name__ == "__main__":
    main()
