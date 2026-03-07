#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
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
    return out


def run_trial(root: Path, cfg_path: Path, *, feature_dropout: float, feature_noise: float) -> Dict[str, Any]:
    cmd = [
        "python",
        "train_physics_first.py",
        "--config",
        str(cfg_path.relative_to(root)),
        "--allow-dev-shortcuts",
        "--ensemble-size",
        "1",
        "--dataset-mode",
        "auto",
        "--feature-dropout",
        str(feature_dropout),
        "--feature-noise-std",
        str(feature_noise),
    ]
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

    summary = load_json(root / "reports" / "training_summary.json")
    iid_raw = summary.get("evaluation_metrics", {}).get("iid", {}).get("raw_space", {})
    ood_raw = summary.get("evaluation_metrics", {}).get("ood", {}).get("raw_space", {})
    phys = summary.get("physics_violation_statistics", {})
    iid_phys = phys.get("iid", {}) if isinstance(phys, dict) else {}
    ood_phys = phys.get("ood", {}) if isinstance(phys, dict) else {}

    iid_cap_r2 = _safe_float((iid_raw.get("capacity_grav") or {}).get("r2"), -1.0)
    ood_cap_r2 = _safe_float((ood_raw.get("capacity_grav") or {}).get("r2"), -1.0)
    iid_overall_r2 = _safe_float((iid_raw.get("overall_micro") or {}).get("r2"), -1.0)
    ood_overall_r2 = _safe_float((ood_raw.get("overall_micro") or {}).get("r2"), -1.0)
    iid_cvol_r2 = _safe_float((iid_raw.get("capacity_vol") or {}).get("r2"), -1.0)
    ood_cvol_r2 = _safe_float((ood_raw.get("capacity_vol") or {}).get("r2"), -1.0)

    violation = 0.0
    violation += _safe_float(iid_phys.get("capacity_violation_rate"), 0.0)
    violation += _safe_float(iid_phys.get("voltage_above_4p4_rate"), 0.0)
    violation += _safe_float(ood_phys.get("capacity_violation_rate"), 0.0)
    violation += _safe_float(ood_phys.get("voltage_above_4p4_rate"), 0.0)

    # Prioritize capacity_grav generalization while preserving global quality and hard-physics validity.
    selection_score = (
        0.40 * iid_cap_r2
        + 0.40 * ood_cap_r2
        + 0.08 * iid_overall_r2
        + 0.08 * ood_overall_r2
        + 0.02 * iid_cvol_r2
        + 0.02 * ood_cvol_r2
        - 3.0 * violation
    )

    out.update(
        {
            "summary_timestamp": summary.get("timestamp"),
            "iid_raw_overall": iid_raw.get("overall_micro", {}),
            "ood_raw_overall": ood_raw.get("overall_micro", {}),
            "iid_raw_capacity_grav": iid_raw.get("capacity_grav", {}),
            "ood_raw_capacity_grav": ood_raw.get("capacity_grav", {}),
            "iid_raw_capacity_vol": iid_raw.get("capacity_vol", {}),
            "ood_raw_capacity_vol": ood_raw.get("capacity_vol", {}),
            "iid_physics": iid_phys,
            "ood_physics": ood_phys,
            "selection_score": float(selection_score),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-12 capacity_grav rescue sweep")
    ap.add_argument("--base-config", default="reports/training_config.json")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--final-epochs", type=int, default=180)
    ap.add_argument("--final-patience", type=int, default=30)
    ap.add_argument("--final-out-config", default="reports/training_config_capacity_rescue_best.json")
    ap.add_argument("--out-json", default="reports/step12_capacity_rescue_results.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_cfg = load_json((root / args.base_config).resolve())
    trials_dir = root / "reports" / "step12_capacity_rescue_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    base_loss = copy.deepcopy(base_cfg.get("loss", {}))
    if not isinstance(base_loss, dict):
        raise RuntimeError("base config missing loss dict")

    candidates: List[Dict[str, Any]] = [
        {
            "name": "tr_baseline",
            "material_interaction": "transformer",
            "num_gnn_layers": 2,
            "hidden_dim": 128,
            "dropout": 0.15,
            "enable_graph_context": True,
            "lr": 8e-4,
            "weight_decay": 5e-5,
            "feature_dropout": 0.10,
            "feature_noise": 0.02,
            "loss_override": {},
        },
        {
            "name": "tr_cap_light",
            "material_interaction": "transformer",
            "num_gnn_layers": 2,
            "hidden_dim": 128,
            "dropout": 0.15,
            "enable_graph_context": True,
            "lr": 8e-4,
            "weight_decay": 5e-5,
            "feature_dropout": 0.10,
            "feature_noise": 0.02,
            "loss_override": {
                "regression_target_weights": {
                    "average_voltage": 1.0,
                    "capacity_grav": 1.2,
                    "capacity_vol": 1.0,
                    "energy_grav": 0.9,
                    "energy_vol": 0.9,
                    "stability_charge": 1.0,
                    "stability_discharge": 1.0,
                },
                "capacity_grav_huber_weight": 0.08,
                "capacity_grav_huber_beta": 0.25,
            },
        },
        {
            "name": "gatv2_cap_medium",
            "material_interaction": "gatv2",
            "num_gnn_layers": 3,
            "hidden_dim": 128,
            "dropout": 0.20,
            "enable_graph_context": True,
            "lr": 7e-4,
            "weight_decay": 5e-5,
            "feature_dropout": 0.08,
            "feature_noise": 0.015,
            "loss_override": {
                "regression_target_weights": {
                    "average_voltage": 1.0,
                    "capacity_grav": 1.35,
                    "capacity_vol": 1.0,
                    "energy_grav": 0.85,
                    "energy_vol": 0.85,
                    "stability_charge": 1.0,
                    "stability_discharge": 1.0,
                },
                "capacity_grav_huber_weight": 0.12,
                "capacity_grav_huber_beta": 0.25,
            },
        },
        {
            "name": "mpnn_cap_medium",
            "material_interaction": "mpnn",
            "num_gnn_layers": 3,
            "hidden_dim": 128,
            "dropout": 0.20,
            "enable_graph_context": True,
            "lr": 7e-4,
            "weight_decay": 5e-5,
            "feature_dropout": 0.08,
            "feature_noise": 0.015,
            "loss_override": {
                "regression_target_weights": {
                    "average_voltage": 1.0,
                    "capacity_grav": 1.35,
                    "capacity_vol": 1.0,
                    "energy_grav": 0.85,
                    "energy_vol": 0.85,
                    "stability_charge": 1.0,
                    "stability_discharge": 1.0,
                },
                "capacity_grav_huber_weight": 0.12,
                "capacity_grav_huber_beta": 0.25,
            },
        },
        {
            "name": "gatv2_cap_light_lowaug",
            "material_interaction": "gatv2",
            "num_gnn_layers": 3,
            "hidden_dim": 128,
            "dropout": 0.20,
            "enable_graph_context": True,
            "lr": 7e-4,
            "weight_decay": 5e-5,
            "feature_dropout": 0.03,
            "feature_noise": 0.005,
            "loss_override": {
                "regression_target_weights": {
                    "average_voltage": 1.0,
                    "capacity_grav": 1.2,
                    "capacity_vol": 1.0,
                    "energy_grav": 0.9,
                    "energy_vol": 0.9,
                    "stability_charge": 1.0,
                    "stability_discharge": 1.0,
                },
                "capacity_grav_huber_weight": 0.08,
                "capacity_grav_huber_beta": 0.25,
            },
        },
    ]

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        cfg = copy.deepcopy(base_cfg)
        cfg["epochs"] = int(args.epochs)
        cfg["early_stopping_patience"] = int(args.patience)
        cfg["lr_patience"] = min(int(cfg.get("lr_patience", 10)), int(args.patience))
        cfg["material_interaction"] = str(cand.get("material_interaction", cfg.get("material_interaction", "transformer")))
        cfg["num_gnn_layers"] = int(cand.get("num_gnn_layers", cfg.get("num_gnn_layers", 2)))
        cfg["hidden_dim"] = int(cand.get("hidden_dim", cfg.get("hidden_dim", 128)))
        cfg["dropout"] = float(cand.get("dropout", cfg.get("dropout", 0.15)))
        cfg["enable_graph_context"] = bool(cand.get("enable_graph_context", cfg.get("enable_graph_context", True)))
        cfg["learning_rate"] = float(cand["lr"])
        cfg["weight_decay"] = float(cand["weight_decay"])
        cfg["loss"] = {**base_loss, **cand.get("loss_override", {})}

        cfg_path = trials_dir / f"{cand['name']}.json"
        save_json(cfg_path, cfg)

        print(f"[trial] {cand['name']}")
        r = run_trial(
            root=root,
            cfg_path=cfg_path,
            feature_dropout=float(cand["feature_dropout"]),
            feature_noise=float(cand["feature_noise"]),
        )
        r["candidate"] = cand
        results.append(r)
        print(f"[trial] rc={r['returncode']} score={r.get('selection_score')}")

    ok = [r for r in results if int(r.get("returncode", 1)) == 0]
    ranked = sorted(ok, key=lambda r: float(r.get("selection_score", -1e9)), reverse=True)
    payload = {
        "search_epochs": int(args.epochs),
        "search_patience": int(args.patience),
        "candidates_tested": len(candidates),
        "successful_trials": len(ok),
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
        "all_results": results,
    }
    if ranked:
        best = ranked[0]["candidate"]
        final_cfg = copy.deepcopy(base_cfg)
        final_cfg["epochs"] = int(args.final_epochs)
        final_cfg["early_stopping_patience"] = int(args.final_patience)
        final_cfg["lr_patience"] = min(int(final_cfg.get("lr_patience", 10)), int(args.final_patience))
        final_cfg["material_interaction"] = str(best.get("material_interaction", final_cfg.get("material_interaction", "transformer")))
        final_cfg["num_gnn_layers"] = int(best.get("num_gnn_layers", final_cfg.get("num_gnn_layers", 2)))
        final_cfg["hidden_dim"] = int(best.get("hidden_dim", final_cfg.get("hidden_dim", 128)))
        final_cfg["dropout"] = float(best.get("dropout", final_cfg.get("dropout", 0.15)))
        final_cfg["enable_graph_context"] = bool(best.get("enable_graph_context", final_cfg.get("enable_graph_context", True)))
        final_cfg["learning_rate"] = float(best.get("lr", final_cfg.get("learning_rate", 8e-4)))
        final_cfg["weight_decay"] = float(best.get("weight_decay", final_cfg.get("weight_decay", 5e-5)))
        final_cfg["loss"] = {**base_loss, **best.get("loss_override", {})}
        final_cfg_path = (root / args.final_out_config).resolve()
        save_json(final_cfg_path, final_cfg)
        payload["recommended_final_config"] = str(final_cfg_path)
        payload["recommended_final_candidate"] = best

    out = (root / args.out_json).resolve()
    save_json(out, payload)
    print(f"[done] {out}")
    if ranked:
        print(f"[best] {ranked[0]['candidate']['name']} score={ranked[0]['selection_score']:.6f}")
        print(f"[final-config] {payload.get('recommended_final_config')}")


if __name__ == "__main__":
    main()
