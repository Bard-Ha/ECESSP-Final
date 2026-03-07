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


def run_trial(root: Path, cfg_path: Path) -> Dict[str, Any]:
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
    ]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    dt = time.time() - t0

    result: Dict[str, Any] = {
        "returncode": int(p.returncode),
        "runtime_sec": float(dt),
        "stdout_tail": "\n".join(p.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(p.stderr.splitlines()[-20:]),
    }
    if p.returncode != 0:
        return result

    s = load_json(root / "reports" / "training_summary.json")
    iid = s.get("evaluation_metrics", {}).get("iid", {}).get("raw_space", {}).get("overall_micro", {})
    ood = s.get("evaluation_metrics", {}).get("ood", {}).get("raw_space", {}).get("overall_micro", {})
    pstat = s.get("physics_violation_statistics", {})
    iid_p = pstat.get("iid", {}) if isinstance(pstat, dict) else {}
    ood_p = pstat.get("ood", {}) if isinstance(pstat, dict) else {}
    unc = s.get("uncertainty_statistics", {})

    iid_r2 = float(iid.get("r2", -1.0))
    ood_r2 = float(ood.get("r2", -1.0))
    iid_rmse = float(iid.get("rmse", 1e9))
    ood_rmse = float(ood.get("rmse", 1e9))
    unc_corr = float(unc.get("corr_uncertainty_abs_error", 0.0) or 0.0)
    violation = float(iid_p.get("capacity_violation_rate", 0.0) or 0.0)
    violation += float(iid_p.get("voltage_above_4p4_rate", 0.0) or 0.0)
    violation += float(ood_p.get("capacity_violation_rate", 0.0) or 0.0)
    violation += float(ood_p.get("voltage_above_4p4_rate", 0.0) or 0.0)

    # Hyperparameter score focused on robust generalization + calibration + physical validity.
    score = (
        0.45 * iid_r2
        + 0.45 * ood_r2
        + 0.10 * max(-1.0, min(1.0, unc_corr))
        - 0.00004 * iid_rmse
        - 0.00004 * ood_rmse
        - 3.0 * violation
    )

    result.update(
        {
            "summary_timestamp": s.get("timestamp"),
            "iid_raw_overall": iid,
            "ood_raw_overall": ood,
            "iid_physics": iid_p,
            "ood_physics": ood_p,
            "uncertainty_corr_abs_err": unc_corr,
            "selection_score": float(score),
        }
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-6 micro-sweep around best architecture+loss.")
    ap.add_argument("--base-config", default="reports/training_config.json")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out-json", default="reports/hyperparam_micro_sweep_results.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base = load_json((root / args.base_config).resolve())
    trials_dir = root / "reports" / "hyperparam_micro_sweep_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Keep architecture and loss fixed; vary only hyperparameters requested for Step 6.
    candidates: List[Dict[str, Any]] = [
        {"name": "hp_base", "learning_rate": 8e-4, "weight_decay": 1e-4, "dropout": 0.15, "batch_size": 32},
        {"name": "hp_lr6e4_wd1e4_do015_bs32", "learning_rate": 6e-4, "weight_decay": 1e-4, "dropout": 0.15, "batch_size": 32},
        {"name": "hp_lr1e3_wd1e4_do015_bs32", "learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.15, "batch_size": 32},
        {"name": "hp_lr8e4_wd5e5_do015_bs32", "learning_rate": 8e-4, "weight_decay": 5e-5, "dropout": 0.15, "batch_size": 32},
        {"name": "hp_lr8e4_wd2e4_do015_bs32", "learning_rate": 8e-4, "weight_decay": 2e-4, "dropout": 0.15, "batch_size": 32},
        {"name": "hp_lr8e4_wd1e4_do010_bs32", "learning_rate": 8e-4, "weight_decay": 1e-4, "dropout": 0.10, "batch_size": 32},
        {"name": "hp_lr8e4_wd1e4_do020_bs32", "learning_rate": 8e-4, "weight_decay": 1e-4, "dropout": 0.20, "batch_size": 32},
        {"name": "hp_lr8e4_wd1e4_do015_bs48", "learning_rate": 8e-4, "weight_decay": 1e-4, "dropout": 0.15, "batch_size": 48},
    ]

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        cfg = copy.deepcopy(base)
        cfg["epochs"] = int(args.epochs)
        cfg["early_stopping_patience"] = int(args.patience)
        cfg["lr_patience"] = min(int(cfg.get("lr_patience", 10)), int(args.patience))
        cfg["learning_rate"] = float(cand["learning_rate"])
        cfg["weight_decay"] = float(cand["weight_decay"])
        cfg["dropout"] = float(cand["dropout"])
        cfg["batch_size"] = int(cand["batch_size"])

        cfg_path = trials_dir / f"{cand['name']}.json"
        save_json(cfg_path, cfg)
        print(f"[trial] {cand['name']}")
        r = run_trial(root, cfg_path)
        r["candidate"] = cand
        results.append(r)
        print(f"[trial] rc={r['returncode']} score={r.get('selection_score')}")

    ok = [r for r in results if int(r.get("returncode", 1)) == 0]
    ranked = sorted(ok, key=lambda r: float(r.get("selection_score", -1e9)), reverse=True)
    payload = {
        "search_epochs": int(args.epochs),
        "candidates_tested": len(candidates),
        "successful_trials": len(ok),
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
        "all_results": results,
    }
    out = (root / args.out_json).resolve()
    save_json(out, payload)
    print(f"[done] {out}")
    if ranked:
        print(f"[best] {ranked[0]['candidate']['name']} score={ranked[0]['selection_score']:.6f}")


if __name__ == "__main__":
    main()
