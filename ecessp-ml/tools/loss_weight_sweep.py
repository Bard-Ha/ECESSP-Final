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
    phys = s.get("physics_violation_statistics", {})
    iid_phys = phys.get("iid", {}) if isinstance(phys, dict) else {}
    ood_phys = phys.get("ood", {}) if isinstance(phys, dict) else {}
    unc = s.get("uncertainty_statistics", {})
    unc_corr = float(unc.get("corr_uncertainty_abs_error", 0.0) or 0.0)

    iid_r2 = float(iid.get("r2", -1.0))
    ood_r2 = float(ood.get("r2", -1.0))
    iid_rmse = float(iid.get("rmse", 1e9))
    ood_rmse = float(ood.get("rmse", 1e9))
    violation = float(iid_phys.get("capacity_violation_rate", 0.0) or 0.0)
    violation += float(iid_phys.get("voltage_above_4p4_rate", 0.0) or 0.0)
    violation += float(ood_phys.get("capacity_violation_rate", 0.0) or 0.0)
    violation += float(ood_phys.get("voltage_above_4p4_rate", 0.0) or 0.0)
    # Prefer strong IID/OOD R2, low RMSE, no violations, and useful uncertainty correlation.
    score = (
        0.45 * iid_r2
        + 0.45 * ood_r2
        + 0.10 * max(-1.0, min(1.0, unc_corr))
        - 0.00005 * iid_rmse
        - 0.00005 * ood_rmse
        - 3.0 * violation
    )

    result.update(
        {
            "summary_timestamp": s.get("timestamp"),
            "iid_raw_overall": iid,
            "ood_raw_overall": ood,
            "iid_physics": iid_phys,
            "ood_physics": ood_phys,
            "uncertainty_corr_abs_err": unc_corr,
            "selection_score": float(score),
        }
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Loss-weight sweep for current best architecture.")
    ap.add_argument("--base-config", default="reports/training_config.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out-json", default="reports/loss_weight_sweep_results.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_cfg = load_json((root / args.base_config).resolve())
    trials_dir = root / "reports" / "loss_weight_sweep_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    base_loss = copy.deepcopy(base_cfg.get("loss", {}))
    if not isinstance(base_loss, dict):
        raise RuntimeError("base config missing loss dict")

    candidates: List[Dict[str, Any]] = [
        {
            "name": "loss_baseline",
            "loss": base_loss,
        },
        {
            "name": "loss_physics_high",
            "loss": {
                **base_loss,
                "capacity_violation_penalty": 14.0,
                "voltage_violation_penalty": 14.0,
                "energy_cap_violation_penalty": 35.0,
                "physics_consistency_loss": 4.0,
                "thermodynamic_consistency_loss": 7.0,
                "mass_balance_penalty": 12.0,
                "ood_penalty": 8.0,
            },
        },
        {
            "name": "loss_physics_medium",
            "loss": {
                **base_loss,
                "capacity_violation_penalty": 12.0,
                "voltage_violation_penalty": 12.0,
                "energy_cap_violation_penalty": 30.0,
                "physics_consistency_loss": 3.2,
                "thermodynamic_consistency_loss": 6.0,
                "mass_balance_penalty": 10.0,
                "ood_penalty": 7.0,
            },
        },
        {
            "name": "loss_ood_emphasis",
            "loss": {
                **base_loss,
                "ood_penalty": 10.0,
                "novelty_weight": 0.08,
                "uncertainty_regularization": 0.15,
            },
        },
        {
            "name": "loss_uncertainty_emphasis",
            "loss": {
                **base_loss,
                "uncertainty_regularization": 0.2,
                "ood_penalty": 8.0,
                "physics_consistency_loss": 3.5,
            },
        },
        {
            "name": "loss_regression_bias",
            "loss": {
                **base_loss,
                "regression_weight": 1.25,
                "capacity_violation_penalty": 8.0,
                "voltage_violation_penalty": 8.0,
                "energy_cap_violation_penalty": 20.0,
                "physics_consistency_loss": 2.0,
                "ood_penalty": 5.0,
            },
        },
    ]

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        cfg = copy.deepcopy(base_cfg)
        cfg["epochs"] = int(args.epochs)
        cfg["early_stopping_patience"] = int(args.patience)
        cfg["lr_patience"] = min(int(cfg.get("lr_patience", 10)), int(args.patience))
        cfg["loss"] = cand["loss"]
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
