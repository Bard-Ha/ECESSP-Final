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


def run_trial(root: Path, trial_cfg_path: Path) -> Dict[str, Any]:
    cmd = [
        "python",
        "train_physics_first.py",
        "--config",
        str(trial_cfg_path.relative_to(root)),
        "--allow-dev-shortcuts",
        "--ensemble-size",
        "1",
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

    s = load_json(root / "reports" / "training_summary.json")
    iid = s.get("evaluation_metrics", {}).get("iid", {}).get("raw_space", {}).get("overall_micro", {})
    ood = s.get("evaluation_metrics", {}).get("ood", {}).get("raw_space", {}).get("overall_micro", {})
    pstat = s.get("physics_violation_statistics", {})
    iid_p = pstat.get("iid", {}) if isinstance(pstat, dict) else {}
    ood_p = pstat.get("ood", {}) if isinstance(pstat, dict) else {}

    iid_r2 = float(iid.get("r2", -1.0))
    ood_r2 = float(ood.get("r2", -1.0))
    iid_rmse = float(iid.get("rmse", 1e9))
    ood_rmse = float(ood.get("rmse", 1e9))
    violation = float(iid_p.get("capacity_violation_rate", 0.0) or 0.0) + float(iid_p.get("voltage_above_4p4_rate", 0.0) or 0.0)
    violation += float(ood_p.get("capacity_violation_rate", 0.0) or 0.0) + float(ood_p.get("voltage_above_4p4_rate", 0.0) or 0.0)
    score = 0.5 * iid_r2 + 0.5 * ood_r2 - 0.00005 * iid_rmse - 0.00005 * ood_rmse - 2.0 * violation

    out.update(
        {
            "summary_timestamp": s.get("timestamp"),
            "iid_raw_overall": iid,
            "ood_raw_overall": ood,
            "iid_physics": iid_p,
            "ood_physics": ood_p,
            "selection_score": float(score),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Architecture sweep on current graph/data.")
    ap.add_argument("--base-config", default="reports/training_config.json")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--out-json", default="reports/architecture_sweep_results.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base = load_json((root / args.base_config).resolve())
    trials_dir = root / "reports" / "architecture_sweep_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Dict[str, Any]] = [
        {"name": "mlp_ctx0_l3_h128_d020_lr7e4", "material_interaction": "mlp", "enable_graph_context": False, "num_gnn_layers": 3, "hidden_dim": 128, "dropout": 0.20, "learning_rate": 7e-4},
        {"name": "mlp_ctx1_l3_h128_d020_lr7e4", "material_interaction": "mlp", "enable_graph_context": True, "num_gnn_layers": 3, "hidden_dim": 128, "dropout": 0.20, "learning_rate": 7e-4},
        {"name": "tr_ctx0_l3_h128_d020_lr7e4", "material_interaction": "transformer", "enable_graph_context": False, "num_gnn_layers": 3, "hidden_dim": 128, "dropout": 0.20, "learning_rate": 7e-4},
        {"name": "tr_ctx1_l3_h128_d020_lr7e4", "material_interaction": "transformer", "enable_graph_context": True, "num_gnn_layers": 3, "hidden_dim": 128, "dropout": 0.20, "learning_rate": 7e-4},
        {"name": "tr_ctx1_l4_h160_d015_lr5e4", "material_interaction": "transformer", "enable_graph_context": True, "num_gnn_layers": 4, "hidden_dim": 160, "dropout": 0.15, "learning_rate": 5e-4},
        {"name": "tr_ctx1_l2_h128_d015_lr8e4", "material_interaction": "transformer", "enable_graph_context": True, "num_gnn_layers": 2, "hidden_dim": 128, "dropout": 0.15, "learning_rate": 8e-4},
    ]

    results: List[Dict[str, Any]] = []
    for c in candidates:
        cfg = copy.deepcopy(base)
        cfg["epochs"] = int(args.epochs)
        cfg["early_stopping_patience"] = int(args.patience)
        cfg["lr_patience"] = min(int(cfg.get("lr_patience", 10)), int(args.patience))
        cfg["graph_context_dim"] = int(cfg.get("battery_feature_dim", 7))
        for k, v in c.items():
            if k != "name":
                cfg[k] = v
        cfg_path = trials_dir / f"{c['name']}.json"
        save_json(cfg_path, cfg)
        print(f"[trial] {c['name']}")
        r = run_trial(root, cfg_path)
        r["candidate"] = c
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
