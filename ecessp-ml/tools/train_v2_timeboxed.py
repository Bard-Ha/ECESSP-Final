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


def _metric(d: Dict[str, Any], name: str, metric: str, default: float = -1.0) -> float:
    return _safe_float((d.get(name) or {}).get(metric), default)


def selection_score(summary: Dict[str, Any]) -> float:
    iid = summary.get("evaluation_metrics", {}).get("iid", {}).get("raw_space", {})
    ood = summary.get("evaluation_metrics", {}).get("ood", {}).get("raw_space", {})
    phys = summary.get("physics_violation_statistics", {})
    iid_phys = phys.get("iid", {}) if isinstance(phys, dict) else {}
    ood_phys = phys.get("ood", {}) if isinstance(phys, dict) else {}
    unc = summary.get("uncertainty_statistics", {})

    iid_overall = _metric(iid, "overall_micro", "r2")
    ood_overall = _metric(ood, "overall_micro", "r2")
    iid_cap = _metric(iid, "capacity_grav", "r2")
    ood_cap = _metric(ood, "capacity_grav", "r2")
    iid_e_g = _metric(iid, "energy_grav", "r2")
    ood_e_g = _metric(ood, "energy_grav", "r2")
    iid_e_v = _metric(iid, "energy_vol", "r2")
    ood_e_v = _metric(ood, "energy_vol", "r2")
    iid_v = _metric(iid, "average_voltage", "r2")
    ood_v = _metric(ood, "average_voltage", "r2")

    iid_cap_violation = _safe_float(iid_phys.get("capacity_violation_rate"), 1.0)
    ood_cap_violation = _safe_float(ood_phys.get("capacity_violation_rate"), 1.0)
    iid_volt_violation = _safe_float(iid_phys.get("voltage_above_4p4_rate"), 1.0)
    ood_volt_violation = _safe_float(ood_phys.get("voltage_above_4p4_rate"), 1.0)

    gap_penalty = abs(iid_overall - ood_overall)
    unc_corr = _safe_float(unc.get("corr_uncertainty_abs_error"), 0.0)

    return float(
        0.20 * iid_overall
        + 0.15 * ood_overall
        + 0.18 * iid_cap
        + 0.12 * ood_cap
        + 0.15 * iid_e_g
        + 0.08 * ood_e_g
        + 0.06 * iid_e_v
        + 0.04 * ood_e_v
        + 0.02 * iid_v
        + 0.02 * ood_v
        + 0.02 * unc_corr
        - 2.2 * iid_cap_violation
        - 2.2 * ood_cap_violation
        - 1.2 * iid_volt_violation
        - 1.2 * ood_volt_violation
        - 0.4 * gap_penalty
    )


def _softmax_weights(scores: List[float], temperature: float = 0.7) -> List[float]:
    if not scores:
        return []
    t = max(1e-6, float(temperature))
    smax = max(scores)
    exps = [math.exp((s - smax) / t) for s in scores]
    z = sum(exps)
    if z <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [float(v / z) for v in exps]


def _candidate_space() -> List[Dict[str, Any]]:
    return [
        {
            "name": "v2_transformer_l2_h128",
            "material_interaction": "transformer",
            "num_gnn_layers": 2,
            "hidden_dim": 128,
            "dropout": 0.15,
            "learning_rate": 8e-4,
            "weight_decay": 5e-5,
            "feature_dropout": 0.08,
            "feature_noise_std": 0.015,
        },
        {
            "name": "v2_gatv2_l3_h128",
            "material_interaction": "gatv2",
            "num_gnn_layers": 3,
            "hidden_dim": 128,
            "dropout": 0.18,
            "learning_rate": 7e-4,
            "weight_decay": 1e-4,
            "feature_dropout": 0.08,
            "feature_noise_std": 0.015,
        },
        {
            "name": "v2_mpnn_l3_h128",
            "material_interaction": "mpnn",
            "num_gnn_layers": 3,
            "hidden_dim": 128,
            "dropout": 0.18,
            "learning_rate": 7e-4,
            "weight_decay": 1e-4,
            "feature_dropout": 0.08,
            "feature_noise_std": 0.015,
        },
        {
            "name": "v2_transformer_l3_h160",
            "material_interaction": "transformer",
            "num_gnn_layers": 3,
            "hidden_dim": 160,
            "dropout": 0.18,
            "learning_rate": 6e-4,
            "weight_decay": 1e-4,
            "feature_dropout": 0.10,
            "feature_noise_std": 0.02,
        },
    ]


def apply_candidate(base: Dict[str, Any], cand: Dict[str, Any], *, epochs: int, patience: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["material_interaction"] = cand["material_interaction"]
    cfg["num_gnn_layers"] = int(cand["num_gnn_layers"])
    cfg["hidden_dim"] = int(cand["hidden_dim"])
    cfg["dropout"] = float(cand["dropout"])
    cfg["learning_rate"] = float(cand["learning_rate"])
    cfg["weight_decay"] = float(cand["weight_decay"])
    cfg["epochs"] = int(epochs)
    cfg["early_stopping_patience"] = int(patience)
    cfg["lr_patience"] = min(int(cfg.get("lr_patience", 10)), max(3, int(patience // 2)))
    return cfg


def run_train(
    *,
    root: Path,
    cfg_path: Path,
    ensemble_size: int,
    feature_dropout: float,
    feature_noise_std: float,
    use_amp: bool,
    timeout_sec: int,
) -> Dict[str, Any]:
    cmd = [
        "python",
        "train_physics_first.py",
        "--config",
        str(cfg_path.relative_to(root)),
        "--ensemble-size",
        str(max(1, int(ensemble_size))),
        "--allow-dev-shortcuts",
        "--dataset-mode",
        "auto",
        "--feature-dropout",
        str(float(feature_dropout)),
        "--feature-noise-std",
        str(float(feature_noise_std)),
    ]
    if use_amp:
        cmd.append("--amp")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=max(60, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "runtime_sec": float(time.time() - t0),
            "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((exc.stderr or "").splitlines()[-20:]),
            "error": f"timeout_after_{timeout_sec}s",
        }

    out: Dict[str, Any] = {
        "returncode": int(proc.returncode),
        "runtime_sec": float(time.time() - t0),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }
    if proc.returncode != 0:
        return out

    summary_path = root / "reports" / "training_summary.json"
    if not summary_path.exists():
        out["returncode"] = 1
        out["error"] = "training_summary_missing"
        return out

    summary = load_json(summary_path)
    out["summary"] = summary
    out["selection_score"] = selection_score(summary)
    return out


def _load_existing_hgt(root: Path) -> Dict[str, Any] | None:
    manifest_path = root / "reports" / "final_family_ensemble_manifest.json"
    if not manifest_path.exists():
        return None
    payload = load_json(manifest_path)
    models = payload.get("models", [])
    if not isinstance(models, list):
        return None
    for item in models:
        if not isinstance(item, dict):
            continue
        if str(item.get("family", "")).lower() == "hetero_hgt":
            return item
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Time-boxed v2 training with rigorous ranking")
    ap.add_argument("--base-config", default="reports/training_config_v2_optimal_time.json")
    ap.add_argument("--stage1-epochs", type=int, default=36)
    ap.add_argument("--stage1-patience", type=int, default=8)
    ap.add_argument("--stage1-ensemble", type=int, default=1)
    ap.add_argument("--stage2-epochs", type=int, default=84)
    ap.add_argument("--stage2-patience", type=int, default=14)
    ap.add_argument("--stage2-ensemble", type=int, default=2)
    ap.add_argument("--stage2-topk", type=int, default=2)
    ap.add_argument("--time-budget-min", type=float, default=70.0)
    ap.add_argument("--trial-timeout-min", type=float, default=90.0)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--include-existing-hgt", action="store_true")
    ap.add_argument("--hgt-weight", type=float, default=0.25)
    ap.add_argument("--promote-final-manifest", action="store_true")
    ap.add_argument("--promote-only-if-better", action="store_true")
    ap.add_argument("--out-dir", default="reports/v2_timeboxed")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve()
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    base_cfg = load_json((root / args.base_config).resolve())
    candidates = _candidate_space()

    time_budget_sec = max(60.0, float(args.time_budget_min) * 60.0)
    trial_timeout_sec = max(180, int(float(args.trial_timeout_min) * 60))
    t_start = time.time()

    stage1_results: list[dict[str, Any]] = []
    stage2_results: list[dict[str, Any]] = []

    print(f"[v2] start stage-1 screening | candidates={len(candidates)} budget_min={args.time_budget_min:.1f}")
    for cand in candidates:
        elapsed = time.time() - t_start
        if elapsed >= time_budget_sec:
            print("[v2] budget reached before finishing stage-1")
            break

        trial_name = f"stage1_{cand['name']}_{timestamp}"
        cfg = apply_candidate(
            base_cfg,
            cand,
            epochs=int(args.stage1_epochs),
            patience=int(args.stage1_patience),
        )
        cfg_path = runs_dir / f"{trial_name}.json"
        save_json(cfg_path, cfg)

        print(f"[stage1] {cand['name']} ...")
        result = run_train(
            root=root,
            cfg_path=cfg_path,
            ensemble_size=int(args.stage1_ensemble),
            feature_dropout=float(cand["feature_dropout"]),
            feature_noise_std=float(cand["feature_noise_std"]),
            use_amp=bool(args.use_amp),
            timeout_sec=trial_timeout_sec,
        )
        result["candidate"] = cand
        result["stage"] = "stage1"
        result["trial_name"] = trial_name
        result["config_path"] = str(cfg_path)
        if int(result.get("returncode", 1)) == 0:
            summary_copy = runs_dir / f"{trial_name}_summary.json"
            save_json(summary_copy, result["summary"])
            result["summary_path"] = str(summary_copy)
            print(f"[stage1] done {cand['name']} score={result['selection_score']:.4f}")
        else:
            print(f"[stage1] failed {cand['name']} rc={result.get('returncode')}")
        stage1_results.append(result)

    stage1_success = [r for r in stage1_results if int(r.get("returncode", 1)) == 0]
    stage1_ranked = sorted(stage1_success, key=lambda r: float(r.get("selection_score", -1e9)), reverse=True)
    shortlisted = stage1_ranked[: max(1, int(args.stage2_topk))]

    print(f"[v2] start stage-2 refine | shortlisted={len(shortlisted)}")
    for r in shortlisted:
        elapsed = time.time() - t_start
        if elapsed >= time_budget_sec:
            print("[v2] budget reached before finishing stage-2")
            break

        cand = dict(r.get("candidate", {}))
        trial_name = f"stage2_{cand.get('name', 'candidate')}_{timestamp}"
        cfg = apply_candidate(
            base_cfg,
            cand,
            epochs=int(args.stage2_epochs),
            patience=int(args.stage2_patience),
        )
        cfg_path = runs_dir / f"{trial_name}.json"
        save_json(cfg_path, cfg)

        print(f"[stage2] {cand.get('name')} ...")
        result = run_train(
            root=root,
            cfg_path=cfg_path,
            ensemble_size=int(args.stage2_ensemble),
            feature_dropout=float(cand.get("feature_dropout", 0.08)),
            feature_noise_std=float(cand.get("feature_noise_std", 0.015)),
            use_amp=bool(args.use_amp),
            timeout_sec=trial_timeout_sec,
        )
        result["candidate"] = cand
        result["stage"] = "stage2"
        result["trial_name"] = trial_name
        result["config_path"] = str(cfg_path)
        if int(result.get("returncode", 1)) == 0:
            summary_copy = runs_dir / f"{trial_name}_summary.json"
            save_json(summary_copy, result["summary"])
            result["summary_path"] = str(summary_copy)
            print(f"[stage2] done {cand.get('name')} score={result['selection_score']:.4f}")
        else:
            print(f"[stage2] failed {cand.get('name')} rc={result.get('returncode')}")
        stage2_results.append(result)

    ranked_source = [r for r in stage2_results if int(r.get("returncode", 1)) == 0]
    if not ranked_source:
        ranked_source = stage1_success
    ranked = sorted(ranked_source, key=lambda r: float(r.get("selection_score", -1e9)), reverse=True)
    ranked = ranked[:3]

    weights = _softmax_weights([float(r.get("selection_score", 0.0)) for r in ranked], temperature=0.7)
    masked_models: list[dict[str, Any]] = []
    for i, r in enumerate(ranked):
        summary = r.get("summary", {})
        ckpt = str(summary.get("best_checkpoint", "")).strip()
        if not ckpt:
            continue
        cand = r.get("candidate", {})
        masked_models.append(
            {
                "name": str(cand.get("name", f"masked_{i}")),
                "family": "masked_gnn",
                "checkpoint_path": ckpt,
                "weight": float(weights[i]) if i < len(weights) else 0.0,
                "material_interaction": str(cand.get("material_interaction", "")),
                "selection_score": float(r.get("selection_score", 0.0)),
                "summary_path": str(r.get("summary_path", "")),
                "summary_timestamp": str(summary.get("timestamp", "")),
            }
        )

    model_entries: list[dict[str, Any]] = []
    if args.include_existing_hgt:
        hgt = _load_existing_hgt(root)
        if hgt is not None:
            hgt_weight = float(max(0.0, min(0.8, args.hgt_weight)))
            hgt_entry = dict(hgt)
            hgt_entry["weight"] = hgt_weight
            model_entries.append(hgt_entry)
            remaining = max(0.0, 1.0 - hgt_weight)
            wsum = sum(float(m.get("weight", 0.0)) for m in masked_models)
            for m in masked_models:
                base_w = float(m.get("weight", 0.0))
                m["weight"] = (base_w / wsum) * remaining if wsum > 0 else (remaining / max(1, len(masked_models)))

    model_entries.extend(masked_models)
    wsum_final = sum(float(m.get("weight", 0.0)) for m in model_entries)
    if wsum_final > 0:
        for m in model_entries:
            m["weight"] = float(m["weight"] / wsum_final)

    manifest = {
        "version": "v2_timeboxed",
        "created_at_epoch_sec": int(time.time()),
        "source": "tools/train_v2_timeboxed.py",
        "base_config": str((root / args.base_config).resolve()),
        "models": model_entries,
        "primary_model": model_entries[0]["name"] if model_entries else None,
        "notes": [
            "Selection optimized for capacity/energy realism + OOD robustness + physics validity.",
            "This manifest is runtime-compatible with backend checkpoint resolver.",
        ],
    }

    results_payload = {
        "timestamp": timestamp,
        "time_budget_min": float(args.time_budget_min),
        "elapsed_sec": float(time.time() - t_start),
        "stage1": stage1_results,
        "stage2": stage2_results,
        "ranked_final": ranked,
        "manifest_preview": manifest,
    }

    manifest_path = out_dir / "manifest.json"
    results_path = out_dir / "results.json"
    save_json(manifest_path, manifest)
    save_json(results_path, results_payload)

    print(f"[done] results: {results_path}")
    print(f"[done] manifest: {manifest_path}")

    if args.promote_final_manifest:
        promote_ok = True
        if args.promote_only_if_better and model_entries:
            current_summary_path = root / "reports" / "training_summary.json"
            if current_summary_path.exists():
                current_summary = load_json(current_summary_path)
                current_score = selection_score(current_summary)
                candidate_best = float((ranked[0] if ranked else {}).get("selection_score", -1e9))
                promote_ok = candidate_best > current_score
                print(
                    f"[promote-check] candidate_best={candidate_best:.4f} current={current_score:.4f} "
                    f"promote={promote_ok}"
                )
        if promote_ok:
            target = root / "reports" / "final_family_ensemble_manifest.json"
            if target.exists():
                backup = target.with_name(f"final_family_ensemble_manifest.backup_{timestamp}.json")
                shutil.copy2(target, backup)
                print(f"[promote] backup: {backup}")
            shutil.copy2(manifest_path, target)
            print(f"[promote] updated: {target}")
        else:
            print("[promote] skipped (not better than current summary score)")


if __name__ == "__main__":
    main()
