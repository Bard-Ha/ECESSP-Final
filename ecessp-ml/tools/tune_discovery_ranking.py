#!/usr/bin/env python
"""
Tune discovery ranking and blending weights for top-5 quality.

This script performs a bounded runtime sweep over orchestrator knobs:
- ECESSP_ROLE_HEAD_BLEND
- ECESSP_COMPAT_HEAD_BLEND
- ECESSP_UNCERTAINTY_MODEL_WEIGHT
- ECESSP_PARETO_OBJECTIVE_WEIGHT
- ECESSP_PARETO_FEASIBILITY_WEIGHT
- ECESSP_PARETO_UNCERTAINTY_WEIGHT

Outputs:
- reports/discovery_ranking_tuning.json
- reports/discovery_ranking_env_recommended.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.discovery_service import DiscoveryService


@dataclass
class Scenario:
    name: str
    working_ion: str
    objective: Dict[str, float]


def _seed_system(working_ion: str) -> Dict[str, Any]:
    return {
        "battery_id": f"rank_tune_seed_{working_ion}",
        "battery_type": "insertion",
        "working_ion": working_ion,
        "average_voltage": 3.6,
        "capacity_grav": 180.0,
        "capacity_vol": 560.0,
        "energy_grav": 648.0,
        "energy_vol": 2016.0,
        "stability_charge": 0.12,
        "stability_discharge": 0.08,
    }


def _scenario_set() -> List[Scenario]:
    return [
        Scenario(
            name="li_high_energy",
            working_ion="Li",
            objective={
                "average_voltage": 3.95,
                "capacity_grav": 230.0,
                "capacity_vol": 720.0,
                "stability_charge": 0.10,
                "stability_discharge": 0.06,
            },
        ),
        Scenario(
            name="li_balanced",
            working_ion="Li",
            objective={
                "average_voltage": 3.70,
                "capacity_grav": 190.0,
                "capacity_vol": 640.0,
                "stability_charge": 0.08,
                "stability_discharge": 0.05,
            },
        ),
        Scenario(
            name="na_balanced",
            working_ion="Na",
            objective={
                "average_voltage": 3.20,
                "capacity_grav": 170.0,
                "capacity_vol": 550.0,
                "stability_charge": 0.08,
                "stability_discharge": 0.05,
            },
        ),
    ]


def _candidate_configs() -> List[Dict[str, float]]:
    pareto_presets = [
        {"obj": 0.50, "feas": 0.35, "unc": 0.15},
        {"obj": 0.60, "feas": 0.25, "unc": 0.15},
        {"obj": 0.45, "feas": 0.30, "unc": 0.25},
    ]
    micro_presets = [
        {"role": 0.30, "compat": 0.60, "unc_model": 0.45},
        {"role": 0.45, "compat": 0.75, "unc_model": 0.45},
        {"role": 0.45, "compat": 0.60, "unc_model": 0.65},
        {"role": 0.60, "compat": 0.75, "unc_model": 0.65},
    ]
    out: List[Dict[str, float]] = []
    for p, m in itertools.product(pareto_presets, micro_presets):
        out.append(
            {
                "role_head_blend": float(m["role"]),
                "compat_head_blend": float(m["compat"]),
                "uncertainty_model_weight": float(m["unc_model"]),
                "pareto_objective_weight": float(p["obj"]),
                "pareto_feasibility_weight": float(p["feas"]),
                "pareto_uncertainty_weight": float(p["unc"]),
            }
        )
    return out


def _set_env_from_cfg(cfg: Dict[str, float]) -> None:
    os.environ["ECESSP_ROLE_HEAD_BLEND"] = f"{cfg['role_head_blend']:.6f}"
    os.environ["ECESSP_COMPAT_HEAD_BLEND"] = f"{cfg['compat_head_blend']:.6f}"
    os.environ["ECESSP_UNCERTAINTY_MODEL_WEIGHT"] = f"{cfg['uncertainty_model_weight']:.6f}"
    os.environ["ECESSP_PARETO_OBJECTIVE_WEIGHT"] = f"{cfg['pareto_objective_weight']:.6f}"
    os.environ["ECESSP_PARETO_FEASIBILITY_WEIGHT"] = f"{cfg['pareto_feasibility_weight']:.6f}"
    os.environ["ECESSP_PARETO_UNCERTAINTY_WEIGHT"] = f"{cfg['pareto_uncertainty_weight']:.6f}"


def _evaluate_config(
    svc: DiscoveryService,
    *,
    cfg: Dict[str, float],
    scenarios: List[Scenario],
    num_candidates: int,
    optimize_steps: int,
) -> Dict[str, Any]:
    _set_env_from_cfg(cfg)
    objective_hit_threshold = float(os.getenv("ECESSP_OBJECTIVE_HIT_THRESHOLD", "0.30"))
    feasibility_hit_threshold = float(os.getenv("ECESSP_FEASIBILITY_HIT_THRESHOLD", "0.70"))

    total_topk = 0
    hit_count = 0
    valid_count = 0
    top1_scores: List[float] = []
    top1_obj: List[float] = []
    top1_unc: List[float] = []
    scenario_rows: List[Dict[str, Any]] = []

    for sc in scenarios:
        res = svc.discover(
            base_system_data=_seed_system(sc.working_ion),
            objective=sc.objective,
            explain=False,
            mode="generative",
            discovery_params={
                "num_candidates": int(num_candidates),
                "optimize_steps": int(optimize_steps),
                "diversity_weight": 0.45,
                "novelty_weight": 0.35,
                "extrapolation_strength": 0.30,
            },
        )
        history = res.get("history") or []
        topk = history[:5]
        total_topk += len(topk)
        for h in topk:
            obj = float(h.get("objective_alignment_score") or 0.0)
            feas = float(h.get("feasibility_score") or 0.0)
            valid = bool(h.get("valid", False)) and bool(h.get("physics_hard_valid", False))
            if valid:
                valid_count += 1
            if obj >= objective_hit_threshold and feas >= feasibility_hit_threshold and valid:
                hit_count += 1

        top = history[0] if history else {}
        top1_scores.append(float(top.get("score") or 0.0))
        top1_obj.append(float(top.get("objective_alignment_score") or 0.0))
        top1_unc.append(float(top.get("uncertainty_penalty") or 0.0))

        scenario_rows.append(
            {
                "name": sc.name,
                "candidate_count": int((res.get("metadata") or {}).get("candidate_count") or 0),
                "top_score": float(top.get("score") or 0.0),
                "top_objective_alignment": float(top.get("objective_alignment_score") or 0.0),
                "top_feasibility_score": float(top.get("feasibility_score") or 0.0),
                "top_uncertainty_penalty": float(top.get("uncertainty_penalty") or 0.0),
            }
        )

    top5_hit_rate = float(hit_count / max(1, total_topk))
    top5_valid_rate = float(valid_count / max(1, total_topk))
    top1_mean_score = float(sum(top1_scores) / max(1, len(top1_scores)))
    top1_mean_objective = float(sum(top1_obj) / max(1, len(top1_obj)))
    top1_mean_uncertainty = float(sum(top1_unc) / max(1, len(top1_unc)))

    composite = (
        0.45 * top5_hit_rate
        + 0.30 * top5_valid_rate
        + 0.15 * top1_mean_objective
        + 0.10 * top1_mean_score
        - 0.05 * max(0.0, top1_mean_uncertainty - 0.40)
    )

    return {
        "config": cfg,
        "metrics": {
            "top5_hit_rate": top5_hit_rate,
            "top5_valid_rate": top5_valid_rate,
            "top1_mean_score": top1_mean_score,
            "top1_mean_objective": top1_mean_objective,
            "top1_mean_uncertainty": top1_mean_uncertainty,
            "composite_score": float(composite),
        },
        "scenarios": scenario_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune discovery ranking/blending knobs")
    parser.add_argument("--num-candidates", type=int, default=24)
    parser.add_argument("--optimize-steps", type=int, default=8)
    parser.add_argument("--max-configs", type=int, default=12)
    parser.add_argument(
        "--objective-hit-threshold",
        type=float,
        default=-1.0,
        help="If <0, auto-calibrate from baseline objective alignment.",
    )
    parser.add_argument("--feasibility-hit-threshold", type=float, default=0.70)
    parser.add_argument("--out", default="reports/discovery_ranking_tuning.json")
    parser.add_argument("--recommended-env-out", default="reports/discovery_ranking_env_recommended.json")
    args = parser.parse_args()

    scenarios = _scenario_set()
    all_cfg = _candidate_configs()[: max(1, int(args.max_configs))]
    svc = DiscoveryService()

    if float(args.objective_hit_threshold) < 0.0:
        baseline = _evaluate_config(
            svc=svc,
            cfg=all_cfg[0],
            scenarios=scenarios,
            num_candidates=int(args.num_candidates),
            optimize_steps=int(args.optimize_steps),
        )
        baseline_obj = float((baseline.get("metrics") or {}).get("top1_mean_objective", 0.20))
        objective_hit_threshold = max(0.08, min(0.45, 0.95 * baseline_obj))
    else:
        objective_hit_threshold = float(args.objective_hit_threshold)
    feasibility_hit_threshold = float(args.feasibility_hit_threshold)
    os.environ["ECESSP_OBJECTIVE_HIT_THRESHOLD"] = f"{objective_hit_threshold:.6f}"
    os.environ["ECESSP_FEASIBILITY_HIT_THRESHOLD"] = f"{feasibility_hit_threshold:.6f}"

    rows: List[Dict[str, Any]] = []
    for i, cfg in enumerate(all_cfg, start=1):
        row = _evaluate_config(
            svc=svc,
            cfg=cfg,
            scenarios=scenarios,
            num_candidates=int(args.num_candidates),
            optimize_steps=int(args.optimize_steps),
        )
        row["trial"] = i
        rows.append(row)
        print(
            f"[trial {i}/{len(all_cfg)}] "
            f"score={row['metrics']['composite_score']:.6f} "
            f"top5_hit={row['metrics']['top5_hit_rate']:.6f} "
            f"top1={row['metrics']['top1_mean_score']:.6f}"
        )

    rows.sort(key=lambda r: float((r.get("metrics") or {}).get("composite_score", 0.0)), reverse=True)
    best = rows[0] if rows else {}
    best_cfg = best.get("config") or {}

    summary = {
        "tuning_target": "maximize top-5 hit/validity with bounded uncertainty",
        "trial_count": len(rows),
        "num_candidates": int(args.num_candidates),
        "optimize_steps": int(args.optimize_steps),
        "objective_hit_threshold": float(objective_hit_threshold),
        "feasibility_hit_threshold": float(feasibility_hit_threshold),
        "best": best,
        "trials": rows,
    }

    out_path = (PROJECT_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    env_recommended = {
        "ECESSP_ROLE_HEAD_BLEND": best_cfg.get("role_head_blend"),
        "ECESSP_COMPAT_HEAD_BLEND": best_cfg.get("compat_head_blend"),
        "ECESSP_UNCERTAINTY_MODEL_WEIGHT": best_cfg.get("uncertainty_model_weight"),
        "ECESSP_PARETO_OBJECTIVE_WEIGHT": best_cfg.get("pareto_objective_weight"),
        "ECESSP_PARETO_FEASIBILITY_WEIGHT": best_cfg.get("pareto_feasibility_weight"),
        "ECESSP_PARETO_UNCERTAINTY_WEIGHT": best_cfg.get("pareto_uncertainty_weight"),
        "ECESSP_OBJECTIVE_HIT_THRESHOLD": float(objective_hit_threshold),
        "ECESSP_FEASIBILITY_HIT_THRESHOLD": float(feasibility_hit_threshold),
    }
    env_out_path = (PROJECT_ROOT / args.recommended_env_out).resolve()
    env_out_path.parent.mkdir(parents=True, exist_ok=True)
    env_out_path.write_text(json.dumps(env_recommended, indent=2), encoding="utf-8")

    print(out_path)
    print(env_out_path)


if __name__ == "__main__":
    main()
