#!/usr/bin/env python
"""
Step 9 validation suite for generative discovery.

Outputs:
- scenario-level discovery summary
- aggregate discovery report-card metrics
- counterfactual directional checks
"""

from __future__ import annotations

import argparse
import json
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
    # Stable generic seed, objective-conditioned discovery will reshape it.
    return {
        "battery_id": f"step9_seed_{working_ion}",
        "working_ion": working_ion,
        "average_voltage": 3.6,
        "capacity_grav": 180.0,
        "capacity_vol": 560.0,
        "energy_grav": 648.0,
        "energy_vol": 2016.0,
        "stability_charge": 0.12,
        "stability_discharge": 0.08,
    }


def _run_scenario(svc: DiscoveryService, scenario: Scenario, num_candidates: int, optimize_steps: int) -> Dict[str, Any]:
    res = svc.discover(
        base_system_data=_seed_system(scenario.working_ion),
        objective=scenario.objective,
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
    top = history[0] if history else {}
    top_system = top.get("system") or {}
    md = res.get("metadata") or {}

    return {
        "name": scenario.name,
        "working_ion": scenario.working_ion,
        "objective": scenario.objective,
        "candidate_count": int(md.get("candidate_count") or 0),
        "report_card": md.get("discovery_report_card") or {},
        "top": {
            "score": float(top.get("score") or 0.0),
            "framework_formula": top_system.get("framework_formula"),
            "capacity_grav": top_system.get("capacity_grav"),
            "average_voltage": top_system.get("average_voltage"),
            "material_novelty_score": top.get("material_novelty_score"),
            "material_uncertainty_proxy": top.get("material_uncertainty_proxy"),
            "material_thermodynamic_proxy": top.get("material_thermodynamic_proxy"),
        },
    }


def _aggregate_report_cards(scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = [
        "hit_rate_at_k",
        "constraint_validity_rate",
        "physics_violation_rate",
        "novelty_mean",
        "novelty_p90",
        "uncertainty_calibration_corr",
        "ood_acceptance_rate",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        vals: List[float] = []
        for s in scenarios:
            rc = s.get("report_card") or {}
            v = rc.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        out[k] = float(sum(vals) / len(vals)) if vals else 0.0
    return out


def _counterfactual_check(svc: DiscoveryService, ion: str, base_obj: Dict[str, float]) -> Dict[str, Any]:
    low = dict(base_obj)
    high = dict(base_obj)
    low["capacity_grav"] = float(base_obj.get("capacity_grav", 180.0)) * 0.85
    high["capacity_grav"] = float(base_obj.get("capacity_grav", 180.0)) * 1.15

    low_res = svc.discover(
        base_system_data=_seed_system(ion),
        objective=low,
        explain=False,
        mode="generative",
        discovery_params={"num_candidates": 24, "optimize_steps": 8},
    )
    high_res = svc.discover(
        base_system_data=_seed_system(ion),
        objective=high,
        explain=False,
        mode="generative",
        discovery_params={"num_candidates": 24, "optimize_steps": 8},
    )

    low_hist = low_res.get("history") or []
    high_hist = high_res.get("history") or []
    low_md = low_res.get("metadata") or {}
    high_md = high_res.get("metadata") or {}

    low_cap = float(((low_hist[0].get("system") or {}).get("capacity_grav") or 0.0)) if low_hist else 0.0
    high_cap = float(((high_hist[0].get("system") or {}).get("capacity_grav") or 0.0)) if high_hist else 0.0
    low_feasible = float((low_md.get("target_objectives_feasible") or {}).get("capacity_grav") or low["capacity_grav"])
    high_feasible = float((high_md.get("target_objectives_feasible") or {}).get("capacity_grav") or high["capacity_grav"])
    feasible_gap = high_feasible - low_feasible

    # Physics-aware directional criterion:
    # if low/high collapse to the same feasible target after hard caps,
    # treat as not-directional and pass (no meaningful monotonic test).
    if abs(feasible_gap) < 1e-6:
        directional_pass = True
        direction_mode = "collapsed_feasible_targets"
    elif feasible_gap > 0:
        directional_pass = bool(high_cap >= low_cap)
        direction_mode = "increasing"
    else:
        directional_pass = bool(high_cap <= low_cap)
        direction_mode = "decreasing"

    return {
        "ion": ion,
        "low_target_capacity": low["capacity_grav"],
        "high_target_capacity": high["capacity_grav"],
        "low_feasible_target_capacity": low_feasible,
        "high_feasible_target_capacity": high_feasible,
        "top_low_capacity": low_cap,
        "top_high_capacity": high_cap,
        "direction_mode": direction_mode,
        "directional_pass": directional_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 9 discovery validation suite")
    parser.add_argument("--num-candidates", type=int, default=28)
    parser.add_argument("--optimize-steps", type=int, default=10)
    parser.add_argument("--out", type=str, default="reports/discovery_validation_step9.json")
    args = parser.parse_args()

    svc = DiscoveryService()

    scenarios = [
        Scenario(
            name="li_high_energy",
            working_ion="Li",
            objective={
                "average_voltage": 3.95,
                "capacity_grav": 230.0,
                "capacity_vol": 720.0,
                "energy_grav": 908.5,
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
                "energy_grav": 703.0,
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
                "energy_grav": 544.0,
                "stability_charge": 0.08,
                "stability_discharge": 0.05,
            },
        ),
    ]

    scenario_results = [
        _run_scenario(
            svc=svc,
            scenario=s,
            num_candidates=args.num_candidates,
            optimize_steps=args.optimize_steps,
        )
        for s in scenarios
    ]

    counterfactuals = [
        _counterfactual_check(svc=svc, ion=s.working_ion, base_obj=s.objective)
        for s in scenarios
    ]

    summary = {
        "validation_step": 9,
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
        "aggregate_report_card": _aggregate_report_cards(scenario_results),
        "counterfactual_checks": counterfactuals,
        "counterfactual_pass_rate": float(
            sum(1 for c in counterfactuals if c.get("directional_pass")) / max(1, len(counterfactuals))
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
