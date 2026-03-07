#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import List


def run_step(root: Path, cmd: List[str], name: str) -> None:
    print(f"[step] {name}")
    print("[cmd] " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(root))
    dt = time.time() - t0
    print(f"[step] {name} rc={p.returncode} runtime_sec={dt:.1f}")
    if p.returncode != 0:
        raise RuntimeError(f"step failed: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full hybrid battery pipeline: masked trio + hetero HGT + fusion")
    ap.add_argument("--ensemble-epochs", type=int, default=140)
    ap.add_argument("--ensemble-patience", type=int, default=25)
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--hgt-epochs", type=int, default=140)
    ap.add_argument("--hgt-patience", type=int, default=30)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    if not args.skip_baseline:
        run_step(
            root,
            [
                "python",
                "train_physics_first.py",
                "--dataset-mode",
                "auto",
            ],
            "baseline_masked_transformer",
        )

    run_step(
        root,
        [
            "python",
            "tools/train_three_model_ensemble.py",
            "--epochs",
            str(int(args.ensemble_epochs)),
            "--patience",
            str(int(args.ensemble_patience)),
            "--ensemble-size",
            str(int(args.ensemble_size)),
        ],
        "masked_trio_train_and_manifest",
    )

    run_step(
        root,
        [
            "python",
            "tools/build_benchmark_splits.py",
            "--parsed-csv",
            "data/processed/batteries_parsed_curated.csv",
            "--output-json",
            "reports/benchmark_splits.json",
            "--seed",
            "42",
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.15",
            "--group-keys",
            "chemsys",
            "framework_formula",
            "working_ion",
            "--ood-enabled",
            "--holdout-ion",
            "Na",
            "--holdout-chemsys-fraction",
            "0.06",
            "--min-ion-count",
            "80",
        ],
        "build_shared_benchmark_splits",
    )

    run_step(
        root,
        [
            "python",
            "tools/build_true_hetero_graph.py",
            "--split-json",
            "reports/benchmark_splits.json",
        ],
        "build_true_hetero_graph",
    )

    run_step(
        root,
        [
            "python",
            "train_hetero_hgt.py",
            "--epochs",
            str(int(args.hgt_epochs)),
            "--patience",
            str(int(args.hgt_patience)),
        ],
        "train_hetero_hgt",
    )

    run_step(
        root,
        [
            "python",
            "tools/fuse_family_ensemble.py",
        ],
        "fuse_family_ensemble_manifest",
    )

    print("[done] full hybrid pipeline complete")
    print("[artifact] reports/three_model_ensemble_manifest.json")
    print("[artifact] reports/training_summary_hgt.json")
    print("[artifact] reports/final_family_ensemble_manifest.json")


if __name__ == "__main__":
    main()
