#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _safe_text_series(df: pd.DataFrame, column: str, fallback: str) -> pd.Series:
    if column in df.columns:
        s = df[column]
    else:
        s = pd.Series([fallback] * len(df))
    s = s.fillna(fallback).astype(str).str.strip()
    return s.replace("", fallback)


def _build_group_labels(parsed_df: pd.DataFrame, group_keys: List[str]) -> np.ndarray:
    cols = [_safe_text_series(parsed_df, key, f"NA_{key}") for key in group_keys]
    g = cols[0]
    for c in cols[1:]:
        g = g + "||" + c
    return g.to_numpy()


def _grouped_partition(
    indices: np.ndarray,
    group_labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(indices) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    test_ratio = max(0.0, 1.0 - float(train_ratio) - float(val_ratio))
    target_counts = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64) * float(len(indices))

    group_to_rows: Dict[str, List[int]] = {}
    for ridx in indices.tolist():
        g = str(group_labels[int(ridx)])
        group_to_rows.setdefault(g, []).append(int(ridx))

    groups = list(group_to_rows.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    buckets: List[List[int]] = [[], [], []]
    bucket_counts = np.zeros(3, dtype=np.float64)
    for g in groups:
        rows = group_to_rows[g]
        pressure = np.array(
            [
                bucket_counts[i] / max(target_counts[i], 1.0)
                if target_counts[i] > 0.0
                else (1e9 if i < 2 else bucket_counts[i] / max(len(indices), 1.0))
                for i in range(3)
            ],
            dtype=np.float64,
        )
        b = int(np.argmin(pressure))
        buckets[b].extend(rows)
        bucket_counts[b] += float(len(rows))

    return (
        np.array(sorted(buckets[0]), dtype=np.int64),
        np.array(sorted(buckets[1]), dtype=np.int64),
        np.array(sorted(buckets[2]), dtype=np.int64),
    )


def build_splits(
    parsed_df: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    group_keys: List[str],
    ood_enabled: bool,
    holdout_ion: str,
    holdout_chemsys_fraction: float,
    min_ion_count: int,
    seed: int,
) -> dict:
    n = int(len(parsed_df))
    all_idx = np.arange(n, dtype=np.int64)

    ion_series = _safe_text_series(parsed_df, "working_ion", "unknown")
    chemsys_series = _safe_text_series(parsed_df, "chemsys", "unknown")
    ion_counts = ion_series.value_counts()

    rng = np.random.default_rng(seed + 17)
    picked_ion = "none"
    ood_ion_mask = np.zeros(n, dtype=bool)
    if ood_enabled and len(ion_counts):
        candidates = [ion for ion, c in ion_counts.items() if int(c) >= int(min_ion_count)]
        if str(holdout_ion).lower() != "auto":
            picked_ion = str(holdout_ion)
        else:
            major = str(ion_counts.index[0])
            non_major = [ion for ion in candidates if str(ion) != major]
            pool = non_major or candidates or [str(ion_counts.index[-1])]
            picked_ion = str(pool[int(rng.integers(0, len(pool)))])
        ood_ion_mask = (ion_series.to_numpy() == picked_ion)

    ood_chemsys: List[str] = []
    ood_chemsys_mask = np.zeros(n, dtype=bool)
    frac = float(np.clip(holdout_chemsys_fraction, 0.0, 0.5))
    if ood_enabled and frac > 0.0:
        remaining_chemsys = chemsys_series.to_numpy()[~ood_ion_mask]
        unique_chemsys = np.unique(remaining_chemsys)
        if unique_chemsys.size > 0:
            k = int(round(float(unique_chemsys.size) * frac))
            k = max(1, min(int(unique_chemsys.size), k))
            picked = rng.choice(unique_chemsys, size=k, replace=False)
            ood_chemsys = sorted([str(x) for x in picked.tolist()])
            ood_chemsys_mask = np.isin(chemsys_series.to_numpy(), np.array(ood_chemsys, dtype=object))

    ood_mask = (ood_ion_mask | ood_chemsys_mask) if ood_enabled else np.zeros(n, dtype=bool)
    iid_pool = all_idx[~ood_mask]

    group_labels = _build_group_labels(parsed_df, group_keys)
    tr_idx, va_idx, te_iid_idx = _grouped_partition(iid_pool, group_labels, train_ratio, val_ratio, seed)
    te_ood_idx = all_idx[ood_mask]

    return {
        "train": tr_idx.tolist(),
        "val": va_idx.tolist(),
        "test_iid": te_iid_idx.tolist(),
        "test_ood": te_ood_idx.tolist(),
        "metadata": {
            "group_keys": group_keys,
            "train_ratio_target": float(train_ratio),
            "val_ratio_target": float(val_ratio),
            "ood_enabled": bool(ood_enabled),
            "holdout_ion": picked_ion,
            "holdout_chemsys_fraction": frac,
            "holdout_chemsys_count": int(len(ood_chemsys)),
            "holdout_chemsys_preview": ood_chemsys[:20],
            "counts": {
                "total": int(n),
                "train": int(len(tr_idx)),
                "val": int(len(va_idx)),
                "test_iid": int(len(te_iid_idx)),
                "test_ood": int(len(te_ood_idx)),
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build grouped IID + OOD benchmark split indices.")
    ap.add_argument("--parsed-csv", default="data/processed/batteries_parsed.csv")
    ap.add_argument("--output-json", default="reports/benchmark_splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--group-keys", nargs="+", default=["chemsys", "framework_formula", "working_ion"])
    ap.add_argument("--ood-enabled", action="store_true")
    ap.add_argument("--holdout-ion", default="auto")
    ap.add_argument("--holdout-chemsys-fraction", type=float, default=0.08)
    ap.add_argument("--min-ion-count", type=int, default=120)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed_path = (root / args.parsed_csv).resolve()
    out_path = (root / args.output_json).resolve()

    parsed_df = pd.read_csv(parsed_path)
    split_payload = build_splits(
        parsed_df,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        group_keys=list(args.group_keys),
        ood_enabled=bool(args.ood_enabled),
        holdout_ion=str(args.holdout_ion),
        holdout_chemsys_fraction=float(args.holdout_chemsys_fraction),
        min_ion_count=int(args.min_ion_count),
        seed=int(args.seed),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote benchmark splits: {out_path}")
    print(f"[ok] counts: {split_payload['metadata']['counts']}")


if __name__ == "__main__":
    main()
