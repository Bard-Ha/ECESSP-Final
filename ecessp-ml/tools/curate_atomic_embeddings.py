#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def find_embedding_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("atom_emb_")]
    cols = sorted(cols, key=lambda x: int(x.split("_")[-1]))
    return cols


def _safe_scale(x: float, floor: float = 1e-6) -> float:
    v = float(x)
    if not np.isfinite(v):
        return 1.0
    return max(abs(v), floor)


def _normalize_embedding_column(v: np.ndarray) -> tuple[np.ndarray, Dict[str, float | str]]:
    x = np.asarray(v, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32), {"method": "identity", "center": 0.0, "scale": 1.0, "skew": 0.0}

    if np.nanstd(x) < 1e-12:
        return np.zeros_like(x, dtype=np.float32), {"method": "constant_zero", "center": float(np.nanmean(x)), "scale": 1.0, "skew": 0.0}

    s = pd.Series(x)
    skew = float(s.skew())
    q01 = float(s.quantile(0.01))
    q99 = float(s.quantile(0.99))
    xw = np.clip(x, q01, q99)

    # Adaptive per-column transform:
    # - positive heavy-skew: log1p + robust z
    # - skewed: robust z
    # - near-symmetric: standard z
    if float(np.min(xw)) >= 0.0 and abs(skew) > 1.5:
        z = np.log1p(xw)
        zs = pd.Series(z)
        median = float(zs.median())
        iqr = float(zs.quantile(0.75) - zs.quantile(0.25))
        scale = _safe_scale(iqr / 1.349 if iqr > 0.0 else float(zs.std(ddof=0)))
        out = (z - median) / scale
        method = "log1p_robust_z"
        center = median
    elif abs(skew) > 1.0:
        median = float(np.median(xw))
        q1 = float(np.quantile(xw, 0.25))
        q3 = float(np.quantile(xw, 0.75))
        iqr = float(q3 - q1)
        scale = _safe_scale(iqr / 1.349 if iqr > 0.0 else float(np.std(xw)))
        out = (xw - median) / scale
        method = "robust_z"
        center = median
    else:
        mean = float(np.mean(xw))
        std = _safe_scale(float(np.std(xw)))
        out = (xw - mean) / std
        method = "standard_z"
        center = mean
        scale = std

    out = np.clip(out, -8.0, 8.0)
    return out.astype(np.float32), {
        "method": method,
        "center": float(center),
        "scale": float(scale),
        "skew": float(skew),
        "q01": q01,
        "q99": q99,
    }


def curate_atomic_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, object]]:
    if "material_id" not in df.columns:
        raise RuntimeError("atomic embeddings file must contain material_id")
    emb_cols = find_embedding_columns(df)
    if len(emb_cols) != 64:
        raise RuntimeError(f"Expected 64 atom_emb_* columns, found {len(emb_cols)}")

    out = df.copy()
    rows_in = int(len(out))

    out["material_id"] = out["material_id"].astype(str).str.strip()
    out = out[out["material_id"].str.len() > 0].copy()

    # Convert to numeric and detect invalid values.
    raw_mat = out[emb_cols].apply(pd.to_numeric, errors="coerce")
    non_finite_mask = ~np.isfinite(raw_mat.to_numpy(dtype=np.float64))
    non_finite_count = int(non_finite_mask.sum())

    # Impute NaN/Inf with per-dimension median.
    med = raw_mat.median(axis=0, skipna=True)
    raw_mat = raw_mat.fillna(med)
    raw_mat = raw_mat.replace([np.inf, -np.inf], np.nan).fillna(med)

    # Column-wise adaptive normalization with robust outlier handling.
    norm_cols: Dict[str, np.ndarray] = {}
    norm_specs: Dict[str, Dict[str, float | str]] = {}
    clipped_count = 0
    for col in emb_cols:
        arr = raw_mat[col].to_numpy(dtype=np.float64)
        arr_norm, spec = _normalize_embedding_column(arr)
        norm_cols[col] = arr_norm
        norm_specs[col] = spec
        q01 = float(spec.get("q01", np.nan))
        q99 = float(spec.get("q99", np.nan))
        if np.isfinite(q01) and np.isfinite(q99):
            clipped_count += int(((arr < q01) | (arr > q99)).sum())

    normalized = pd.DataFrame(norm_cols, index=raw_mat.index)

    # Collapse duplicate material IDs by mean embedding.
    dedup = pd.concat([out[["material_id"]], normalized], axis=1)
    dup_rows = int(dedup.duplicated(subset=["material_id"]).sum())
    dedup = dedup.groupby("material_id", as_index=False)[emb_cols].mean()

    # Norm diagnostics.
    em = dedup[emb_cols].to_numpy(dtype=np.float64)
    l2 = np.linalg.norm(em, axis=1)
    near_zero = l2 < 1e-8
    dedup["emb_l2_norm"] = l2
    dedup["emb_is_near_zero"] = near_zero

    summary = {
        "rows_in": rows_in,
        "rows_after_id_cleaning": int(len(out)),
        "rows_out": int(len(dedup)),
        "embedding_dims": int(len(emb_cols)),
        "non_finite_values_imputed": non_finite_count,
        "values_clipped_0.1pct_99.9pct": clipped_count,
        "normalization_policy": {
            "type": "column_adaptive_embedding_normalization",
            "clip_bounds": "p01-p99 then optional transform-specific scaling",
            "per_column": norm_specs,
        },
        "duplicate_material_rows_collapsed": dup_rows,
        "near_zero_embeddings": int(near_zero.sum()),
        "l2_norm_stats": {
            "min": float(np.min(l2)) if l2.size else 0.0,
            "p01": float(np.quantile(l2, 0.01)) if l2.size else 0.0,
            "p50": float(np.quantile(l2, 0.50)) if l2.size else 0.0,
            "p99": float(np.quantile(l2, 0.99)) if l2.size else 0.0,
            "max": float(np.max(l2)) if l2.size else 0.0,
        },
    }
    return dedup, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate and audit atomic_embeddings.csv")
    ap.add_argument("--input", default="data/processed/atomic_embeddings.csv")
    ap.add_argument("--output", default="data/processed/atomic_embeddings_curated.csv")
    ap.add_argument("--summary", default="reports/atomic_embeddings_curation_summary.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    inp = (root / args.input).resolve()
    out = (root / args.output).resolve()
    rep = (root / args.summary).resolve()

    df = pd.read_csv(inp)
    curated, summary = curate_atomic_embeddings(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    rep.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(out, index=False)
    rep.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] curated embeddings: {out}")
    print(f"[ok] summary: {rep}")
    print(f"[ok] rows_out={summary['rows_out']} near_zero={summary['near_zero_embeddings']}")


if __name__ == "__main__":
    main()
