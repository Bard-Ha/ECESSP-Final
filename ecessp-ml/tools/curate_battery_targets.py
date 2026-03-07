#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TARGET_COLS = [
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "stability_charge",
    "stability_discharge",
    "max_delta_volume",
]
NORM_TARGET_COLS = [
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "stability_charge",
    "stability_discharge",
    "max_delta_volume",
]


def _winsor_bounds(series: pd.Series, q_lo: float = 0.005, q_hi: float = 0.995) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce")
    lo = float(s.quantile(q_lo))
    hi = float(s.quantile(q_hi))
    if not np.isfinite(lo):
        lo = float(np.nanmin(s.values))
    if not np.isfinite(hi):
        hi = float(np.nanmax(s.values))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _clip_with_flag(df: pd.DataFrame, col: str, lo: float, hi: float, flag_prefix: str) -> pd.Series:
    raw = pd.to_numeric(df[col], errors="coerce")
    clipped = raw.clip(lower=lo, upper=hi)
    df[f"{flag_prefix}_was_clipped"] = (raw != clipped).fillna(False)
    return clipped


def _safe_scale(x: float, floor: float = 1e-6) -> float:
    v = float(x)
    if not np.isfinite(v):
        return 1.0
    return max(abs(v), floor)


def _fit_affine_normalizer(series: pd.Series) -> Dict[str, float | str]:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s.to_numpy(dtype=np.float64))]
    if len(s) == 0:
        return {
            "method": "standard_z",
            "center": 0.0,
            "scale": 1.0,
            "mean": 0.0,
            "std": 1.0,
            "median": 0.0,
            "iqr": 1.0,
            "skew": 0.0,
        }

    mean = float(s.mean())
    std = float(s.std(ddof=0))
    median = float(s.median())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = float(q3 - q1)
    skew = float(s.skew())

    # Column-adaptive affine normalization policy.
    # Strongly skewed distributions use robust stats; near-symmetric use mean/std.
    if abs(skew) > 1.0 or iqr <= 0.0:
        method = "robust_z"
        center = median
        scale = _safe_scale(iqr / 1.349 if iqr > 0.0 else std)
    else:
        method = "standard_z"
        center = mean
        scale = _safe_scale(std)

    return {
        "method": method,
        "center": float(center),
        "scale": float(scale),
        "mean": mean,
        "std": std,
        "median": median,
        "iqr": iqr,
        "skew": skew,
    }


def curate(parsed_df: pd.DataFrame, ml_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    df = parsed_df.copy()
    ml = ml_df.copy()

    if len(df) != len(ml):
        raise RuntimeError(f"Row mismatch parsed vs ml: {len(df)} vs {len(ml)}")

    for col in TARGET_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column in parsed dataset: {col}")

    if "battery_id" not in ml.columns and "battery_id" in df.columns:
        ml["battery_id"] = df["battery_id"].astype(str)

    # Keep originals for traceability.
    df["energy_grav_reported"] = pd.to_numeric(df["energy_grav"], errors="coerce")
    df["energy_vol_reported"] = pd.to_numeric(df["energy_vol"], errors="coerce")

    # Hard physical clips + winsorization.
    df["average_voltage"] = _clip_with_flag(df, "average_voltage", 0.5, 6.0, "average_voltage")
    for col in ["capacity_grav", "capacity_vol", "stability_charge", "stability_discharge", "max_delta_volume"]:
        lo, hi = _winsor_bounds(df[col], 0.005, 0.995)
        if col in {"capacity_grav", "capacity_vol", "max_delta_volume"}:
            lo = max(0.0, lo)
        if col in {"stability_charge", "stability_discharge"}:
            lo = max(0.0, lo)
        df[col] = _clip_with_flag(df, col, lo, hi, col)

    # Target redesign: treat energies as derived targets from curated voltage and capacity.
    expected_energy_grav = pd.to_numeric(df["average_voltage"], errors="coerce") * pd.to_numeric(df["capacity_grav"], errors="coerce")
    expected_energy_vol = pd.to_numeric(df["average_voltage"], errors="coerce") * pd.to_numeric(df["capacity_vol"], errors="coerce")

    rel_err_g = (df["energy_grav_reported"] - expected_energy_grav).abs() / np.maximum(expected_energy_grav.abs(), 1.0)
    rel_err_v = (df["energy_vol_reported"] - expected_energy_vol).abs() / np.maximum(expected_energy_vol.abs(), 1.0)
    df["energy_grav_inconsistent"] = rel_err_g > 0.35
    df["energy_vol_inconsistent"] = rel_err_v > 0.35
    df["energy_grav_negative"] = pd.to_numeric(df["energy_grav_reported"], errors="coerce") < 0.0
    df["energy_vol_negative"] = pd.to_numeric(df["energy_vol_reported"], errors="coerce") < 0.0

    df["energy_grav"] = expected_energy_grav
    df["energy_vol"] = expected_energy_vol

    # Row-level noise indicator for future robust training/sample weighting.
    correction_flags = [
        "average_voltage_was_clipped",
        "capacity_grav_was_clipped",
        "capacity_vol_was_clipped",
        "stability_charge_was_clipped",
        "stability_discharge_was_clipped",
        "max_delta_volume_was_clipped",
        "energy_grav_inconsistent",
        "energy_vol_inconsistent",
        "energy_grav_negative",
        "energy_vol_negative",
    ]
    for flag in correction_flags:
        if flag not in df.columns:
            df[flag] = False

    df["target_curation_corrections"] = df[correction_flags].astype(int).sum(axis=1)
    df["target_noise_band"] = np.select(
        [
            df["target_curation_corrections"] <= 1,
            df["target_curation_corrections"] <= 3,
        ],
        ["low", "medium"],
        default="high",
    )
    df["target_sample_weight"] = np.clip(np.exp(-0.35 * df["target_curation_corrections"]), 0.2, 1.0)

    ml["target_curation_corrections"] = df["target_curation_corrections"].to_numpy()
    ml["target_noise_band"] = df["target_noise_band"].to_numpy()
    ml["target_sample_weight"] = df["target_sample_weight"].to_numpy()

    # Recompute normalized columns from curated parsed targets using
    # column-adaptive affine normalizers.
    normalization_specs: Dict[str, Dict[str, float | str]] = {}
    for col in NORM_TARGET_COLS:
        if col not in df.columns:
            continue
        spec = _fit_affine_normalizer(df[col])
        normalization_specs[col] = spec
        norm_col = f"{col}_norm"
        center = float(spec["center"])
        scale = float(spec["scale"])
        ml[norm_col] = ((pd.to_numeric(df[col], errors="coerce") - center) / scale).astype(float)

    summary = {
        "rows": int(len(df)),
        "corrections": {
            flag: int(df[flag].sum()) for flag in correction_flags
        },
        "noise_band_counts": {k: int(v) for k, v in df["target_noise_band"].value_counts().to_dict().items()},
        "target_quantiles_after_curation": {
            c: {
                "q01": float(pd.to_numeric(df[c], errors="coerce").quantile(0.01)),
                "q50": float(pd.to_numeric(df[c], errors="coerce").quantile(0.50)),
                "q99": float(pd.to_numeric(df[c], errors="coerce").quantile(0.99)),
            }
            for c in TARGET_COLS
        },
        "normalization_policy": {
            "type": "column_adaptive_affine",
            "normalized_columns": {f"{k}_norm": v for k, v in normalization_specs.items()},
        },
    }
    return df, ml, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate battery targets with physics-consistent redesign.")
    ap.add_argument("--parsed-in", default="data/processed/batteries_parsed.csv")
    ap.add_argument("--ml-in", default="data/processed/batteries_ml.csv")
    ap.add_argument("--parsed-out", default="data/processed/batteries_parsed_curated.csv")
    ap.add_argument("--ml-out", default="data/processed/batteries_ml_curated.csv")
    ap.add_argument("--summary-out", default="reports/data_curation_summary.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed_in = (root / args.parsed_in).resolve()
    ml_in = (root / args.ml_in).resolve()
    parsed_out = (root / args.parsed_out).resolve()
    ml_out = (root / args.ml_out).resolve()
    summary_out = (root / args.summary_out).resolve()

    parsed_df = pd.read_csv(parsed_in)
    ml_df = pd.read_csv(ml_in)
    parsed_cur, ml_cur, summary = curate(parsed_df, ml_df)

    parsed_out.parent.mkdir(parents=True, exist_ok=True)
    ml_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    parsed_cur.to_csv(parsed_out, index=False)
    ml_cur.to_csv(ml_out, index=False)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] parsed curated -> {parsed_out}")
    print(f"[ok] ml curated -> {ml_out}")
    print(f"[ok] summary -> {summary_out}")
    print(f"[ok] noise bands: {summary['noise_band_counts']}")


if __name__ == "__main__":
    main()
