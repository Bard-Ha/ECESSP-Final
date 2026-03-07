#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

from models.masked_gnn import MaskedGNN, PROPERTY_INDEX
from design.physics_chemistry import solve_oxidation_states, theoretical_capacity_mAh_per_g

TARGET_ORDER = [
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "stability_charge",
    "stability_discharge",
]

@dataclass
class AffineMap:
    scale: float
    shift: float
    r2: float
    def to_norm(self, x): return self.scale * x + self.shift
    def to_raw(self, y): return y if abs(self.scale) < 1e-12 else (y - self.shift) / self.scale

class NodeDataset(Dataset):
    def __init__(self, X, M, mask, C_theo, n_e, ion, comp, ctx, role_targets, compatibility_targets, sample_weight):
        self.X, self.M, self.mask = X, M, mask
        self.C_theo, self.n_e, self.ion, self.comp = C_theo, n_e, ion, comp
        self.ctx = ctx
        self.role_targets = role_targets
        self.compatibility_targets = compatibility_targets
        self.sample_weight = sample_weight
    def __len__(self): return int(self.X.size(0))
    def __getitem__(self, i):
        return {
            'x': self.X[i], 'm': self.M[i], 'mask': self.mask[i], 'y': self.X[i],
            'c_theo': self.C_theo[i], 'n_e': self.n_e[i], 'ion': self.ion[i], 'comp': self.comp[i],
            'role_target': self.role_targets[i], 'compat_target': self.compatibility_targets[i],
            'sample_weight': self.sample_weight[i],
            'ctx': self.ctx[i], 'idx': torch.tensor(i, dtype=torch.long),
        }

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _safe_text_series(df: pd.DataFrame, column: str, fallback: str) -> pd.Series:
    if column in df.columns:
        s = df[column]
    else:
        s = pd.Series([fallback] * len(df))
    s = s.fillna(fallback).astype(str).str.strip()
    return s.replace("", fallback)

def _build_group_labels(parsed_df: pd.DataFrame, group_keys: List[str]) -> np.ndarray:
    if not group_keys:
        group_keys = ["chemsys", "framework_formula", "working_ion"]
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

    train_idx = np.array(sorted(buckets[0]), dtype=np.int64)
    val_idx = np.array(sorted(buckets[1]), dtype=np.int64)
    test_idx = np.array(sorted(buckets[2]), dtype=np.int64)
    return train_idx, val_idx, test_idx

def build_benchmark_split_indices(parsed_df: pd.DataFrame, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    n = int(len(parsed_df))
    all_idx = np.arange(n, dtype=np.int64)
    split_cfg = cfg.get("split_protocol", {}) if isinstance(cfg, dict) else {}
    group_keys = list(split_cfg.get("group_keys", ["chemsys", "framework_formula", "working_ion"]))
    train_ratio = float(split_cfg.get("train_ratio", cfg.get("train_ratio", 0.7)))
    val_ratio = float(split_cfg.get("val_ratio", cfg.get("val_ratio", 0.15)))

    ood_cfg = split_cfg.get("ood", {}) if isinstance(split_cfg, dict) else {}
    ood_enabled = bool(ood_cfg.get("enabled", True))
    min_ion_count = int(ood_cfg.get("min_ion_count", 120))
    holdout_ion_cfg = str(ood_cfg.get("holdout_ion", "auto"))
    holdout_chemsys_fraction = float(ood_cfg.get("holdout_chemsys_fraction", 0.08))
    holdout_chemsys_fraction = float(np.clip(holdout_chemsys_fraction, 0.0, 0.5))

    ion_series = _safe_text_series(parsed_df, "working_ion", "unknown")
    chemsys_series = _safe_text_series(parsed_df, "chemsys", "unknown")

    ion_counts = ion_series.value_counts()
    rng = np.random.default_rng(seed + 17)
    holdout_ion = "none"
    ood_ion_mask = np.zeros(n, dtype=bool)
    if ood_enabled and len(ion_counts):
        candidates = [ion for ion, c in ion_counts.items() if int(c) >= min_ion_count]
        if holdout_ion_cfg.lower() != "auto":
            holdout_ion = holdout_ion_cfg
        else:
            major = str(ion_counts.index[0])
            non_major = [ion for ion in candidates if str(ion) != major]
            pool = non_major or candidates or [str(ion_counts.index[-1])]
            holdout_ion = str(pool[int(rng.integers(0, len(pool)))])
        ood_ion_mask = (ion_series.to_numpy() == holdout_ion)

    ood_chemsys: List[str] = []
    ood_chemsys_mask = np.zeros(n, dtype=bool)
    if ood_enabled and holdout_chemsys_fraction > 0.0:
        remaining_chemsys = chemsys_series.to_numpy()[~ood_ion_mask]
        unique_chemsys = np.unique(remaining_chemsys)
        if unique_chemsys.size > 0:
            k = int(round(float(unique_chemsys.size) * holdout_chemsys_fraction))
            k = max(1, min(int(unique_chemsys.size), k))
            picked = rng.choice(unique_chemsys, size=k, replace=False)
            ood_chemsys = sorted([str(x) for x in picked.tolist()])
            ood_chemsys_mask = np.isin(chemsys_series.to_numpy(), np.array(ood_chemsys, dtype=object))

    ood_mask = (ood_ion_mask | ood_chemsys_mask) if ood_enabled else np.zeros(n, dtype=bool)
    iid_pool = all_idx[~ood_mask]
    group_labels = _build_group_labels(parsed_df, group_keys)
    train_idx, val_idx, test_iid_idx = _grouped_partition(
        indices=iid_pool,
        group_labels=group_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    ood_idx = all_idx[ood_mask]

    if test_iid_idx.size == 0 and val_idx.size > 0:
        cut = max(1, int(round(0.5 * val_idx.size)))
        test_iid_idx = val_idx[-cut:]
        val_idx = val_idx[:-cut]

    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test_iid": test_iid_idx.tolist(),
        "test_ood": ood_idx.tolist(),
        "metadata": {
            "protocol_name": str(split_cfg.get("name", "grouped_iid_plus_ood_v1")),
            "group_keys": group_keys,
            "train_ratio_target": train_ratio,
            "val_ratio_target": val_ratio,
            "ood_enabled": ood_enabled,
            "holdout_ion": holdout_ion,
            "holdout_chemsys_fraction": holdout_chemsys_fraction,
            "holdout_chemsys_count": int(len(ood_chemsys)),
            "holdout_chemsys_preview": ood_chemsys[:20],
            "counts": {
                "total": n,
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test_iid": int(len(test_iid_idx)),
                "test_ood": int(len(ood_idx)),
            },
        },
    }

def _sanitize_indices(raw_indices: Any, n: int) -> List[int]:
    if raw_indices is None:
        return []
    out: list[int] = []
    for v in list(raw_indices):
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            out.append(i)
    return sorted(set(out))

def load_benchmark_split_indices(path: Path, n: int) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    train = _sanitize_indices(payload.get("train"), n)
    val = _sanitize_indices(payload.get("val"), n)
    test_iid = _sanitize_indices(payload.get("test_iid"), n)
    test_ood = _sanitize_indices(payload.get("test_ood"), n)
    used = set(train) | set(val) | set(test_iid) | set(test_ood)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not train or not val or not test_iid:
        raise RuntimeError(f"Invalid split file (empty train/val/test_iid): {path}")
    if len(used) != (len(train) + len(val) + len(test_iid) + len(test_ood)):
        raise RuntimeError(f"Split file has overlapping indices: {path}")
    return {
        "train": train,
        "val": val,
        "test_iid": test_iid,
        "test_ood": test_ood,
        "metadata": {
            "protocol_name": str(metadata.get("protocol_name", "external_benchmark_split")),
            "group_keys": list(metadata.get("group_keys", ["chemsys", "framework_formula", "working_ion"])),
            "ood_enabled": bool(metadata.get("ood_enabled", len(test_ood) > 0)),
            "holdout_ion": str(metadata.get("holdout_ion", "unknown")),
            "holdout_chemsys_fraction": float(metadata.get("holdout_chemsys_fraction", 0.0)),
            "holdout_chemsys_count": int(metadata.get("holdout_chemsys_count", 0)),
            "holdout_chemsys_preview": list(metadata.get("holdout_chemsys_preview", [])),
            "counts": {
                "total": int(n),
                "train": int(len(train)),
                "val": int(len(val)),
                "test_iid": int(len(test_iid)),
                "test_ood": int(len(test_ood)),
            },
            "source": str(path),
        },
    }

def build_graph_context_features(graph: Dict[str, Any], X: torch.Tensor) -> torch.Tensor:
    n, d = int(X.size(0)), int(X.size(1))
    if "fused_system_edge_index" in graph and "fused_system_edge_weight" in graph:
        edge_index = graph["fused_system_edge_index"].long()
        edge_weight = graph["fused_system_edge_weight"].float().view(-1)
    else:
        edge_index = None
        for k in sorted(graph.get("edge_index_dict", {}).keys()):
            ei = graph["edge_index_dict"][k]
            if isinstance(ei, torch.Tensor) and ei.numel():
                edge_index = ei.long()
                break
        if edge_index is None:
            return X.clone()
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    if edge_index.numel() == 0:
        return X.clone()

    src = edge_index[0].long().clamp(0, n - 1)
    dst = edge_index[1].long().clamp(0, n - 1)
    w = edge_weight[: src.size(0)].float().view(-1, 1)

    ctx = torch.zeros((n, d), dtype=X.dtype)
    denom = torch.zeros((n, 1), dtype=X.dtype)
    ctx.index_add_(0, src, X[dst] * w)
    denom.index_add_(0, src, w)
    ctx = ctx / denom.clamp_min(1e-6)

    no_nbr = (denom.view(-1) <= 0)
    if no_nbr.any():
        ctx[no_nbr] = X[no_nbr]
    return ctx

def fit_map(raw, norm) -> AffineMap:
    raw = raw.astype(np.float64); norm = norm.astype(np.float64)
    x = np.vstack([raw, np.ones_like(raw)]).T
    c, *_ = np.linalg.lstsq(x, norm, rcond=None)
    a, b = float(c[0]), float(c[1])
    p = a * raw + b
    ss_res = float(np.sum((norm - p) ** 2)); ss_tot = float(np.sum((norm - np.mean(norm)) ** 2))
    return AffineMap(a, b, 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

def build_maps(parsed_df: pd.DataFrame, norm_df: pd.DataFrame) -> Dict[str, AffineMap]:
    pairs = [
        ('average_voltage', 'average_voltage_norm'),
        ('capacity_grav', 'capacity_grav_norm'),
        ('capacity_vol', 'capacity_vol_norm'),
        ('energy_grav', 'energy_grav_norm'),
        ('energy_vol', 'energy_vol_norm'),
        ('stability_charge', 'stability_charge_norm'),
        ('stability_discharge', 'stability_discharge_norm'),
    ]
    return {r: fit_map(parsed_df[r].to_numpy(), norm_df[n].to_numpy()) for r, n in pairs}

def build_maps_from_graph_targets(parsed_df: pd.DataFrame, battery_features: torch.Tensor) -> Dict[str, AffineMap]:
    """
    Alternative map builder when user selects parsed-only source.
    Uses graph battery_features columns as normalized targets.
    """
    pairs = [
        ("average_voltage", 0),
        ("capacity_grav", 1),
        ("capacity_vol", 2),
        ("energy_grav", 3),
        ("energy_vol", 4),
        ("stability_charge", 5),
        ("stability_discharge", 6),
    ]
    maps = {}
    bf_np = battery_features.detach().cpu().numpy()
    for raw_col, idx in pairs:
        maps[raw_col] = fit_map(parsed_df[raw_col].to_numpy(), bf_np[:, idx])
    return maps

def build_formula_features(parsed_df: pd.DataFrame, cap_map: AffineMap):
    n = len(parsed_df)
    c_theo = np.zeros(n, dtype=np.float32)
    n_e = np.zeros(n, dtype=np.float32)
    ion_code = np.zeros(n, dtype=np.int64)
    ion_map = {'Li': 0, 'Na': 1, 'K': 2, 'Mg': 3, 'Ca': 4, 'Zn': 5, 'other': 6}
    comps: List[Dict[str, float]] = []
    for i, row in parsed_df.iterrows():
        ion = str(row.get('working_ion') or '')
        ion_code[i] = ion_map.get(ion, ion_map['other'])
        comp, c_raw, ne = {}, 0.0, 0.0
        f = str(row.get('battery_formula') or '')
        if f:
            s = solve_oxidation_states(f)
            if s.valid:
                comp = s.composition
                ne = float(s.n_electrons_max or 0.0)
                if s.molar_mass and s.molar_mass > 0 and ne > 0:
                    try: c_raw = theoretical_capacity_mAh_per_g(ne, s.molar_mass)
                    except Exception: c_raw = 0.0
        comps.append(comp); n_e[i] = ne; c_theo[i] = float(cap_map.to_norm(c_raw))
    vocab = sorted({el for c in comps for el in c.keys()})
    vidx = {e: j for j, e in enumerate(vocab)}
    comp_vec = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, comp in enumerate(comps):
        s = float(sum(comp.values()))
        if s <= 0: continue
        for el, v in comp.items(): comp_vec[i, vidx[el]] = float(v) / s
    return torch.tensor(c_theo), torch.tensor(n_e), torch.tensor(ion_code), torch.tensor(comp_vec), vocab


def build_role_targets(graph: Dict[str, Any], parsed_df: pd.DataFrame) -> torch.Tensor:
    """
    Multi-label role targets: [cathode, anode, electrolyte_candidate].
    """
    n = int(len(parsed_df))
    out = np.zeros((n, 3), dtype=np.float32)
    role_assignments = graph.get("role_assignments", [])

    def _mark(role_text: str, vec: np.ndarray) -> None:
        t = str(role_text or "").strip().lower()
        if not t:
            return
        if "cathode" in t:
            vec[0] = 1.0
        elif "anode" in t:
            vec[1] = 1.0
        elif "electrolyte" in t or "separator" in t or "additive" in t:
            vec[2] = 1.0

    if isinstance(role_assignments, list) and len(role_assignments) == n:
        for i, entry in enumerate(role_assignments):
            if isinstance(entry, dict):
                for _, role_name in entry.items():
                    _mark(str(role_name), out[i])
            elif isinstance(entry, (list, tuple, set)):
                for role_name in entry:
                    _mark(str(role_name), out[i])
            else:
                _mark(str(entry), out[i])

    # Deterministic fallback when graph role metadata is sparse.
    voltage_series = pd.to_numeric(parsed_df.get("average_voltage"), errors="coerce").fillna(0.0).to_numpy()
    for i in range(n):
        if out[i].sum() <= 0:
            out[i, 0] = 1.0 if voltage_series[i] >= 3.0 else 0.0
            out[i, 1] = 1.0 if voltage_series[i] < 3.0 else 1.0
        # Keep electrolyte candidate soft-present in training to avoid single-task collapse.
        if out[i, 2] <= 0:
            out[i, 2] = 0.25

    return torch.tensor(out, dtype=torch.float32)


def build_compatibility_targets(parsed_df: pd.DataFrame) -> torch.Tensor:
    """
    Pseudo compatibility targets in [0,1]:
    [voltage_window_overlap_score, chemical_stability_score, mechanical_strain_risk]
    """
    n = int(len(parsed_df))
    voltage = pd.to_numeric(parsed_df.get("average_voltage"), errors="coerce").fillna(3.0).to_numpy()
    st_c = pd.to_numeric(parsed_df.get("stability_charge"), errors="coerce").fillna(0.0).to_numpy()
    st_d = pd.to_numeric(parsed_df.get("stability_discharge"), errors="coerce").fillna(0.0).to_numpy()
    dvol = pd.to_numeric(parsed_df.get("max_delta_volume"), errors="coerce").fillna(0.1).to_numpy()

    vscore = np.clip((voltage - 1.0) / (4.4 - 1.0), 0.0, 1.0)
    chem = np.exp(-np.clip(np.abs(st_c) + np.abs(st_d), 0.0, 6.0))
    strain_risk = np.clip(dvol / 0.25, 0.0, 1.0)

    targets = np.stack([vscore, chem, strain_risk], axis=1).astype(np.float32)
    if targets.shape != (n, 3):
        raise RuntimeError("compatibility targets shape mismatch")
    return torch.tensor(targets, dtype=torch.float32)

def ood_ref(M: torch.Tensor, mask: torch.Tensor):
    rep = (M * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mu = rep.mean(dim=0)
    c = rep - mu.unsqueeze(0)
    if c.size(0) < 2:
        cov = 1e-3 * torch.eye(c.size(1), dtype=rep.dtype)
    else:
        cov = torch.cov(c.T) + 1e-4 * torch.eye(c.size(1), dtype=rep.dtype)
    return mu, torch.inverse(cov)

def maha(rep, mu, inv_cov):
    d = rep - mu.unsqueeze(0)
    l = torch.matmul(d, inv_cov)
    return torch.sqrt((l * d).sum(dim=1).clamp_min(0) + 1e-12)

def affine_to_raw_t(x: torch.Tensor, m: AffineMap) -> torch.Tensor:
    if abs(float(m.scale)) < 1e-12:
        return x
    return (x - float(m.shift)) / float(m.scale)

def affine_to_norm_t(x: torch.Tensor, m: AffineMap) -> torch.Tensor:
    return x * float(m.scale) + float(m.shift)

def deterministic_cell(
    pred: torch.Tensor,
    maps: Dict[str, AffineMap],
    np_rng: Tuple[float, float] = (1.05, 1.15),
    am_rng: Tuple[float, float] = (0.6, 0.75),
    lim: Optional[Dict[str, float]] = None,
):
    out = pred.clone()
    iv, icg, icv, ieg, iev = PROPERTY_INDEX['average_voltage'], PROPERTY_INDEX['capacity_grav'], PROPERTY_INDEX['capacity_vol'], PROPERTY_INDEX['energy_grav'], PROPERTY_INDEX['energy_vol']
    isc, isd = PROPERTY_INDEX['stability_charge'], PROPERTY_INDEX['stability_discharge']
    v_pre = affine_to_raw_t(pred[:, iv], maps['average_voltage'])
    cg_pre = affine_to_raw_t(pred[:, icg], maps['capacity_grav'])
    cv_pre = affine_to_raw_t(pred[:, icv], maps['capacity_vol'])

    v_min = float((lim or {}).get('min_voltage', 1.0))
    v_max = float((lim or {}).get('max_voltage', 4.4))
    cap_max = float((lim or {}).get('max_capacity_grav', 350.0))
    egrav_max = float((lim or {}).get('max_energy_grav', 450.0))
    evol_max = float((lim or {}).get('max_energy_vol', 1200.0))

    # Explicit hard clamps to prevent physically impossible outputs.
    v = torch.clamp(v_pre, min=v_min, max=v_max)
    cg = torch.clamp(cg_pre, min=0.0, max=cap_max)
    cv = torch.clamp(cv_pre, min=0.0)

    # Keep a pre-clamp path for penalties while enforcing bounded deployment outputs.
    np_ratio_pre = np_rng[0] + (np_rng[1] - np_rng[0]) * pred[:, isc]
    amf_pre = am_rng[0] + (am_rng[1] - am_rng[0]) * pred[:, isd]
    np_ratio = torch.clamp(np_ratio_pre, min=np_rng[0], max=np_rng[1])
    amf = torch.clamp(amf_pre, min=am_rng[0], max=am_rng[1])

    ca_g, ca_v = torch.clamp(cg * 1.2, min=1.0), torch.clamp(cv * 1.2, min=1.0)
    cc_g, cc_v = torch.clamp(cg, min=1.0), torch.clamp(cv, min=1.0)
    ccell_g = 1.0 / (1.0 / cc_g + np_ratio / ca_g)
    ccell_v = 1.0 / (1.0 / cc_v + np_ratio / ca_v)
    eg_pre = v * ccell_g * amf
    ev_pre = v * ccell_v * amf
    eg = torch.clamp(eg_pre, min=0.0, max=egrav_max)
    ev = torch.clamp(ev_pre, min=0.0, max=evol_max)
    out[:, ieg] = affine_to_norm_t(eg, maps['energy_grav'])
    out[:, iev] = affine_to_norm_t(ev, maps['energy_vol'])
    return out, {
        'v': v, 'cg': cg, 'cv': cv, 'eg': eg, 'ev': ev,
        'v_pre': v_pre, 'cg_pre': cg_pre, 'cv_pre': cv_pre, 'eg_pre': eg_pre, 'ev_pre': ev_pre,
        'np_ratio': np_ratio, 'np_ratio_pre': np_ratio_pre,
        'amf': amf, 'amf_pre': amf_pre,
        'ccell_g': ccell_g, 'ccell_v': ccell_v,
    }


def capacity_violation_rate(
    pred_cg_raw: np.ndarray,
    c_theoretical_raw: np.ndarray,
    *,
    max_capacity_grav: float,
) -> tuple[float, float]:
    """
    Robust capacity violation metric.

    Uses theoretical limit when valid; otherwise falls back to global hard cap.
    Returns:
      (violation_rate, theoretical_valid_fraction)
    """
    pred = np.asarray(pred_cg_raw, dtype=np.float64)
    theo = np.asarray(c_theoretical_raw, dtype=np.float64)
    if pred.size == 0 or theo.size == 0:
        return 0.0, 0.0
    n = min(pred.size, theo.size)
    pred = pred[:n]
    theo = theo[:n]
    valid = np.isfinite(theo) & (theo > 1.0)
    cap_lim = np.where(valid, np.minimum(theo, float(max_capacity_grav)), float(max_capacity_grav))
    rate = float(np.mean(pred > cap_lim)) if n > 0 else 0.0
    valid_frac = float(np.mean(valid)) if n > 0 else 0.0
    return rate, valid_frac

def loss_terms(
    pred,
    y,
    aux,
    c_theo_norm,
    n_e,
    ion,
    comp,
    train_comp,
    rep,
    mu,
    inv_cov,
    maps,
    w,
    lim,
    ood_thresh,
    pred_logvar=None,
    role_logits=None,
    role_targets=None,
    compatibility_scores=None,
    compatibility_targets=None,
    sample_weight=None,
):
    sw = None
    sw_col = None
    if sample_weight is not None:
        sw = torch.clamp(sample_weight.view(-1), min=1e-6).to(device=pred.device, dtype=pred.dtype)
        sw = sw / sw.mean().clamp_min(1e-6)
        sw_col = sw.view(-1, 1)
    if sw_col is not None:
        mse = (((pred - y).pow(2)) * sw_col).sum() / (sw_col.sum() * float(pred.size(1)))
    else:
        mse = F.mse_loss(pred, y)
    iv, icg = PROPERTY_INDEX['average_voltage'], PROPERTY_INDEX['capacity_grav']

    tw_cfg = w.get('regression_target_weights', {})
    target_weights = torch.tensor(
        [
            float(tw_cfg.get('average_voltage', 1.0)),
            float(tw_cfg.get('capacity_grav', 1.0)),
            float(tw_cfg.get('capacity_vol', 1.0)),
            float(tw_cfg.get('energy_grav', 1.0)),
            float(tw_cfg.get('energy_vol', 1.0)),
            float(tw_cfg.get('stability_charge', 1.0)),
            float(tw_cfg.get('stability_discharge', 1.0)),
        ],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, -1)
    if sw_col is not None:
        weighted_mse = (((pred - y).pow(2) * target_weights) * sw_col).sum() / (sw_col.sum() * float(pred.size(1)))
    else:
        weighted_mse = ((pred - y).pow(2) * target_weights).mean()
    cg_beta = max(1e-4, float(w.get('capacity_grav_huber_beta', 0.2)))
    cg_huber_raw = F.smooth_l1_loss(pred[:, icg], y[:, icg], beta=cg_beta, reduction='none')
    if sw is not None:
        cg_huber = (cg_huber_raw * sw).sum() / sw.sum()
    else:
        cg_huber = cg_huber_raw.mean()
    regression_loss = weighted_mse + float(w.get('capacity_grav_huber_weight', 0.0)) * cg_huber

    t_v = affine_to_raw_t(y[:, iv], maps['average_voltage'])
    c_theo = affine_to_raw_t(c_theo_norm, maps['capacity_grav'])
    # Robust theoretical-capacity gating:
    # many rows may not yield a valid oxidation-state-derived C_theo.
    # When invalid, fall back to global hard cap instead of collapsing to ~0.
    cap_hard = torch.full_like(c_theo, float(lim['max_capacity_grav']))
    theo_valid = torch.isfinite(c_theo) & (c_theo > 1.0)
    cap_lim = torch.where(theo_valid, torch.minimum(c_theo, cap_hard), cap_hard)
    cap_pen = torch.relu(aux['cg_pre'] - cap_lim).pow(2).mean()

    li = (ion == 0).float(); na = (ion == 1).float()
    max_v = li * 4.4 + na * 4.3 + (1.0 - li - na).clamp_min(0.0) * float(lim['max_voltage'])
    volt_pen = (torch.relu(aux['v_pre'] - max_v).pow(2) + torch.relu(torch.full_like(max_v, lim['min_voltage']) - aux['v_pre']).pow(2)).mean()

    ecap_pen = (
        torch.relu(aux['eg_pre'] - lim['max_energy_grav']).pow(2).mean()
        + torch.relu(aux['ev_pre'] - lim['max_energy_vol']).pow(2).mean()
    )
    econs = (aux['eg'] - (aux['v'] * aux['ccell_g'] * aux['amf'])).pow(2).mean()
    thermo = (((aux['v'] - t_v) ** 2) * (n_e + 1.0)).mean()
    mb_pen = (
        torch.relu(torch.tensor(1.05, device=pred.device) - aux['np_ratio_pre']).pow(2).mean()
        + torch.relu(aux['np_ratio_pre'] - torch.tensor(1.15, device=pred.device)).pow(2).mean()
        + torch.relu(torch.tensor(0.6, device=pred.device) - aux['amf_pre']).pow(2).mean()
        + torch.relu(aux['amf_pre'] - torch.tensor(0.75, device=pred.device)).pow(2).mean()
    )

    d = maha(rep, mu.to(rep.device), inv_cov.to(rep.device))
    ood_pen = torch.relu(d - float(ood_thresh)).pow(2).mean()

    if comp.numel() and train_comp.numel():
        q = F.normalize(comp, dim=1)
        t = F.normalize(train_comp.to(q.device), dim=1)
        max_sim, _ = torch.matmul(q, t.T).max(dim=1)
        novelty = 1.0 - max_sim
    else:
        novelty = torch.zeros(pred.size(0), device=pred.device)

    if pred_logvar is not None:
        unc_reg = torch.exp(torch.clamp(pred_logvar, min=-8.0, max=8.0)).mean()
    else:
        unc_reg = torch.tensor(0.0, device=pred.device)
    if role_logits is not None and role_targets is not None and role_logits.numel() > 0:
        role_raw = F.binary_cross_entropy_with_logits(role_logits, role_targets, reduction='none').mean(dim=1)
        if sw is not None:
            role_loss = (role_raw * sw).sum() / sw.sum()
        else:
            role_loss = role_raw.mean()
    else:
        role_loss = torch.tensor(0.0, device=pred.device)
    if compatibility_scores is not None and compatibility_targets is not None and compatibility_scores.numel() > 0:
        compat_raw = (compatibility_scores - compatibility_targets).pow(2).mean(dim=1)
        if sw is not None:
            compatibility_loss = (compat_raw * sw).sum() / sw.sum()
        else:
            compatibility_loss = compat_raw.mean()
    else:
        compatibility_loss = torch.tensor(0.0, device=pred.device)
    novelty_weight = float(w.get('novelty_weight', 0.05))

    total = (
        w['regression_mse'] * regression_loss
        + w['capacity_violation_penalty'] * cap_pen
        + w['voltage_violation_penalty'] * volt_pen
        + w['energy_cap_violation_penalty'] * ecap_pen
        + float(w.get('physics_consistency_loss', 0.0)) * econs
        + w['thermodynamic_consistency_loss'] * thermo
        + w['mass_balance_penalty'] * mb_pen
        + w['ood_penalty'] * ood_pen
        + w['uncertainty_regularization'] * unc_reg
        + float(w.get('role_loss_weight', 0.0)) * role_loss
        + float(w.get('compatibility_loss_weight', 0.0)) * compatibility_loss
        - novelty_weight * novelty.mean()
    )
    m = {
        'total': total, 'mse': mse, 'regression_loss': regression_loss, 'capacity_grav_huber': cg_huber,
        'capacity_violation': cap_pen, 'voltage_violation': volt_pen,
        'energy_cap_violation': ecap_pen, 'energy_consistency': econs,
        'thermodynamic_consistency_loss': thermo, 'mass_balance_penalty': mb_pen,
        'ood_penalty': ood_pen, 'uncertainty_regularization': unc_reg,
        'role_loss': role_loss, 'compatibility_loss': compatibility_loss,
        'novelty_mean': novelty.mean(),
        'sample_weight_mean': (sw.mean() if sw is not None else torch.tensor(1.0, device=pred.device)),
    }
    a = {
        'pred_v_raw': aux['v'], 'pred_cg_raw': aux['cg'], 'pred_eg_raw': aux['eg'],
        'true_v_raw': t_v,
        'true_cg_raw': affine_to_raw_t(y[:, icg], maps['capacity_grav']),
        'c_theoretical_raw': c_theo,
        'c_theoretical_valid_rate': theo_valid.float().mean(),
        'ood_distance': d, 'novelty': novelty,
    }
    return total, m, a

def epoch_run(model, loader, device, maps, w, lim, mu, inv_cov, train_comp, ood_thresh, np_rng, am_rng, opt=None, feat_drop=0.0, feat_noise=0.0, store=False, amp_enabled=False):
    train = opt is not None
    model.train(train)
    sums, n = {}, 0
    skipped_non_finite = 0
    dump = {k: [] for k in ['pred_v_raw', 'pred_cg_raw', 'pred_eg_raw', 'true_v_raw', 'true_cg_raw', 'c_theoretical_raw', 'ood_distance', 'novelty']}
    use_amp = bool(amp_enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=bool(use_amp and train))
    for b in loader:
        x, M, mask, y = b['x'].to(device), b['m'].to(device), b['mask'].to(device), b['y'].to(device)
        ctx = b['ctx'].to(device)
        c_theo, n_e, ion, comp = b['c_theo'].to(device), b['n_e'].to(device), b['ion'].to(device), b['comp'].to(device)
        role_target = b['role_target'].to(device)
        compat_target = b['compat_target'].to(device)
        sample_weight = b['sample_weight'].to(device)
        xin = x.clone()
        if train:
            if feat_drop > 0:
                keep = (torch.rand_like(xin) > feat_drop).float(); xin = xin * keep
            if feat_noise > 0:
                xin = xin + torch.randn_like(xin) * feat_noise

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            mo = model(xin, M, mask, graph_context=ctx)
            pred0 = mo['properties']
            pred, aux = deterministic_cell(pred0, maps, np_rng=np_rng, am_rng=am_rng, lim=lim)
            rep = (M * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            total, metrics, extra = loss_terms(
                pred, y, aux, c_theo, n_e, ion, comp, train_comp, rep, mu, inv_cov, maps, w, lim,
                ood_thresh=ood_thresh, pred_logvar=mo.get('property_log_variance'),
                role_logits=mo.get('role_logits'), role_targets=role_target,
                compatibility_scores=mo.get('compatibility_scores'), compatibility_targets=compat_target,
                sample_weight=sample_weight,
            )
        if not torch.isfinite(total):
            skipped_non_finite += 1
            continue
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()

        bs = int(x.size(0)); n += bs
        for k, v in metrics.items(): sums[k] = sums.get(k, 0.0) + float(v.detach().cpu()) * bs
        if store:
            for k in dump: dump[k].append(extra[k].detach().cpu().numpy())

    avg = {k: v / max(1, n) for k, v in sums.items()}
    avg['skipped_non_finite_batches'] = float(skipped_non_finite)
    flat = {k: (np.concatenate(v) if v else np.array([])) for k, v in dump.items()}
    return avg, flat

def save_line(x, ys, title, out, ylabel='value'):
    plt.figure(figsize=(8, 6), dpi=300)
    for k, y in ys.items(): plt.plot(x, y, label=k)
    plt.title(title); plt.xlabel('epoch'); plt.ylabel(ylabel); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def save_scatter(x, y, title, out, xlabel, ylabel, diag=False):
    plt.figure(figsize=(8, 6), dpi=300); plt.scatter(x, y, s=8, alpha=0.5)
    if diag and len(x):
        lo, hi = float(min(np.min(x), np.min(y))), float(max(np.max(x), np.max(y))); plt.plot([lo, hi], [lo, hi], 'r--')
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out); plt.close()

def save_hist(data, title, out, xlabel):
    plt.figure(figsize=(8, 6), dpi=300)
    for k, v in data.items(): plt.hist(v, bins=40, alpha=0.45, label=k)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel('count'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def reliability(pred_std, abs_err, bins=10):
    if len(pred_std) == 0: return np.array([]), np.array([])
    q = np.linspace(0, 1, bins + 1); edges = np.quantile(pred_std, q)
    xs, ys = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pred_std >= lo) & ((pred_std <= hi) if i == bins - 1 else (pred_std < hi))
        if m.sum() < 5: continue
        xs.append(float(pred_std[m].mean())); ys.append(float(abs_err[m].mean()))
    return np.array(xs), np.array(ys)

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape_percent": 0.0, "r2": 0.0, "pearson_r": 0.0}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = 0.0
    return {"mae": mae, "rmse": rmse, "mape_percent": mape, "r2": r2, "pearson_r": pearson}

def per_property_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(TARGET_ORDER):
        out[name] = regression_metrics(y_true[:, i], y_pred[:, i])
    out["overall_micro"] = regression_metrics(y_true.reshape(-1), y_pred.reshape(-1))
    return out

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else 0.0

def mk_dirs(root: Path):
    d = {
        'root': root,
        'training_curves': root / 'training_curves',
        'physics_diagnostics': root / 'physics_diagnostics',
        'uncertainty_analysis': root / 'uncertainty_analysis',
        'distribution_analysis': root / 'distribution_analysis',
        'benchmark_validation': root / 'benchmark_validation',
    }
    for p in d.values(): p.mkdir(parents=True, exist_ok=True)
    return d

def load_model(path: Path, cfg, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = MaskedGNN(
        battery_feature_dim=int(cfg['battery_feature_dim']), material_embedding_dim=int(cfg['material_embedding_dim']),
        num_materials=int(cfg['num_materials']), hidden_dim=int(cfg['hidden_dim']), latent_dim=int(cfg['latent_dim']),
        num_gnn_layers=int(cfg['num_gnn_layers']), dropout=float(cfg['dropout']),
        decoder_mode=str(cfg.get('decoder_mode', 'legacy')),
        enable_uncertainty=bool(cfg.get('enable_uncertainty', False)),
        material_interaction=str(cfg.get('material_interaction', 'mlp')),
        enable_graph_context=bool(cfg.get('enable_graph_context', False)),
        graph_context_dim=int(cfg.get('graph_context_dim', int(cfg.get('battery_feature_dim', 7)))),
        enable_role_head=bool(cfg.get('enable_role_head', False)),
        enable_compatibility_head=bool(cfg.get('enable_compatibility_head', False)),
        role_output_dim=int(cfg.get('role_output_dim', 3)),
        compatibility_output_dim=int(cfg.get('compatibility_output_dim', 3)),
    ).to(device)
    m.load_state_dict(ck['model_state_dict'], strict=False); m.eval(); return m

def main():
    ap = argparse.ArgumentParser(description='Physics-first ensemble training + reporting')
    ap.add_argument('--config', default='reports/training_config.json')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--ensemble-size', type=int, default=5)
    ap.add_argument(
        '--allow-dev-shortcuts',
        action='store_true',
        help='Allow quick developer shortcuts (e.g., --dry-run or ensemble-size < 5).',
    )
    ap.add_argument('--feature-dropout', type=float, default=0.10)
    ap.add_argument('--feature-noise-std', type=float, default=0.02)
    ap.add_argument('--amp', action='store_true', help='Enable mixed precision on CUDA')
    ap.add_argument('--resume', default='', help='Resume from a specific checkpoint (recommended with --ensemble-size 1)')
    ap.add_argument(
        '--dataset-mode',
        choices=['ml', 'parsed', 'auto'],
        default='auto',
        help='ml: use batteries_ml.csv for normalization mapping; parsed: map parsed raw columns directly to graph targets.',
    )
    ap.add_argument(
        '--split-json',
        default='',
        help='Optional benchmark split JSON path; overrides config-driven split generation.',
    )
    args = ap.parse_args()

    # Production-by-default behavior:
    # - disable dry-run
    # - enforce ensemble size >= 5
    if not args.allow_dev_shortcuts:
        if args.dry_run:
            print('[mode] production default active: ignoring --dry-run (running full epochs).')
            args.dry_run = False
        if int(args.ensemble_size) < 5:
            print(f"[mode] production default active: raising ensemble-size {args.ensemble_size} -> 5.")
            args.ensemble_size = 5

    root = Path(__file__).resolve().parent
    cfg = json.loads((root / args.config).read_text(encoding='utf-8-sig'))

    set_seed(args.seed)
    device = torch.device('cuda' if bool(cfg.get('use_cuda', True)) and torch.cuda.is_available() else 'cpu')

    g = torch.load(root / cfg['graph_path'], map_location='cpu', weights_only=False)
    X, M, mask = g['battery_features'].float(), g['material_embeddings'].float(), g['node_masks'].float()
    n = int(X.size(0))
    use_graph_context = bool(cfg.get('enable_graph_context', False))
    if use_graph_context:
        X_ctx = build_graph_context_features(g, X).float()
    else:
        X_ctx = X.clone()

    parsed_path = root / str(cfg.get('parsed_data_path', 'data/processed/batteries_parsed.csv'))
    parsed = pd.read_csv(parsed_path)
    if len(parsed) != n:
            raise RuntimeError(f'Dataset size mismatch: {parsed_path.name} vs graph nodes')

    resolved_dataset_mode = args.dataset_mode
    if resolved_dataset_mode == 'auto':
        resolved_dataset_mode = str(cfg.get('dataset_mode', 'ml')).strip().lower()
        if resolved_dataset_mode not in {'ml', 'parsed'}:
            resolved_dataset_mode = 'ml'

    normdf = None
    if resolved_dataset_mode == 'ml':
        normdf = pd.read_csv(root / cfg['batteries_data_path'])
        if len(normdf) != n:
            raise RuntimeError('Dataset size mismatch: batteries_ml.csv vs graph nodes')
        if 'battery_id' in parsed.columns and 'battery_id' in normdf.columns:
            left = parsed['battery_id'].astype(str).reset_index(drop=True)
            right = normdf['battery_id'].astype(str).reset_index(drop=True)
            if not left.equals(right):
                raise RuntimeError('Row alignment mismatch: parsed vs batteries_ml battery_id order')
        maps = build_maps(parsed, normdf)
        data_source_mode = f"ml ({cfg['batteries_data_path']})"
    else:
        maps = build_maps_from_graph_targets(parsed, X)
        data_source_mode = "parsed (mapped to graph battery_features)"

    print(f"[data] dataset_mode={resolved_dataset_mode} | mapping_source={data_source_mode}")
    C_theo, n_e, ion, comp_vec, vocab = build_formula_features(parsed, maps['capacity_grav'])
    role_targets = build_role_targets(g, parsed)
    compatibility_targets = build_compatibility_targets(parsed)

    sample_cfg = cfg.get('sample_weighting', {}) if isinstance(cfg, dict) else {}
    sample_enabled = bool(sample_cfg.get('enabled', True))
    sample_col = str(sample_cfg.get('column', 'target_sample_weight'))
    sample_min = float(sample_cfg.get('min', 0.2))
    sample_max = float(sample_cfg.get('max', 1.0))
    sample_norm_mean = bool(sample_cfg.get('normalize_mean_to_one', True))
    sample_weight_source = 'uniform'
    sample_weight_np = np.ones((n,), dtype=np.float32)
    if sample_enabled:
        if sample_col in parsed.columns:
            sample_weight_np = pd.to_numeric(parsed[sample_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)
            sample_weight_source = f'parsed:{sample_col}'
        elif normdf is not None and sample_col in normdf.columns:
            sample_weight_np = pd.to_numeric(normdf[sample_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)
            sample_weight_source = f'ml:{sample_col}'
        sample_weight_np = np.nan_to_num(sample_weight_np, nan=1.0, posinf=1.0, neginf=sample_min).astype(np.float32)
        sample_weight_np = np.clip(sample_weight_np, sample_min, sample_max)
        if sample_norm_mean:
            sample_weight_np = sample_weight_np / max(1e-6, float(np.mean(sample_weight_np)))
    sample_weight_t = torch.tensor(sample_weight_np, dtype=torch.float32)

    split_source = 'generated_from_config'
    split_json_raw = str(args.split_json or '').strip() or str(cfg.get('split_indices_path', '')).strip()
    if split_json_raw:
        split_path = Path(split_json_raw)
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()
        if not split_path.exists():
            raise RuntimeError(f'split file not found: {split_path}')
        split_info = load_benchmark_split_indices(split_path, n)
        split_source = str(split_path)
    else:
        split_info = build_benchmark_split_indices(parsed, cfg, args.seed)
    tr_idx = np.array(split_info['train'], dtype=np.int64)
    va_idx = np.array(split_info['val'], dtype=np.int64)
    te_iid_idx = np.array(split_info['test_iid'], dtype=np.int64)
    te_ood_idx = np.array(split_info['test_ood'], dtype=np.int64)
    split_meta = split_info.get('metadata', {})

    scope_cfg = cfg.get('scope', {}) if isinstance(cfg, dict) else {}
    insertion_only_training = bool(scope_cfg.get('insertion_only_training', False))
    if insertion_only_training and 'battery_type' in parsed.columns:
        battery_type = parsed['battery_type'].fillna('').astype(str).str.lower().to_numpy()
        allowed_mask = (battery_type == 'insertion')
        tr_idx = tr_idx[allowed_mask[tr_idx]]
        va_idx = va_idx[allowed_mask[va_idx]]
        te_iid_idx = te_iid_idx[allowed_mask[te_iid_idx]]
        te_ood_idx = te_ood_idx[allowed_mask[te_ood_idx]]

    if len(tr_idx) == 0 or len(va_idx) == 0 or len(te_iid_idx) == 0:
        raise RuntimeError(
            f"Invalid benchmark split sizes train/val/test_iid="
            f"{len(tr_idx)}/{len(va_idx)}/{len(te_iid_idx)}"
        )
    print(
        "[split] "
        f"protocol={split_meta.get('protocol_name', 'grouped_iid_plus_ood_v1')} "
        f"train={len(tr_idx)} val={len(va_idx)} test_iid={len(te_iid_idx)} test_ood={len(te_ood_idx)} "
        f"holdout_ion={split_meta.get('holdout_ion', 'none')} "
        f"insertion_only_training={insertion_only_training} "
        f"source={split_source}"
    )
    train_sw = sample_weight_np[tr_idx] if len(tr_idx) else np.array([], dtype=np.float32)
    print(
        "[sample-weight] "
        f"enabled={sample_enabled} source={sample_weight_source} "
        f"train_mean={float(np.mean(train_sw)) if len(train_sw) else 1.0:.4f} "
        f"train_p10={float(np.quantile(train_sw, 0.10)) if len(train_sw) else 1.0:.4f} "
        f"train_p90={float(np.quantile(train_sw, 0.90)) if len(train_sw) else 1.0:.4f}"
    )

    base = NodeDataset(
        X,
        M,
        mask,
        C_theo,
        n_e,
        ion,
        comp_vec,
        X_ctx,
        role_targets,
        compatibility_targets,
        sample_weight_t,
    )
    bs = int(cfg.get('batch_size', 32))
    tr_loader = DataLoader(Subset(base, tr_idx.tolist()), batch_size=bs, shuffle=True)
    va_loader = DataLoader(Subset(base, va_idx.tolist()), batch_size=bs, shuffle=False)
    te_iid_loader = DataLoader(Subset(base, te_iid_idx.tolist()), batch_size=bs, shuffle=False)
    te_ood_loader = DataLoader(Subset(base, te_ood_idx.tolist()), batch_size=bs, shuffle=False) if len(te_ood_idx) else None

    mu, inv_cov = ood_ref(M[torch.tensor(tr_idx, dtype=torch.long)], mask[torch.tensor(tr_idx, dtype=torch.long)])
    train_comp = comp_vec[torch.tensor(tr_idx, dtype=torch.long)]

    loss_cfg = cfg.get('loss', {})
    reg_target_w_cfg = loss_cfg.get('regression_target_weights', {})
    reg_target_w_default = {
        'average_voltage': 1.0,
        'capacity_grav': 1.0,
        'capacity_vol': 1.0,
        'energy_grav': 1.0,
        'energy_vol': 1.0,
        'stability_charge': 1.0,
        'stability_discharge': 1.0,
    }
    reg_target_w = {
        k: float(reg_target_w_cfg.get(k, v))
        for k, v in reg_target_w_default.items()
    }
    w = {
        'regression_mse': float(loss_cfg.get('regression_weight', 1.0)),
        'regression_target_weights': reg_target_w,
        'capacity_grav_huber_weight': float(loss_cfg.get('capacity_grav_huber_weight', 0.0)),
        'capacity_grav_huber_beta': float(loss_cfg.get('capacity_grav_huber_beta', 0.20)),
        'capacity_violation_penalty': float(loss_cfg.get('capacity_violation_penalty', 10.0)),
        'voltage_violation_penalty': float(loss_cfg.get('voltage_violation_penalty', 10.0)),
        'energy_cap_violation_penalty': float(loss_cfg.get('energy_cap_violation_penalty', 25.0)),
        'physics_consistency_loss': float(loss_cfg.get('physics_consistency_loss', 0.0)),
        'thermodynamic_consistency_loss': float(loss_cfg.get('thermodynamic_consistency_loss', 5.0)),
        'mass_balance_penalty': float(loss_cfg.get('mass_balance_penalty', 8.0)),
        'ood_penalty': float(loss_cfg.get('ood_penalty', 6.0)),
        'uncertainty_regularization': float(loss_cfg.get('uncertainty_regularization', 0.1)),
        'role_loss_weight': float(loss_cfg.get('role_loss_weight', 0.0)),
        'compatibility_loss_weight': float(loss_cfg.get('compatibility_loss_weight', 0.0)),
        'novelty_weight': float(loss_cfg.get('novelty_weight', 0.05)),
    }
    lim = {
        'min_voltage': float(cfg.get('physics_limits', {}).get('min_voltage', 1.0)),
        'max_voltage': float(cfg.get('physics_limits', {}).get('max_voltage', 4.4)),
        'max_capacity_grav': float(cfg.get('physics_limits', {}).get('max_capacity_grav', 350.0)),
        'max_energy_grav': float(cfg.get('physics_limits', {}).get('max_energy_grav', 450.0)),
        'max_energy_vol': float(cfg.get('physics_limits', {}).get('max_energy_vol', 1200.0)),
    }
    fcm = cfg.get('full_cell_model', {})
    np_rng = tuple(fcm.get('np_ratio_range', [1.05, 1.15]))
    am_rng = tuple(fcm.get('active_mass_fraction_range', [0.6, 0.75]))
    if len(np_rng) != 2 or len(am_rng) != 2:
        raise RuntimeError('full_cell_model ranges must each contain two values')
    np_rng = (float(np_rng[0]), float(np_rng[1]))
    am_rng = (float(am_rng[0]), float(am_rng[1]))
    if not (np_rng[0] < np_rng[1] and am_rng[0] < am_rng[1]):
        raise RuntimeError('full_cell_model ranges must be strictly increasing')
    train_rep = (M[torch.tensor(tr_idx)] * mask[torch.tensor(tr_idx)].unsqueeze(-1)).sum(dim=1) / mask[torch.tensor(tr_idx)].sum(dim=1, keepdim=True).clamp_min(1.0)
    train_d = maha(train_rep, mu, inv_cov)
    ood_thresh = float(torch.quantile(train_d, 0.95).cpu()) if train_d.numel() else 0.0

    epochs = 2 if args.dry_run else int(cfg.get('epochs', 180))
    stage1 = max(1, int(epochs * 0.6)); stage2 = max(1, epochs - stage1)
    ens = max(1, int(args.ensemble_size))

    stamp = time.strftime('%Y%m%d_%H%M%S')
    model_dir = root / 'reports/models'; model_dir.mkdir(parents=True, exist_ok=True)
    fig_dirs = mk_dirs(root / 'reports/figures' / stamp)

    histories = []; ckpts = []
    best_val_global, best_epoch_global, best_ckpt = math.inf, -1, ''

    for mi in range(ens):
        set_seed(args.seed + 101 * mi)
        model = MaskedGNN(
            battery_feature_dim=int(cfg['battery_feature_dim']), material_embedding_dim=int(cfg['material_embedding_dim']),
            num_materials=int(cfg['num_materials']), hidden_dim=int(cfg['hidden_dim']), latent_dim=int(cfg['latent_dim']),
            num_gnn_layers=int(cfg['num_gnn_layers']), dropout=float(cfg['dropout']),
            decoder_mode=str(cfg.get('decoder_mode', 'legacy')),
            enable_uncertainty=bool(cfg.get('enable_uncertainty', False)),
            material_interaction=str(cfg.get('material_interaction', 'mlp')),
            enable_graph_context=bool(cfg.get('enable_graph_context', False)),
            graph_context_dim=int(cfg.get('graph_context_dim', int(cfg.get('battery_feature_dim', 7)))),
            enable_role_head=bool(cfg.get('enable_role_head', False)),
            enable_compatibility_head=bool(cfg.get('enable_compatibility_head', False)),
            role_output_dim=int(cfg.get('role_output_dim', 3)),
            compatibility_output_dim=int(cfg.get('compatibility_output_dim', 3)),
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('learning_rate', 7e-4)), weight_decay=float(cfg.get('weight_decay', 1e-4)))
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=int(cfg.get('lr_patience', 10)), factor=0.5, min_lr=1e-6)
        patience = int(cfg.get('early_stopping_patience', 30)); wait = 0; best_val = math.inf
        start_epoch = 0

        if args.resume and ens == 1 and mi == 0:
            rpath = Path(args.resume)
            if not rpath.is_absolute():
                rpath = (root / rpath).resolve()
            if rpath.exists():
                rck = torch.load(rpath, map_location=device, weights_only=False)
                if 'model_state_dict' in rck:
                    model.load_state_dict(rck['model_state_dict'], strict=False)
                if 'optimizer_state_dict' in rck:
                    opt.load_state_dict(rck['optimizer_state_dict'])
                if 'scheduler_state_dict' in rck:
                    sch.load_state_dict(rck['scheduler_state_dict'])
                start_epoch = int(rck.get('epoch', -1)) + 1
                best_val = float(rck.get('best_val_loss', math.inf))
                print(f"[resume] loaded {rpath} | start_epoch={start_epoch} best_val={best_val:.6f}")
            else:
                print(f"[resume] checkpoint not found: {rpath}")

        h = {
            k: []
            for k in [
                'train_total',
                'val_total',
                'train_mse',
                'val_mse',
                'capacity_violation',
                'voltage_violation',
                'energy_cap_violation',
                'energy_consistency',
                'uncertainty_regularization',
                'role_loss',
                'compatibility_loss',
                'learning_rate',
            ]
        }

        for ep in range(start_epoch, stage1 + stage2):
            if ep == stage1:
                for p in model.material_encoder.parameters(): p.requires_grad = False
                for layer in model.gnn_layers:
                    for p in layer.parameters(): p.requires_grad = False
                if getattr(model, 'material_transformer', None) is not None:
                    for p in model.material_transformer.parameters():
                        p.requires_grad = False
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.get('learning_rate', 7e-4)) * 0.5, weight_decay=float(cfg.get('weight_decay', 1e-4)))
                sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=max(3, int(cfg.get('lr_patience', 10)) // 2), factor=0.5, min_lr=1e-6)

            tr, _ = epoch_run(
                model, tr_loader, device, maps, w, lim, mu, inv_cov, train_comp, ood_thresh, np_rng, am_rng,
                opt=opt, feat_drop=args.feature_dropout, feat_noise=args.feature_noise_std, store=False, amp_enabled=args.amp,
            )
            va, _ = epoch_run(
                model, va_loader, device, maps, w, lim, mu, inv_cov, train_comp, ood_thresh, np_rng, am_rng,
                opt=None, store=False, amp_enabled=False,
            )
            sch.step(va['total'])

            h['train_total'].append(float(tr['total'])); h['val_total'].append(float(va['total']))
            h['train_mse'].append(float(tr['mse'])); h['val_mse'].append(float(va['mse']))
            h['capacity_violation'].append(float(va['capacity_violation']))
            h['voltage_violation'].append(float(va['voltage_violation']))
            h['energy_cap_violation'].append(float(va['energy_cap_violation']))
            h['energy_consistency'].append(float(va['energy_consistency']))
            h['uncertainty_regularization'].append(float(va['uncertainty_regularization']))
            h['role_loss'].append(float(va.get('role_loss', 0.0)))
            h['compatibility_loss'].append(float(va.get('compatibility_loss', 0.0)))
            h['learning_rate'].append(float(opt.param_groups[0]['lr']))

            print(
                f"[model {mi+1}/{ens}] [epoch {ep:03d}] "
                f"train={tr['total']:.5f} val={va['total']:.5f} mse={va['mse']:.5f} "
                f"lr={opt.param_groups[0]['lr']:.2e} "
                f"skip_non_finite(train/val)={int(tr.get('skipped_non_finite_batches',0))}/{int(va.get('skipped_non_finite_batches',0))}"
            )

            if va['total'] < best_val:
                best_val = float(va['total']); wait = 0
                ckpt = model_dir / f'ensemble_{mi}_best_{stamp}.pt'
                torch.save(
                    {
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': sch.state_dict(),
                        'best_val_loss': best_val,
                        'seed': args.seed + 101 * mi,
                        'config': cfg,
                        'split_indices': {
                            'train': tr_idx.tolist(),
                            'val': va_idx.tolist(),
                            'test_iid': te_iid_idx.tolist(),
                            'test_ood': te_ood_idx.tolist(),
                        },
                        'normalization_maps': {
                            k: {'scale': v.scale, 'shift': v.shift, 'r2': v.r2}
                            for k, v in maps.items()
                        },
                    },
                    ckpt,
                )
                if best_val < best_val_global:
                    best_val_global, best_epoch_global, best_ckpt = best_val, ep, str(ckpt)
            else:
                wait += 1
            if wait >= patience:
                print(f'[model {mi+1}] early stop at epoch {ep}')
                break

        histories.append(h)
        ckpts.append(model_dir / f'ensemble_{mi}_best_{stamp}.pt')

    models = [load_model(p, cfg, device) for p in ckpts if p.exists()]
    if not models: raise RuntimeError('No checkpoints found for ensemble')

    def ens_predict(loader):
        Pm, Pv, Y = [], [], []
        aux = {k: [] for k in ['pred_v_raw', 'pred_cg_raw', 'pred_eg_raw', 'true_v_raw', 'true_cg_raw', 'c_theoretical_raw', 'ood_distance', 'novelty']}
        with torch.no_grad():
            for b in loader:
                x, M, mask, y = b['x'].to(device), b['m'].to(device), b['mask'].to(device), b['y'].to(device)
                ctx = b['ctx'].to(device)
                c_theo, n_e, ion, comp = b['c_theo'].to(device), b['n_e'].to(device), b['ion'].to(device), b['comp'].to(device)
                preds = []
                for mdl in models:
                    p0 = mdl(x, M, mask, graph_context=ctx)['properties']
                    p, _ = deterministic_cell(p0, maps, np_rng=np_rng, am_rng=am_rng, lim=lim)
                    preds.append(p)
                S = torch.stack(preds, dim=0)
                mean, var = S.mean(dim=0), S.var(dim=0, unbiased=False)
                rep = (M * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                _, _, ex = loss_terms(
                    mean, y, deterministic_cell(mean, maps, np_rng=np_rng, am_rng=am_rng, lim=lim)[1],
                    c_theo, n_e, ion, comp, train_comp, rep, mu, inv_cov, maps, w, lim, ood_thresh=ood_thresh,
                    pred_logvar=None,
                )
                Pm.append(mean.detach().cpu().numpy()); Pv.append(var.detach().cpu().numpy()); Y.append(y.detach().cpu().numpy())
                for k in aux: aux[k].append(ex[k].detach().cpu().numpy())
        out = {'pred_mean': np.concatenate(Pm), 'pred_var': np.concatenate(Pv), 'true': np.concatenate(Y)}
        out.update({k: np.concatenate(v) if v else np.array([]) for k, v in aux.items()}); return out

    valp = ens_predict(va_loader)
    test_iid_p = ens_predict(te_iid_loader)
    test_ood_p = ens_predict(te_ood_loader) if te_ood_loader is not None else None

    # aggregate history
    ep_max = max(len(h['train_total']) for h in histories)
    ep_axis = list(range(ep_max))
    mh = {}
    for k in histories[0].keys():
        arrs = []
        for h in histories:
            a = np.array(h[k], dtype=np.float64)
            if len(a) < ep_max: a = np.pad(a, (0, ep_max - len(a)), constant_values=np.nan)
            arrs.append(a)
        mh[k] = np.nanmean(np.stack(arrs), axis=0)

    save_line(ep_axis, {'train_total': mh['train_total'], 'val_total': mh['val_total']}, 'Total Loss vs Epoch', fig_dirs['training_curves'] / 'loss_curve.png', 'loss')
    save_line(ep_axis, {'train_mse': mh['train_mse'], 'val_mse': mh['val_mse']}, 'MSE vs Epoch', fig_dirs['training_curves'] / 'mse_curve.png', 'mse')
    save_line(ep_axis, {'learning_rate': mh['learning_rate']}, 'Learning Rate Schedule', fig_dirs['training_curves'] / 'lr_schedule.png', 'lr')
    save_line(ep_axis, {'capacity_violation': mh['capacity_violation'], 'voltage_violation': mh['voltage_violation'], 'energy_cap_violation': mh['energy_cap_violation'], 'energy_consistency': mh['energy_consistency']}, 'Physics Violation Terms During Training', fig_dirs['physics_diagnostics'] / 'physics_penalties.png', 'penalty')

    save_scatter(valp['true'][:, PROPERTY_INDEX['capacity_grav']], valp['pred_mean'][:, PROPERTY_INDEX['capacity_grav']], 'Predicted vs True Gravimetric Capacity', fig_dirs['physics_diagnostics'] / 'capacity_scatter.png', 'true capacity (norm)', 'pred capacity (norm)', True)
    save_scatter(valp['c_theoretical_raw'], valp['pred_cg_raw'], 'Predicted Capacity vs Theoretical Ceiling', fig_dirs['physics_diagnostics'] / 'theoretical_limit_check.png', 'theoretical capacity (mAh/g)', 'pred capacity (mAh/g)', False)

    save_hist({'true_voltage': valp['true_v_raw'], 'predicted_voltage': valp['pred_v_raw']}, 'Voltage Distribution Comparison', fig_dirs['distribution_analysis'] / 'voltage_distribution.png', 'voltage (V)')
    save_hist({'true_energy_grav(norm)': valp['true'][:, PROPERTY_INDEX['energy_grav']], 'pred_energy_grav(norm)': valp['pred_mean'][:, PROPERTY_INDEX['energy_grav']]}, 'Energy Density Distribution', fig_dirs['distribution_analysis'] / 'energy_distribution.png', 'energy (norm)')

    pred_unc = np.sqrt(np.maximum(0.0, valp['pred_var'])).mean(axis=1)
    abs_err = np.abs(valp['pred_mean'] - valp['true']).mean(axis=1)
    save_scatter(pred_unc, abs_err, 'Uncertainty Calibration', fig_dirs['uncertainty_analysis'] / 'uncertainty_vs_error.png', 'pred uncertainty', 'absolute error', False)

    cx, cy = reliability(pred_unc, abs_err, bins=10)
    plt.figure(figsize=(8, 6), dpi=300)
    if len(cx):
        plt.plot(cx, cy, marker='o', label='empirical')
        lo, hi = min(cx.min(), cy.min()), max(cx.max(), cy.max())
        plt.plot([lo, hi], [lo, hi], 'r--', label='ideal')
    plt.title('Prediction Calibration Curve'); plt.xlabel('pred uncertainty'); plt.ylabel('observed abs error'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(fig_dirs['uncertainty_analysis'] / 'calibration_curve.png'); plt.close()

    save_hist({'ood_distance': valp['ood_distance']}, 'Out-of-Distribution Distance Histogram', fig_dirs['distribution_analysis'] / 'ood_distance.png', 'mahalanobis distance')

    formulas = parsed.iloc[te_iid_idx]['battery_formula'].fillna('').astype(str).str.replace(' ', '', regex=False).str.lower().reset_index(drop=True)
    pred_cg = test_iid_p['pred_cg_raw']; true_cg = maps['capacity_grav'].to_raw(test_iid_p['true'][:, PROPERTY_INDEX['capacity_grav']])
    systems = ['lifepo4', 'nafepo4', 'nmc']
    tv, pv = [], []
    for s in systems:
        m = formulas.str.contains(s, regex=False).to_numpy()
        tv.append(float(np.nanmean(true_cg[m])) if m.sum() else np.nan)
        pv.append(float(np.nanmean(pred_cg[m])) if m.sum() else np.nan)
    x = np.arange(len(systems)); wbar = 0.35
    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(x - wbar / 2, tv, wbar, label='true_capacity')
    plt.bar(x + wbar / 2, pv, wbar, label='predicted_capacity')
    plt.xticks(x, ['LiFePO4', 'NaFePO4', 'NMC811']); plt.title('Benchmark System Reproduction'); plt.ylabel('Capacity (mAh/g)'); plt.grid(True, axis='y', alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(fig_dirs['benchmark_validation'] / 'benchmark_reproduction.png'); plt.close()

    def to_raw_matrix(arr_norm: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                maps['average_voltage'].to_raw(arr_norm[:, PROPERTY_INDEX['average_voltage']]),
                maps['capacity_grav'].to_raw(arr_norm[:, PROPERTY_INDEX['capacity_grav']]),
                maps['capacity_vol'].to_raw(arr_norm[:, PROPERTY_INDEX['capacity_vol']]),
                maps['energy_grav'].to_raw(arr_norm[:, PROPERTY_INDEX['energy_grav']]),
                maps['energy_vol'].to_raw(arr_norm[:, PROPERTY_INDEX['energy_vol']]),
                maps['stability_charge'].to_raw(arr_norm[:, PROPERTY_INDEX['stability_charge']]),
                maps['stability_discharge'].to_raw(arr_norm[:, PROPERTY_INDEX['stability_discharge']]),
            ],
            axis=1,
        )

    def metric_bundle(pred_pack: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Dict[str, float]]]:
        norm = per_property_metrics(pred_pack['true'], pred_pack['pred_mean'])
        raw_true_m = to_raw_matrix(pred_pack['true'])
        raw_pred_m = to_raw_matrix(pred_pack['pred_mean'])
        raw = per_property_metrics(raw_true_m, raw_pred_m)
        return {'normalized_space': norm, 'raw_space': raw}

    iid_metrics = metric_bundle(test_iid_p)
    ood_metrics = metric_bundle(test_ood_p) if test_ood_p is not None else None

    overall_raw = iid_metrics['raw_space'].get('overall_micro', {})
    print(
        "[eval/raw] "
        f"R2={overall_raw.get('r2', 0.0):.4f} "
        f"MAE={overall_raw.get('mae', 0.0):.4f} "
        f"RMSE={overall_raw.get('rmse', 0.0):.4f} "
        f"MAPE%={overall_raw.get('mape_percent', 0.0):.2f} "
        f"PearsonR={overall_raw.get('pearson_r', 0.0):.4f}"
    )

    iid_cap_violation_rate, iid_theo_valid_frac = capacity_violation_rate(
        test_iid_p['pred_cg_raw'],
        test_iid_p['c_theoretical_raw'],
        max_capacity_grav=float(lim['max_capacity_grav']),
    )
    if test_ood_p is not None and len(test_ood_p['pred_cg_raw']):
        ood_cap_violation_rate, ood_theo_valid_frac = capacity_violation_rate(
            test_ood_p['pred_cg_raw'],
            test_ood_p['c_theoretical_raw'],
            max_capacity_grav=float(lim['max_capacity_grav']),
        )
    else:
        ood_cap_violation_rate, ood_theo_valid_frac = None, None

    summary = {
        'timestamp': stamp,
        'device': str(device),
        'ensemble_size': ens,
        'dataset_mode': resolved_dataset_mode,
        'parsed_data_path': str(parsed_path),
        'dataset_mapping_source': data_source_mode,
        'target_policy': cfg.get('target_policy', {}),
        'model_architecture': {
            'material_interaction': str(cfg.get('material_interaction', 'mlp')),
            'enable_graph_context': bool(cfg.get('enable_graph_context', False)),
            'graph_context_dim': int(cfg.get('graph_context_dim', int(cfg.get('battery_feature_dim', 7)))),
            'enable_role_head': bool(cfg.get('enable_role_head', False)),
            'enable_compatibility_head': bool(cfg.get('enable_compatibility_head', False)),
            'role_output_dim': int(cfg.get('role_output_dim', 3)),
            'compatibility_output_dim': int(cfg.get('compatibility_output_dim', 3)),
            'graph_version': str(g.get('metadata', {}).get('version', 'unknown')),
        },
        'benchmark_protocol': {
            'name': split_meta.get('protocol_name', 'grouped_iid_plus_ood_v1'),
            'group_keys': split_meta.get('group_keys', ['chemsys', 'framework_formula', 'working_ion']),
            'counts': split_meta.get('counts', {}),
            'source': split_source,
            'ood': {
                'enabled': bool(split_meta.get('ood_enabled', False)),
                'holdout_ion': split_meta.get('holdout_ion', 'none'),
                'holdout_chemsys_fraction': float(split_meta.get('holdout_chemsys_fraction', 0.0)),
                'holdout_chemsys_count': int(split_meta.get('holdout_chemsys_count', 0)),
                'holdout_chemsys_preview': split_meta.get('holdout_chemsys_preview', []),
            },
        },
        'sample_weighting': {
            'enabled': sample_enabled,
            'source': sample_weight_source,
            'column': sample_col,
            'normalize_mean_to_one': sample_norm_mean,
            'train_mean': float(np.mean(train_sw)) if len(train_sw) else 1.0,
            'train_min': float(np.min(train_sw)) if len(train_sw) else 1.0,
            'train_max': float(np.max(train_sw)) if len(train_sw) else 1.0,
            'train_p10': float(np.quantile(train_sw, 0.10)) if len(train_sw) else 1.0,
            'train_p90': float(np.quantile(train_sw, 0.90)) if len(train_sw) else 1.0,
        },
        'best_val_loss': float(best_val_global),
        'final_test_loss_iid': float(np.mean((test_iid_p['pred_mean'] - test_iid_p['true']) ** 2)),
        'final_test_loss_ood': float(np.mean((test_ood_p['pred_mean'] - test_ood_p['true']) ** 2)) if test_ood_p is not None else None,
        'best_epoch': int(best_epoch_global),
        'best_checkpoint': best_ckpt,
        'evaluation_metrics': {
            'iid': iid_metrics,
            'ood': ood_metrics,
        },
        'physics_violation_statistics': {
            'iid': {
                'capacity_violation_rate': iid_cap_violation_rate,
                'theoretical_capacity_valid_fraction': iid_theo_valid_frac,
                'voltage_above_4p4_rate': float(np.mean(test_iid_p['pred_v_raw'] > 4.4)),
            },
            'ood': {
                'capacity_violation_rate': ood_cap_violation_rate,
                'theoretical_capacity_valid_fraction': ood_theo_valid_frac,
                'voltage_above_4p4_rate': float(np.mean(test_ood_p['pred_v_raw'] > 4.4)) if test_ood_p is not None and len(test_ood_p['pred_v_raw']) else None,
            },
        },
        'uncertainty_statistics': {
            'pred_uncertainty_mean': float(np.mean(pred_unc)),
            'pred_uncertainty_std': float(np.std(pred_unc)),
            'corr_uncertainty_abs_error': safe_corr(pred_unc, abs_err),
        },
        'distribution_shift_statistics': {
            'ood_distance_mean': float(np.mean(valp['ood_distance'])) if len(valp['ood_distance']) else 0.0,
            'ood_distance_p95': float(np.quantile(valp['ood_distance'], 0.95)) if len(valp['ood_distance']) else 0.0,
            'novelty_mean': float(np.mean(valp['novelty'])) if len(valp['novelty']) else 0.0,
        },
        'report_folder': str(fig_dirs['root']),
    }
    (root / 'reports' / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f"[done] reports generated at: {fig_dirs['root']}")
    print(f"[done] summary: {root / 'reports' / 'training_summary.json'}")

if __name__ == '__main__':
    main()

