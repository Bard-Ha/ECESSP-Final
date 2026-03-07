#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (PROJECT_ROOT, TOOLS_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


ROLE_NAMES = ["cathode", "anode", "electrolyte", "separator", "additive"]
ROLE_INDEX = {r: i for i, r in enumerate(ROLE_NAMES)}
TARGET_ORDER = [
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "stability_charge",
    "stability_discharge",
]
SYSTEM_FEATURE_ORDER = [
    "n_voltage_pairs",
    "frac_charge_mean",
    "frac_discharge_mean",
    "delta_frac_A",
    "max_volume_expansion_ratio",
    "charge_discharge_consistent",
    "nelements",
    "num_steps",
    "material_count",
]


def _safe_parse(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    try:
        v = ast.literal_eval(s)
    except Exception:
        return []
    return v if isinstance(v, list) else []


def _as_edge_index(edges: List[List[int]]) -> torch.Tensor:
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _fit_affine(raw: np.ndarray, norm: np.ndarray) -> Dict[str, float]:
    x = np.vstack([raw, np.ones_like(raw)]).T
    c, *_ = np.linalg.lstsq(x, norm, rcond=None)
    a, b = float(c[0]), float(c[1])
    return {"scale": a, "shift": b}


def _numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full((len(df),), float(default), dtype=np.float32)
    s = pd.to_numeric(df[col], errors="coerce").fillna(float(default))
    return s.to_numpy(dtype=np.float32)


def _build_split_indices(parsed_df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    from build_benchmark_splits import build_splits

    return build_splits(
        parsed_df,
        train_ratio=0.7,
        val_ratio=0.15,
        group_keys=["chemsys", "framework_formula", "working_ion"],
        ood_enabled=True,
        holdout_ion="auto",
        holdout_chemsys_fraction=0.08,
        min_ion_count=120,
        seed=int(seed),
    )


def _sanitize_indices(raw_indices: Any, n: int) -> list[int]:
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


def _load_split_indices(path: Path, n: int) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    train = _sanitize_indices(payload.get("train"), n)
    val = _sanitize_indices(payload.get("val"), n)
    test_iid = _sanitize_indices(payload.get("test_iid"), n)
    test_ood = _sanitize_indices(payload.get("test_ood"), n)
    if not train or not val or not test_iid:
        raise RuntimeError(f"invalid split file: {path}")
    used = set(train) | set(val) | set(test_iid) | set(test_ood)
    if len(used) != (len(train) + len(val) + len(test_iid) + len(test_ood)):
        raise RuntimeError(f"split file contains overlapping indices: {path}")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return {
        "train": train,
        "val": val,
        "test_iid": test_iid,
        "test_ood": test_ood,
        "metadata": {
            **(metadata if isinstance(metadata, dict) else {}),
            "source": str(path),
            "counts": {
                "total": int(n),
                "train": int(len(train)),
                "val": int(len(val)),
                "test_iid": int(len(test_iid)),
                "test_ood": int(len(test_ood)),
            },
        },
    }


def _infer_role(material_id: str, row_parsed: pd.Series, row_ml: pd.Series, used_roles: set[str]) -> str:
    id_charge = str(row_parsed.get("id_charge") or "")
    id_discharge = str(row_parsed.get("id_discharge") or "")
    if material_id and material_id == id_discharge:
        return "cathode"
    if material_id and material_id == id_charge:
        return "anode"

    working_ion = str(row_ml.get("working_ion") or "")
    elems = {str(e) for e in _safe_parse(row_ml.get("elements"))}
    if working_ion and working_ion in elems and "electrolyte" not in used_roles:
        return "electrolyte"
    if "separator" not in used_roles:
        return "separator"
    return "additive"


def _cosine_knn(x: torch.Tensor, k: int) -> List[List[int]]:
    if x.size(0) <= 1:
        return [[] for _ in range(int(x.size(0)))]
    x = F.normalize(x, dim=1)
    sim = torch.matmul(x, x.T)
    n = x.size(0)
    k = max(1, min(int(k), n - 1))
    out: List[List[int]] = []
    for i in range(n):
        _, idx = torch.topk(sim[i], k=k + 1)
        nbrs = [int(j) for j in idx.tolist() if int(j) != i][:k]
        out.append(nbrs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build true hetero graph for HGT training")
    ap.add_argument("--batteries-ml", default="data/processed/batteries_ml_curated.csv")
    ap.add_argument("--batteries-parsed", default="data/processed/batteries_parsed_curated.csv")
    ap.add_argument("--atomic-emb", default="data/processed/atomic_embeddings_curated.csv")
    ap.add_argument("--out", default="graphs/battery_hetero_graph_v1.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split-json",
        default="",
        help="Optional shared benchmark split JSON. If set, reuse this split instead of regenerating.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    ml = pd.read_csv((root / args.batteries_ml).resolve())
    parsed = pd.read_csv((root / args.batteries_parsed).resolve())
    atomic = pd.read_csv((root / args.atomic_emb).resolve())

    if len(ml) != len(parsed):
        raise RuntimeError(f"row mismatch ml vs parsed: {len(ml)} vs {len(parsed)}")
    n_system = int(len(ml))

    emb_cols = [c for c in atomic.columns if c.startswith("atom_emb_")]
    material_dim = int(len(emb_cols))
    if material_dim <= 0:
        raise RuntimeError("no atomic embedding columns found")

    atomic_map = {
        str(r["material_id"]): r[emb_cols].to_numpy(dtype=np.float32)
        for _, r in atomic.iterrows()
    }

    all_materials: List[str] = []
    for i in range(n_system):
        mats = [str(x) for x in _safe_parse(ml.iloc[i].get("material_ids")) if str(x)]
        if not mats:
            fallback = [str(parsed.iloc[i].get("id_charge") or ""), str(parsed.iloc[i].get("id_discharge") or "")]
            mats = [m for m in fallback if m]
        all_materials.extend(mats)
    unique_materials = sorted(set(all_materials))
    midx = {m: j for j, m in enumerate(unique_materials)}
    n_material = len(unique_materials)

    material_x = torch.zeros((n_material, material_dim), dtype=torch.float32)
    for m, j in midx.items():
        vec = atomic_map.get(m)
        if vec is not None:
            material_x[j] = torch.tensor(vec, dtype=torch.float32)

    ions = sorted(set(str(v) for v in ml["working_ion"].fillna("unknown").astype(str).tolist()))
    iidx = {ion: k for k, ion in enumerate(ions)}
    ion_x = torch.zeros((len(ions), len(ions)), dtype=torch.float32)
    for ion, j in iidx.items():
        ion_x[j, j] = 1.0

    role_x = torch.eye(len(ROLE_NAMES), dtype=torch.float32)

    material_count = np.zeros((n_system,), dtype=np.float32)
    for i in range(n_system):
        mats_i = [str(x) for x in _safe_parse(ml.iloc[i].get("material_ids")) if str(x)]
        if not mats_i:
            fallback = [str(parsed.iloc[i].get("id_charge") or ""), str(parsed.iloc[i].get("id_discharge") or "")]
            mats_i = [m for m in fallback if m]
        material_count[i] = float(len(set(mats_i)))

    raw_system = np.column_stack(
        [
            _numeric_series(ml, "n_voltage_pairs", default=1.0),
            _numeric_series(ml, "frac_charge_mean", default=0.0),
            _numeric_series(ml, "frac_discharge_mean", default=0.0),
            _numeric_series(ml, "delta_frac_A", default=0.0),
            _numeric_series(ml, "max_volume_expansion_ratio", default=1.0),
            _numeric_series(ml, "charge_discharge_consistent", default=1.0),
            _numeric_series(parsed, "nelements", default=1.0),
            _numeric_series(parsed, "num_steps", default=1.0),
            material_count,
        ]
    ).astype(np.float32, copy=False)
    raw_mu = raw_system.mean(axis=0, dtype=np.float64)
    raw_std = raw_system.std(axis=0, dtype=np.float64)
    raw_std = np.where(raw_std < 1e-6, 1.0, raw_std)
    system_arr = (raw_system - raw_mu) / raw_std
    system_x = torch.tensor(system_arr, dtype=torch.float32)

    targets_norm = torch.tensor(
        ml[
            [
                "average_voltage_norm",
                "capacity_grav_norm",
                "capacity_vol_norm",
                "energy_grav_norm",
                "energy_vol_norm",
                "stability_charge_norm",
                "stability_discharge_norm",
            ]
        ].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    edge_sm: List[List[int]] = []
    edge_si: List[List[int]] = []
    edge_sr: List[List[int]] = []
    edge_rm: List[List[int]] = []

    for i in range(n_system):
        row_ml = ml.iloc[i]
        row_parsed = parsed.iloc[i]
        mats = [str(x) for x in _safe_parse(row_ml.get("material_ids")) if str(x)]
        if not mats:
            fallback = [str(row_parsed.get("id_charge") or ""), str(row_parsed.get("id_discharge") or "")]
            mats = [m for m in fallback if m]
        mats = list(dict.fromkeys(mats))

        used_roles: set[str] = set()
        for m in mats:
            j = midx.get(m)
            if j is None:
                continue
            edge_sm.append([i, j])
            role = _infer_role(m, row_parsed, row_ml, used_roles)
            used_roles.add(role)
            rid = ROLE_INDEX[role]
            edge_sr.append([i, rid])
            edge_rm.append([rid, j])

        ion = str(row_ml.get("working_ion") or "unknown")
        j_ion = iidx.get(ion)
        if j_ion is not None:
            edge_si.append([i, j_ion])

    mm_edges: List[List[int]] = []
    for i, nbrs in enumerate(_cosine_knn(material_x, k=20)):
        mm_edges.extend([[i, j] for j in nbrs if i != j])

    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {
        ("system", "has_material", "material"): _as_edge_index(edge_sm),
        ("material", "rev_has_material", "system"): _as_edge_index([[b, a] for a, b in edge_sm]),
        ("system", "uses_ion", "ion"): _as_edge_index(edge_si),
        ("ion", "rev_uses_ion", "system"): _as_edge_index([[b, a] for a, b in edge_si]),
        ("system", "has_role", "role"): _as_edge_index(edge_sr),
        ("role", "rev_has_role", "system"): _as_edge_index([[b, a] for a, b in edge_sr]),
        ("role", "connects_material", "material"): _as_edge_index(edge_rm),
        ("material", "rev_connects_material", "role"): _as_edge_index([[b, a] for a, b in edge_rm]),
        ("material", "similar_material", "material"): _as_edge_index(mm_edges),
    }

    norm_maps: Dict[str, Dict[str, float]] = {}
    for p in TARGET_ORDER:
        raw = parsed[p].to_numpy(dtype=np.float64)
        norm = ml[f"{p}_norm"].to_numpy(dtype=np.float64)
        norm_maps[p] = _fit_affine(raw, norm)

    split_json = str(args.split_json or "").strip()
    if split_json:
        split_path = Path(split_json)
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()
        if not split_path.exists():
            raise RuntimeError(f"split file not found: {split_path}")
        split_payload = _load_split_indices(split_path, n_system)
    else:
        split_payload = _build_split_indices(parsed_df=parsed, seed=int(args.seed))

    out_graph = {
        "node_features": {
            "system": system_x,
            "material": material_x,
            "ion": ion_x,
            "role": role_x,
        },
        "edge_index_dict": edge_index_dict,
        "targets_norm": targets_norm,
        "normalization_maps": norm_maps,
        "split_indices": {
            "train": torch.tensor(split_payload["train"], dtype=torch.long),
            "val": torch.tensor(split_payload["val"], dtype=torch.long),
            "test_iid": torch.tensor(split_payload["test_iid"], dtype=torch.long),
            "test_ood": torch.tensor(split_payload["test_ood"], dtype=torch.long),
        },
        "metadata": {
            "version": "battery_hetero_graph_v2_no_target_leakage",
            "num_system_nodes": int(n_system),
            "num_material_nodes": int(n_material),
            "num_ion_nodes": int(len(ions)),
            "num_role_nodes": int(len(ROLE_NAMES)),
            "system_feature_order": SYSTEM_FEATURE_ORDER,
            "system_feature_stats": {
                "mean": [float(v) for v in raw_mu.tolist()],
                "std": [float(v) for v in raw_std.tolist()],
            },
            "target_order": TARGET_ORDER,
            "role_names": ROLE_NAMES,
            "material_id_to_index": midx,
            "ion_to_index": iidx,
            "split_metadata": split_payload.get("metadata", {}),
            "sources": {
                "batteries_ml": str((root / args.batteries_ml).resolve()),
                "batteries_parsed": str((root / args.batteries_parsed).resolve()),
                "atomic_embeddings": str((root / args.atomic_emb).resolve()),
            },
        },
    }

    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_graph, out_path)
    print(f"[ok] wrote hetero graph: {out_path}")
    print(f"[ok] systems={n_system} materials={n_material} ions={len(ions)} roles={len(ROLE_NAMES)}")


if __name__ == "__main__":
    main()
