#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, total=None):  # type: ignore
        return x


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graph_v2")


DATA_ML = Path("data/processed/batteries_ml_curated.csv")
DATA_PARSED = Path("data/processed/batteries_parsed_curated.csv")
ATOMIC_EMB = Path("data/processed/atomic_embeddings_curated.csv")
OUT_GRAPH = Path("graphs/masked_battery_graph_normalized_v2.pt")

NUM_MATERIALS = 5
MATERIAL_DIM = 64
BATTERY_DIM = 7
BATTERY_FEATURE_ORDER = [
    "average_voltage_norm",
    "capacity_grav_norm",
    "capacity_vol_norm",
    "energy_grav_norm",
    "energy_vol_norm",
    "stability_charge_norm",
    "stability_discharge_norm",
]
ROLE_NAMES = ["cathode", "anode", "electrolyte", "separator", "additive"]
ROLE_INDEX = {r: i for i, r in enumerate(ROLE_NAMES)}

K_ATOMIC = 12
K_CHEMSYS = 12
K_ION = 10
K_STOICH = 15
K_PHYSICS = 20
K_MATERIAL = 20


def parse_obj(x: Any) -> Any:
    if isinstance(x, (list, dict)):
        return x
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return []


def cosine_knn(x: torch.Tensor, k: int) -> List[List[int]]:
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


def euclid_knn(x: torch.Tensor, k: int) -> List[List[int]]:
    n = x.size(0)
    k = max(1, min(int(k), n - 1))
    out: List[List[int]] = []
    for i in range(n):
        d = torch.norm(x - x[i : i + 1], dim=1)
        idx = torch.argsort(d)[: k + 1]
        nbrs = [int(j) for j in idx.tolist() if int(j) != i][:k]
        out.append(nbrs)
    return out


def role_scores_for_material(mat_id: str, row_parsed: pd.Series, row_ml: pd.Series) -> np.ndarray:
    s = np.zeros(len(ROLE_NAMES), dtype=np.float32)
    id_charge = str(row_parsed.get("id_charge") or "").strip()
    id_discharge = str(row_parsed.get("id_discharge") or "").strip()
    ion = str(row_ml.get("working_ion") or "")
    chemsys = str(row_ml.get("chemsys") or "")
    elements = set(parse_obj(row_ml.get("elements", "[]")))
    formula = str(row_parsed.get("battery_formula") or "")

    if mat_id and mat_id == id_discharge:
        s[ROLE_INDEX["cathode"]] += 0.95
    if mat_id and mat_id == id_charge:
        s[ROLE_INDEX["anode"]] += 0.95
    if ion and (ion in formula or ion in chemsys or ion in elements):
        s[ROLE_INDEX["electrolyte"]] += 0.45
    if "O" in elements or "F" in elements:
        s[ROLE_INDEX["cathode"]] += 0.15
    if "C" in elements or "Si" in elements or "Sn" in elements:
        s[ROLE_INDEX["anode"]] += 0.12

    s[ROLE_INDEX["separator"]] += 0.05
    s[ROLE_INDEX["additive"]] += 0.05
    return s


def as_edge_index(edges: List[List[int]]) -> torch.Tensor:
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_dataset_fingerprint(paths: List[Path], row_count: int) -> str:
    h = hashlib.sha256()
    h.update(f"rows:{int(row_count)}".encode("utf-8"))
    for p in paths:
        rp = p.resolve()
        h.update(str(rp).encode("utf-8"))
        try:
            st = rp.stat()
            h.update(str(int(st.st_size)).encode("utf-8"))
            h.update(str(int(st.st_mtime_ns)).encode("utf-8"))
        except OSError:
            h.update(b"missing")
    return h.hexdigest()


def build() -> Dict[str, Any]:
    logger.info("Loading curated battery/system datasets and atomic embeddings...")
    ml = pd.read_csv(DATA_ML)
    parsed = pd.read_csv(DATA_PARSED)
    emb = pd.read_csv(ATOMIC_EMB)

    if len(ml) != len(parsed):
        raise RuntimeError(f"Row mismatch ml vs parsed: {len(ml)} vs {len(parsed)}")
    n = int(len(ml))

    emb_cols = [c for c in emb.columns if c.startswith("atom_emb_")]
    if len(emb_cols) != MATERIAL_DIM:
        raise RuntimeError(f"Expected {MATERIAL_DIM} embedding dims, found {len(emb_cols)}")
    emb_map: Dict[str, np.ndarray] = {
        str(r["material_id"]): r[emb_cols].to_numpy(dtype=np.float32)
        for _, r in emb.iterrows()
    }

    battery_feat = torch.zeros((n, BATTERY_DIM), dtype=torch.float32)
    material_emb = torch.zeros((n, NUM_MATERIALS, MATERIAL_DIM), dtype=torch.float32)
    node_mask = torch.zeros((n, NUM_MATERIALS), dtype=torch.float32)
    role_conf = torch.zeros((n, NUM_MATERIALS), dtype=torch.float32)
    role_name_slots: List[Dict[int, str]] = []

    elem_vocab = sorted({str(e) for items in ml["elements"].fillna("[]").tolist() for e in parse_obj(items) if str(e)})
    eidx = {e: i for i, e in enumerate(elem_vocab)}
    element_frac = torch.zeros((n, len(elem_vocab)), dtype=torch.float32)

    covered_material_refs = 0
    total_material_refs = 0
    logger.info("Building node tensors with atomic embeddings...")
    for i in tqdm(range(n), total=n):
        r_ml = ml.iloc[i]
        r_parsed = parsed.iloc[i]
        battery_feat[i] = torch.tensor([
            float(r_ml.get("average_voltage_norm", 0.0)),
            float(r_ml.get("capacity_grav_norm", 0.0)),
            float(r_ml.get("capacity_vol_norm", 0.0)),
            float(r_ml.get("energy_grav_norm", 0.0)),
            float(r_ml.get("energy_vol_norm", 0.0)),
            float(r_ml.get("stability_charge_norm", 0.0)),
            float(r_ml.get("stability_discharge_norm", 0.0)),
        ], dtype=torch.float32)

        elems = [str(x) for x in parse_obj(r_ml.get("elements", "[]"))]
        if elems:
            w = 1.0 / len(elems)
            for e in elems:
                if e in eidx:
                    element_frac[i, eidx[e]] += w

        mats = [str(x) for x in parse_obj(r_ml.get("material_ids", "[]")) if str(x)]
        if not mats:
            fallback = [str(r_parsed.get("id_charge") or ""), str(r_parsed.get("id_discharge") or "")]
            mats = [m for m in fallback if m]
        mats = list(dict.fromkeys(mats))

        scored: List[Tuple[float, str, str]] = []
        for m in mats:
            total_material_refs += 1
            vec = emb_map.get(m)
            if vec is None:
                continue
            covered_material_refs += 1
            rs = role_scores_for_material(m, r_parsed, r_ml)
            rid = int(np.argmax(rs))
            scored.append((float(rs[rid]), ROLE_NAMES[rid], m))

        scored.sort(reverse=True, key=lambda t: t[0])
        slot_roles: Dict[int, str] = {}
        for pos, (score, role, mid) in enumerate(scored[:NUM_MATERIALS]):
            material_emb[i, pos] = torch.tensor(emb_map[mid], dtype=torch.float32)
            node_mask[i, pos] = 1.0
            role_conf[i, pos] = float(np.clip(score, 0.0, 1.0))
            slot_roles[pos] = role
        role_name_slots.append(slot_roles)

    logger.info("Building edge sets...")
    present = material_emb * node_mask.unsqueeze(-1)
    avg_emb = present.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    std_emb = torch.sqrt(((present - avg_emb.unsqueeze(1)) ** 2 * node_mask.unsqueeze(-1)).sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0))

    chemsys = ml["chemsys"].fillna("unknown").astype(str).tolist()
    ions = ml["working_ion"].fillna("unknown").astype(str).tolist()

    edges_atomic: List[List[int]] = []
    for i, nbrs in enumerate(cosine_knn(avg_emb, K_ATOMIC)):
        edges_atomic.extend([[i, j] for j in nbrs])

    edges_chemsys: List[List[int]] = []
    by_chemsys: Dict[str, List[int]] = {}
    for i, cs in enumerate(chemsys):
        by_chemsys.setdefault(cs, []).append(i)
    for idxs in by_chemsys.values():
        if len(idxs) < 2:
            continue
        local = avg_emb[idxs]
        knn = cosine_knn(local, min(K_CHEMSYS, len(idxs) - 1))
        for li, nbrs in enumerate(knn):
            src = idxs[li]
            edges_chemsys.extend([[src, idxs[lj]] for lj in nbrs if idxs[lj] != src])

    edges_ion: List[List[int]] = []
    by_ion: Dict[str, List[int]] = {}
    for i, ion in enumerate(ions):
        by_ion.setdefault(ion, []).append(i)
    for idxs in by_ion.values():
        if len(idxs) < 2:
            continue
        local = avg_emb[idxs]
        knn = cosine_knn(local, min(K_ION, len(idxs) - 1))
        for li, nbrs in enumerate(knn):
            src = idxs[li]
            edges_ion.extend([[src, idxs[lj]] for lj in nbrs if idxs[lj] != src])

    edges_stoich: List[List[int]] = []
    if element_frac.size(1) > 0:
        for i, nbrs in enumerate(cosine_knn(element_frac, K_STOICH)):
            edges_stoich.extend([[i, j] for j in nbrs])

    physics_feat = torch.stack([battery_feat[:, 0], battery_feat[:, 1], battery_feat[:, 2], battery_feat[:, 5], battery_feat[:, 6]], dim=1)
    edges_physics: List[List[int]] = []
    for i, nbrs in enumerate(euclid_knn(physics_feat, K_PHYSICS)):
        edges_physics.extend([[i, j] for j in nbrs])

    edge_index_dict = {
        0: as_edge_index(edges_atomic),
        1: as_edge_index(edges_chemsys),
        2: as_edge_index(edges_ion),
        3: as_edge_index(edges_stoich),
        4: as_edge_index(edges_physics),
    }

    role_onehot = torch.zeros((n, NUM_MATERIALS, len(ROLE_NAMES)), dtype=torch.float32)
    for i, slot_map in enumerate(role_name_slots):
        for pos, r in slot_map.items():
            role_onehot[i, pos, ROLE_INDEX[r]] = 1.0

    sets_elems = [set(parse_obj(v)) for v in ml["elements"].fillna("[]").tolist()]
    sets_mats = [set(parse_obj(v)) for v in ml["material_ids"].fillna("[]").tolist()]

    def edge_features(edges: torch.Tensor) -> torch.Tensor:
        if edges.numel() == 0:
            return torch.zeros((0, 8), dtype=torch.float32)
        feats = []
        for k in range(edges.size(1)):
            a = int(edges[0, k].item())
            b = int(edges[1, k].item())
            sa, sb = sets_mats[a], sets_mats[b]
            ea, eb = sets_elems[a], sets_elems[b]
            mat_j = len(sa & sb) / max(1, len(sa | sb))
            el_j = len(ea & eb) / max(1, len(ea | eb))
            emb_sim = float(F.cosine_similarity(avg_emb[a : a + 1], avg_emb[b : b + 1]).item())
            v_sim = 1.0 / (1.0 + abs(float(battery_feat[a, 0] - battery_feat[b, 0])))
            c_sim = 1.0 / (1.0 + abs(float(battery_feat[a, 1] - battery_feat[b, 1])))
            ion_match = 1.0 if ions[a] == ions[b] else 0.0
            role_overlap = float((role_onehot[a] * role_onehot[b]).sum().item()) / max(1.0, float(role_onehot[a].sum().item() + role_onehot[b].sum().item()))
            vol_sim = 1.0 / (1.0 + abs(float(battery_feat[a, 2] - battery_feat[b, 2])))
            feats.append([mat_j, el_j, emb_sim, v_sim, c_sim, ion_match, role_overlap, vol_sim])
        return torch.tensor(feats, dtype=torch.float32)

    edge_features_dict = {et: edge_features(ei) for et, ei in edge_index_dict.items()}

    # Fused multi-view system graph for downstream hetero GNN/transformer models.
    fused_pairs: Dict[Tuple[int, int], float] = {}
    view_weights = {0: 0.34, 1: 0.18, 2: 0.12, 3: 0.16, 4: 0.20}
    for et, ei in edge_index_dict.items():
        w = float(view_weights.get(et, 0.1))
        if ei.numel() == 0:
            continue
        for k in range(ei.size(1)):
            a = int(ei[0, k].item())
            b = int(ei[1, k].item())
            fused_pairs[(a, b)] = fused_pairs.get((a, b), 0.0) + w

    fused_edges = [[a, b] for (a, b) in fused_pairs.keys()]
    fused_weights = torch.tensor([fused_pairs[(a, b)] for (a, b) in fused_pairs.keys()], dtype=torch.float32)
    if fused_weights.numel() > 0:
        fused_weights = fused_weights / fused_weights.max().clamp_min(1e-6)
    fused_edge_index = as_edge_index(fused_edges)

    unique_mats = sorted({m for s in sets_mats for m in s if str(m)})
    midx = {m: i for i, m in enumerate(unique_mats)}
    material_nodes = torch.zeros((len(unique_mats), MATERIAL_DIM), dtype=torch.float32)
    for m, i in midx.items():
        vec = emb_map.get(m)
        if vec is not None:
            material_nodes[i] = torch.tensor(vec, dtype=torch.float32)

    sm_edges = []
    for i, mats in enumerate(sets_mats):
        for m in mats:
            j = midx.get(str(m))
            if j is not None:
                sm_edges.append([i, j])

    # Material-material graph from curated atomic embeddings.
    mm_edges: List[List[int]] = []
    if len(unique_mats) > 1:
        mm_knn = cosine_knn(material_nodes, min(K_MATERIAL, len(unique_mats) - 1))
        for i, nbrs in enumerate(mm_knn):
            mm_edges.extend([[i, j] for j in nbrs if i != j])

    # Ion nodes and system->ion links.
    unique_ions = sorted(set(ions))
    ion_idx = {ion: i for i, ion in enumerate(unique_ions)}
    ion_features = torch.zeros((len(unique_ions), len(unique_ions)), dtype=torch.float32)
    for ion, iidx in ion_idx.items():
        ion_features[iidx, iidx] = 1.0
    system_to_ion_edges = [[i, ion_idx[ions[i]]] for i in range(n) if ions[i] in ion_idx]

    # Context features for each battery/system (kept separate from 7D canonical targets).
    ion_onehot = torch.zeros((n, len(unique_ions)), dtype=torch.float32)
    for i, ion in enumerate(ions):
        j = ion_idx.get(ion)
        if j is not None:
            ion_onehot[i, j] = 1.0
    role_conf_stats = torch.stack(
        [
            role_conf.mean(dim=1),
            role_conf.max(dim=1).values,
            role_conf.min(dim=1).values,
        ],
        dim=1,
    )
    mask_count = node_mask.sum(dim=1, keepdim=True)
    context_features = torch.cat(
        [
            mask_count / float(NUM_MATERIALS),
            role_conf_stats,
            std_emb.mean(dim=1, keepdim=True),
            std_emb.max(dim=1, keepdim=True).values,
            ion_onehot,
            element_frac,
        ],
        dim=1,
    )

    battery_np = battery_feat.detach().cpu().numpy()
    feature_mean = [float(v) for v in battery_np.mean(axis=0).tolist()]
    feature_std = [float(max(v, 1e-8)) for v in battery_np.std(axis=0).tolist()]
    feature_min = [float(v) for v in battery_np.min(axis=0).tolist()]
    feature_max = [float(v) for v in battery_np.max(axis=0).tolist()]
    dataset_fingerprint = build_dataset_fingerprint(
        paths=[DATA_ML, DATA_PARSED, ATOMIC_EMB],
        row_count=n,
    )

    coverage = float(covered_material_refs / max(1, total_material_refs))
    graph = {
        "battery_features": battery_feat,
        "material_embeddings": material_emb,
        "node_masks": node_mask,
        "edge_index_dict": edge_index_dict,
        "edge_features_dict": edge_features_dict,
        "fused_system_edge_index": fused_edge_index,
        "fused_system_edge_weight": fused_weights,
        "role_assignments": role_name_slots,
        "role_confidence": role_conf,
        "system_context_features": context_features,
        "hetero_material_node_features": material_nodes,
        "hetero_system_to_material_edge_index": as_edge_index(sm_edges),
        "hetero_material_to_material_edge_index": as_edge_index(mm_edges),
        "hetero_material_id_to_index": midx,
        "hetero_ion_node_features": ion_features,
        "hetero_ion_to_index": ion_idx,
        "hetero_system_to_ion_edge_index": as_edge_index(system_to_ion_edges),
        "metadata": {
            "version": "masked_graph_v2.1_multiview",
            "num_nodes": n,
            "battery_feature_dim": BATTERY_DIM,
            "system_feature_order": list(BATTERY_FEATURE_ORDER),
            "dataset_fingerprint": dataset_fingerprint,
            "scaler_type": "standard_scaler",
            "scaler_parameters": {
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "feature_min": feature_min,
                "feature_max": feature_max,
            },
            "feature_engineering_steps": [
                "curated_target_normalization_v2",
                "multiview_system_graph_fusion",
                "role_conditioned_material_slotting",
            ],
            "clipping_rules": {
                key: {"min": feature_min[i], "max": feature_max[i], "space": "normalized"}
                for i, key in enumerate(BATTERY_FEATURE_ORDER)
            },
            "log_transform_flags": {key: False for key in BATTERY_FEATURE_ORDER},
            "unit_definitions": {key: "dimensionless_normalized" for key in BATTERY_FEATURE_ORDER},
            "num_material_slots": NUM_MATERIALS,
            "material_embedding_dim": MATERIAL_DIM,
            "edge_types": {
                "0": "atomic_similarity",
                "1": "chemsys_similarity",
                "2": "working_ion_similarity",
                "3": "stoichiometry_similarity",
                "4": "physics_similarity",
            },
            "sources": {
                "battery_features": str(DATA_ML),
                "battery_parsed": str(DATA_PARSED),
                "atomic_embeddings": str(ATOMIC_EMB),
            },
            "atomic_embedding_coverage": {
                "covered_material_refs": int(covered_material_refs),
                "total_material_refs": int(total_material_refs),
                "coverage_fraction": coverage,
                "unique_material_nodes": int(len(unique_mats)),
            },
            "construction": {
                "safe_parsing": "ast.literal_eval/json fallback",
                "knn": {
                    "atomic": K_ATOMIC,
                    "chemsys": K_CHEMSYS,
                    "ion": K_ION,
                    "stoichiometry": K_STOICH,
                    "physics": K_PHYSICS,
                    "material": K_MATERIAL,
                },
                "fused_view_weights": view_weights,
                "context_feature_dim": int(context_features.size(1)),
            },
        },
    }
    return graph


def main() -> None:
    graph = build()
    OUT_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, OUT_GRAPH)
    logger.info("Saved graph: %s", OUT_GRAPH)
    meta = graph.get("metadata", {})
    logger.info("Graph nodes=%s edge_types=%s", meta.get("num_nodes"), len(graph.get("edge_index_dict", {})))
    logger.info("Atomic embedding coverage=%.4f", float(meta.get("atomic_embedding_coverage", {}).get("coverage_fraction", 0.0)))


if __name__ == "__main__":
    main()
