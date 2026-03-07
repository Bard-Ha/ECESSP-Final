#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from models.hetero_hgt import HeteroBatteryHGT


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

    def to_raw(self, y: np.ndarray) -> np.ndarray:
        if abs(float(self.scale)) < 1e-12:
            return y.astype(np.float64)
        return (y.astype(np.float64) - float(self.shift)) / float(self.scale)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


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


def _to_raw(arr_norm: np.ndarray, maps: Dict[str, AffineMap]) -> np.ndarray:
    arr = arr_norm.astype(np.float64, copy=True)
    for i, name in enumerate(TARGET_ORDER):
        arr[:, i] = maps[name].to_raw(arr[:, i])
    return arr


def forward_all(model: HeteroBatteryHGT, graph: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    node = graph["node_features"]
    edges = graph["edge_index_dict"]
    node_dev = {k: v.to(device) for k, v in node.items()}
    edges_dev = {k: v.to(device) for k, v in edges.items()}
    return model(
        system_x=node_dev["system"],
        material_x=node_dev["material"],
        ion_x=node_dev["ion"],
        role_x=node_dev["role"],
        edge_index_dict=edges_dev,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HGT on true hetero battery graph")
    ap.add_argument("--graph", default="graphs/battery_hetero_graph_v1.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--out-summary", default="reports/training_summary_hgt.json")
    ap.add_argument("--split-json", default="", help="Optional external split JSON override.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    graph_path = (root / args.graph).resolve()
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps_raw = graph.get("normalization_maps", {})
    maps = {
        k: AffineMap(scale=_safe_float(v.get("scale"), 1.0), shift=_safe_float(v.get("shift"), 0.0))
        for k, v in maps_raw.items()
        if k in TARGET_ORDER
    }
    if len(maps) != len(TARGET_ORDER):
        raise RuntimeError("hetero graph missing normalization_maps for canonical targets")

    split = graph["split_indices"]
    if str(args.split_json or "").strip():
        split_path = Path(str(args.split_json))
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()
        if not split_path.exists():
            raise RuntimeError(f"split file not found: {split_path}")
        split_payload = json.loads(split_path.read_text(encoding="utf-8-sig"))

        def _idx(name: str) -> torch.Tensor:
            vals = split_payload.get(name, [])
            if not isinstance(vals, list):
                return torch.zeros((0,), dtype=torch.long, device=device)
            return torch.tensor([int(v) for v in vals], dtype=torch.long, device=device)

        idx_train = _idx("train")
        idx_val = _idx("val")
        idx_iid = _idx("test_iid")
        idx_ood = _idx("test_ood")
    else:
        idx_train = split["train"].long().to(device)
        idx_val = split["val"].long().to(device)
        idx_iid = split["test_iid"].long().to(device)
        idx_ood = split["test_ood"].long().to(device)

    y = graph["targets_norm"].float().to(device)
    system_x = graph["node_features"]["system"].float()
    system_dim = int(system_x.size(1))
    if tuple(system_x.shape) == tuple(graph["targets_norm"].shape):
        try:
            if torch.allclose(system_x, graph["targets_norm"].float(), atol=1e-6, rtol=1e-5):
                raise RuntimeError(
                    "Detected target leakage: system node features are identical to targets_norm. "
                    "Rebuild hetero graph with non-target system features."
                )
        except RuntimeError:
            raise
        except Exception:
            pass
    n_ion = int(graph["node_features"]["ion"].size(1))
    n_role = int(graph["node_features"]["role"].size(1))

    model = HeteroBatteryHGT(
        system_dim=system_dim,
        property_dim=len(TARGET_ORDER),
        material_dim=int(graph["node_features"]["material"].size(1)),
        ion_dim=n_ion,
        role_dim=n_role,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        enable_uncertainty=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(4, int(args.patience) // 4), min_lr=1e-6)

    best_val = math.inf
    best_epoch = -1
    wait = 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = root / "reports" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / f"hgt_best_{stamp}.pt"

    for ep in range(int(args.epochs)):
        model.train()
        out = forward_all(model, graph, device)
        pred = out["properties"]
        logvar = out["property_log_variance"]

        reg = F.mse_loss(pred[idx_train], y[idx_train])
        unc = torch.exp(torch.clamp(logvar[idx_train], min=-8.0, max=8.0)).mean()
        loss = reg + 0.08 * unc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            out_val = forward_all(model, graph, device)
            pred_val = out_val["properties"]
            val_loss = F.mse_loss(pred_val[idx_val], y[idx_val]).item()
        sch.step(val_loss)

        print(
            f"[epoch {ep:03d}] train={float(loss.item()):.5f} val={float(val_loss):.5f} "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = int(ep)
            wait = 0
            torch.save(
                {
                    "epoch": int(ep),
                    "best_val_loss": float(best_val),
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "system_dim": system_dim,
                        "property_dim": len(TARGET_ORDER),
                        "material_dim": int(graph["node_features"]["material"].size(1)),
                        "ion_dim": n_ion,
                        "role_dim": n_role,
                        "hidden_dim": int(args.hidden_dim),
                        "num_layers": int(args.num_layers),
                        "dropout": float(args.dropout),
                    },
                    "graph_path": str(graph_path),
                    "normalization_maps": maps_raw,
                },
                ckpt_path,
            )
        else:
            wait += 1
            if wait >= int(args.patience):
                print(f"[early-stop] epoch={ep}")
                break

    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        out_all = forward_all(model, graph, device)
        pred_norm = out_all["properties"].detach().cpu().numpy()
        y_norm = y.detach().cpu().numpy()

    idx_iid_np = idx_iid.detach().cpu().numpy()
    idx_ood_np = idx_ood.detach().cpu().numpy()
    y_iid_norm = y_norm[idx_iid_np]
    p_iid_norm = pred_norm[idx_iid_np]
    y_ood_norm = y_norm[idx_ood_np] if idx_ood_np.size else np.zeros((0, len(TARGET_ORDER)))
    p_ood_norm = pred_norm[idx_ood_np] if idx_ood_np.size else np.zeros((0, len(TARGET_ORDER)))

    y_iid_raw = _to_raw(y_iid_norm, maps)
    p_iid_raw = _to_raw(p_iid_norm, maps)
    y_ood_raw = _to_raw(y_ood_norm, maps) if idx_ood_np.size else y_ood_norm
    p_ood_raw = _to_raw(p_ood_norm, maps) if idx_ood_np.size else p_ood_norm

    iid_norm_metrics = per_property_metrics(y_iid_norm, p_iid_norm)
    iid_raw_metrics = per_property_metrics(y_iid_raw, p_iid_raw)
    if idx_ood_np.size:
        ood_norm_metrics = per_property_metrics(y_ood_norm, p_ood_norm)
        ood_raw_metrics = per_property_metrics(y_ood_raw, p_ood_raw)
    else:
        ood_norm_metrics = {"overall_micro": regression_metrics([], [])}
        ood_raw_metrics = {"overall_micro": regression_metrics([], [])}

    summary = {
        "timestamp": stamp,
        "model_family": "hetero_hgt",
        "graph_path": str(graph_path),
        "device": str(device),
        "system_feature_dim": system_dim,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_checkpoint": str(ckpt_path.resolve()),
        "evaluation_metrics": {
            "iid": {"normalized_space": iid_norm_metrics, "raw_space": iid_raw_metrics},
            "ood": {"normalized_space": ood_norm_metrics, "raw_space": ood_raw_metrics},
        },
        "physics_violation_statistics": {
            "iid": {
                "capacity_violation_rate": float(np.mean(p_iid_raw[:, 1] > 350.0)) if p_iid_raw.size else 0.0,
                "voltage_above_4p4_rate": float(np.mean(p_iid_raw[:, 0] > 4.4)) if p_iid_raw.size else 0.0,
            },
            "ood": {
                "capacity_violation_rate": float(np.mean(p_ood_raw[:, 1] > 350.0)) if p_ood_raw.size else 0.0,
                "voltage_above_4p4_rate": float(np.mean(p_ood_raw[:, 0] > 4.4)) if p_ood_raw.size else 0.0,
            },
        },
    }

    out_summary = (root / args.out_summary).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] summary: {out_summary}")


if __name__ == "__main__":
    main()
