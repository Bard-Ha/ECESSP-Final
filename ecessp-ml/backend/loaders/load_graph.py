# backend/loaders/load_graph.py
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Any

import torch
import logging

from backend.config import GRAPH_CONFIG

logger = logging.getLogger(__name__)


__all__ = [
    "load_graph",
    "describe_graph",
]


REQUIRED_GRAPH_KEYS: Set[str] = {
    "x",
    "edge_index_dict",
}

# MaskedGNN graph keys
MASKED_GNN_GRAPH_KEYS: Set[str] = {
    "battery_features",
    "material_embeddings", 
    "node_masks",
    "edge_index_dict",
    "edge_features_dict",
    "role_assignments",
    "metadata",
}

REQUIRED_NODE_TYPE = "battery"


# ============================================================
# Path Resolution
# ============================================================

def _resolve_graph_path() -> Path:
    graph_path = GRAPH_CONFIG.graph_dir / GRAPH_CONFIG.filename

    logger.debug("Resolving graph path: %s", graph_path)

    if not graph_path.exists():
        raise FileNotFoundError(
            f"[GraphLoader] Graph file not found: {graph_path}"
        )

    if not graph_path.is_file():
        raise RuntimeError(
            f"[GraphLoader] Graph path is not a file: {graph_path}"
        )

    return graph_path


# ============================================================
# Schema Validation
# ============================================================

def _validate_graph_object(graph: Dict[str, Any]) -> None:
    # Check if it's a MaskedGNN graph format
    if MASKED_GNN_GRAPH_KEYS.issubset(graph.keys()):
        logger.info("MaskedGNN graph format detected")
        return
    
    # Check if it's the old format
    missing = REQUIRED_GRAPH_KEYS - graph.keys()
    if missing:
        raise RuntimeError(
            f"[GraphLoader] Graph missing required keys: {missing}"
        )

    x = graph["x"]
    edge_index_dict = graph["edge_index_dict"]

    # -----------------------------
    # Node features
    # -----------------------------
    if not isinstance(x, torch.Tensor):
        raise TypeError("[GraphLoader] graph['x'] must be torch.Tensor")

    if x.ndim != 2:
        raise ValueError(
            "[GraphLoader] graph['x'] must have shape [num_nodes, num_features]"
        )

    num_nodes, num_features = x.shape

    if num_nodes <= 0 or num_features <= 0:
        raise ValueError("[GraphLoader] Invalid node feature matrix")

    # -----------------------------
    # Edges
    # -----------------------------
    if not isinstance(edge_index_dict, dict):
        raise TypeError(
            "[GraphLoader] graph['edge_index_dict'] must be dict"
        )

    if not edge_index_dict:
        raise ValueError("[GraphLoader] edge_index_dict is empty")

    for key, edge_index in edge_index_dict.items():

        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(
                f"[GraphLoader] Invalid edge key {key}"
            )

        src, rel, dst = key

        if src != REQUIRED_NODE_TYPE or dst != REQUIRED_NODE_TYPE:
            raise ValueError(
                f"[GraphLoader] Invalid node type in relation {key}"
            )

        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(
                f"[GraphLoader] Edge index for '{rel}' must be torch.Tensor"
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                f"[GraphLoader] Edge index for '{rel}' must be torch.long"
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"[GraphLoader] Edge index for '{rel}' "
                "must have shape [2, num_edges]"
            )

        if edge_index.numel() == 0:
            raise ValueError(
                f"[GraphLoader] Relation '{rel}' has no edges"
            )

        min_idx = int(edge_index.min().item())
        max_idx = int(edge_index.max().item())

        if min_idx < 0 or max_idx >= num_nodes:
            raise ValueError(
                f"[GraphLoader] Relation '{rel}' has invalid node indices"
            )


# ============================================================
# Public Loader
# ============================================================

def load_graph(
    device: torch.device | None = None,
) -> Dict[str, Any]:

    path = _resolve_graph_path()

    try:
        logger.info("Loading graph from %s", path)
        graph = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        logger.exception("Failed to load graph from %s", path)
        raise

    if not isinstance(graph, dict):
        raise RuntimeError(
            "[GraphLoader] Loaded graph object is not a dict"
        )

    _validate_graph_object(graph)

    # Handle MaskedGNN format - preserve it for the enhanced engine
    if MASKED_GNN_GRAPH_KEYS.issubset(graph.keys()):
        logger.info("MaskedGNN graph format detected - preserving format")
        # Just copy to prevent accidental mutation of original loaded object
        graph = {
            "battery_features": graph["battery_features"],
            "material_embeddings": graph["material_embeddings"],
            "node_masks": graph["node_masks"],
            "edge_index_dict": dict(graph["edge_index_dict"]),
            "edge_features_dict": graph["edge_features_dict"],
            "role_assignments": graph["role_assignments"],
            "metadata": graph["metadata"],
        }
    else:
        # Handle old format
        # Copy to prevent accidental mutation of original loaded object
        graph = {
            "x": graph["x"],
            "edge_index_dict": dict(graph["edge_index_dict"]),
        }

    # --------------------------------------------------------
    # Device move
    # --------------------------------------------------------
    if device is not None:
        with torch.no_grad():
            if "battery_features" in graph:
                # MaskedGNN format
                graph["battery_features"] = graph["battery_features"].to(device).contiguous()
                graph["material_embeddings"] = graph["material_embeddings"].to(device).contiguous()
                graph["node_masks"] = graph["node_masks"].to(device).contiguous()
                graph["edge_index_dict"] = {
                    k: v.to(device).contiguous()
                    for k, v in graph["edge_index_dict"].items()
                }
            else:
                # Old format
                graph["x"] = graph["x"].to(device).contiguous()
                graph["edge_index_dict"] = {
                    k: v.to(device).contiguous()
                    for k, v in graph["edge_index_dict"].items()
                }

    return graph


# ============================================================
# Introspection Utility
# ============================================================

def describe_graph(graph: Dict[str, Any]) -> Dict[str, Any]:

    x = graph["x"]
    edge_index_dict = graph["edge_index_dict"]

    return {
        "num_nodes": int(x.size(0)),
        "node_feature_dim": int(x.size(1)),
        "num_relations": len(edge_index_dict),
        "relations": sorted({k[1] for k in edge_index_dict}),
    }
