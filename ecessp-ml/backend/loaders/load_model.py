# backend/loaders/load_model.py
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Type

import torch
import logging

from backend.config import MODEL_CONFIG

logger = logging.getLogger(__name__)


# ============================================================
# Model Registry (Lazy)
# ============================================================

MODEL_REGISTRY: Dict[str, Type[torch.nn.Module]] = {}


# ============================================================
# Checkpoint Resolution
# ============================================================

def _resolve_checkpoint(path: Optional[str | Path]) -> Path:
    if path is None:
        raise RuntimeError(
            "[ModelLoader] MODEL_CONFIG.checkpoint_path is not set."
        )

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"[ModelLoader] Checkpoint not found: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"[ModelLoader] Checkpoint path is not a file: {checkpoint_path}"
        )

    return checkpoint_path


# ============================================================
# Checkpoint Validation
# ============================================================

def _validate_checkpoint(checkpoint: dict) -> None:
    # Support both old format (model_kwargs, state_dict) and new MaskedGNN format
    if "model_kwargs" in checkpoint and "state_dict" in checkpoint:
        # Old format
        required = {"model_kwargs", "state_dict"}
        missing = required - checkpoint.keys()
        if missing:
            raise RuntimeError(
                f"[ModelLoader] Invalid checkpoint format. Missing keys: {missing}"
            )
        if not isinstance(checkpoint["model_kwargs"], dict):
            raise RuntimeError("[ModelLoader] model_kwargs must be a dict")
        if not isinstance(checkpoint["state_dict"], dict):
            raise RuntimeError("[ModelLoader] state_dict must be a dict")
    elif "config" in checkpoint and "model_state_dict" in checkpoint:
        # MaskedGNN format
        if not isinstance(checkpoint["config"], dict):
            raise RuntimeError("[ModelLoader] config must be a dict")
        if not isinstance(checkpoint["model_state_dict"], dict):
            raise RuntimeError("[ModelLoader] model_state_dict must be a dict")
    else:
        raise RuntimeError("[ModelLoader] Invalid checkpoint format. Expected either {'model_kwargs', 'state_dict'} or {'config', 'model_state_dict'}")


# ============================================================
# Registry Population
# ============================================================

def _populate_registry_once() -> None:
    if MODEL_REGISTRY:
        return

    try:
        from models.masked_gnn import MaskedGNN
        MODEL_REGISTRY["MASKED_GNN"] = MaskedGNN
    except Exception:
        logger.debug("MaskedGNN import failed", exc_info=True)


def _infer_decoder_flags_from_state_dict(state_dict: dict) -> dict:
    keys = set(state_dict.keys())
    is_multihead = any(k.startswith("decoder_trunk.") for k in keys) or any(
        k.startswith("voltage_head.") or k.startswith("capacity_head.") or k.startswith("stability_head.")
        for k in keys
    )
    has_logvar = any(k.startswith("logvar_head.") for k in keys)
    has_role = any(k.startswith("role_head.") for k in keys)
    has_compat = any(k.startswith("compatibility_head.") for k in keys)

    role_dim = 3
    compat_dim = 3
    try:
        rw = state_dict.get("role_head.weight")
        if isinstance(rw, torch.Tensor) and rw.ndim == 2:
            role_dim = int(rw.shape[0])
    except Exception:
        role_dim = 3
    try:
        cw = state_dict.get("compatibility_head.weight")
        if isinstance(cw, torch.Tensor) and cw.ndim == 2:
            compat_dim = int(cw.shape[0])
    except Exception:
        compat_dim = 3
    return {
        "decoder_mode": "multihead" if is_multihead else "legacy",
        "enable_uncertainty": bool(has_logvar),
        "enable_role_head": bool(has_role),
        "enable_compatibility_head": bool(has_compat),
        "role_output_dim": int(role_dim),
        "compatibility_output_dim": int(compat_dim),
    }


# ============================================================
# Public Loader
# ============================================================

def load_model(
    *,
    checkpoint_path: Optional[str | Path] = None,
    model_type: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:

    _populate_registry_once()

    model_type = model_type or MODEL_CONFIG.model_type
    device = device or torch.device("cpu")

    if model_type not in MODEL_REGISTRY:
        raise RuntimeError(
            f"[ModelLoader] Model '{model_type}' not available. "
            f"Registered: {list(MODEL_REGISTRY.keys())}"
        )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "[ModelLoader] CUDA requested but not available."
        )

    checkpoint_path = _resolve_checkpoint(
        checkpoint_path or MODEL_CONFIG.checkpoint_path
    )

    # --------------------------------------------------------
    # Load checkpoint safely on CPU
    # --------------------------------------------------------
    try:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,   # PyTorch 2.x safety
        )
    except Exception as exc:
        logger.exception("Checkpoint load failed")
        raise RuntimeError(
            f"[ModelLoader] Failed to load checkpoint {checkpoint_path}: {exc}"
        ) from exc

    _validate_checkpoint(checkpoint)

    model_cls = MODEL_REGISTRY[model_type]

    try:
        if "model_kwargs" in checkpoint:
            # Old format
            model_kwargs = checkpoint["model_kwargs"]
            state_dict = checkpoint["state_dict"]
        elif "config" in checkpoint:
            # MaskedGNN format - filter config to only include model parameters
            config = checkpoint["config"]
            valid_params = [
                "battery_feature_dim",
                "material_embedding_dim",
                "num_materials",
                "hidden_dim",
                "latent_dim",
                "num_gnn_layers",
                "gnn_type",
                "dropout",
                "decoder_mode",
                "enable_uncertainty",
                "material_interaction",
                "enable_graph_context",
                "graph_context_dim",
                "enable_role_head",
                "enable_compatibility_head",
                "role_output_dim",
                "compatibility_output_dim",
            ]
            model_kwargs = {k: v for k, v in config.items() if k in valid_params}
            state_dict = checkpoint["model_state_dict"]
            inferred = _infer_decoder_flags_from_state_dict(state_dict)
            model_kwargs.setdefault("decoder_mode", inferred["decoder_mode"])
            model_kwargs.setdefault("enable_uncertainty", inferred["enable_uncertainty"])
        else:
            raise RuntimeError("[ModelLoader] Unknown checkpoint format")

        # Safety net for older checkpoints/configs without explicit decoder flags.
        inferred = _infer_decoder_flags_from_state_dict(state_dict)
        model_kwargs.setdefault("decoder_mode", inferred["decoder_mode"])
        model_kwargs.setdefault("enable_uncertainty", inferred["enable_uncertainty"])
        
        model = model_cls(**model_kwargs)
    except Exception as exc:
        logger.exception("Model construction failed")
        raise RuntimeError(
            f"[ModelLoader] Failed to construct model '{model_type}': {exc}"
        ) from exc

    # Strict load
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"[ModelLoader] State dict mismatch. "
            f"Missing: {missing}, Unexpected: {unexpected}"
        )

    # --------------------------------------------------------
    # Device transfer (after load)
    # --------------------------------------------------------
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------------
    # Deterministic inference mode
    # --------------------------------------------------------
    if MODEL_CONFIG.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return model


# ============================================================
# Introspection
# ============================================================

def describe_model(model: torch.nn.Module) -> Dict[str, int | str]:
    return {
        "class": model.__class__.__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "device": next(model.parameters()).device.type,
    }
