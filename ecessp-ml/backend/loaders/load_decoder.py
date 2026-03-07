# ============================================================
# Decoder Loader (Embedding → Canonical Properties)
# ============================================================
# Purpose:
#   - Map HGT embedding → canonical battery properties (7 targets)
#   - Load decoder weights (if available)
#   - Remain independent from model loader
#
# NO inference logic here.
# ============================================================

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import logging

from backend.config import MODEL_CONFIG

logger = logging.getLogger(__name__)

# ============================================================
# Canonical Target Properties (LOCKED)
# ============================================================
CANONICAL_PROPERTIES = [
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "stability_charge",
    "stability_discharge",
]

# ============================================================
# Decoder Architecture
# ============================================================

class PropertyDecoder(nn.Module):
    """
    Simple MLP decoder:
        embedding → hidden → canonical properties
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(CANONICAL_PROPERTIES)),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor shaped [batch, 7] (or [7] for single item).
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # add batch

        output = self.network(embedding)
        return output

# ============================================================
# Loader Function
# ============================================================

def load_decoder(device: torch.device | str = "cpu") -> PropertyDecoder:
    """
    Initialize the property decoder and optionally load checkpoint.

    Parameters
    ----------
    device : torch.device or str

    Returns
    -------
    PropertyDecoder
    """
    if isinstance(device, str):
        device = torch.device(device)

    embedding_dim = getattr(MODEL_CONFIG, "latent_dim", 128)
    decoder = PropertyDecoder(embedding_dim=embedding_dim)

    # --------------------------------------------------------
    # Optional: load decoder checkpoint if provided
    # --------------------------------------------------------
    checkpoint_path = getattr(MODEL_CONFIG, "decoder_checkpoint_path", None)
    if checkpoint_path:
        try:
            if checkpoint_path.exists():
                state = torch.load(checkpoint_path, map_location=device)
                decoder.load_state_dict(state)
                logger.info("Decoder weights loaded from %s", checkpoint_path)
            else:
                logger.warning("Decoder checkpoint not found: %s", checkpoint_path)
        except Exception as exc:
            logger.exception("Failed to load decoder checkpoint: %s", exc)

    decoder.to(device)
    decoder.eval()
    return decoder

# ============================================================
# Introspection Utility
# ============================================================

def describe_decoder() -> dict:
    """
    Return read-only decoder metadata.
    """
    return {
        "num_targets": len(CANONICAL_PROPERTIES),
        "targets": CANONICAL_PROPERTIES.copy(),
        "hidden_dim": 256,
        "locked": True,
        "compatible_models": ["BatteryHGT", "BatteryRGCN", "BatteryHGT_VAE"],
    }
