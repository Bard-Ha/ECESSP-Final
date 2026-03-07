# ============================================================
# Property Decoder (Embedding → Canonical Properties)
# ============================================================

from typing import Dict, Any
import torch


class PropertyDecoder(torch.nn.Module):
    """
    Maps latent embedding → canonical battery properties.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 8),  # 8 canonical properties
        )

        self.property_order = [
            "average_voltage",
            "capacity_grav",
            "capacity_vol",
            "energy_grav",
            "energy_vol",
            "stability_charge",
            "stability_discharge",
            "max_delta_volume",
        ]

    def forward(self, embedding: torch.Tensor) -> Dict[str, Any]:
        out = self.head(embedding)

        if out.dim() == 2:
            out = out.squeeze(0)

        values = out.detach().cpu().numpy().tolist()

        return dict(zip(self.property_order, values))