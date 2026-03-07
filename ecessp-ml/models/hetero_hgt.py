#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class HeteroBatteryHGT(nn.Module):
    """
    Lightweight hetero-graph transformer for battery-system prediction.
    Node types:
    - system
    - material
    - ion
    - role
    """

    def __init__(
        self,
        system_dim: int = 7,
        property_dim: int = 7,
        material_dim: int = 64,
        ion_dim: int = 8,
        role_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
        enable_uncertainty: bool = True,
    ):
        super().__init__()
        self.system_dim = int(system_dim)
        self.property_dim = int(property_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.enable_uncertainty = bool(enable_uncertainty)

        self.system_encoder = nn.Sequential(
            nn.Linear(system_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.material_encoder = nn.Sequential(
            nn.Linear(material_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ion_encoder = nn.Sequential(
            nn.Linear(ion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.role_encoder = nn.Sequential(
            nn.Linear(role_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.material_msg_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4 if hidden_dim % 4 == 0 else 2,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.token_transformer = nn.TransformerEncoder(enc_layer, num_layers=max(1, num_layers))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.property_dim),
        )
        if self.enable_uncertainty:
            self.logvar_head = nn.Linear(hidden_dim * 2, self.property_dim)
        else:
            self.logvar_head = None

    @staticmethod
    def _aggregate_from_edges(
        src_size: int,
        dst_x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros((src_size, dst_x.size(1)), device=dst_x.device, dtype=dst_x.dtype)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        out = torch.zeros((src_size, dst_x.size(1)), device=dst_x.device, dtype=dst_x.dtype)
        cnt = torch.zeros((src_size, 1), device=dst_x.device, dtype=dst_x.dtype)
        out.index_add_(0, src, dst_x[dst])
        cnt.index_add_(0, src, torch.ones((src.size(0), 1), device=dst_x.device, dtype=dst_x.dtype))
        return out / cnt.clamp_min(1.0)

    @staticmethod
    def _material_message_pass(
        material_x: torch.Tensor,
        edge_index: torch.Tensor,
        msg_layer: nn.Module,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return material_x
        src = edge_index[0].long()
        dst = edge_index[1].long()
        msg = msg_layer(material_x[src])
        out = torch.zeros_like(material_x)
        cnt = torch.zeros((material_x.size(0), 1), device=material_x.device, dtype=material_x.dtype)
        out.index_add_(0, dst, msg)
        cnt.index_add_(0, dst, torch.ones((dst.size(0), 1), device=material_x.device, dtype=material_x.dtype))
        return 0.5 * material_x + 0.5 * (out / cnt.clamp_min(1.0))

    def forward(
        self,
        *,
        system_x: torch.Tensor,
        material_x: torch.Tensor,
        ion_x: torch.Tensor,
        role_x: torch.Tensor,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        s = self.system_encoder(system_x)
        m = self.material_encoder(material_x)
        i = self.ion_encoder(ion_x)
        r = self.role_encoder(role_x)

        mm_key = ("material", "similar_material", "material")
        mm_edges = edge_index_dict.get(mm_key, torch.zeros((2, 0), device=material_x.device, dtype=torch.long))
        for layer in self.material_msg_mlp:
            m = self._material_message_pass(m, mm_edges, layer)

        sm = self._aggregate_from_edges(
            src_size=s.size(0),
            dst_x=m,
            edge_index=edge_index_dict.get(("system", "has_material", "material"), torch.zeros((2, 0), device=s.device, dtype=torch.long)),
        )
        si = self._aggregate_from_edges(
            src_size=s.size(0),
            dst_x=i,
            edge_index=edge_index_dict.get(("system", "uses_ion", "ion"), torch.zeros((2, 0), device=s.device, dtype=torch.long)),
        )
        sr = self._aggregate_from_edges(
            src_size=s.size(0),
            dst_x=r,
            edge_index=edge_index_dict.get(("system", "has_role", "role"), torch.zeros((2, 0), device=s.device, dtype=torch.long)),
        )

        tokens = torch.stack([s, sm, si, sr], dim=1)
        h = self.token_transformer(tokens)
        pooled = torch.cat([h[:, 0, :], h.mean(dim=1)], dim=1)

        properties = self.head(pooled)
        if self.logvar_head is not None:
            logvar = self.logvar_head(pooled)
        else:
            logvar = torch.zeros_like(properties)

        return {
            "properties": properties,
            "property_log_variance": logvar,
            "system_embedding": h[:, 0, :],
        }
