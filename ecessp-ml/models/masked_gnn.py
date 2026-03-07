#!/usr/bin/env python3
"""
Masked GNN model used by ECESSP runtime.

This implementation is aligned to the training checkpoint layout:
- material_encoder.*
- gnn_layers.*
- decoder.*
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Canonical property order used by model outputs and training targets.
PROPERTY_INDEX = {
    "average_voltage": 0,
    "capacity_grav": 1,
    "capacity_vol": 2,
    "energy_grav": 3,
    "energy_vol": 4,
    "stability_charge": 5,
    "stability_discharge": 6,
}

DEFAULT_PHYSICS_LIMITS = {
    "min_voltage": 1.0,
    "max_voltage": 4.4,
    "max_capacity_grav": 350.0,
    "max_energy_grav": 450.0,
    "max_energy_vol": 1200.0,
}

DEFAULT_LOSS_WEIGHTS = {
    "regression_mse": 1.0,
    "capacity_violation_penalty": 8.0,
    "voltage_violation_penalty": 8.0,
    "energy_cap_violation_penalty": 20.0,
    "physics_consistency_loss": 2.5,
    "uncertainty_regularization": 0.05,
}


def compute_physics_constrained_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    limits: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    c_theoretical: Optional[torch.Tensor] = None,
    pred_variance: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Physics-aware training loss for 7-property battery prediction.

    Expected tensor shape: [batch, 7]
    Property order must follow PROPERTY_INDEX.
    """
    if pred.ndim != 2 or pred.size(-1) != 7:
        raise ValueError(f"pred must be [N,7], got {tuple(pred.shape)}")
    if target.ndim != 2 or target.size(-1) != 7:
        raise ValueError(f"target must be [N,7], got {tuple(target.shape)}")

    cfg_limits = {**DEFAULT_PHYSICS_LIMITS, **(limits or {})}
    cfg_w = {**DEFAULT_LOSS_WEIGHTS, **(weights or {})}

    mse = F.mse_loss(pred, target)

    v = pred[:, PROPERTY_INDEX["average_voltage"]]
    c_grav = pred[:, PROPERTY_INDEX["capacity_grav"]]
    e_grav = pred[:, PROPERTY_INDEX["energy_grav"]]
    e_vol = pred[:, PROPERTY_INDEX["energy_vol"]]

    # Hard physical boundary penalties.
    voltage_low_pen = torch.relu(cfg_limits["min_voltage"] - v).pow(2).mean()
    voltage_high_pen = torch.relu(v - cfg_limits["max_voltage"]).pow(2).mean()
    voltage_violation = voltage_low_pen + voltage_high_pen

    capacity_limit = cfg_limits["max_capacity_grav"]
    if c_theoretical is not None:
        if c_theoretical.ndim > 1:
            c_theoretical = c_theoretical.view(-1)
        capacity_limit_tensor = torch.minimum(
            torch.full_like(c_grav, float(capacity_limit)),
            c_theoretical.to(c_grav.device, dtype=c_grav.dtype),
        )
    else:
        capacity_limit_tensor = torch.full_like(c_grav, float(capacity_limit))
    capacity_violation = torch.relu(c_grav - capacity_limit_tensor).pow(2).mean()

    energy_cap_violation = (
        torch.relu(e_grav - cfg_limits["max_energy_grav"]).pow(2).mean()
        + torch.relu(e_vol - cfg_limits["max_energy_vol"]).pow(2).mean()
    )

    # Physics consistency: E ~= V * C for gravimetric quantities.
    energy_consistency = (e_grav - (v * c_grav)).pow(2).mean()

    # Optional uncertainty term (for MC dropout / ensemble variance).
    if pred_variance is not None:
        if pred_variance.shape != pred.shape:
            raise ValueError("pred_variance must match pred shape")
        uncertainty_reg = pred_variance.mean()
    else:
        uncertainty_reg = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    total = (
        cfg_w["regression_mse"] * mse
        + cfg_w["capacity_violation_penalty"] * capacity_violation
        + cfg_w["voltage_violation_penalty"] * voltage_violation
        + cfg_w["energy_cap_violation_penalty"] * energy_cap_violation
        + cfg_w["physics_consistency_loss"] * energy_consistency
        + cfg_w["uncertainty_regularization"] * uncertainty_reg
    )

    return {
        "total": total,
        "mse": mse,
        "capacity_violation": capacity_violation,
        "voltage_violation": voltage_violation,
        "energy_cap_violation": energy_cap_violation,
        "energy_consistency": energy_consistency,
        "uncertainty_regularization": uncertainty_reg,
    }


class MaskedGNN(nn.Module):
    """
    Masked graph model for battery property prediction.
    """

    def __init__(
        self,
        battery_feature_dim: int = 7,
        material_embedding_dim: int = 64,
        num_materials: int = 5,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_gnn_layers: int = 3,
        gnn_type: str = "GIN",
        dropout: float = 0.2,
        decoder_mode: str = "legacy",
        enable_uncertainty: bool = False,
        material_interaction: str = "mlp",
        enable_graph_context: bool = False,
        graph_context_dim: int = 7,
        enable_role_head: bool = False,
        enable_compatibility_head: bool = False,
        role_output_dim: int = 3,
        compatibility_output_dim: int = 3,
        enable_insertion_probability_head: bool = False,
        enable_redox_potential_head: bool = False,
        enable_structure_type_head: bool = False,
        enable_volume_expansion_head: bool = False,
        structure_type_output_dim: int = 6,
    ):
        super().__init__()

        self.battery_feature_dim = battery_feature_dim
        self.material_embedding_dim = material_embedding_dim
        self.num_materials = num_materials
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.decoder_mode = decoder_mode.lower()
        self.enable_uncertainty = bool(enable_uncertainty)
        self.material_interaction = str(material_interaction).lower()
        self.enable_graph_context = bool(enable_graph_context)
        self.graph_context_dim = int(graph_context_dim)
        self.enable_role_head = bool(enable_role_head)
        self.enable_compatibility_head = bool(enable_compatibility_head)
        self.role_output_dim = int(role_output_dim)
        self.compatibility_output_dim = int(compatibility_output_dim)
        self.enable_insertion_probability_head = bool(enable_insertion_probability_head)
        self.enable_redox_potential_head = bool(enable_redox_potential_head)
        self.enable_structure_type_head = bool(enable_structure_type_head)
        self.enable_volume_expansion_head = bool(enable_volume_expansion_head)
        self.structure_type_output_dim = int(structure_type_output_dim)

        self.material_encoder = nn.Sequential(
            nn.Linear(material_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if self.material_interaction == "transformer":
            nhead = 4 if hidden_dim % 4 == 0 else (2 if hidden_dim % 2 == 0 else 1)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.material_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_gnn_layers)
            self.gnn_layers = nn.ModuleList()
            self.gat_q_layers = nn.ModuleList()
            self.gat_k_layers = nn.ModuleList()
            self.gat_v_layers = nn.ModuleList()
            self.gat_edge_layers = nn.ModuleList()
            self.gat_out_layers = nn.ModuleList()
            self.mpnn_message_layers = nn.ModuleList()
            self.mpnn_update_layers = nn.ModuleList()
        elif self.material_interaction == "gatv2":
            self.material_transformer = None
            self.gnn_layers = nn.ModuleList()
            self.gat_q_layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
            )
            self.gat_k_layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
            )
            self.gat_v_layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
            )
            self.gat_edge_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(num_gnn_layers)
                ]
            )
            self.gat_out_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_gnn_layers)
                ]
            )
            self.mpnn_message_layers = nn.ModuleList()
            self.mpnn_update_layers = nn.ModuleList()
        elif self.material_interaction == "mpnn":
            self.material_transformer = None
            self.gnn_layers = nn.ModuleList()
            self.gat_q_layers = nn.ModuleList()
            self.gat_k_layers = nn.ModuleList()
            self.gat_v_layers = nn.ModuleList()
            self.gat_edge_layers = nn.ModuleList()
            self.gat_out_layers = nn.ModuleList()
            self.mpnn_message_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_gnn_layers)
                ]
            )
            self.mpnn_update_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_gnn_layers)
                ]
            )
        else:
            self.material_transformer = None
            self.gnn_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_gnn_layers)
                ]
            )
            self.gat_q_layers = nn.ModuleList()
            self.gat_k_layers = nn.ModuleList()
            self.gat_v_layers = nn.ModuleList()
            self.gat_edge_layers = nn.ModuleList()
            self.gat_out_layers = nn.ModuleList()
            self.mpnn_message_layers = nn.ModuleList()
            self.mpnn_update_layers = nn.ModuleList()

        if self.enable_graph_context:
            if self.graph_context_dim != self.battery_feature_dim:
                self.context_proj = nn.Linear(self.graph_context_dim, self.battery_feature_dim)
            else:
                self.context_proj = nn.Identity()
            self.context_gate = nn.Sequential(
                nn.Linear(self.battery_feature_dim * 2, self.battery_feature_dim),
                nn.Sigmoid(),
            )
        else:
            self.context_proj = nn.Identity()
            self.context_gate = None

        if self.decoder_mode == "legacy":
            self.decoder = nn.Sequential(
                nn.Linear(battery_feature_dim + hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, battery_feature_dim),
            )
            if (
                self.enable_role_head
                or self.enable_compatibility_head
                or self.enable_insertion_probability_head
                or self.enable_redox_potential_head
                or self.enable_structure_type_head
                or self.enable_volume_expansion_head
            ):
                self.aux_trunk = nn.Sequential(
                    nn.Linear(battery_feature_dim + hidden_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
        else:
            # Shared decoder trunk, then explicit task heads.
            self.decoder_trunk = nn.Sequential(
                nn.Linear(battery_feature_dim + hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.voltage_head = nn.Linear(latent_dim, 1)
            self.capacity_head = nn.Linear(latent_dim, 2)  # grav, vol
            self.stability_head = nn.Linear(latent_dim, 2)  # charge, discharge
            if self.enable_uncertainty:
                self.logvar_head = nn.Linear(latent_dim, battery_feature_dim)

        head_input_dim = latent_dim
        if self.enable_role_head:
            self.role_head = nn.Linear(head_input_dim, self.role_output_dim)
        if self.enable_compatibility_head:
            self.compatibility_head = nn.Linear(head_input_dim, self.compatibility_output_dim)
        if self.enable_insertion_probability_head:
            self.insertion_probability_head = nn.Linear(head_input_dim, 1)
        if self.enable_redox_potential_head:
            self.redox_potential_head = nn.Linear(head_input_dim, 1)
        if self.enable_structure_type_head:
            self.structure_type_head = nn.Linear(head_input_dim, self.structure_type_output_dim)
        if self.enable_volume_expansion_head:
            self.volume_expansion_head = nn.Linear(head_input_dim, 1)

    def forward(
        self,
        battery_features: torch.Tensor,
        material_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        graph_context: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,  # kept for API compatibility
        batch: Optional[torch.Tensor] = None,  # kept for API compatibility
    ) -> Dict[str, torch.Tensor]:
        batch_size = battery_features.size(0)

        if self.enable_graph_context:
            if graph_context is None:
                graph_context = battery_features
            graph_context = self.context_proj(graph_context)
            gate = self.context_gate(torch.cat([battery_features, graph_context], dim=1))
            battery_features = gate * battery_features + (1.0 - gate) * graph_context

        material_flat = material_embeddings.view(-1, self.material_embedding_dim)
        material_encoded = self.material_encoder(material_flat)
        material_encoded = material_encoded.view(batch_size, self.num_materials, self.hidden_dim)
        material_encoded = material_encoded * node_mask.unsqueeze(-1)

        if self.material_transformer is not None:
            key_padding_mask = (node_mask <= 0.0)
            material_encoded = self.material_transformer(material_encoded, src_key_padding_mask=key_padding_mask)
            material_encoded = material_encoded * node_mask.unsqueeze(-1)
        elif self.material_interaction == "gatv2":
            n_nodes = material_encoded.size(1)
            node_mask_f = node_mask.float()
            for q_layer, k_layer, v_layer, edge_layer, out_layer in zip(
                self.gat_q_layers,
                self.gat_k_layers,
                self.gat_v_layers,
                self.gat_edge_layers,
                self.gat_out_layers,
            ):
                q = q_layer(material_encoded)
                k = k_layer(material_encoded)
                v = v_layer(material_encoded)

                score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(self.hidden_dim))
                qi = q.unsqueeze(2).expand(-1, -1, n_nodes, -1)
                kj = k.unsqueeze(1).expand(-1, n_nodes, -1, -1)
                edge_bias = edge_layer(torch.cat([qi, kj], dim=-1)).squeeze(-1)
                score = score + edge_bias

                key_valid = node_mask_f.unsqueeze(1).expand(-1, n_nodes, -1)
                score = score.masked_fill(key_valid <= 0.0, -1e9)
                attn = torch.softmax(score, dim=-1)
                out = torch.matmul(attn, v)
                out = out_layer(out)
                material_encoded = out * node_mask_f.unsqueeze(-1)
        elif self.material_interaction == "mpnn":
            node_mask_f = node_mask.float()
            for msg_layer, upd_layer in zip(self.mpnn_message_layers, self.mpnn_update_layers):
                n_nodes = material_encoded.size(1)
                xi = material_encoded.unsqueeze(2).expand(-1, -1, n_nodes, -1)
                xj = material_encoded.unsqueeze(1).expand(-1, n_nodes, -1, -1)
                msg = msg_layer(torch.cat([xi, xj], dim=-1))
                pair_mask = node_mask_f.unsqueeze(1) * node_mask_f.unsqueeze(2)
                msg = msg * pair_mask.unsqueeze(-1)
                denom = pair_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
                agg = msg.sum(dim=2) / denom
                material_encoded = upd_layer(torch.cat([material_encoded, agg], dim=-1))
                material_encoded = material_encoded * node_mask_f.unsqueeze(-1)
        else:
            for layer in self.gnn_layers:
                material_encoded = layer(material_encoded)
                material_encoded = material_encoded * node_mask.unsqueeze(-1)

        material_sum = material_encoded.sum(dim=1)
        material_count = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        latent = material_sum / material_count

        combined_features = torch.cat([battery_features, latent], dim=1)
        if self.decoder_mode == "legacy":
            properties = self.decoder(combined_features)
            voltage = properties[:, PROPERTY_INDEX["average_voltage"]]
            capacity_grav = properties[:, PROPERTY_INDEX["capacity_grav"]]
            capacity_vol = properties[:, PROPERTY_INDEX["capacity_vol"]]
            energy_grav = properties[:, PROPERTY_INDEX["energy_grav"]]
            energy_vol = properties[:, PROPERTY_INDEX["energy_vol"]]
            stability_charge = properties[:, PROPERTY_INDEX["stability_charge"]]
            stability_discharge = properties[:, PROPERTY_INDEX["stability_discharge"]]
            logvar = torch.zeros_like(properties)
            if self.enable_role_head or self.enable_compatibility_head:
                h_aux = self.aux_trunk(combined_features)
            else:
                h_aux = None
        else:
            h = self.decoder_trunk(combined_features)
            voltage = self.voltage_head(h).squeeze(-1)
            caps = self.capacity_head(h)
            capacity_grav = caps[:, 0]
            capacity_vol = caps[:, 1]
            stabs = self.stability_head(h)
            stability_charge = stabs[:, 0]
            stability_discharge = stabs[:, 1]
            # Deterministic energy construction (not freely predicted).
            energy_grav = voltage * capacity_grav
            energy_vol = voltage * capacity_vol
            properties = torch.stack(
                [
                    voltage,
                    capacity_grav,
                    capacity_vol,
                    energy_grav,
                    energy_vol,
                    stability_charge,
                    stability_discharge,
                ],
                dim=1,
            )
            if self.enable_uncertainty:
                logvar = self.logvar_head(h)
            else:
                logvar = torch.zeros_like(properties)
            h_aux = h

        if self.enable_role_head and h_aux is not None:
            role_logits = self.role_head(h_aux)
        else:
            role_logits = torch.zeros(
                (batch_size, self.role_output_dim),
                device=properties.device,
                dtype=properties.dtype,
            )

        if self.enable_compatibility_head and h_aux is not None:
            compatibility_logits = self.compatibility_head(h_aux)
            compatibility_scores = torch.sigmoid(compatibility_logits)
        else:
            compatibility_logits = torch.zeros(
                (batch_size, self.compatibility_output_dim),
                device=properties.device,
                dtype=properties.dtype,
            )
            compatibility_scores = torch.zeros_like(compatibility_logits)

        if self.enable_insertion_probability_head and h_aux is not None:
            insertion_probability = torch.sigmoid(self.insertion_probability_head(h_aux)).squeeze(-1)
        else:
            insertion_probability = torch.zeros((batch_size,), device=properties.device, dtype=properties.dtype)

        if self.enable_redox_potential_head and h_aux is not None:
            redox_potential = self.redox_potential_head(h_aux).squeeze(-1)
        else:
            redox_potential = torch.zeros((batch_size,), device=properties.device, dtype=properties.dtype)

        if self.enable_structure_type_head and h_aux is not None:
            structure_type_logits = self.structure_type_head(h_aux)
        else:
            structure_type_logits = torch.zeros(
                (batch_size, self.structure_type_output_dim),
                device=properties.device,
                dtype=properties.dtype,
            )

        if self.enable_volume_expansion_head and h_aux is not None:
            volume_expansion = self.volume_expansion_head(h_aux).squeeze(-1)
        else:
            volume_expansion = torch.zeros((batch_size,), device=properties.device, dtype=properties.dtype)

        return {
            "properties": properties,
            "latent": latent,
            "embedding": latent,  # compatibility alias
            "voltage": voltage,
            "capacity_grav": capacity_grav,
            "capacity_vol": capacity_vol,
            "stability_charge": stability_charge,
            "stability_discharge": stability_discharge,
            "property_log_variance": logvar,
            "role_logits": role_logits,
            "compatibility_logits": compatibility_logits,
            "compatibility_scores": compatibility_scores,
            "insertion_probability": insertion_probability,
            "redox_potential": redox_potential,
            "structure_type_logits": structure_type_logits,
            "volume_expansion": volume_expansion,
            "material_interaction_mode": torch.tensor(
                1 if self.material_interaction == "transformer" else 0,
                device=properties.device,
            ),
            "decoder_mode": torch.tensor(0 if self.decoder_mode == "legacy" else 1, device=properties.device),
        }

    def predict_properties(
        self,
        battery_features: torch.Tensor,
        material_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(battery_features, material_embeddings, node_mask)["properties"]

    def get_latent_representation(
        self,
        battery_features: torch.Tensor,
        material_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(battery_features, material_embeddings, node_mask)["latent"]


class MaskedBatteryGNN(MaskedGNN):
    """
    Backward-compatible alias.
    """


class BatteryGraphBuilder:
    """
    Utility for building per-system tensors used by tests/tools.
    """

    MATERIAL_COLUMNS = [
        "cathode_material_id",
        "anode_material_id",
        "electrolyte_material_id",
        "separator_material_id",
        "additives_material_id",
    ]

    def __init__(self, material_embeddings_df: pd.DataFrame, batteries_df: pd.DataFrame):
        self.material_embeddings_df = material_embeddings_df
        self.batteries_df = batteries_df
        self.material_id_to_embedding = self._create_embedding_mapping()
        self.embedding_dim = self._detect_embedding_dim()

    def _detect_embedding_dim(self) -> int:
        cols = [c for c in self.material_embeddings_df.columns if c != "material_id"]
        return len(cols) if cols else 64

    def _create_embedding_mapping(self) -> Dict[str, torch.Tensor]:
        mapping: Dict[str, torch.Tensor] = {}
        for _, row in self.material_embeddings_df.iterrows():
            material_id = str(row["material_id"])
            vec = torch.tensor(row.drop("material_id").values.astype(float), dtype=torch.float32)
            mapping[material_id] = vec
        return mapping

    def build_node_features(self, battery_row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        battery_features = torch.tensor(
            [
                float(battery_row.get("average_voltage", 0.0)),
                float(battery_row.get("capacity_grav", 0.0)),
                float(battery_row.get("capacity_vol", 0.0)),
                float(battery_row.get("energy_grav", 0.0)),
                float(battery_row.get("energy_vol", 0.0)),
                float(battery_row.get("stability_charge", 0.0)),
                float(battery_row.get("stability_discharge", 0.0)),
            ],
            dtype=torch.float32,
        )

        material_embeddings = torch.zeros(5, self.embedding_dim, dtype=torch.float32)
        node_mask = torch.zeros(5, dtype=torch.float32)

        for i, col in enumerate(self.MATERIAL_COLUMNS):
            if col in battery_row and pd.notna(battery_row[col]):
                mat_id = str(battery_row[col])
                vec = self.material_id_to_embedding.get(mat_id)
                if vec is not None:
                    material_embeddings[i] = vec
                    node_mask[i] = 1.0

        return battery_features, material_embeddings, node_mask

    def build_edge_index(self, num_nodes: int) -> torch.Tensor:
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def build_graph_for_battery(self, battery_id: str) -> Dict[str, torch.Tensor | str]:
        row = self.batteries_df[self.batteries_df["battery_id"] == battery_id].iloc[0]
        battery_features, material_embeddings, node_mask = self.build_node_features(row)
        edge_index = self.build_edge_index(material_embeddings.size(0))
        return {
            "battery_features": battery_features,
            "material_embeddings": material_embeddings,
            "node_mask": node_mask,
            "edge_index": edge_index,
            "battery_id": battery_id,
        }


def create_masked_gnn(config: Dict[str, int | float | str]) -> MaskedGNN:
    return MaskedGNN(
        battery_feature_dim=int(config.get("battery_feature_dim", 7)),
        material_embedding_dim=int(config.get("material_embedding_dim", 64)),
        num_materials=int(config.get("num_materials", 5)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        num_gnn_layers=int(config.get("num_gnn_layers", 3)),
        gnn_type=str(config.get("gnn_type", "GIN")),
        dropout=float(config.get("dropout", 0.2)),
        decoder_mode=str(config.get("decoder_mode", "legacy")),
        enable_uncertainty=bool(config.get("enable_uncertainty", False)),
        material_interaction=str(config.get("material_interaction", "mlp")),
        enable_graph_context=bool(config.get("enable_graph_context", False)),
        graph_context_dim=int(config.get("graph_context_dim", int(config.get("battery_feature_dim", 7)))),
        enable_role_head=bool(config.get("enable_role_head", False)),
        enable_compatibility_head=bool(config.get("enable_compatibility_head", False)),
        role_output_dim=int(config.get("role_output_dim", 3)),
        compatibility_output_dim=int(config.get("compatibility_output_dim", 3)),
        enable_insertion_probability_head=bool(config.get("enable_insertion_probability_head", False)),
        enable_redox_potential_head=bool(config.get("enable_redox_potential_head", False)),
        enable_structure_type_head=bool(config.get("enable_structure_type_head", False)),
        enable_volume_expansion_head=bool(config.get("enable_volume_expansion_head", False)),
        structure_type_output_dim=int(config.get("structure_type_output_dim", 6)),
    )
