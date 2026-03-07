"""
Build Masked Battery Graph with Enhanced Material Embeddings and Normalized Data

This script constructs a graph using the properly normalized batteries_ml.csv data
for optimal ML training performance.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration - Using Normalized Data
# ============================================================
DATA_FILE = "data/processed/batteries_ml.csv"  # Use normalized data
EMBEDDING_FILE = "data/processed/atomic_embeddings.csv"  # Enhanced 64D embeddings
OUTPUT_GRAPH_FILE = "graphs/masked_battery_graph_normalized.pt"
MASK_TYPES = ["cathode", "anode", "electrolyte", "separator", "additives"]
EDGE_TYPES = ["atomic_similarity", "chemsys_similarity", "role_similarity", "physics_similarity", "electrochemical_similarity"]

# Configuration - Optimized dimensions
NUM_MATERIALS = 5  # cathode, anode, electrolyte, separator, additives
MATERIAL_EMBEDDING_DIM = 64  # Enhanced embeddings
BATTERY_FEATURE_DIM = 7  # normalized properties

# Edge construction parameters - Optimized for battery discovery
K_ATOMIC = 12      # Atomic similarity neighbors
K_CHEMSYS = 18     # Chemsys similarity neighbors  
K_ROLE = 15        # Role similarity neighbors
K_PHYSICS = 25     # Physics similarity neighbors
K_ELECTRO = 15     # Electrochemical similarity neighbors

# Material role assignment rules (same as before)
WORKING_ION_CATHODE_ELEMENTS = {
    'Li': ['Co', 'Ni', 'Mn', 'Fe', 'V', 'Cr', 'Mo', 'W', 'O', 'F'],
    'Na': ['Fe', 'Mn', 'V', 'Ti', 'O', 'F'],
    'K': ['Fe', 'Mn', 'V', 'O', 'F'],
    'Mg': ['Ti', 'V', 'Mn', 'O', 'S'],
    'Ca': ['Fe', 'Mn', 'V', 'O'],
    'Zn': ['Mn', 'V', 'O']
}

WORKING_ION_ANODE_ELEMENTS = {
    'Li': ['C', 'Si', 'Sn', 'Ge', 'Ti', 'Nb', 'O'],
    'Na': ['C', 'Sn', 'P', 'Sb', 'O'],
    'K': ['C', 'Sn', 'P', 'O'],
    'Mg': ['Ti', 'Nb', 'O', 'S'],
    'Ca': ['Si', 'Sn', 'O'],
    'Zn': ['C', 'O']
}

# ============================================================
# Helper Functions - Enhanced for Battery Chemistry
# ============================================================
def safe_eval(x):
    """Safely evaluate string representations of lists/dicts."""
    try:
        return eval(x)
    except Exception:
        return []

def cosine_similarity_torch(x, y):
    """Compute cosine similarity between two tensors."""
    x_norm = F.normalize(x, p=2, dim=0)
    y_norm = F.normalize(y, p=2, dim=0)
    return torch.dot(x_norm, y_norm).item()

def euclidean_distance_torch(x, y):
    """Compute Euclidean distance between two tensors."""
    return torch.norm(x - y).item()

def find_k_nearest_neighbors(query_vec, candidate_matrix, k, metric='cosine'):
    """Find k nearest neighbors using PyTorch operations."""
    if metric == 'cosine':
        similarities = F.cosine_similarity(query_vec.unsqueeze(0), candidate_matrix, dim=1)
        _, indices = torch.topk(similarities, k=k+1)
    else:  # euclidean
        distances = torch.norm(candidate_matrix - query_vec, dim=1)
        _, indices = torch.topk(distances, k=k+1, largest=False)
    
    return indices[1:]  # Skip self (first element)

def infer_material_roles(material_ids, battery_row):
    """
    Intelligently infer material roles based on battery chemistry.
    """
    roles = {}
    working_ion = str(battery_row.get('working_ion', 'Li'))
    elements = set(safe_eval(battery_row.get('elements', '[]')))
    
    # Get material compositions
    material_compositions = {}
    for mat_id in material_ids:
        material_compositions[mat_id] = elements  # Simplified
    
    # Rule-based role assignment
    for mat_id in material_ids:
        mat_elements = material_compositions[mat_id]
        
        # Cathode inference
        if any(elem in mat_elements for elem in WORKING_ION_CATHODE_ELEMENTS.get(working_ion, [])):
            if 'cathode' not in roles:
                roles[mat_id] = 'cathode'
                continue
        
        # Anode inference  
        if any(elem in mat_elements for elem in WORKING_ION_ANODE_ELEMENTS.get(working_ion, [])):
            if 'anode' not in roles:
                roles[mat_id] = 'anode'
                continue
        
        # Electrolyte inference (contains working ion)
        if working_ion in mat_elements:
            if 'electrolyte' not in roles:
                roles[mat_id] = 'electrolyte'
                continue
        
        # Default assignment for remaining materials
        if 'separator' not in roles:
            roles[mat_id] = 'separator'
        elif 'additives' not in roles:
            roles[mat_id] = 'additives'
    
    return roles

def build_material_embedding_matrix(material_ids, material_embeddings_df, battery_row):
    """
    Build material embedding matrix with intelligent role-based ordering.
    """
    num_materials = NUM_MATERIALS
    material_embeddings = torch.zeros(num_materials, MATERIAL_EMBEDDING_DIM, dtype=torch.float32)
    node_mask = torch.zeros(num_materials, dtype=torch.float32)
    role_assignments = {}
    
    # Infer material roles
    roles = infer_material_roles(material_ids, battery_row)
    
    # Role ordering priority: cathode, anode, electrolyte, separator, additives
    role_priority = ['cathode', 'anode', 'electrolyte', 'separator', 'additives']
    
    # Assign materials to positions based on roles
    position = 0
    for role in role_priority:
        for mat_id in material_ids:
            if roles.get(mat_id) == role and position < num_materials:
                try:
                    row = material_embeddings_df[material_embeddings_df['material_id'] == mat_id]
                    if not row.empty:
                        embedding = row.iloc[0][[f'atom_emb_{j}' for j in range(MATERIAL_EMBEDDING_DIM)]].values
                        material_embeddings[position] = torch.tensor(embedding, dtype=torch.float32)
                        node_mask[position] = 1.0
                        role_assignments[position] = role
                        position += 1
                        break
                except Exception as e:
                    logger.warning(f"Failed to get embedding for material {mat_id}: {e}")
                    continue
    
    # Fill remaining positions with any remaining materials
    for mat_id in material_ids:
        if position >= num_materials:
            break
        if mat_id not in [k for k, v in roles.items() if v in role_assignments.values()]:
            try:
                row = material_embeddings_df[material_embeddings_df['material_id'] == mat_id]
                if not row.empty:
                    embedding = row.iloc[0][[f'atom_emb_{j}' for j in range(MATERIAL_EMBEDDING_DIM)]].values
                    material_embeddings[position] = torch.tensor(embedding, dtype=torch.float32)
                    node_mask[position] = 1.0
                    role_assignments[position] = 'additives'  # Default to additives
                    position += 1
            except Exception as e:
                logger.warning(f"Failed to get embedding for material {mat_id}: {e}")
                continue
    
    return material_embeddings, node_mask, role_assignments

def compute_advanced_edge_features(battery_i, battery_j, material_embeddings_i, material_embeddings_j, 
                                  mask_i, mask_j, material_embeddings_df, roles_i, roles_j):
    """
    Compute advanced edge features between two battery systems.
    """
    features = []
    
    # 1. Shared materials fraction
    mat_ids_i = set(safe_eval(battery_i.get('material_ids', '[]')))
    mat_ids_j = set(safe_eval(battery_j.get('material_ids', '[]')))
    shared_materials = mat_ids_i.intersection(mat_ids_j)
    shared_material_fraction = len(shared_materials) / max(len(mat_ids_i), len(mat_ids_j), 1)
    features.append(shared_material_fraction)
    
    # 2. Shared elements fraction
    elements_i = set(safe_eval(battery_i.get('elements', '[]')))
    elements_j = set(safe_eval(battery_j.get('elements', '[]')))
    shared_elements = elements_i.intersection(elements_j)
    shared_element_fraction = len(shared_elements) / max(len(elements_i), len(elements_j), 1)
    features.append(shared_element_fraction)
    
    # 3. Chemical similarity (cosine similarity of material embeddings)
    present_i = material_embeddings_i * mask_i.unsqueeze(1)
    present_j = material_embeddings_j * mask_j.unsqueeze(1)
    
    avg_emb_i = present_i.sum(dim=0) / mask_i.sum().clamp(min=1)
    avg_emb_j = present_j.sum(dim=0) / mask_j.sum().clamp(min=1)
    
    if mask_i.sum() > 0 and mask_j.sum() > 0:
        cos_sim = cosine_similarity_torch(avg_emb_i, avg_emb_j)
    else:
        cos_sim = 0.0
    features.append(cos_sim)
    
    # 4. Battery property similarity (cosine similarity of normalized battery features)
    battery_features_i = torch.tensor([
        float(battery_i.get('average_voltage_norm', 0)),
        float(battery_i.get('capacity_grav_norm', 0)),
        float(battery_i.get('capacity_vol_norm', 0)),
        float(battery_i.get('energy_grav_norm', 0)),
        float(battery_i.get('energy_vol_norm', 0)),
        float(battery_i.get('stability_charge_norm', 0)),
        float(battery_i.get('stability_discharge_norm', 0))
    ], dtype=torch.float32)
    
    battery_features_j = torch.tensor([
        float(battery_j.get('average_voltage_norm', 0)),
        float(battery_j.get('capacity_grav_norm', 0)),
        float(battery_j.get('capacity_vol_norm', 0)),
        float(battery_j.get('energy_grav_norm', 0)),
        float(battery_j.get('energy_vol_norm', 0)),
        float(battery_j.get('stability_charge_norm', 0)),
        float(battery_j.get('stability_discharge_norm', 0))
    ], dtype=torch.float32)
    
    # Normalize battery features
    battery_features_i_norm = F.normalize(battery_features_i.unsqueeze(0), p=2, dim=1)
    battery_features_j_norm = F.normalize(battery_features_j.unsqueeze(0), p=2, dim=1)
    
    battery_sim = F.cosine_similarity(battery_features_i_norm, battery_features_j_norm).item()
    features.append(battery_sim)
    
    # 5. Working ion similarity
    working_ion_i = str(battery_i.get('working_ion', ''))
    working_ion_j = str(battery_j.get('working_ion', ''))
    working_ion_sim = 1.0 if working_ion_i == working_ion_j else 0.0
    features.append(working_ion_sim)
    
    # 6. Role compatibility score
    roles_i_set = set(roles_i.values()) if roles_i else set()
    roles_j_set = set(roles_j.values()) if roles_j else set()
    role_overlap = len(roles_i_set.intersection(roles_j_set)) / max(len(roles_i_set), len(roles_j_set), 1)
    features.append(role_overlap)
    
    # 7. Voltage compatibility (normalized difference)
    voltage_i = float(battery_i.get('average_voltage_norm', 0))
    voltage_j = float(battery_j.get('average_voltage_norm', 0))
    voltage_diff = abs(voltage_i - voltage_j)
    voltage_compat = 1.0 / (1.0 + voltage_diff)  # Higher score for similar voltages
    features.append(voltage_compat)
    
    # 8. Capacity compatibility
    capacity_i = float(battery_i.get('capacity_grav_norm', 0))
    capacity_j = float(battery_j.get('capacity_grav_norm', 0))
    capacity_diff = abs(capacity_i - capacity_j)
    capacity_compat = 1.0 / (1.0 + capacity_diff / 10.0)  # Normalize by typical capacity scale
    features.append(capacity_compat)
    
    return torch.tensor(features, dtype=torch.float32)

# ============================================================
# Main Graph Construction - Using Normalized Data
# ============================================================
def main():
    logger.info("🔋 Loading normalized datasets for optimized battery graph construction...")
    
    # Load data
    batteries_df = pd.read_csv(DATA_FILE)
    material_embeddings_df = pd.read_csv(EMBEDDING_FILE)
    
    N = len(batteries_df)
    logger.info(f"📦 Processing {N} battery systems with normalized data")
    logger.info(f"📊 Using enhanced {MATERIAL_EMBEDDING_DIM}D material embeddings")
    logger.info(f"🎯 Optimized for battery material discovery and property prediction")
    
    # Initialize storage for node features
    battery_features_list = []
    material_embeddings_list = []
    node_masks_list = []
    role_assignments_list = []
    
    # Build node features
    logger.info("🏗️  Building optimized node features with normalized data...")
    for idx, battery_row in tqdm(batteries_df.iterrows(), total=N, desc="Processing batteries"):
        # Extract normalized battery features (7 properties)
        battery_features = torch.tensor([
            float(battery_row.get('average_voltage_norm', 0)),
            float(battery_row.get('capacity_grav_norm', 0)),
            float(battery_row.get('capacity_vol_norm', 0)),
            float(battery_row.get('energy_grav_norm', 0)),
            float(battery_row.get('energy_vol_norm', 0)),
            float(battery_row.get('stability_charge_norm', 0)),
            float(battery_row.get('stability_discharge_norm', 0))
        ], dtype=torch.float32)
        
        # Get material IDs
        material_ids = safe_eval(battery_row.get('material_ids', '[]'))
        
        # Build material embeddings and masks with intelligent role assignment
        material_embeddings, node_mask, roles = build_material_embedding_matrix(
            material_ids, material_embeddings_df, battery_row
        )
        
        battery_features_list.append(battery_features)
        material_embeddings_list.append(material_embeddings)
        node_masks_list.append(node_mask)
        role_assignments_list.append(roles)
    
    # Stack into tensors
    battery_features_tensor = torch.stack(battery_features_list)
    material_embeddings_tensor = torch.stack(material_embeddings_list)
    node_masks_tensor = torch.stack(node_masks_list)
    
    logger.info(f"🧠 Optimized node feature dimensions:")
    logger.info(f"  - Battery features: {battery_features_tensor.shape}")
    logger.info(f"  - Material embeddings: {material_embeddings_tensor.shape}")
    logger.info(f"  - Node masks: {node_masks_tensor.shape}")
    logger.info(f"  - Total features per node: {BATTERY_FEATURE_DIM + NUM_MATERIALS * MATERIAL_EMBEDDING_DIM + NUM_MATERIALS}")
    
    # Build edges
    logger.info("🔗 Building multi-type edges...")
    
    # Edge type 0: Atomic similarity (based on enhanced material embeddings)
    logger.info("  - Edge type 0: Atomic similarity (using enhanced 64D embeddings)")
    avg_embeddings = []
    for i in range(N):
        present = material_embeddings_tensor[i] * node_masks_tensor[i].unsqueeze(1)
        avg_emb = present.sum(dim=0) / node_masks_tensor[i].sum().clamp(min=1)
        avg_embeddings.append(avg_emb)
    
    avg_embeddings_matrix = torch.stack(avg_embeddings)
    edges_atomic = []
    
    for i in range(N):
        neighbors = find_k_nearest_neighbors(avg_embeddings_matrix[i], avg_embeddings_matrix, K_ATOMIC, 'cosine')
        for j in neighbors:
            if i != j:  # Skip self
                edges_atomic.append([i, j])
    
    # Edge type 1: Chemsys similarity
    logger.info("  - Edge type 1: Chemsys similarity")
    chemsys_groups = batteries_df.groupby('chemsys')
    edges_chemsys = []
    
    for chemsys, group in chemsys_groups:
        group_indices = group.index.tolist()
        if len(group_indices) < 2:
            continue
        
        # Get embeddings for this chemsys group
        group_embeddings = avg_embeddings_matrix[group_indices]
        k = min(K_CHEMSYS, len(group_indices) - 1)  # Ensure k is valid
        
        for local_i, global_i in enumerate(group_indices):
            if len(group_indices) > 1:
                neighbors = find_k_nearest_neighbors(avg_embeddings_matrix[global_i], avg_embeddings_matrix[group_indices], k, 'cosine')
                for local_j in neighbors:
                    global_j = group_indices[local_j]
                    if global_i != global_j:
                        edges_chemsys.append([global_i, global_j])
    
    # Edge type 2: Role similarity (based on working ion and elements)
    logger.info("  - Edge type 2: Role similarity")
    role_features = []
    for idx, battery_row in batteries_df.iterrows():
        working_ion = str(battery_row.get('working_ion', ''))
        elements = safe_eval(battery_row.get('elements', '[]'))
        
        # One-hot encode working ion
        ion_features = [1.0 if working_ion == ion else 0.0 for ion in ['Li', 'Na', 'K', 'Mg', 'Ca', 'Zn']]
        
        # Element features (simplified)
        element_features = [1.0 if elem in elements else 0.0 for elem in ['Li', 'Na', 'K', 'Mg', 'Ca', 'Zn', 'C', 'O', 'F', 'S']]
        
        role_features.append(ion_features + element_features + [len(elements)])
    
    role_features_matrix = torch.tensor(role_features, dtype=torch.float32)
    edges_role = []
    
    for i in range(N):
        neighbors = find_k_nearest_neighbors(role_features_matrix[i], role_features_matrix, K_ROLE, 'cosine')
        for j in neighbors:
            if i != j:
                edges_role.append([i, j])
    
    # Edge type 3: Physics similarity (based on electrochemical properties)
    logger.info("  - Edge type 3: Physics similarity")
    physics_features = []
    for idx, battery_row in batteries_df.iterrows():
        physics_features.append([
            float(battery_row.get('average_voltage_norm', 0)),
            float(battery_row.get('stability_charge_norm', 0)),
            float(battery_row.get('stability_discharge_norm', 0))
        ])
    
    physics_features_matrix = torch.tensor(physics_features, dtype=torch.float32)
    edges_physics = []
    
    for i in range(N):
        neighbors = find_k_nearest_neighbors(physics_features_matrix[i], physics_features_matrix, K_PHYSICS, 'euclidean')
        for j in neighbors:
            if i != j:
                edges_physics.append([i, j])
    
    # Edge type 4: Electrochemical similarity
    logger.info("  - Edge type 4: Electrochemical similarity")
    electro_features = []
    for idx, battery_row in batteries_df.iterrows():
        electro_features.append([
            float(battery_row.get('average_voltage_norm', 0)),
            float(battery_row.get('capacity_grav_norm', 0)),
            float(battery_row.get('capacity_vol_norm', 0)),
            float(battery_row.get('energy_grav_norm', 0)),
            float(battery_row.get('energy_vol_norm', 0))
        ])
    
    electro_features_matrix = torch.tensor(electro_features, dtype=torch.float32)
    edges_electro = []
    
    for i in range(N):
        neighbors = find_k_nearest_neighbors(electro_features_matrix[i], electro_features_matrix, K_ELECTRO, 'euclidean')
        for j in neighbors:
            if i != j:
                edges_electro.append([i, j])
    
    # Build edge index dictionary
    edge_index_dict = {
        0: torch.tensor(edges_atomic, dtype=torch.long).t().contiguous(),
        1: torch.tensor(edges_chemsys, dtype=torch.long).t().contiguous(),
        2: torch.tensor(edges_role, dtype=torch.long).t().contiguous(),
        3: torch.tensor(edges_physics, dtype=torch.long).t().contiguous(),
        4: torch.tensor(edges_electro, dtype=torch.long).t().contiguous(),
    }
    
    # Compute advanced edge features
    logger.info("🔧 Computing advanced edge features...")
    edge_features_dict = {}
    
    for edge_type, edge_index in edge_index_dict.items():
        edge_features = []
        
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            
            edge_feat = compute_advanced_edge_features(
                batteries_df.iloc[src], batteries_df.iloc[dst],
                material_embeddings_tensor[src], material_embeddings_tensor[dst],
                node_masks_tensor[src], node_masks_tensor[dst],
                material_embeddings_df,
                role_assignments_list[src], role_assignments_list[dst]
            )
            edge_features.append(edge_feat)
        
        edge_features_dict[edge_type] = torch.stack(edge_features)
    
    # Save graph
    logger.info("💾 Saving optimized masked battery graph with normalized data...")
    Path(OUTPUT_GRAPH_FILE).parent.mkdir(exist_ok=True, parents=True)
    
    graph_data = {
        'battery_features': battery_features_tensor,
        'material_embeddings': material_embeddings_tensor,
        'node_masks': node_masks_tensor,
        'edge_index_dict': edge_index_dict,
        'edge_features_dict': edge_features_dict,
        'role_assignments': role_assignments_list,
        'metadata': {
            'num_nodes': N,
            'num_materials': NUM_MATERIALS,
            'material_embedding_dim': MATERIAL_EMBEDDING_DIM,
            'battery_feature_dim': BATTERY_FEATURE_DIM,
            'edge_types': len(edge_index_dict),
            'construction_params': {
                'K_ATOMIC': K_ATOMIC,
                'K_CHEMSYS': K_CHEMSYS,
                'K_ROLE': K_ROLE,
                'K_PHYSICS': K_PHYSICS,
                'K_ELECTRO': K_ELECTRO
            },
            'mask_types': MASK_TYPES,
            'edge_types_names': EDGE_TYPES,
            'embedding_source': 'enhanced_atomic_embeddings_64d',
            'optimization_target': 'battery_material_discovery',
            'data_source': 'batteries_ml.csv (normalized)',
            'total_features_per_node': BATTERY_FEATURE_DIM + NUM_MATERIALS * MATERIAL_EMBEDDING_DIM + NUM_MATERIALS
        }
    }
    
    torch.save(graph_data, OUTPUT_GRAPH_FILE)
    
    logger.info("✅ OPTIMIZED MASKED BATTERY GRAPH BUILT SUCCESSFULLY WITH NORMALIZED DATA!")
    logger.info(f"📊 Graph Statistics:")
    logger.info(f"  - Nodes: {N}")
    logger.info(f"  - Battery features per node: {BATTERY_FEATURE_DIM} (normalized)")
    logger.info(f"  - Material embeddings per node: {NUM_MATERIALS} × {MATERIAL_EMBEDDING_DIM} (ENHANCED 64D)")
    logger.info(f"  - Node masks per node: {NUM_MATERIALS}")
    logger.info(f"  - Total features per node: {BATTERY_FEATURE_DIM + NUM_MATERIALS * MATERIAL_EMBEDDING_DIM + NUM_MATERIALS}")
    logger.info(f"  - Edge types: {len(edge_index_dict)}")
    
    for edge_type, edge_index in edge_index_dict.items():
        logger.info(f"  - Edge type {edge_type} ({EDGE_TYPES[edge_type]}): {edge_index.size(1)} edges")
        if edge_type in edge_features_dict:
            logger.info(f"    - Edge features: {edge_features_dict[edge_type].shape[1]} dimensions")
    
    logger.info("🎯 Graph optimized for:")
    logger.info("  - Battery material discovery with normalized data")
    logger.info("  - Property prediction with proper scaling")
    logger.info("  - Multi-scale representation learning")
    logger.info("  - Role-aware material relationships")

if __name__ == "__main__":
    main()