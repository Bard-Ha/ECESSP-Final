# backend_client.py
# ============================================================
# Python helper for frontend to interact with ecessp-ml backend
# ============================================================

import requests
from typing import Dict, Any, List
import os

# ------------------------------------------------------------
# Configure your backend URL here
# ------------------------------------------------------------
BACKEND_URL = os.getenv("ECESSP_ML_URL", "http://127.0.0.1:8000")


# ------------------------------------------------------------
# Battery properties
# ------------------------------------------------------------
BATTERY_PROPERTIES = [
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "max_delta_volume",
    "stability_charge",
    "stability_discharge"
]


# ------------------------------------------------------------
# Discovery API
# ------------------------------------------------------------
def discover_systems(targets: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Run discovery on backend.
    Input:
        targets: dict of 7 battery properties
    Returns:
        list of candidate systems in frontend-compatible format
    """
    url = f"{BACKEND_URL}/api/discover"
    payload = {
        "system": {
            "battery_id": "backend_client_discovery",
            **{k: targets.get(k, 0.0) for k in BATTERY_PROPERTIES},
        },
        "objective": {
            "objectives": {k: targets.get(k, 0.0) for k in BATTERY_PROPERTIES},
        },
        "explain": True,
        "mode": "generative",
        "discovery_params": {
            "num_candidates": 120,
            "diversity_weight": 0.45,
            "novelty_weight": 0.35,
            "extrapolation_strength": 0.35,
            "optimize_steps": 32
        },
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    data = response.json()
    history = data.get("history") or []
    if isinstance(history, list) and history:
        # Frontend-friendly list of ranked candidate entries.
        return history
    # Fallback: keep prior behavior if history is absent.
    return [data]


# ------------------------------------------------------------
# Prediction API
# ------------------------------------------------------------
def predict_system(components: Dict[str, str]) -> Dict[str, Any]:
    """
    Predict battery properties from selected components.
    Input:
        components: dict with 'cathode', 'anode', 'electrolyte'
    Returns:
        dict with predicted properties + system_name + confidence_score
    """
    url = f"{BACKEND_URL}/api/predict"
    payload = {"components": components}
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    data = response.json()
    return data


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example discovery
    targets_example = {
        "average_voltage": 3.7,
        "capacity_grav": 200,
        "capacity_vol": 700,
        "energy_grav": 400,
        "energy_vol": 1000,
        "max_delta_volume": 0.12,
        "stability_charge": 0.05,
        "stability_discharge": 0.05
    }
    systems = discover_systems(targets_example)
    print("Discovered systems:", systems)

    # Example prediction
    components_example = {
        "cathode": "1",  # can be ID or name, adjust as backend expects
        "anode": "2",
        "electrolyte": "3"
    }
    prediction = predict_system(components_example)
    print("Prediction:", prediction)
