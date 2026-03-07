# ============================================================
# Feature Encoder Loader (PRODUCTION-LOCKED, 7 TARGETS)
# ============================================================

from __future__ import annotations
from typing import Callable, List, Dict
import numpy as np
import logging

from design.system_template import BatterySystem
from backend.config import MODEL_CONFIG

logger = logging.getLogger(__name__)

# ============================================================
# Encoder Specification (LOCKED)
# ============================================================
FEATURE_ORDER: List[str] = [
    # --- Electrochemical ---
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",

    # --- Structural / Stability ---
    "stability_charge",
    "stability_discharge",
]

NUM_FEATURES: int = len(FEATURE_ORDER)

# ============================================================
# Normalization Ranges (Training-Derived)
# ============================================================
NORMALIZATION_RANGES: Dict[str, tuple[float, float]] = {
    "average_voltage": (0.0, 8.0),
    "capacity_grav": (0.0, 500.0),
    "capacity_vol": (0.0, 3000.0),
    "energy_grav": (0.0, 900.0),
    "energy_vol": (0.0, 3000.0),
    "stability_charge": (-0.5, 0.5),
    "stability_discharge": (-0.5, 0.5),
}

# ============================================================
# Internal Utilities
# ============================================================

def _normalize(value: float | None, vmin: float, vmax: float) -> float:
    """Deterministic min-max normalization to [0,1]. None → 0."""
    if value is None:
        return 0.0
    if not isinstance(value, (int, float)):
        raise TypeError(f"Encoder received non-numeric value: {value} ({type(value)})")
    clipped = max(min(float(value), vmax), vmin)
    return (clipped - vmin) / (vmax - vmin + 1e-12)

def _validate_system(system: BatterySystem) -> None:
    """Ensure BatterySystem exposes all required features."""
    if not isinstance(system, BatterySystem):
        raise TypeError(f"Expected BatterySystem, got {type(system)}")
    for feature in FEATURE_ORDER:
        if not hasattr(system, feature):
            raise AttributeError(f"BatterySystem missing required feature '{feature}'")

# ============================================================
# Core Encoder
# ============================================================

def encode_battery_system(system: BatterySystem) -> np.ndarray:
    _validate_system(system)
    values: List[float] = []
    for feature in FEATURE_ORDER:
        raw_value = getattr(system, feature)
        vmin, vmax = NORMALIZATION_RANGES[feature]
        normalized = _normalize(raw_value, vmin, vmax)
        values.append(normalized)
    vector = np.asarray(values, dtype=np.float32)

    if vector.ndim != 1 or vector.shape[0] != NUM_FEATURES:
        raise RuntimeError(f"Encoded feature vector has invalid shape: {vector.shape}, expected ({NUM_FEATURES},)")

    if not np.isfinite(vector).all():
        raise RuntimeError("Encoded feature vector contains NaN or Inf values")

    return vector

# ============================================================
# Public Loader API
# ============================================================

def load_feature_encoder() -> Callable[[BatterySystem], np.ndarray]:
    expected_dim = MODEL_CONFIG.input_dim
    if expected_dim != NUM_FEATURES:
        logger.error("Model / encoder mismatch: model expects %s, encoder provides %s", expected_dim, NUM_FEATURES)
        raise RuntimeError(f"[EncoderLoader] Model / encoder mismatch: model expects input_dim={expected_dim}, encoder provides {NUM_FEATURES}")
    return encode_battery_system

# ============================================================
# Introspection Utility
# ============================================================

def describe_encoder() -> dict:
    return {
        "num_features": NUM_FEATURES,
        "feature_order": FEATURE_ORDER.copy(),
        "normalization_ranges": NORMALIZATION_RANGES.copy(),
        "locked": True,
        "expects_system": "BatterySystem",
        "compatible_models": ["BatteryHGT", "BatteryRGCN", "BatteryHGT_VAE"],
    }
