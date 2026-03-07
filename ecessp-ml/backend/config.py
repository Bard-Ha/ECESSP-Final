# backend/config.py
# ============================================================
# Backend Configuration & Environment Control (FINAL · FIXED)
# ============================================================

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Environment Detection
# ============================================================

def _detect_env() -> str:
    if os.getenv("REPL_ID"):
        return "replit"
    if os.getenv("ECESSP_ENV"):
        return os.getenv("ECESSP_ENV")
    return "local"


ENVIRONMENT = _detect_env()


# ============================================================
# Base Paths (FIXED)
# ============================================================

# config.py lives in: ecessp-ml/backend/config.py
# parents[1] → ecessp-ml
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

DESIGN_DIR = PROJECT_ROOT / "design"
MATERIALS_DIR = PROJECT_ROOT / "materials"


def _resolve_allowed_origins() -> list[str]:
    configured = os.getenv("ECESSP_ALLOWED_ORIGINS", "").strip()
    if configured:
        origins = [item.strip() for item in configured.split(",") if item.strip()]
        return origins or ["*"]
    if ENVIRONMENT == "local":
        return ["*"]
    return [
        "https://*.replit.app",
        "https://*.replit.dev",
    ]


def _resolve_runtime_checkpoint() -> Path:
    """
    Resolve the checkpoint used by /predict and /discover at backend startup.

    Priority:
    1) ECESSP_CHECKPOINT env override (absolute or project-relative)
    2) reports/training_summary.json best_checkpoint if it exists and looks usable
    3) most recent best_model_layer_norm_*.pt
    4) legacy fallback checkpoint name(s)
    """
    models_dir = REPORTS_DIR / "models"
    legacy_candidates = [
        "best_model_layer_norm_20260214_133567.pt",  # user-requested legacy name
        "best_model_layer_norm_20260214_133657.pt",  # existing known legacy file
    ]

    def _resolve_masked_checkpoint_from_manifest() -> Path | None:
        manifest_paths = [
            REPORTS_DIR / "final_family_ensemble_manifest.json",
            REPORTS_DIR / "three_model_ensemble_manifest.json",
        ]
        for manifest_path in manifest_paths:
            if not manifest_path.exists():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                models = payload.get("models", [])
                if not isinstance(models, list) or not models:
                    continue

                primary_name = str(payload.get("primary_model", "")).strip()
                if primary_name:
                    models = sorted(
                        models,
                        key=lambda m: 0 if str((m or {}).get("name", "")).strip() == primary_name else 1,
                    )

                for item in models:
                    if not isinstance(item, dict):
                        continue
                    family = str(item.get("family", "masked_gnn")).strip().lower()
                    interaction = str(item.get("material_interaction", "")).strip().lower()
                    name = str(item.get("name", "")).strip().lower()
                    if family and family != "masked_gnn":
                        continue
                    if interaction == "hgt" or name == "hetero_hgt":
                        continue

                    ckpt = str(item.get("checkpoint_path", "")).strip()
                    if not ckpt:
                        continue
                    p = Path(ckpt)
                    if not p.is_absolute():
                        p = (PROJECT_ROOT / p).resolve()
                    if p.exists() and p.is_file():
                        logger.info(
                            "Using masked checkpoint from ensemble manifest %s: %s",
                            manifest_path.name,
                            p,
                        )
                        return p
            except Exception as exc:
                logger.warning("Failed to parse ensemble manifest %s: %s", manifest_path, exc)
        return None

    env_ckpt = os.getenv("ECESSP_CHECKPOINT", "").strip()
    if env_ckpt:
        p = Path(env_ckpt)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if p.exists() and p.is_file():
            logger.info("Using checkpoint from ECESSP_CHECKPOINT: %s", p)
            return p
        logger.warning("ECESSP_CHECKPOINT path not found: %s", p)

    summary_path = REPORTS_DIR / "training_summary.json"
    r2_gate = float(os.getenv("ECESSP_MIN_OVERALL_R2", "0.20"))
    summary_gate_failed = False
    summary_overall_r2 = None

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            eval_metrics = summary.get("evaluation_metrics", {})
            # New format: evaluation_metrics.{iid,ood}.raw_space.overall_micro.r2
            iid_r2 = (
                eval_metrics.get("iid", {})
                .get("raw_space", {})
                .get("overall_micro", {})
                .get("r2", None)
            )
            ood_r2 = (
                eval_metrics.get("ood", {})
                .get("raw_space", {})
                .get("overall_micro", {})
                .get("r2", None)
            )
            if iid_r2 is not None and ood_r2 is not None:
                overall_r2 = float(0.5 * (float(iid_r2) + float(ood_r2)))
            else:
                # Backward compatibility with older summary schema.
                overall_r2 = float(
                    eval_metrics.get("raw_space", {})
                    .get("overall_micro", {})
                    .get("r2", -1.0)
                )
            summary_overall_r2 = overall_r2
            best_ckpt = Path(str(summary.get("best_checkpoint", "")).strip())
            if str(best_ckpt):
                if not best_ckpt.is_absolute():
                    best_ckpt = (PROJECT_ROOT / best_ckpt).resolve()
                if best_ckpt.exists() and best_ckpt.is_file() and overall_r2 >= r2_gate:
                    logger.info("Using checkpoint from training summary: %s", best_ckpt)
                    return best_ckpt
                if overall_r2 < r2_gate:
                    summary_gate_failed = True
        except Exception as exc:
            logger.warning("Failed to parse training_summary.json for checkpoint: %s", exc)

    manifest_ckpt = _resolve_masked_checkpoint_from_manifest()
    if manifest_ckpt is not None:
        return manifest_ckpt

    if summary_gate_failed and summary_overall_r2 is not None:
        logger.warning(
            "training_summary overall_micro.r2=%.4f < gate %.4f; "
            "falling back to legacy/baseline checkpoint.",
            summary_overall_r2,
            r2_gate,
        )

    if models_dir.exists():
        candidates = sorted(models_dir.glob("best_model_layer_norm_*.pt"))
        if candidates:
            chosen = candidates[-1]
            logger.info("Using latest best_model_layer_norm checkpoint: %s", chosen)
            return chosen

    for name in legacy_candidates:
        p = models_dir / name
        if p.exists():
            logger.info("Using legacy fallback checkpoint: %s", p)
            return p

    # Keep previous behavior as final fallback path (may not exist; warning handled below).
    return models_dir / "best_model_layer_norm_20260214_133657.pt"


# ============================================================
# Runtime Configuration
# ============================================================

class RuntimeConfig:
    use_gpu: bool = os.getenv("USE_GPU", "1") == "1"
    require_gpu: bool = False

    allowed_origins: list[str] = _resolve_allowed_origins()


RUNTIME_CONFIG = RuntimeConfig()


# ============================================================
# Model Configuration
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "MASKED_GNN"

    # REQUIRED by encoder - must match encoder's 7 features
    input_dim: int = 7

    # Preferred device (runtime resolves final)
    device: str = "cuda" if os.getenv("USE_GPU", "1") == "1" else "cpu"

    # Checkpoint is resolved dynamically for runtime predict/discover.
    checkpoint_path: Path = _resolve_runtime_checkpoint()
    model_name: str = checkpoint_path.name

    deterministic: bool = True
    allow_speculative: bool = True


MODEL_CONFIG = ModelConfig()

if not MODEL_CONFIG.checkpoint_path.exists():
    logger.warning(
        "Model checkpoint not found at %s",
        MODEL_CONFIG.checkpoint_path,
    )


# ============================================================
# Graph Configuration (FIXED PATH)
# ============================================================

@dataclass(frozen=True)
class GraphConfig:
    filename: str = "masked_battery_graph_normalized_v2.pt"
    graph_dir: Path = GRAPHS_DIR  # FIXED (was graphs/graph_objects)
    lazy_load: bool = True
    cache_graph: bool = True


GRAPH_CONFIG = GraphConfig()


# ============================================================
# Discovery Configuration
# ============================================================

@dataclass(frozen=True)
class DiscoveryConfig:
    max_candidates: int = 64
    elite_size: int = 5
    max_iterations: int = 6

    mutation_strength: float = 0.15
    extrapolation_strength: float = 0.30

    speculative_voltage_threshold: float = 5.5
    hard_voltage_cap: float = 7.0


DISCOVERY_CONFIG = DiscoveryConfig()


# ============================================================
# Backend / API Configuration
# ============================================================

@dataclass(frozen=True)
class BackendConfig:
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", 8000))

    enable_chat: bool = True
    enable_cif_upload: bool = True
    enable_discovery: bool = True

    max_cif_size_mb: int = 5
    request_timeout_sec: int = int(os.getenv("REQUEST_TIMEOUT_SEC", 60))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


BACKEND_CONFIG = BackendConfig()


# ============================================================
# Security / Reliability Configuration
# ============================================================

@dataclass(frozen=True)
class SecurityConfig:
    api_key: str = os.getenv("ECESSP_API_KEY", "").strip()
    require_api_key: bool = (os.getenv("REQUIRE_API_KEY", "0") == "1") or bool(os.getenv("ECESSP_API_KEY", "").strip())
    bearer_token: str = os.getenv("ECESSP_BEARER_TOKEN", "").strip()
    require_bearer_token: bool = (os.getenv("REQUIRE_BEARER_TOKEN", "0") == "1") or bool(os.getenv("ECESSP_BEARER_TOKEN", "").strip())

    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", 120))
    rate_limit_window_sec: int = int(os.getenv("RATE_LIMIT_WINDOW_SEC", 60))


SECURITY_CONFIG = SecurityConfig()


# ============================================================
# Validation Utilities
# ============================================================

def validate_paths(strict: bool = False) -> list[str]:

    required_dirs = [
        DATA_DIR,
        MODELS_DIR,
        GRAPHS_DIR,
        DESIGN_DIR,
        MATERIALS_DIR,
    ]

    missing = [str(p) for p in required_dirs if not p.exists()]
    if missing:
        msg = "Missing required project directories:\n" + "\n".join(missing)
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    return missing


def summary() -> dict:
    return {
        "environment": ENVIRONMENT,
        "project_root": str(PROJECT_ROOT),
        "model": {
            "name": MODEL_CONFIG.model_name,
            "checkpoint": str(MODEL_CONFIG.checkpoint_path),
        },
        "graph": {
            "filename": GRAPH_CONFIG.filename,
            "graph_dir": str(GRAPH_CONFIG.graph_dir),
        },
        "discovery": DISCOVERY_CONFIG.__dict__,
        "backend": BACKEND_CONFIG.__dict__,
    }


# ============================================================
# Safe Local Validation
# ============================================================

if ENVIRONMENT != "replit":
    try:
        validate_paths(strict=False)
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during path validation: %s", exc)
