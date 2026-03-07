# ============================================================
# Global Runtime Context (Corrected & Thread-Safe)
# ============================================================

from __future__ import annotations

from typing import Optional, Dict, Any
import threading
import logging

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..config import MODEL_CONFIG, RUNTIME_CONFIG


class RuntimeContext:
    """
    Singleton, thread-safe runtime registry for ML objects:
    - Model
    - Graph
    - Encoder
    - Decoder
    """

    _instance: Optional["RuntimeContext"] = None
    _lock = threading.RLock()  # Reentrant to prevent deadlocks during init

    def __init__(self):
        if RuntimeContext._instance is not None:
            raise RuntimeError(
                "RuntimeContext is a singleton. Use get_runtime_context()."
            )

        # Device determined immediately
        self.device: torch.device | str = self._resolve_device()

        # ML objects
        self.model: Optional[object] = None
        self.graph: Optional[dict] = None
        self.encoder: Optional[object] = None
        self.decoder: Optional[object] = None

        # Metadata & readiness
        self.metadata: Dict[str, Any] = {}
        self._initialized: bool = False

    # --------------------------------------------------------
    # Device Resolution
    # --------------------------------------------------------
    def _resolve_device(self) -> torch.device | str:
        if torch is None:
            return "cpu"
        if RUNTIME_CONFIG.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # --------------------------------------------------------
    # Initialization (thread-safe)
    # --------------------------------------------------------
    def initialize(self) -> None:
        if self._initialized:
            return

        with RuntimeContext._lock:
            if self._initialized:  # double-check inside lock
                return

            if torch is None:
                logger.info("PyTorch not available; ML runtime disabled.")
                self.metadata = {
                    "device": str(self.device),
                    "model_name": MODEL_CONFIG.model_name,
                    "note": "PyTorch not available",
                }
                return

            logger.info("Initializing RuntimeContext (ML resources)")

            from ..loaders.load_model import load_model
            from ..loaders.load_graph import load_graph
            from ..loaders.load_encoder import load_feature_encoder
            from ..loaders.load_decoder import load_decoder

            errors: list[str] = []

            # -----------------------------
            # Load Model
            # -----------------------------
            try:
                self.model = load_model(
                    checkpoint_path=MODEL_CONFIG.checkpoint_path,
                    device=self.device,
                )
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
                logger.info("Model loaded")
            except Exception as exc:
                msg = f"Model load failed: {exc}"
                logger.exception(msg)
                errors.append(msg)

            # -----------------------------
            # Load Graph
            # -----------------------------
            try:
                self.graph = load_graph(device=self.device)
                logger.info("Graph loaded")
            except Exception as exc:
                msg = f"Graph load failed: {exc}"
                logger.exception(msg)
                errors.append(msg)

            # -----------------------------
            # Load Encoder
            # -----------------------------
            try:
                self.encoder = load_feature_encoder()
                if hasattr(self.encoder, "to"):
                    self.encoder.to(self.device)
                logger.info("Encoder loaded")
            except Exception as exc:
                msg = f"Encoder load failed: {exc}"
                logger.exception(msg)
                errors.append(msg)

            # -----------------------------
            # Load Decoder
            # -----------------------------
            try:
                self.decoder = load_decoder(device=self.device)
                self.decoder.eval()
                for p in self.decoder.parameters():
                    p.requires_grad = False
                logger.info("Decoder loaded")
            except Exception as exc:
                msg = f"Decoder load failed: {exc}"
                logger.exception(msg)
                errors.append(msg)

            # -----------------------------
            # Metadata
            # -----------------------------
            self.metadata = {
                "device": str(self.device),
                "model_name": MODEL_CONFIG.model_name,
                "input_dim": MODEL_CONFIG.input_dim,
                "model_class": self.model.__class__.__name__ if self.model else None,
                "encoder": getattr(self.encoder, "__name__", type(self.encoder).__name__) if self.encoder else None,
                "decoder": self.decoder.__class__.__name__ if self.decoder else None,
            }

            if errors:
                self.metadata["errors"] = errors

            # -----------------------------
            # Initialization Status
            # -----------------------------
            self._initialized = all(
                obj is not None for obj in [self.model, self.graph, self.encoder, self.decoder]
            )

            if not self._initialized:
                logger.error("RuntimeContext partially initialized")
            else:
                logger.info("RuntimeContext fully initialized")

    # --------------------------------------------------------
    # Guards
    # --------------------------------------------------------
    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "RuntimeContext not fully initialized; ML objects missing."
            )

    # --------------------------------------------------------
    # Accessors
    # --------------------------------------------------------
    def get_model(self):
        self._require_initialized()
        return self.model

    def get_graph(self):
        self._require_initialized()
        return self.graph

    def get_encoder(self):
        self._require_initialized()
        return self.encoder

    def get_decoder(self):
        self._require_initialized()
        return self.decoder

    def get_device(self):
        return self.device

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def is_ready_for_discovery(self) -> bool:
        return self._initialized

    def get_model_path(self) -> Optional[str]:
        """Get the path to the model checkpoint file."""
        if MODEL_CONFIG.checkpoint_path and MODEL_CONFIG.checkpoint_path.exists():
            return str(MODEL_CONFIG.checkpoint_path)
        return None

    # --------------------------------------------------------
    # Safe Inference Context
    # --------------------------------------------------------
    def get_inference_context(self):
        self._require_initialized()
        if torch is None:
            return None
        return torch.no_grad()

# ============================================================
# Global Singleton Access
# ============================================================
def get_runtime_context() -> RuntimeContext:
    if RuntimeContext._instance is None:
        with RuntimeContext._lock:
            if RuntimeContext._instance is None:
                RuntimeContext._instance = RuntimeContext()
                try:
                    if MODEL_CONFIG.checkpoint_path and MODEL_CONFIG.checkpoint_path.exists():
                        RuntimeContext._instance.initialize()
                    else:
                        logger.warning(
                            "Checkpoint not found; ML runtime not initialized."
                        )
                except Exception as exc:
                    logger.exception(
                        "RuntimeContext initialization failed: %s", exc
                    )
    return RuntimeContext._instance
