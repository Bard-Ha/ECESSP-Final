# backend/main.py
# ============================================================
# ECESSP Backend Application Entry Point
# ============================================================

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import threading
import time
from collections import defaultdict, deque
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.api.routes import router as api_router
from backend.runtime.context import get_runtime_context
from backend.config import RUNTIME_CONFIG, BACKEND_CONFIG, SECURITY_CONFIG


class _InMemoryRateLimiter:
    def __init__(self, limit: int, window_sec: int):
        self.limit = limit
        self.window_sec = window_sec
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, int]:
        now = time.time()
        with self._lock:
            bucket = self._buckets[key]
            cutoff = now - self.window_sec
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                retry_after = max(1, int(self.window_sec - (now - bucket[0])))
                return False, retry_after, 0

            bucket.append(now)
            remaining = max(0, self.limit - len(bucket))
            return True, 0, remaining


_RATE_LIMITER = _InMemoryRateLimiter(
    limit=max(1, SECURITY_CONFIG.rate_limit_requests),
    window_sec=max(1, SECURITY_CONFIG.rate_limit_window_sec),
)


class _MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests_total: dict[tuple[str, str, int], int] = defaultdict(int)
        self.request_latency_sum_sec: dict[tuple[str, str], float] = defaultdict(float)

    def observe(self, method: str, path: str, status_code: int, duration_sec: float) -> None:
        route_key = (method, path)
        code_key = (method, path, status_code)
        with self._lock:
            self.requests_total[code_key] += 1
            self.request_latency_sum_sec[route_key] += duration_sec

    def render_prometheus(self) -> str:
        lines: list[str] = []
        now = time.time()
        uptime = max(0.0, now - self.started_at)

        lines.append("# HELP ecessp_uptime_seconds Process uptime in seconds")
        lines.append("# TYPE ecessp_uptime_seconds gauge")
        lines.append(f"ecessp_uptime_seconds {uptime:.3f}")

        lines.append("# HELP ecessp_http_requests_total Total HTTP requests")
        lines.append("# TYPE ecessp_http_requests_total counter")
        for (method, path, status_code), count in sorted(self.requests_total.items()):
            safe_path = path.replace('"', '\\"')
            lines.append(
                f'ecessp_http_requests_total{{method="{method}",path="{safe_path}",status="{status_code}"}} {count}'
            )

        lines.append("# HELP ecessp_http_request_duration_seconds_sum Cumulative request duration")
        lines.append("# TYPE ecessp_http_request_duration_seconds_sum counter")
        for (method, path), total in sorted(self.request_latency_sum_sec.items()):
            safe_path = path.replace('"', '\\"')
            lines.append(
                f'ecessp_http_request_duration_seconds_sum{{method="{method}",path="{safe_path}"}} {total:.6f}'
            )

        return "\n".join(lines) + "\n"


_METRICS = _MetricsCollector()


def _is_authorized(request: Request) -> bool:
    require_any_auth = SECURITY_CONFIG.require_api_key or SECURITY_CONFIG.require_bearer_token
    if not require_any_auth:
        return True

    valid_api_key = False
    valid_bearer = False

    if SECURITY_CONFIG.api_key:
        valid_api_key = request.headers.get("x-api-key", "") == SECURITY_CONFIG.api_key

    if SECURITY_CONFIG.bearer_token:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            provided = auth_header[7:].strip()
            valid_bearer = provided == SECURITY_CONFIG.bearer_token

    return valid_api_key or valid_bearer


# ============================================================
# Lifespan (Modern Startup/Shutdown Handling)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initializes heavy runtime resources once.
    Safe for:
    - Uvicorn
    - Gunicorn
    - Docker
    - Testing
    """
    print("Initializing ECESSP Runtime Context...")

    # Force runtime initialization
    runtime = get_runtime_context()
    runtime.initialize()

    print("Runtime initialized successfully.")

    yield

    print("Shutting down ECESSP backend...")


# ============================================================
# Application Factory
# ============================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="ECESSP ML Backend",
        description=(
            "Electrochemical System Synthesis & Prediction Platform\n\n"
            "Provides battery discovery, reasoning, and optimization APIs "
            "powered by graph neural networks and physics-aware constraints."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --------------------------------------------------------
    # CORS Configuration
    # --------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=RUNTIME_CONFIG.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def security_and_timeout_middleware(request: Request, call_next):
        start = time.time()
        path = request.url.path
        exempt_paths = {
            "/",
            "/health",
            "/api/health",
            "/api/runtime-diagnostics",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

        if path.startswith("/api") and path not in exempt_paths:
            if not _is_authorized(request):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized"},
                )

            if SECURITY_CONFIG.rate_limit_enabled:
                forwarded = request.headers.get("x-forwarded-for")
                client_ip = (forwarded.split(",")[0].strip() if forwarded else None) or (request.client.host if request.client else "unknown")
                allowed, retry_after, remaining = _RATE_LIMITER.check(client_ip)
                if not allowed:
                    response = JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                    )
                    response.headers["Retry-After"] = str(retry_after)
                    response.headers["X-RateLimit-Limit"] = str(SECURITY_CONFIG.rate_limit_requests)
                    response.headers["X-RateLimit-Remaining"] = "0"
                    return response
                request.state.rate_limit_remaining = remaining

        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=max(1, BACKEND_CONFIG.request_timeout_sec),
            )
        except asyncio.TimeoutError:
            response = JSONResponse(
                status_code=504,
                content={"detail": "Request timed out"},
            )
            _METRICS.observe(request.method, path, 504, time.time() - start)
            return response

        if path.startswith("/api") and SECURITY_CONFIG.rate_limit_enabled:
            remaining = getattr(request.state, "rate_limit_remaining", SECURITY_CONFIG.rate_limit_requests)
            response.headers["X-RateLimit-Limit"] = str(SECURITY_CONFIG.rate_limit_requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        _METRICS.observe(request.method, path, int(response.status_code), time.time() - start)
        return response

    # --------------------------------------------------------
    # API Router
    # --------------------------------------------------------
    app.include_router(api_router, prefix="/api")

    # --------------------------------------------------------
    # Root Endpoint
    # --------------------------------------------------------
    @app.get("/", tags=["system"])
    async def root():
        return {
            "status": "ok",
            "service": "ecessp-ml",
            "version": "1.0.0",
        }

    # --------------------------------------------------------
    # Health Endpoint
    # --------------------------------------------------------
    @app.get("/health", tags=["system"])
    async def health():
        return {
            "status": "ok",
            "service": "ecessp-ml",
        }

    @app.get("/metrics", tags=["system"])
    async def metrics():
        return PlainTextResponse(
            content=_METRICS.render_prometheus(),
            media_type="text/plain; version=0.0.4",
        )

    return app


# ============================================================
# ASGI Application Instance
# ============================================================

app = create_app()

