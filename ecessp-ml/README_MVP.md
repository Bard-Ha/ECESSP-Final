# ECESSP MVP Runbook

Operating mode: **Mode A (live discovery/prediction)**.

## Local startup (recommended)

From `ecessp-ml`:

```powershell
.\run_mvp.ps1
```

This opens two windows:

1. ML backend (`FastAPI`) at `http://127.0.0.1:8000`
2. Frontend server at `http://localhost:5000`

Frontend proxy forwards `/api/*` to the ML backend using `ECESSP_ML_URL`.

Security/reliability envs (optional):

- `ECESSP_API_KEY` (backend API key; enabling this enforces `x-api-key`)
- `ECESSP_BEARER_TOKEN` (backend bearer token; enables `Authorization: Bearer ...`)
- `AUTH_ENABLED` (frontend app auth gate; `1` enables login-protected discovery/predict routes)
- `AUTH_USERNAME` / `AUTH_PASSWORD` (frontend login credentials)
- `AUTH_TOKEN_TTL_SEC` (frontend app token lifetime; default `43200`)
- `RATE_LIMIT_ENABLED` (`1`/`0`)
- `RATE_LIMIT_REQUESTS` (default `120`)
- `RATE_LIMIT_WINDOW_SEC` (default `60`)
- `REQUEST_TIMEOUT_SEC` (default `60`)

## Backend only

From `ecessp-ml`:

```powershell
.\start_backend.ps1
```

Optional:

```powershell
.\start_backend.ps1 -Reload
.\start_backend.ps1 -BindHost 127.0.0.1 -Port 8001
```

## Health checks

ML backend:

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/runtime-diagnostics
curl http://127.0.0.1:8000/metrics
```

Frontend proxy health:

```powershell
curl http://localhost:5000/api/health
curl http://localhost:5000/api/runtime-diagnostics
```

Frontend auth endpoints (when `AUTH_ENABLED=1`):

```powershell
curl -X POST http://localhost:5000/api/auth/login -H "Content-Type: application/json" -d "{\"username\":\"admin\",\"password\":\"change-me\"}"
curl http://localhost:5000/api/auth/me -H "Authorization: Bearer <token>"
curl -X POST http://localhost:5000/api/auth/logout -H "Authorization: Bearer <token>"
```

Client behavior:

- When `AUTH_ENABLED=0`, the UI loads directly.
- When `AUTH_ENABLED=1`, the UI shows a sign-in gate before discovery/prediction routes.

## Required folders/files

- Model checkpoint: `ecessp-ml/reports/models/best_model_layer_norm_20260214_133657.pt`
- Graph: `ecessp-ml/graphs/masked_battery_graph_normalized.pt`
- Fast material catalog: `ecessp-ml/data/processed/material_catalog.csv`

## Common issues

1. Port already in use:
   - Change `-Port` in `start_backend.ps1`, or free the existing process.
2. Python env not found:
   - Ensure `.venv` exists in `ecessp-ml`, or run backend from the repo root virtualenv.
3. Runtime not ready:
   - Check `http://127.0.0.1:8000/api/runtime-diagnostics` for missing checkpoint/graph details.

## E2E smoke test

From repo root:

```powershell
.\.venv\Scripts\python.exe e2e_smoke_discovery_proxy.py
```

## Docker deployment (baseline)

From repo root:

```powershell
docker compose up --build
```

Services:

- ML backend: `http://localhost:8000`
- Frontend: `http://localhost:5000`
