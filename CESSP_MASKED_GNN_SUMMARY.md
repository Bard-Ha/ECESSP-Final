# ECESSP Masked GNN Project Summary (Current State)

## 1. Executive Status

This document reflects the **current repository state** after cleanup and consolidation.

- Runtime mode: **MVP inference/discovery service** (not training pipeline)
- Backend framework: **FastAPI**
- Core model: **MaskedGNN** (`ecessp-ml/models/masked_gnn.py`)
- Active checkpoint: `ecessp-ml/reports/models/best_model_layer_norm_20260214_133657.pt`
- Active graph artifact: `ecessp-ml/graphs/masked_battery_graph_normalized.pt`
- Material catalog used by API: `ecessp-ml/data/processed/material_catalog.csv`
- Physics-governance baseline (added **February 20, 2026**):
  - canonical spec: `ecessp-ml/design/physics_first_spec.json`
  - enforced report path: `evaluate_system(...)[\"physics_first\"]`

## 2. What Changed Since Older Summary

The previous summary referenced files/workflows that are no longer part of the active codebase.

Removed from active repo:
- old training/test scripts (`train_*`, `test_*`, standalone validation helpers)
- old implementation note files no longer aligned with current structure
- stale generated plots/figure folders not needed for runtime

Kept intentionally:
- processed and raw data assets (`atomic_embeddings.csv`, `batteries_ml.csv`, `batteries.csv`, `mp_atomic*.csv`, etc.)
- graph build script `ecessp-ml/graphs/build_masked_battery_graph_normalized.py`
- all `ecessp-ml/reports/*` assets
- all `ecessp-ml/tools/*` diagram/report tools
- `ecessp-ml/backend/backend_client.py`

## 3. Current System Architecture

### 3.1 Service Layer

- Entry point: `ecessp-ml/backend/main.py`
- Routers: `ecessp-ml/backend/api/routes.py`
- Service modules:
  - `ecessp-ml/backend/services/discovery_service.py`
  - `ecessp-ml/backend/services/chat_service.py`
  - `ecessp-ml/backend/services/cif_service.py`
  - `ecessp-ml/backend/services/explain_service.py`

### 3.2 Runtime Context and Loaders

- Global singleton runtime: `ecessp-ml/backend/runtime/context.py`
- Loaders:
  - model loader: `ecessp-ml/backend/loaders/load_model.py`
  - graph loader: `ecessp-ml/backend/loaders/load_graph.py`
  - feature encoder: `ecessp-ml/backend/loaders/load_encoder.py`
  - decoder: `ecessp-ml/backend/loaders/load_decoder.py`

`RuntimeContext` initializes and freezes:
- model
- graph
- encoder
- decoder

Readiness gates are enforced before discovery endpoints execute.

### 3.3 Domain/Design Layer

- battery schema/template logic: `ecessp-ml/design/system_template.py`
- generation/scoring/reasoning/constraints:
  - `ecessp-ml/design/system_generator.py`
  - `ecessp-ml/design/system_scorer.py`
  - `ecessp-ml/design/system_reasoner.py`
  - `ecessp-ml/design/system_constraints.py`

### 3.4 Materials Layer

- CIF parsing and descriptor support:
  - `ecessp-ml/materials/cif_parser.py`
  - `ecessp-ml/materials/material_descriptors.py`
  - `ecessp-ml/materials/material_to_system.py`
  - `ecessp-ml/materials/role_inference.py`

## 4. Masked GNN Technical Structure

Model file: `ecessp-ml/models/masked_gnn.py`

Key model characteristics:
- predicts exactly **7 battery properties**
  - `average_voltage`
  - `capacity_grav`
  - `capacity_vol`
  - `energy_grav`
  - `energy_vol`
  - `stability_charge`
  - `stability_discharge`
- input concept:
  - battery-level feature vector (7)
  - up to 5 material slots with mask (`cathode/anode/electrolyte/separator/additives`)
- architecture blocks:
  - material encoder (MLP + LayerNorm + ReLU + Dropout)
  - stacked latent transformation blocks (`num_gnn_layers`)
  - masked aggregation (mean over present material nodes)
  - decoder head outputting 7 properties

Checkpoint compatibility is handled in `load_model.py` for both legacy and current checkpoint key formats.

## 4.1 Physics-First Guardrails (New)

Constraint evaluation now includes a dedicated physics-first block in:
- `ecessp-ml/design/system_constraints.py`

Hard-gate behavior:
- applies voltage stability-window checks
- applies gravimetric/volumetric energy hard caps
- applies gravimetric capacity hard cap
- enforces energy consistency (`E ~= V * C`) when values are present
- runs formula oxidation-state solving and charge-neutrality validation
- computes theoretical capacity (`C_max`) and clips predicted `capacity_grav` to `C_max` (no-override policy)
- contributes directly to `overall_valid` (hard reject on violation)

Output structure:
- `constraints.physics_first.enforced`
- `constraints.physics_first.hard_valid`
- `constraints.physics_first.hard_violations`
- `constraints.physics_first.limits_used`
- `constraints.physics_first.output_requirements`
- `constraints.physics_first.derived` (oxidation states, molar mass, redox-active elements, `C_theoretical`, clipping trace)

## 5. API Workflow (Current)

### 5.1 Health and Diagnostics

Endpoints:
- `GET /health`
- `GET /api/health`
- `GET /api/runtime-diagnostics`
- `GET /metrics`

`/api/runtime-diagnostics` reports:
- checkpoint existence
- graph artifact existence
- runtime ready/not-ready reasons
- metadata/errors from runtime initialization

### 5.2 Materials Retrieval

Endpoint:
- `GET /api/materials?query=&limit=&offset=`

Behavior:
- reads `material_catalog.csv`
- caches results in-memory
- supports substring filtering on `material_id`, `name`, `formula`

### 5.3 Discovery / Prediction Path

Primary endpoints:
- `POST /api/discover`
- `POST /api/discover-from-cif`
- prediction route is served from API router with same runtime context stack

Execution flow:
1. request validation via Pydantic schemas
2. runtime readiness check
3. conversion to `BatterySystem` domain object
4. candidate generation and objective alignment in `DiscoveryService`
5. constraint checks and scoring
6. optional reasoning/explanation layer
7. structured response payload

## 6. Operational Controls

Configured in `ecessp-ml/backend/config.py`:
- auth (API key / bearer token)
- rate limiting
- request timeout
- CORS behavior
- device selection (CPU/GPU preference)

Middleware in `backend/main.py` enforces:
- auth for API paths (except exempt endpoints)
- in-memory rate limiting
- timeout protection
- Prometheus-style metrics emission

## 7. Deployment and Run Modes

### Local MVP
- `ecessp-ml/run_mvp.ps1` launches backend + frontend dev windows
- `ecessp-ml/start_backend.ps1` runs backend only
- `ecessp-ml/stop_mvp.ps1` stop helper

### Docker
- repo root `docker-compose.yml`
- services:
  - `ecessp-ml` on port `8000`
  - `ecessp-frontend` on port `5000`

## 8. Current Project Artifacts for Reporting

Diagrams/reports retained:
- `ecessp-ml/reports/diagrams/Project Structure Map/project_structure_map.png`
- `ecessp-ml/reports/diagrams/Project Structure Map/project_structure_map.mmd`
- `ecessp-ml/reports/diagrams/Project Structure Map/project_structure_map.json`
- `ecessp-ml/reports/diagrams/Project Structure Map/project_structure_map.md`
- `ecessp-ml/reports/cleanup_audit.md`

Utility scripts retained:
- `ecessp-ml/tools/export_project_tree.py`
- `ecessp-ml/tools/generate_project_structure_diagram.py`
- `ecessp-ml/tools/render_project_structure_png.py`

## 9. Known Notes

- The repository is now oriented toward **serving/inference/discovery**, with training pipelines intentionally removed from active scope.

## 10. Conclusion

The ECESSP Masked GNN project is currently in a **clean MVP-serving state**:
- model + graph + runtime loaders are wired and validated
- API endpoints for health/materials/discovery are active
- observability and reliability controls are integrated
- project assets and reporting artifacts are preserved for academic progress updates
