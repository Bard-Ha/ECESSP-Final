# ECESSP End-to-End Workflow Guide

This guide maps the exact runtime path for **Run Discovery** and **Run Prediction**, from UI action to model output.

Generated artifacts:
- `ECESSP_end_to_end_workflow.png`
- `ECESSP_end_to_end_workflow.svg`

## Run Discovery: Step-by-step
1. **D1. UI Trigger**
   - What happens: User clicks Run Discovery in DiscoveryEcesspMl page.
   - Code anchor: `ecessp-frontend/client/src/pages/DiscoveryEcesspMl.tsx:264`
2. **D2. Request Build**
   - What happens: createDiscoveryRequest builds canonical payload with system + objective + discovery_params.
   - Code anchor: `ecessp-frontend/client/src/hooks/use-ecessp-ml.ts:223`
3. **D3. Frontend API Call**
   - What happens: useDiscover posts to /api/discover with JSON body.
   - Code anchor: `ecessp-frontend/client/src/hooks/use-ecessp-ml.ts:125`
4. **D4. Proxy Route**
   - What happens: Express backendProxy validates canonical contract then forwards to Python backend /api/discover.
   - Code anchor: `ecessp-frontend/server/backendProxy.ts:227`
5. **D5. FastAPI Route**
   - What happens: routes.discover_system validates DiscoveryRequest and calls DiscoveryService.discover(..., mode='generative').
   - Code anchor: `ecessp-ml/backend/api/routes.py:151`
6. **D6. Orchestration Core**
   - What happens: DiscoveryService initializes runtime, runs DiscoveryOrchestrator staged pipeline, assembles candidates + metadata/history.
   - Code anchor: `ecessp-ml/backend/services/discovery_service.py:820`
7. **D7. Generative Stages**
   - What happens: MaterialGenerator -> ChemistryValidator -> RoleClassifier -> CompatibilityModel -> FullCellAssembler -> model inference + ranking + guardrails.
   - Code anchor: `ecessp-ml/backend/services/discovery_orchestrator.py:1009`
8. **D8. Response to UI**
   - What happens: Best system + score + explanation + candidate history + rich metadata returned; UI renders ranked cards and diagnostics.
   - Code anchor: `ecessp-frontend/client/src/hooks/use-ecessp-ml.ts:282`

## Run Prediction: Step-by-step
1. **P1. UI Trigger**
   - What happens: User selects 5 components and clicks Predict Performance.
   - Code anchor: `ecessp-frontend/client/src/pages/Prediction.tsx:175`
2. **P2. Frontend API Call**
   - What happens: usePredict posts {components} to /api/predict.
   - Code anchor: `ecessp-frontend/client/src/hooks/use-ecessp-ml.ts:169`
3. **P3. Proxy Route**
   - What happens: Express backendProxy forwards /predict to Python backend /api/predict.
   - Code anchor: `ecessp-frontend/server/backendProxy.ts:248`
4. **P4. FastAPI Route**
   - What happens: routes.predict_system builds deterministic feature hint and resolves formulas from material catalog.
   - Code anchor: `ecessp-ml/backend/api/routes.py:673`
5. **P5. Predictive Service Call**
   - What happens: predict_system invokes DiscoveryService.discover(..., mode='predictive') for unified inference path.
   - Code anchor: `ecessp-ml/backend/api/routes.py:715`
6. **P6. Model Inference**
   - What happens: EnhancedInferenceEngine.infer runs MaskedGNN forward, computes auxiliary heads, and populates uncertainty/material/cell-level outputs.
   - Code anchor: `ecessp-ml/backend/runtime/enhanced_engine.py:1693`
7. **P7. Chemistry Guardrails**
   - What happens: Predictive chemistry gate + electrolyte stability + N/P checks adjust validity, score, and confidence.
   - Code anchor: `ecessp-ml/backend/api/routes.py:387`
8. **P8. Response to UI**
   - What happens: PredictionResponse returns predicted_properties (8 fields), confidence_score, score, diagnostics.
   - Code anchor: `ecessp-ml/backend/api/schemas.py:205`

## Model Architecture and Deep Integration
1. **Runtime loading and readiness**
   - RuntimeContext loads model, graph, encoder, decoder once and serves all requests. Anchor: `ecessp-ml/backend/runtime/context.py:21`
2. **MaskedGNN backbone**
   - Core model is `MaskedGNN` with material encoder + configurable interaction block + decoder. Anchor: `ecessp-ml/models/masked_gnn.py:134`
3. **Decoder and heads**
   - Supports `legacy` or `multihead` decoding and optional uncertainty, role, compatibility heads. Anchor: `ecessp-ml/models/masked_gnn.py:149`
4. **Graph and feature contracts**
   - Graph loader expects MaskedGNN keys (`battery_features`, `material_embeddings`, `node_masks`, `edge_index_dict`, ...). Anchor: `ecessp-ml/backend/loaders/load_graph.py:29`
   - Encoder maps canonical 7-feature system vector into normalized tensor input. Anchor: `ecessp-ml/backend/loaders/load_encoder.py:18`
5. **Service-layer fusion**
   - DiscoveryService merges inference outputs with constraints, scoring, explanations, and guardrail metadata. Anchor: `ecessp-ml/backend/services/discovery_service.py:820`
6. **Generative staged pipeline**
   - DiscoveryOrchestrator executes Stage 1-7 modules: material generation, chemistry validation, role assignment, compatibility scoring, assembly, inference/ranking/reporting. Anchor: `ecessp-ml/backend/services/discovery_orchestrator.py:1009`

## Optimization Playbook (High Impact)
1. Add stage timing instrumentation inside orchestrator `stage_metrics` and expose latency distribution in metadata.
2. Tune candidate pool and optimize steps adaptively by objective difficulty to reduce long-tail runtime.
3. Calibrate predictive confidence against real outcomes using uncertainty penalty + chemistry gate diagnostics.
4. Add cache and memoization for repeated formula parsing and structure classification in chemistry modules.
5. Track quality KPIs over time: validity rate, guardrail reject causes, top-k hit rate, and speculative ratio.
