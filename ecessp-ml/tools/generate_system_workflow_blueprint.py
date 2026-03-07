from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


@dataclass
class FlowStep:
    title: str
    detail: str
    anchor_file: str
    anchor_pattern: str


PALETTE = {
    "bg": "#F9FAFB",
    "text": "#111827",
    "muted": "#374151",
    "discovery": "#DBEAFE",
    "discovery_edge": "#1D4ED8",
    "prediction": "#FEF3C7",
    "prediction_edge": "#B45309",
    "core": "#DCFCE7",
    "core_edge": "#15803D",
    "data": "#EDE9FE",
    "data_edge": "#6D28D9",
    "opt": "#FCE7F3",
    "opt_edge": "#BE185D",
}


DISCOVERY_STEPS = [
    FlowStep(
        title="D1. UI Trigger",
        detail="User clicks Run Discovery in DiscoveryEcesspMl page.",
        anchor_file="ecessp-frontend/client/src/pages/DiscoveryEcesspMl.tsx",
        anchor_pattern=r"handleRunDiscovery",
    ),
    FlowStep(
        title="D2. Request Build",
        detail="createDiscoveryRequest builds canonical payload with system + objective + discovery_params.",
        anchor_file="ecessp-frontend/client/src/hooks/use-ecessp-ml.ts",
        anchor_pattern=r"createDiscoveryRequest",
    ),
    FlowStep(
        title="D3. Frontend API Call",
        detail="useDiscover posts to /api/discover with JSON body.",
        anchor_file="ecessp-frontend/client/src/hooks/use-ecessp-ml.ts",
        anchor_pattern=r"export function useDiscover",
    ),
    FlowStep(
        title="D4. Proxy Route",
        detail="Express backendProxy validates canonical contract then forwards to Python backend /api/discover.",
        anchor_file="ecessp-frontend/server/backendProxy.ts",
        anchor_pattern=r'router\.post\("/discover"',
    ),
    FlowStep(
        title="D5. FastAPI Route",
        detail="routes.discover_system validates DiscoveryRequest and calls DiscoveryService.discover(..., mode='generative').",
        anchor_file="ecessp-ml/backend/api/routes.py",
        anchor_pattern=r"def discover_system",
    ),
    FlowStep(
        title="D6. Orchestration Core",
        detail="DiscoveryService initializes runtime, runs DiscoveryOrchestrator staged pipeline, assembles candidates + metadata/history.",
        anchor_file="ecessp-ml/backend/services/discovery_service.py",
        anchor_pattern=r"def discover\(",
    ),
    FlowStep(
        title="D7. Generative Stages",
        detail="MaterialGenerator -> ChemistryValidator -> RoleClassifier -> CompatibilityModel -> FullCellAssembler -> model inference + ranking + guardrails.",
        anchor_file="ecessp-ml/backend/services/discovery_orchestrator.py",
        anchor_pattern=r"def run_generative",
    ),
    FlowStep(
        title="D8. Response to UI",
        detail="Best system + score + explanation + candidate history + rich metadata returned; UI renders ranked cards and diagnostics.",
        anchor_file="ecessp-frontend/client/src/hooks/use-ecessp-ml.ts",
        anchor_pattern=r"extractCandidateSystems",
    ),
]


PREDICTION_STEPS = [
    FlowStep(
        title="P1. UI Trigger",
        detail="User selects 5 components and clicks Predict Performance.",
        anchor_file="ecessp-frontend/client/src/pages/Prediction.tsx",
        anchor_pattern=r"handlePredict",
    ),
    FlowStep(
        title="P2. Frontend API Call",
        detail="usePredict posts {components} to /api/predict.",
        anchor_file="ecessp-frontend/client/src/hooks/use-ecessp-ml.ts",
        anchor_pattern=r"export function usePredict",
    ),
    FlowStep(
        title="P3. Proxy Route",
        detail="Express backendProxy forwards /predict to Python backend /api/predict.",
        anchor_file="ecessp-frontend/server/backendProxy.ts",
        anchor_pattern=r'router\.post\("/predict"',
    ),
    FlowStep(
        title="P4. FastAPI Route",
        detail="routes.predict_system builds deterministic feature hint and resolves formulas from material catalog.",
        anchor_file="ecessp-ml/backend/api/routes.py",
        anchor_pattern=r"def predict_system",
    ),
    FlowStep(
        title="P5. Predictive Service Call",
        detail="predict_system invokes DiscoveryService.discover(..., mode='predictive') for unified inference path.",
        anchor_file="ecessp-ml/backend/api/routes.py",
        anchor_pattern=r"mode='predictive'",
    ),
    FlowStep(
        title="P6. Model Inference",
        detail="EnhancedInferenceEngine.infer runs MaskedGNN forward, computes auxiliary heads, and populates uncertainty/material/cell-level outputs.",
        anchor_file="ecessp-ml/backend/runtime/enhanced_engine.py",
        anchor_pattern=r"def predict_system",
    ),
    FlowStep(
        title="P7. Chemistry Guardrails",
        detail="Predictive chemistry gate + electrolyte stability + N/P checks adjust validity, score, and confidence.",
        anchor_file="ecessp-ml/backend/api/routes.py",
        anchor_pattern=r"def _predictive_chemistry_gate",
    ),
    FlowStep(
        title="P8. Response to UI",
        detail="PredictionResponse returns predicted_properties (8 fields), confidence_score, score, diagnostics.",
        anchor_file="ecessp-ml/backend/api/schemas.py",
        anchor_pattern=r"class PredictionResponse",
    ),
]


CORE_ARCH_BLOCKS = [
    (
        "Runtime Context",
        "RuntimeContext singleton loads model, graph, encoder, decoder and exposes readiness + metadata.",
        "ecessp-ml/backend/runtime/context.py",
        r"class RuntimeContext",
    ),
    (
        "Model Loader",
        "load_model resolves checkpoint, infers decoder flags, constructs MaskedGNN, strict state_dict load.",
        "ecessp-ml/backend/loaders/load_model.py",
        r"def load_model",
    ),
    (
        "MaskedGNN Architecture",
        "Material encoder + interaction stack (MLP/GATv2/MPNN/Transformer) + decoder (legacy|multihead) + optional role/compatibility/uncertainty heads.",
        "ecessp-ml/models/masked_gnn.py",
        r"class MaskedGNN",
    ),
    (
        "Graph + Features",
        "Graph loader consumes battery_features/material_embeddings/node_masks/edge_index_dict; encoder normalizes canonical 7-feature vector.",
        "ecessp-ml/backend/loaders/load_graph.py",
        r"MASKED_GNN_GRAPH_KEYS",
    ),
    (
        "Scoring + Constraints + Reasoning",
        "evaluate_system hard-gates physics/chemistry; score_system ranks; reason_about_system generates explanation text.",
        "ecessp-ml/backend/services/discovery_service.py",
        r"score_result = score_system",
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_line(path: Path, pattern: str) -> int | None:
    rx = re.compile(pattern)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if rx.search(line):
                return i
    return None


def _anchor(path: str, pattern: str) -> str:
    p = _repo_root() / path
    line = _find_line(p, pattern)
    if line is None:
        return f"{path}:?"
    return f"{path}:{line}"


def _wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    face: str,
    edge: str,
    title_size: int = 10,
    body_size: int = 8,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.01,
        y + h - 0.022,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=PALETTE["text"],
        va="top",
        ha="left",
    )
    ax.text(
        x + 0.01,
        y + h - 0.052,
        _wrap(body, width=54),
        fontsize=body_size,
        color=PALETTE["muted"],
        va="top",
        ha="left",
        linespacing=1.25,
    )


def _draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.3,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def render_diagram(output_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(24, 16), dpi=220)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "ECESSP End-to-End System Blueprint",
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="top",
        color=PALETTE["text"],
    )
    ax.text(
        0.5,
        0.948,
        "Discovery and Prediction request flows, model internals, and optimization points",
        fontsize=12,
        ha="center",
        va="top",
        color=PALETTE["muted"],
    )

    ax.text(0.17, 0.914, "Run Discovery Flow", fontsize=14, fontweight="bold", color=PALETTE["discovery_edge"], ha="center")
    ax.text(0.83, 0.914, "Run Prediction Flow", fontsize=14, fontweight="bold", color=PALETTE["prediction_edge"], ha="center")
    ax.text(0.50, 0.914, "Shared Core + Model Architecture", fontsize=14, fontweight="bold", color=PALETTE["core_edge"], ha="center")

    left_x, mid_x, right_x = 0.03, 0.355, 0.69
    col_w = 0.28
    box_h = 0.08
    gap = 0.012
    start_y = 0.885

    left_centers: list[tuple[float, float]] = []
    for idx, step in enumerate(DISCOVERY_STEPS):
        y = start_y - idx * (box_h + gap) - box_h
        _draw_box(
            ax,
            left_x,
            y,
            col_w,
            box_h,
            step.title,
            f"{step.detail} [{_anchor(step.anchor_file, step.anchor_pattern)}]",
            face=PALETTE["discovery"],
            edge=PALETTE["discovery_edge"],
        )
        left_centers.append((left_x + col_w / 2, y + box_h / 2))

    right_centers: list[tuple[float, float]] = []
    for idx, step in enumerate(PREDICTION_STEPS):
        y = start_y - idx * (box_h + gap) - box_h
        _draw_box(
            ax,
            right_x,
            y,
            col_w,
            box_h,
            step.title,
            f"{step.detail} [{_anchor(step.anchor_file, step.anchor_pattern)}]",
            face=PALETTE["prediction"],
            edge=PALETTE["prediction_edge"],
        )
        right_centers.append((right_x + col_w / 2, y + box_h / 2))

    core_h = 0.11
    core_gap = 0.018
    core_start_y = 0.865
    core_centers: list[tuple[float, float]] = []
    for idx, (title, body, path, pattern) in enumerate(CORE_ARCH_BLOCKS):
        y = core_start_y - idx * (core_h + core_gap) - core_h
        _draw_box(
            ax,
            mid_x,
            y,
            col_w,
            core_h,
            title,
            f"{body} [{_anchor(path, pattern)}]",
            face=PALETTE["core"],
            edge=PALETTE["core_edge"],
            title_size=11,
            body_size=8,
        )
        core_centers.append((mid_x + col_w / 2, y + core_h / 2))

    opt_y = 0.02
    _draw_box(
        ax,
        0.03,
        opt_y,
        0.94,
        0.145,
        "Performance and Quality Optimization Targets",
        (
            "1) Runtime latency: cache material catalog + warm RuntimeContext. "
            "2) Candidate quality: tune DiscoveryOrchestrator stage thresholds (chemistry gate, role confidence, compatibility hard_valid). "
            "3) Predictive reliability: calibrate uncertainty_penalty + chemistry gate weighting and track false-reject/false-accept rates. "
            "4) Throughput: reduce num_candidates/optimize_steps adaptively by objective difficulty. "
            "5) Observability: persist per-stage timings + acceptance rates into report cards for regression monitoring."
        ),
        face=PALETTE["opt"],
        edge=PALETTE["opt_edge"],
        title_size=12,
        body_size=9,
    )

    for i in range(len(left_centers) - 1):
        _draw_arrow(
            ax,
            (left_centers[i][0], left_centers[i][1] - box_h / 2 + 0.005),
            (left_centers[i + 1][0], left_centers[i + 1][1] + box_h / 2 - 0.005),
            PALETTE["discovery_edge"],
        )

    for i in range(len(right_centers) - 1):
        _draw_arrow(
            ax,
            (right_centers[i][0], right_centers[i][1] - box_h / 2 + 0.005),
            (right_centers[i + 1][0], right_centers[i + 1][1] + box_h / 2 - 0.005),
            PALETTE["prediction_edge"],
        )

    _draw_arrow(ax, (mid_x, core_centers[2][1]), (left_x + col_w, left_centers[5][1]), PALETTE["core_edge"])
    _draw_arrow(ax, (mid_x + col_w, core_centers[2][1]), (right_x, right_centers[5][1]), PALETTE["core_edge"])
    _draw_arrow(ax, (mid_x + col_w / 2, core_centers[-1][1] - core_h / 2 + 0.02), (0.5, opt_y + 0.145), PALETTE["opt_edge"])

    png_path = output_dir / "ECESSP_end_to_end_workflow.png"
    svg_path = output_dir / "ECESSP_end_to_end_workflow.svg"
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def _write_section(
    lines: list[str],
    title: str,
    steps: Iterable[FlowStep],
) -> None:
    lines.append(f"## {title}")
    for i, step in enumerate(steps, start=1):
        anchor = _anchor(step.anchor_file, step.anchor_pattern)
        lines.append(f"{i}. **{step.title}**")
        lines.append(f"   - What happens: {step.detail}")
        lines.append(f"   - Code anchor: `{anchor}`")
    lines.append("")


def write_markdown(output_dir: Path) -> Path:
    lines: list[str] = []
    lines.append("# ECESSP End-to-End Workflow Guide")
    lines.append("")
    lines.append("This guide maps the exact runtime path for **Run Discovery** and **Run Prediction**, from UI action to model output.")
    lines.append("")
    lines.append("Generated artifacts:")
    lines.append("- `ECESSP_end_to_end_workflow.png`")
    lines.append("- `ECESSP_end_to_end_workflow.svg`")
    lines.append("")
    _write_section(lines, "Run Discovery: Step-by-step", DISCOVERY_STEPS)
    _write_section(lines, "Run Prediction: Step-by-step", PREDICTION_STEPS)

    lines.append("## Model Architecture and Deep Integration")
    lines.append("1. **Runtime loading and readiness**")
    lines.append(
        f"   - RuntimeContext loads model, graph, encoder, decoder once and serves all requests. "
        f"Anchor: `{_anchor('ecessp-ml/backend/runtime/context.py', r'class RuntimeContext')}`"
    )
    lines.append("2. **MaskedGNN backbone**")
    lines.append(
        f"   - Core model is `MaskedGNN` with material encoder + configurable interaction block + decoder. "
        f"Anchor: `{_anchor('ecessp-ml/models/masked_gnn.py', r'class MaskedGNN')}`"
    )
    lines.append("3. **Decoder and heads**")
    lines.append(
        f"   - Supports `legacy` or `multihead` decoding and optional uncertainty, role, compatibility heads. "
        f"Anchor: `{_anchor('ecessp-ml/models/masked_gnn.py', r'decoder_mode')}`"
    )
    lines.append("4. **Graph and feature contracts**")
    lines.append(
        f"   - Graph loader expects MaskedGNN keys (`battery_features`, `material_embeddings`, `node_masks`, `edge_index_dict`, ...). "
        f"Anchor: `{_anchor('ecessp-ml/backend/loaders/load_graph.py', r'MASKED_GNN_GRAPH_KEYS')}`"
    )
    lines.append(
        f"   - Encoder maps canonical 7-feature system vector into normalized tensor input. "
        f"Anchor: `{_anchor('ecessp-ml/backend/loaders/load_encoder.py', r'FEATURE_ORDER')}`"
    )
    lines.append("5. **Service-layer fusion**")
    discover_anchor = _anchor(
        "ecessp-ml/backend/services/discovery_service.py",
        r"def discover\(",
    )
    lines.append(
        f"   - DiscoveryService merges inference outputs with constraints, scoring, explanations, and guardrail metadata. "
        f"Anchor: `{discover_anchor}`"
    )
    lines.append("6. **Generative staged pipeline**")
    lines.append(
        f"   - DiscoveryOrchestrator executes Stage 1-7 modules: material generation, chemistry validation, role assignment, compatibility scoring, assembly, inference/ranking/reporting. "
        f"Anchor: `{_anchor('ecessp-ml/backend/services/discovery_orchestrator.py', r'def run_generative')}`"
    )
    lines.append("")

    lines.append("## Optimization Playbook (High Impact)")
    lines.append("1. Add stage timing instrumentation inside orchestrator `stage_metrics` and expose latency distribution in metadata.")
    lines.append("2. Tune candidate pool and optimize steps adaptively by objective difficulty to reduce long-tail runtime.")
    lines.append("3. Calibrate predictive confidence against real outcomes using uncertainty penalty + chemistry gate diagnostics.")
    lines.append("4. Add cache and memoization for repeated formula parsing and structure classification in chemistry modules.")
    lines.append("5. Track quality KPIs over time: validity rate, guardrail reject causes, top-k hit rate, and speculative ratio.")
    lines.append("")

    md_path = output_dir / "ECESSP_WORKFLOW_GUIDE.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a visual + textual blueprint for ECESSP discovery and prediction workflows."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ecessp-ml/reports/architecture_blueprint"),
        help="Directory for generated artifacts.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path, svg_path = render_diagram(output_dir)
    md_path = write_markdown(output_dir)

    print(f"Generated: {png_path}")
    print(f"Generated: {svg_path}")
    print(f"Generated: {md_path}")


if __name__ == "__main__":
    main()
