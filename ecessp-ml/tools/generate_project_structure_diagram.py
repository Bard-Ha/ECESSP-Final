#!/usr/bin/env python3
"""
Generate a concise monorepo project structure diagram for ECESSP.

Outputs (under ecessp-ml/reports/diagrams):
- project_structure_map.mmd
- project_structure_map.md
- project_structure_map.json
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, UTC

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ecessp-ml"
FRONTEND_ROOT = REPO_ROOT / "ecessp-frontend"
OUT_DIR = ML_ROOT / "reports" / "diagrams"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "attached_assets",
}

MAX_DEPTH = 2
MAX_CHILDREN_PER_DIR = 14


def list_dir(path: Path, depth: int = 0) -> dict:
    node = {
        "name": path.name,
        "path": str(path.relative_to(REPO_ROOT)),
        "type": "dir",
        "children": [],
    }

    if depth >= MAX_DEPTH:
        return node

    children = []
    for item in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if item.name in EXCLUDE_DIRS:
            continue
        if item.name.startswith(".") and item.name not in {".github"}:
            continue

        if item.is_dir():
            children.append(list_dir(item, depth + 1))
        else:
            children.append({
                "name": item.name,
                "path": str(item.relative_to(REPO_ROOT)),
                "type": "file",
            })

    if len(children) > MAX_CHILDREN_PER_DIR:
        shown = children[:MAX_CHILDREN_PER_DIR]
        shown.append({
            "name": f"... +{len(children) - MAX_CHILDREN_PER_DIR} more",
            "path": str(path.relative_to(REPO_ROOT)),
            "type": "meta",
        })
        node["children"] = shown
    else:
        node["children"] = children

    return node


def collect_top_level_summary() -> dict:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(REPO_ROOT),
        "sections": {
            "ecessp-ml": list_dir(ML_ROOT, depth=0),
            "ecessp-frontend": list_dir(FRONTEND_ROOT, depth=0),
        },
    }


def mermaid_from_summary(summary: dict) -> str:
    lines = [
        "flowchart TB",
        "  ROOT[ECESSP Monorepo]",
        "  ROOT --> ML[ecessp-ml\\nML Backend + Runtime + Training]",
        "  ROOT --> FE[ecessp-frontend\\nReact UI + Express Proxy]",
    ]

    def add_children(parent_id: str, node: dict, prefix: str):
        children = node.get("children", [])
        for idx, child in enumerate(children, start=1):
            cid = f"{prefix}_{idx}"
            label = child["name"].replace('"', "'")

            if child["type"] == "dir":
                lines.append(f"  {parent_id} --> {cid}[{label}/]")
                sub = child.get("children", [])
                for jdx, grand in enumerate(sub, start=1):
                    gid = f"{cid}_{jdx}"
                    glabel = grand["name"].replace('"', "'")
                    if grand["type"] == "dir":
                        lines.append(f"  {cid} --> {gid}[{glabel}/]")
                    else:
                        lines.append(f"  {cid} --> {gid}[{glabel}]")
            elif child["type"] == "meta":
                lines.append(f"  {parent_id} --> {cid}(({label}))")
            else:
                lines.append(f"  {parent_id} --> {cid}[{label}]")

    add_children("ML", summary["sections"]["ecessp-ml"], "ML")
    add_children("FE", summary["sections"]["ecessp-frontend"], "FE")
    return "\n".join(lines) + "\n"


def markdown_guide(summary: dict) -> str:
    ts = summary["generated_at"]
    return f"""# Project Structure Map (Monorepo)\n\nGenerated: {ts}\n\n## Purpose\nThis figure gives a high-level map of where core responsibilities live in the ECESSP repository:\n- `ecessp-ml`: ML backend, runtime, model/training, preprocessing, reports\n- `ecessp-frontend`: React client, Express proxy, shared API contracts\n\n## Files\n- Mermaid source: `ecessp-ml/reports/diagrams/project_structure_map.mmd`\n- Structure JSON: `ecessp-ml/reports/diagrams/project_structure_map.json`\n\n## Render\n1. Open Mermaid source in any Mermaid-compatible renderer.\n2. Export as SVG/PNG for slides/report.\n\n## Notes\n- The diagram is intentionally depth-limited for readability.\n- Hidden/large dependency folders are excluded (e.g., `.venv`, `node_modules`).\n"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = collect_top_level_summary()
    mermaid = mermaid_from_summary(summary)
    guide = markdown_guide(summary)

    (OUT_DIR / "project_structure_map.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "project_structure_map.mmd").write_text(mermaid, encoding="utf-8")
    (OUT_DIR / "project_structure_map.md").write_text(guide, encoding="utf-8")

    print("Generated:")
    print(OUT_DIR / "project_structure_map.json")
    print(OUT_DIR / "project_structure_map.mmd")
    print(OUT_DIR / "project_structure_map.md")


if __name__ == "__main__":
    main()
