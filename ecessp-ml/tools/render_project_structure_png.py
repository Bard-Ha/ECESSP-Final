#!/usr/bin/env python3
"""
Render project_structure_map.json into a readable LEFT->RIGHT hierarchical tree PNG.

Layout:
- Horizontal hierarchy (parent on left, descendants on right)
- Vertically stacked nodes
- Horizontal text with wrapping
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGRAM_DIR = REPO_ROOT / "ecessp-ml" / "reports" / "diagrams"
JSON_FILE = DIAGRAM_DIR / "project_structure_map.json"
PNG_FILE = DIAGRAM_DIR / "project_structure_map.png"

WRAP_WIDTH = 24
X_GAP = 7.0
Y_GAP = 1.2


@dataclass
class TreeNode:
    label: str
    depth: int
    children: list["TreeNode"] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0


def fmt(node: dict) -> str:
    kind = node.get("type", "?")
    name = node.get("name", "?")
    if kind == "dir":
        name = f"{name}/"
    return "\n".join(textwrap.wrap(name, width=WRAP_WIDTH)) or name


def build_tree(summary: dict) -> TreeNode:
    root = TreeNode("ECESSP Monorepo", depth=0)

    ml = TreeNode("ecessp-ml\nML Backend + Runtime + Training", depth=1)
    fe = TreeNode("ecessp-frontend\nReact UI + Express Proxy", depth=1)

    for child in summary["sections"]["ecessp-ml"].get("children", []):
        c = TreeNode(fmt(child), depth=2)
        for grand in child.get("children", []):
            c.children.append(TreeNode(fmt(grand), depth=3))
        ml.children.append(c)

    for child in summary["sections"]["ecessp-frontend"].get("children", []):
        c = TreeNode(fmt(child), depth=2)
        for grand in child.get("children", []):
            c.children.append(TreeNode(fmt(grand), depth=3))
        fe.children.append(c)

    root.children = [ml, fe]
    return root


def max_depth(node: TreeNode) -> int:
    if not node.children:
        return node.depth
    return max(max_depth(c) for c in node.children)


def leaves_in_order(node: TreeNode) -> list[TreeNode]:
    if not node.children:
        return [node]
    out: list[TreeNode] = []
    for c in node.children:
        out.extend(leaves_in_order(c))
    return out


def assign_y_positions(node: TreeNode) -> None:
    leaves = leaves_in_order(node)
    for idx, leaf in enumerate(leaves):
        leaf.y = -idx * Y_GAP

    def postorder(n: TreeNode) -> float:
        if not n.children:
            return n.y
        ys = [postorder(c) for c in n.children]
        n.y = (min(ys) + max(ys)) / 2.0
        return n.y

    postorder(node)


def assign_x_positions(node: TreeNode) -> None:
    def walk(n: TreeNode) -> None:
        n.x = n.depth * X_GAP
        for c in n.children:
            walk(c)

    walk(node)


def walk(node: TreeNode) -> list[TreeNode]:
    out = [node]
    for c in node.children:
        out.extend(walk(c))
    return out


def edges(node: TreeNode) -> list[tuple[TreeNode, TreeNode]]:
    out: list[tuple[TreeNode, TreeNode]] = []
    for c in node.children:
        out.append((node, c))
        out.extend(edges(c))
    return out


def style(depth: int) -> tuple[str, str, int]:
    if depth == 0:
        return ("#0b132b", "#dbeafe", 16)
    if depth == 1:
        return ("#1d4ed8", "#e0f2fe", 13)
    if depth == 2:
        return ("#0f766e", "#dcfce7", 11)
    return ("#334155", "#f8fafc", 10)


def draw(root: TreeNode) -> None:
    nodes = walk(root)
    depth_max = max_depth(root)
    y_vals = [n.y for n in nodes]

    fig_w = max(20, (depth_max + 1) * 4.8)
    fig_h = max(16, (len([n for n in nodes if not n.children]) * 0.62) + 6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#ffffff")

    # Elbow connectors: horizontal then vertical
    for p, c in edges(root):
        mid_x = p.x + (X_GAP * 0.45)
        ax.plot([p.x + 0.7, mid_x], [p.y, p.y], color="#64748b", lw=0.9, alpha=0.9, zorder=1)
        ax.plot([mid_x, mid_x], [p.y, c.y], color="#94a3b8", lw=0.8, alpha=0.9, zorder=1)
        ax.plot([mid_x, c.x - 0.7], [c.y, c.y], color="#64748b", lw=0.9, alpha=0.9, zorder=1)

    for n in nodes:
        tcolor, bcolor, fsize = style(n.depth)
        ax.text(
            n.x,
            n.y,
            n.label,
            ha="center",
            va="center",
            fontsize=fsize,
            color=tcolor,
            family="DejaVu Sans",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=bcolor,
                edgecolor="#94a3b8",
                linewidth=1.0,
            ),
            zorder=2,
        )

    ax.set_title("ECESSP Project Structure (Hierarchical Tree)", fontsize=16, pad=14, color="#0f172a")
    ax.set_xlim(-2, (depth_max * X_GAP) + 8)
    ax.set_ylim(min(y_vals) - 2.5, max(y_vals) + 2.5)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(PNG_FILE, dpi=340)
    plt.close(fig)


def main() -> None:
    if not JSON_FILE.exists():
        raise FileNotFoundError(f"Missing input JSON: {JSON_FILE}")

    summary = json.loads(JSON_FILE.read_text(encoding="utf-8"))
    root = build_tree(summary)
    assign_y_positions(root)
    assign_x_positions(root)
    draw(root)
    print(f"Generated: {PNG_FILE}")


if __name__ == "__main__":
    main()
