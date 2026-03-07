from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("backend", "design", "materials", "models")

DEFERRED_PATH_TOKENS = (
    "crystal_diffusion",
    "reaction_transformer",
    "graph_to_graph_reaction",
)

DEFERRED_CODE_TOKENS = (
    "import atomate2",
    "from atomate2",
    "import fireworks",
    "from fireworks",
    "CrystalDiffusion",
    "ReactionTransformer",
    "GraphToGraphReaction",
)


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_DIRS:
        folder = ROOT / rel
        if not folder.exists():
            continue
        files.extend(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def _main() -> int:
    violations: list[str] = []
    files = _iter_python_files()

    for path in files:
        rel = path.relative_to(ROOT).as_posix().lower()
        for token in DEFERRED_PATH_TOKENS:
            if token in rel:
                violations.append(f"path token '{token}' in {rel}")

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in DEFERRED_CODE_TOKENS:
            if token in text:
                violations.append(f"code token '{token}' in {rel}")

    if violations:
        print("Deferred-feature guard failed. Remove deferred features before merge:")
        for item in violations:
            print(f" - {item}")
        return 1

    print("Deferred-feature guard passed.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
