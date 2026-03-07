import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "project_full_manifest.json"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTENSIONS = {
    ".pt", ".pth", ".ckpt",
    ".csv", ".npy", ".npz",
    ".zip", ".tar", ".gz",
}

MAX_FILE_SIZE_MB = 50  # skip hashing very large files

# ============================================================
# Utilities
# ============================================================
def sha256(path: Path):
    if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return "SKIPPED_LARGE_FILE"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def python_file_stats(path: Path):
    stats = {
        "lines": 0,
        "functions": 0,
        "classes": 0,
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stats["lines"] += 1
                l = line.strip()
                if l.startswith("def "):
                    stats["functions"] += 1
                elif l.startswith("class "):
                    stats["classes"] += 1
    except Exception:
        pass
    return stats


def should_exclude(path: Path):
    if path.name in EXCLUDE_DIRS:
        return True
    if path.suffix in EXCLUDE_EXTENSIONS:
        return True
    return False


# ============================================================
# Tree builder
# ============================================================
def build_tree(path: Path):
    node = {
        "type": "directory" if path.is_dir() else "file",
        "name": path.name,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
    }

    if path.is_dir():
        node["children"] = []
        for item in sorted(path.iterdir()):
            if should_exclude(item):
                continue
            node["children"].append(build_tree(item))
    else:
        stat = path.stat()
        node.update({
            "size_bytes": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "sha256": sha256(path),
            "extension": path.suffix,
        })

        if path.suffix == ".py":
            node["python_stats"] = python_file_stats(path)

    return node


# ============================================================
# Environment snapshot
# ============================================================
def environment_info():
    env = {
        "python_version": sys.version,
        "platform": sys.platform,
        "executable": sys.executable,
    }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            env["pip_packages"] = result.stdout.strip().splitlines()
    except Exception:
        env["pip_packages"] = "UNAVAILABLE"

    return env


# ============================================================
# Main
# ============================================================
def main():
    manifest = {
        "project_name": PROJECT_ROOT.name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "project_root": str(PROJECT_ROOT),
        "environment": environment_info(),
        "structure": build_tree(PROJECT_ROOT),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Full project manifest exported to:")
    print(f"   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
