#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal Frontend Folder Structure Scanner
=========================================

Scans only top-level and one-level deep folders, outputs:
- folder names
- subfolder names
- file names
- file counts

Saves JSON to 'frontend_structure_minimal.json'.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def scan_folder_minimal(folder: Path, depth: int = 0, max_depth: int = 1):
    """Scan folder up to max_depth and return minimal structure."""
    structure = {
        "name": folder.name,
        "files": [],
        "subfolders": []
    }

    try:
        for entry in folder.iterdir():
            if entry.is_dir() and depth < max_depth:
                structure["subfolders"].append(scan_folder_minimal(entry, depth + 1, max_depth))
            elif entry.is_file():
                structure["files"].append(entry.name)
    except Exception as e:
        structure["error"] = str(e)

    structure["file_count"] = len(structure["files"])
    structure["subfolder_count"] = len(structure["subfolders"])
    return structure

def main():
    folder = Path.cwd()
    summary = scan_folder_minimal(folder)

    output_file = folder / "frontend_structure_minimal.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Minimal frontend structure saved to {output_file}")

if __name__ == "__main__":
    main()
