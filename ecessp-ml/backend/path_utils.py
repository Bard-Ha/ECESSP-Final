from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def resolve_project_artifact_path(
    raw_path: str | Path,
    *,
    project_root: Path,
    preferred_dirs: Iterable[Path] = (),
) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return project_root

    candidates: list[Path] = []

    direct = Path(text)
    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.append((project_root / direct).resolve())

    normalized = _WINDOWS_DRIVE_RE.sub("", text.replace("\\", "/"))
    if normalized:
        normalized_rel = Path(normalized)
        candidates.append((project_root / normalized_rel).resolve())

        lowered_parts = [part.lower() for part in normalized_rel.parts]
        if "ecessp-ml" in lowered_parts:
            idx = lowered_parts.index("ecessp-ml")
            suffix_parts = normalized_rel.parts[idx + 1 :]
            if suffix_parts:
                candidates.append((project_root / Path(*suffix_parts)).resolve())

        basename = normalized_rel.name
        if basename:
            candidates.append((project_root / basename).resolve())
            for preferred in preferred_dirs:
                candidates.append((preferred / basename).resolve())

    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return candidate

    return deduped[0] if deduped else project_root
