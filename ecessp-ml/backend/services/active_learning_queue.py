from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from collections import deque
import json

from backend.config import REPORTS_DIR


@dataclass
class ActiveLearningQueueConfig:
    queue_path: Path = REPORTS_DIR / "active_learning_queue.jsonl"
    max_records: int = 20000


class ActiveLearningQueueService:
    """
    Queue-only hook for active learning candidate triage.

    This service intentionally does not run DFT or external orchestration.
    It only persists candidate records for downstream offline selection.
    """

    def __init__(self, config: ActiveLearningQueueConfig | None = None):
        self.config = config or ActiveLearningQueueConfig()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _trim_if_needed(self) -> None:
        path = self.config.queue_path
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self.config.max_records:
            return
        kept = lines[-self.config.max_records :]
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    def enqueue(
        self,
        *,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> int:
        if not items:
            return 0
        self.config.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.queue_path.open("a", encoding="utf-8") as handle:
            for item in items:
                payload = {
                    "queued_at": self._utc_now_iso(),
                    "context": context,
                    "candidate": item,
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._trim_if_needed()
        return len(items)

    def tail(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return the last `limit` queue entries as parsed JSON records.
        Invalid JSON lines are skipped.
        """
        n = max(1, int(limit))
        path = self.config.queue_path
        if not path.exists():
            return []

        tail_lines: deque[str] = deque(maxlen=n)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    tail_lines.append(text)

        records: list[dict[str, Any]] = []
        for line in tail_lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records
