#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-experiment proposition-combination score cache (per sample, on disk)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def combination_cache_key(combination: List[int]) -> str:
    """Stable key: sorted node ids (order of combination list does not matter)."""
    return ",".join(str(int(n)) for n in sorted(combination))


class CombinationScoreCache:
    """Per-sample hash table: combination_key -> relevance score with question."""

    def __init__(self, path: Path, sample_id: str, target_key: str = "question"):
        self.path = path
        self.sample_id = sample_id
        self.target_key = target_key
        self.scores: Dict[str, float] = {}
        self.dirty = False
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        bucket = (data.get("scores_by_target") or {}).get(self.target_key)
        if isinstance(bucket, dict):
            self.scores = {str(k): float(v) for k, v in bucket.items()}

    def get(self, combination: List[int]) -> Optional[float]:
        key = combination_cache_key(combination)
        if key in self.scores:
            self.hits += 1
            return self.scores[key]
        self.misses += 1
        return None

    def set(self, combination: List[int], score: float) -> None:
        key = combination_cache_key(combination)
        value = float(score)
        if self.scores.get(key) != value:
            self.scores[key] = value
            self.dirty = True

    def flush(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict = {}
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        scores_by_target = existing.get("scores_by_target")
        if not isinstance(scores_by_target, dict):
            scores_by_target = {}
        scores_by_target[self.target_key] = self.scores
        payload = {
            "sample_id": self.sample_id,
            "updated_at": datetime.now().isoformat(),
            "scores_by_target": scores_by_target,
            "num_entries": len(self.scores),
            "cache_hits_last_session": self.hits,
            "cache_misses_last_session": self.misses,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.dirty = False
