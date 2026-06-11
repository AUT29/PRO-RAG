"""Shared data models used by the PRO-RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SubQuery:
    query_id: int
    text: str
    dependencies: List[int] = field(default_factory=list)


@dataclass
class SubQueryResult:
    query_id: int
    query: str
    retriever: str
    documents: List[Dict]
    summary: str


class ShortTermMemory:
    def __init__(self):
        self.results: Dict[int, List[SubQueryResult]] = {}

    def add(self, result: SubQueryResult) -> None:
        self.results.setdefault(result.query_id, []).append(result)

    def dependency_context(self, dependency_ids: List[int]) -> str:
        lines = []
        for dependency_id in dependency_ids:
            for result in self.results.get(dependency_id, []):
                if result.summary:
                    lines.append(result.summary)
        return "\n".join(lines)
