"""Sub-query decomposition and dependency-aware rewriting."""

from __future__ import annotations

from typing import List

from ..llm import LLMClient
from ..models import ShortTermMemory, SubQuery


class QueryProcessor:
    def __init__(self, llm: LLMClient, memory: ShortTermMemory):
        self.llm = llm
        self.memory = memory

    def decompose(self, question: str, enabled: bool = True) -> List[SubQuery]:
        if not enabled:
            return [SubQuery(1, question)]
        data = self.llm.complete_json(
            [
                {
                    "role": "system",
                    "content": "Decompose multi-hop questions into only the essential sub-questions.",
                },
                {
                    "role": "user",
                    "content": (
                        "Return JSON only using this schema: "
                        '{"subqueries":[{"id":1,"query":"...","dependencies":[]}]}.\n'
                        f"Question: {question}"
                    ),
                },
            ]
        )
        return [
            SubQuery(
                int(item["id"]),
                str(item["query"]),
                [int(value) for value in item.get("dependencies", [])],
            )
            for item in data.get("subqueries", [])
        ] or [SubQuery(1, question)]

    def rewrite_with_dependencies(self, subquery: SubQuery) -> str:
        context = self.memory.dependency_context(subquery.dependencies)
        if not context:
            return subquery.text
        return self.llm.complete(
            [
                {
                    "role": "system",
                    "content": "Rewrite the sub-question using resolved dependency information. Output only the rewritten question.",
                },
                {
                    "role": "user",
                    "content": f"Sub-question: {subquery.text}\nResolved information:\n{context}",
                },
            ],
            max_tokens=160,
        )
