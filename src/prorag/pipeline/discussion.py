"""Evidence discussion and final answer generation."""

from __future__ import annotations

from typing import Iterable

from ..llm import LLMClient
from ..models import SubQueryResult


class MeetingDiscussion:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def synthesize(self, question: str, results: Iterable[SubQueryResult]) -> str:
        summaries = "\n".join(
            f"- [{result.retriever}] {result.query}: {result.summary}"
            for result in results
        )
        return self.llm.complete(
            [
                {
                    "role": "system",
                    "content": "Integrate the sub-query findings, resolve conflicts, and retain only evidence relevant to the original question.",
                },
                {
                    "role": "user",
                    "content": f"Original question: {question}\nFindings:\n{summaries}",
                },
            ],
            max_tokens=600,
        )

    def answer(self, question: str, discussion: str) -> str:
        return self.llm.complete(
            [
                {
                    "role": "system",
                    "content": "Answer the question concisely using only the supplied discussion.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nDiscussion:\n{discussion}\nFinal answer:",
                },
            ],
            max_tokens=200,
        )
