"""Retriever wrappers and evidence summarization."""

from __future__ import annotations

from typing import Dict, List

from ..llm import LLMClient


class RetrievalAgent:
    def __init__(self, name: str, retriever, llm: LLMClient):
        self.name = name
        self.retriever = retriever
        self.llm = llm

    def retrieve(self, query: str) -> List[Dict]:
        return self.retriever.retrieve_documents(query)

    def summarize(self, query: str, documents: List[Dict]) -> str:
        evidence = "\n".join(
            f"- {document.get('text', '')}" for document in documents
        )
        return self.llm.complete(
            [
                {
                    "role": "system",
                    "content": "Summarize retrieved evidence into a concise answer to the sub-question. Do not add unsupported facts.",
                },
                {
                    "role": "user",
                    "content": f"Sub-question: {query}\nEvidence:\n{evidence}",
                },
            ],
            max_tokens=300,
        )
