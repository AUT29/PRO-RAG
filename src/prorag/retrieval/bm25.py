"""BM25 retrieval over proposition-network nodes."""

from __future__ import annotations

import json
from typing import Dict, List

import bm25s

try:
    import Stemmer
except ImportError:
    Stemmer = None


class BM25RAGSystem:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.retriever = bm25s.BM25()
        self.stemmer = Stemmer.Stemmer("english") if Stemmer else None

    def build_index(self, pnet_file: str) -> bool:
        with open(pnet_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.metadata = [
            {
                "id": node.get("id", index),
                "text": node.get("text", ""),
                "entities": node.get("entities", []),
            }
            for index, node in enumerate(data.get("nodes", []))
            if node.get("text")
        ]
        self.documents = [item["text"] for item in self.metadata]
        self.retriever.index(
            bm25s.tokenize(self.documents, stopwords="en", stemmer=self.stemmer)
        )
        return True

    def retrieve_documents(self, query: str) -> List[Dict]:
        indices, scores = self.retriever.retrieve(
            bm25s.tokenize(query, stemmer=self.stemmer),
            k=min(self.top_k, len(self.documents)),
        )
        results = []
        for rank, (index, score) in enumerate(zip(indices[0], scores[0]), start=1):
            item = dict(self.metadata[int(index)])
            item.update(score=float(score), rank=rank)
            results.append(item)
        return results
