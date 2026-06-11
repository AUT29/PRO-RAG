"""Dense retrieval over proposition-network nodes."""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from .dense import DenseRetriever


class DenseRAGSystem:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.retriever = DenseRetriever()
        self.metadata: List[Dict] = []

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
        self.retriever.corpus = [item["text"] for item in self.metadata]
        raw_embeddings = data.get("embeddings", {}).get("nodes")
        if isinstance(raw_embeddings, dict):
            self.retriever.corpus_embeddings = np.asarray(
                [raw_embeddings[str(item["id"])] for item in self.metadata],
                dtype=np.float32,
            )
        else:
            self.retriever.build_index(self.retriever.corpus)
        return True

    def retrieve_documents(self, query: str) -> List[Dict]:
        texts, scores = self.retriever.retrieve(query, self.top_k)
        by_text = {item["text"]: item for item in self.metadata}
        results = []
        for rank, (text, score) in enumerate(zip(texts, scores), start=1):
            item = dict(by_text[text])
            item.update(score=score, rank=rank)
            results.append(item)
        return results
