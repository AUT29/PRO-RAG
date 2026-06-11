"""Provider-neutral dense embedding retrieval."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from openai import OpenAI

from ..config import APISettings


class DenseRetriever:
    def __init__(self, settings: Optional[APISettings] = None):
        self.settings = settings or APISettings.from_env()
        self.client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
        )
        self.model = self.settings.embedding_model
        self.corpus: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None

    def _get_embeddings(
        self, texts: Union[str, List[str]], max_batch_size: int = 32
    ) -> np.ndarray:
        values = [texts] if isinstance(texts, str) else list(texts)
        values = [str(value).strip() for value in values if str(value).strip()]
        if not values:
            raise ValueError("No valid text to embed")
        batches = []
        for start in range(0, len(values), max_batch_size):
            response = self.client.embeddings.create(
                model=self.model,
                input=values[start : start + max_batch_size],
            )
            batches.extend(item.embedding for item in response.data)
        return np.asarray(batches, dtype=np.float32)

    @staticmethod
    def cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        norms = np.linalg.norm(documents, axis=1, keepdims=True)
        documents = documents / np.maximum(norms, 1e-12)
        return documents @ query

    def build_index(self, corpus: List[str]) -> None:
        self.corpus = list(corpus)
        self.corpus_embeddings = self._get_embeddings(self.corpus)

    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List[str], List[float]]:
        if self.corpus_embeddings is None:
            raise RuntimeError("Dense index is not initialized")
        query_embedding = self._get_embeddings(query)[0]
        scores = self.cosine_similarity(query_embedding, self.corpus_embeddings)
        indices = np.argsort(scores)[::-1][:top_k]
        return [self.corpus[i] for i in indices], [float(scores[i]) for i in indices]
