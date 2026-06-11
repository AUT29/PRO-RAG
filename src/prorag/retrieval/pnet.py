"""Proposition-network retrieval wrapper."""

from __future__ import annotations

from typing import Dict, List

from .dense import DenseRetriever
from .proposition_network import PropositionCombinationRetriever


class PNetRAGSystem:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.retriever = PropositionCombinationRetriever(
            node_retriever=DenseRetriever(),
            max_hops=4,
            initial_props_num=50,
            expansion_size=20,
            max_combinations_per_hop=20,
            final_props_num=top_k,
            alpha=0.6,
            max_no_qualified_rounds=2,
        )

    def build_index(self, pnet_file: str) -> bool:
        self.retriever.load_from_json(pnet_file)
        return True

    def set_current_subquery(self, query: str) -> None:
        self.retriever.set_current_subquery(query)

    def retrieve_documents(self, query: str) -> List[Dict]:
        self.set_current_subquery(query)
        results, _ = self.retriever.iter_retrieve()
        return results[: self.top_k]
