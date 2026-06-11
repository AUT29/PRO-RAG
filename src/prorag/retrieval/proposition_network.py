"""Dynamic proposition-network retrieval."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


class PropositionCombinationRetriever:
    """Retrieve connected proposition combinations with dynamic score updates."""

    def __init__(
        self,
        node_retriever,
        max_hops: int = 4,
        initial_props_num: int = 100,
        expansion_size: int = 20,
        max_combinations_per_hop: int = 20,
        final_props_num: int = 20,
        alpha: float = 0.6,
        max_no_qualified_rounds: int = 2,
    ):
        self.node_retriever = node_retriever
        self.dense_retriever = node_retriever
        self.max_hops = max_hops
        self.initial_props_num = initial_props_num
        self.expansion_size = expansion_size
        self.max_combinations_per_hop = max_combinations_per_hop
        self.final_props_num = final_props_num
        self.alpha = alpha
        self.max_no_qualified_rounds = max_no_qualified_rounds
        self.combination_score_cache = None
        self._reset_network()

    def _reset_network(self) -> None:
        self.graph = nx.Graph()
        self.entity_to_nodes: Dict[str, Set[int]] = defaultdict(set)
        self.question: Optional[str] = None
        self.question_embedding: Optional[np.ndarray] = None
        self.node_embeddings: Dict[int, np.ndarray] = {}
        self.embedding_source = "none"
        self.current_sub_query: Optional[str] = None
        self.current_sub_query_embedding: Optional[np.ndarray] = None
        self.qualified_cache: Set[frozenset] = set()
        self.candidate_cache: Set[frozenset] = set()
        self.combinations_history = []
        self.node_relevance_scores: Dict[int, float] = {}
        self.original_relevance_scores: Dict[int, float] = {}
        self._reset_improvement_statistics()

    def _reset_improvement_statistics(self) -> None:
        self.score_improvements = {
            "total_updates": 0,
            "significant_improvements": 0,
            "improvement_threshold": 0.05,
            "max_improvement": 0.0,
            "improved_nodes": set(),
            "top_improvements": [],
            "rank_changes": {},
        }

    def load_from_json(self, json_path: str) -> None:
        self._reset_network()
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.question = str(data.get("question", "")).strip().lower()

        for node in data.get("nodes", []):
            node_id = int(node["id"])
            entities = {
                str(entity).strip().lower()
                for entity in node.get("entities", [])
                if str(entity).strip()
            }
            self.graph.add_node(
                node_id,
                text=str(node.get("text", "")).strip().lower(),
                entities=entities,
            )
            for entity in entities:
                self.entity_to_nodes[entity].add(node_id)

        for edge in data.get("edges", []):
            source, target = int(edge["source"]), int(edge["target"])
            if source not in self.graph or target not in self.graph:
                raise ValueError(f"PNet edge references an unknown node: {source}, {target}")
            shared = {
                str(entity).strip().lower()
                for entity in edge.get("shared_entities", [])
                if str(entity).strip()
            }
            self.graph.add_edge(
                source,
                target,
                type=edge.get("type", "entity"),
                shared_entities=shared,
                weight=len(shared),
            )

        self._load_precomputed_embeddings(data, json_path)
        self._precompute_relevance_scores()

    def _load_precomputed_embeddings(self, data: Dict, json_path: str) -> None:
        block = data.get("embeddings")
        if not isinstance(block, dict):
            raise ValueError(f"PNet file missing embeddings block: {json_path}")
        if block.get("question") is None:
            raise ValueError(f"PNet file missing embeddings.question: {json_path}")
        self.question_embedding = np.asarray(block["question"], dtype=np.float32)

        raw_nodes = block.get("nodes")
        if isinstance(raw_nodes, dict):
            items = raw_nodes.items()
        elif isinstance(raw_nodes, list):
            items = enumerate(raw_nodes)
        else:
            raise ValueError(f"PNet file missing embeddings.nodes: {json_path}")
        self.node_embeddings = {
            int(node_id): np.asarray(vector, dtype=np.float32)
            for node_id, vector in items
        }
        missing = [node_id for node_id in self.graph if node_id not in self.node_embeddings]
        if missing:
            raise ValueError(f"PNet file is missing embeddings for nodes: {missing[:5]}")
        self.embedding_source = "pnet_json_precomputed"

    def set_current_subquery(self, sub_query: str) -> None:
        self.current_sub_query = sub_query.strip().lower()
        self.current_sub_query_embedding = self.dense_retriever._get_embeddings(
            [self.current_sub_query]
        )[0]
        self._recompute_relevance_with_subquery()

    def _recompute_relevance_with_subquery(self) -> None:
        if not self.graph.nodes:
            self.node_relevance_scores = {}
            self.original_relevance_scores = {}
            return
        scores = self.compute_semantic_similarity(
            self.current_sub_query_embedding,
            self._node_embedding_matrix(),
        )
        self.node_relevance_scores = {
            node_id: float(score) for node_id, score in zip(self.graph.nodes, scores)
        }
        self.original_relevance_scores = dict(self.node_relevance_scores)
        self._reset_improvement_statistics()

    def _precompute_relevance_scores(self) -> None:
        if not self.graph.nodes:
            self.node_relevance_scores = {}
            self.original_relevance_scores = {}
            return
        scores = self.compute_semantic_similarity(
            self.question_embedding,
            self._node_embedding_matrix(),
        )
        self.node_relevance_scores = {
            node_id: float(score) for node_id, score in zip(self.graph.nodes, scores)
        }
        self.original_relevance_scores = dict(self.node_relevance_scores)
        self._reset_improvement_statistics()

    def _node_embedding_matrix(self) -> np.ndarray:
        return np.stack([self.node_embeddings[node_id] for node_id in self.graph.nodes])

    def update_relevance_scores(
        self, combination: List[int], combination_score: float
    ) -> None:
        for node_id in combination:
            original = self.original_relevance_scores[node_id]
            current = self.node_relevance_scores[node_id]
            if combination_score <= current:
                continue
            improvement = combination_score - original
            relative = improvement / max(abs(original), 1e-10)
            self.node_relevance_scores[node_id] = combination_score
            self.score_improvements["total_updates"] += 1
            if relative > self.score_improvements["improvement_threshold"]:
                self.score_improvements["significant_improvements"] += 1
            self.score_improvements["max_improvement"] = max(
                self.score_improvements["max_improvement"], relative
            )
            self.score_improvements["improved_nodes"].add(node_id)
            self.score_improvements["top_improvements"].append(
                {
                    "node_id": node_id,
                    "original_score": original,
                    "new_score": combination_score,
                    "improvement": improvement,
                    "relative_improvement": relative,
                    "combination": combination,
                }
            )
            self.score_improvements["top_improvements"] = sorted(
                self.score_improvements["top_improvements"],
                key=lambda item: item["relative_improvement"],
                reverse=True,
            )[:5]

    def get_improvement_statistics(self) -> Dict:
        stats = dict(self.score_improvements)
        total_nodes = len(self.node_relevance_scores)
        improved_nodes = len(stats["improved_nodes"])
        stats["improvement_ratio"] = improved_nodes / total_nodes if total_nodes else 0.0
        stats["significant_improvement_ratio"] = (
            stats["significant_improvements"] / total_nodes if total_nodes else 0.0
        )
        stats["average_improvement"] = (
            sum(
                self.node_relevance_scores[node_id]
                - self.original_relevance_scores[node_id]
                for node_id in stats["improved_nodes"]
            )
            / improved_nodes
            if improved_nodes
            else 0.0
        )
        stats["improved_nodes"] = sorted(stats["improved_nodes"])
        return stats

    @staticmethod
    def compute_semantic_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        if docs.ndim != 2:
            raise ValueError("docs must be a two-dimensional array")
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-10)
        normalized_docs = docs / np.maximum(
            np.linalg.norm(docs, axis=1, keepdims=True), 1e-10
        )
        return (normalized_docs @ query.T).ravel()

    def compute_node_score(self, node_id: int) -> float:
        own_score = self.node_relevance_scores[node_id]
        neighbors = list(self.graph.neighbors(node_id))
        if not neighbors:
            return own_score
        neighbor_score = max(self.node_relevance_scores[node] for node in neighbors)
        return self.alpha * own_score + (1 - self.alpha) * neighbor_score

    def get_initial_combinations(self) -> List[Tuple[List[int], float]]:
        scored = sorted(
            ((node_id, self.compute_node_score(node_id)) for node_id in self.graph),
            key=lambda item: item[1],
            reverse=True,
        )
        return [([node_id], score) for node_id, score in scored[: self.initial_props_num]]

    def get_valid_neighbors(self, combination: List[int]) -> List[int]:
        neighbors: Set[int] = set()
        for node_id in combination:
            neighbors.update(self.graph.neighbors(node_id))
        scored = sorted(
            (
                (node_id, self.compute_node_score(node_id))
                for node_id in neighbors - set(combination)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top = [node_id for node_id, _ in scored[: self.expansion_size]]
        nearby = [
            node_id
            for node_id, _ in scored[self.expansion_size :]
            if any(abs(node_id - existing) <= 5 for existing in combination)
        ]
        return top + nearby

    def _lookup_cached_combination_scores(
        self, combinations: List[List[int]]
    ) -> Tuple[List[Optional[float]], List[int], List[List[int]]]:
        aligned: List[Optional[float]] = [None] * len(combinations)
        missing_indices: List[int] = []
        missing: List[List[int]] = []
        for index, combination in enumerate(combinations):
            cached = (
                self.combination_score_cache.get(combination)
                if self.combination_score_cache is not None
                else None
            )
            if cached is None:
                missing_indices.append(index)
                missing.append(combination)
            else:
                aligned[index] = float(cached)
        return aligned, missing_indices, missing

    def _store_combination_scores(
        self, combinations: List[List[int]], scores: np.ndarray
    ) -> None:
        if self.combination_score_cache is None:
            return
        for combination, score in zip(combinations, scores):
            self.combination_score_cache.set(combination, float(score))

    def _target_embedding(self) -> np.ndarray:
        return (
            self.current_sub_query_embedding
            if self.current_sub_query_embedding is not None
            else self.question_embedding
        )

    def compute_combination_score(self, combination: List[int]) -> float:
        return float(self.compute_combinations_scores([combination])[0])

    def compute_combinations_scores(self, combinations: List[List[int]]) -> np.ndarray:
        if not combinations:
            return np.asarray([], dtype=np.float32)
        aligned, missing_indices, missing = self._lookup_cached_combination_scores(
            combinations
        )
        if missing:
            texts = [
                " || ".join(self.graph.nodes[node_id]["text"] for node_id in combination)
                for combination in missing
            ]
            embeddings = self.dense_retriever._get_embeddings(texts)
            scores = self.compute_semantic_similarity(self._target_embedding(), embeddings)
            self._store_combination_scores(missing, scores)
            for index, score in zip(missing_indices, scores):
                aligned[index] = float(score)
        return np.asarray(aligned, dtype=np.float32)

    def expand_combinations(
        self,
        current_combinations: List[Tuple[List[int], float]],
        prev_best_score: float,
    ) -> Tuple[List[Tuple[List[int], float]], List[Tuple[List[int], float]]]:
        new_combinations = []
        for combination, _ in current_combinations:
            for neighbor in self.get_valid_neighbors(combination):
                candidate = combination + [neighbor]
                key = frozenset(candidate)
                if key in self.qualified_cache or key in self.candidate_cache:
                    continue
                self.candidate_cache.add(key)
                new_combinations.append(candidate)
        scores = self.compute_combinations_scores(new_combinations)
        qualified = []
        candidates = []
        for combination, score in zip(new_combinations, scores):
            item = (combination, float(score))
            if score > prev_best_score:
                qualified.append(item)
                self.qualified_cache.add(frozenset(combination))
            else:
                candidates.append(item)
        qualified.sort(key=lambda item: item[1], reverse=True)
        candidates.sort(key=lambda item: item[1], reverse=True)
        if len(qualified) >= self.max_combinations_per_hop:
            return qualified[: self.max_combinations_per_hop], []
        remaining = self.max_combinations_per_hop - len(qualified)
        return qualified, candidates[:remaining]

    def iter_retrieve(self) -> Tuple[List[Dict], Dict]:
        current = self.get_initial_combinations()
        if not current:
            return [], self.get_improvement_statistics()
        best_score = max(score for _, score in current)
        self.qualified_cache = {frozenset(combination) for combination, _ in current}
        self.candidate_cache.clear()
        self.combinations_history = [(1, current, [])]
        no_qualified_rounds = 0

        for hop in range(2, self.max_hops + 1):
            qualified, candidates = self.expand_combinations(current, best_score)
            if qualified:
                best_score = qualified[0][1]
                for combination, score in qualified:
                    self.update_relevance_scores(combination, score)
                no_qualified_rounds = 0
            else:
                no_qualified_rounds += 1
            current = qualified + candidates
            self.combinations_history.append((hop, qualified, candidates))
            if not current or no_qualified_rounds >= self.max_no_qualified_rounds:
                break

        initial_ranks = {
            node_id: rank
            for rank, (node_id, _) in enumerate(
                sorted(
                    self.original_relevance_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ),
                start=1,
            )
        }
        sorted_nodes = sorted(
            self.node_relevance_scores.items(), key=lambda item: item[1], reverse=True
        )
        final_ranks = {
            node_id: rank for rank, (node_id, _) in enumerate(sorted_nodes, start=1)
        }
        top = []
        limit = None if self.final_props_num == -1 else self.final_props_num
        for node_id, score in sorted_nodes[:limit]:
            top.append(
                {
                    "id": node_id,
                    "final_rank": final_ranks[node_id],
                    "initial_rank": initial_ranks[node_id],
                    "rank_change": initial_ranks[node_id] - final_ranks[node_id],
                    "final_score": float(score),
                    "initial_score": float(self.original_relevance_scores[node_id]),
                    "score_change": float(score - self.original_relevance_scores[node_id]),
                    "text": self.graph.nodes[node_id]["text"],
                }
            )
        self.score_improvements["rank_changes"] = {
            item["id"]: {
                key: value
                for key, value in item.items()
                if key not in {"id"}
            }
            for item in top
        }
        return top, self.get_improvement_statistics()
