#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configurable internal ablations for the proposition-network retriever.

1. Dynamic proposition scoring update (beam search) vs static dense retrieval.
2. Alpha-weighted node scoring: g_i = alpha * r_i + (1-alpha) * max_neighbor(r_j).
3. Beam combination pruning (max combinations per hop) + early stopping.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from ..retrieval.dense import DenseRetriever
from ..retrieval.proposition_network import PropositionCombinationRetriever


@dataclass
class PNetAblationConfig:
    experiment_name: str = "baseline"
    experiment_group: str = "dynamic"  # dynamic | alpha | early_stop | pruning_m
    use_dynamic_update: bool = True
    use_pruning: bool = True
    use_early_stop: bool = True
    # Paper: initial proposition count n; final retained count (LLM evidence size).
    initial_n: int = 50
    final_n: int = 30
    top_n: int = 30  # plot axis / static sweep label (= final_n for dynamic group)
    alpha: float = 0.5
    max_hops: int = 6
    expansion_size: int = 20  # neighbor expansion k
    max_combinations_per_hop: int = 20  # combinations per hop m
    early_stop_rounds: int = 2  # early stopping threshold r
    unpruned_safety_cap: int = 1000

    def to_dict(self) -> Dict:
        return asdict(self)


class AblationPropositionRetriever(PropositionCombinationRetriever):
    """PNet ablation retriever aligned with the paper definitions."""

    def __init__(self, config: PNetAblationConfig):
        self.ablation_config = config
        # Paper Algorithm 1: stop when no_gain >= T (T = early_stop_rounds).
        # T=0 => no_gain >= 0, stop after the first expansion hop (~2 iterations).
        max_no_qualified = config.early_stop_rounds if config.use_early_stop else 10**9

        super().__init__(
            node_retriever=DenseRetriever(),
            max_hops=config.max_hops,
            initial_props_num=config.initial_n,
            expansion_size=config.expansion_size,
            max_combinations_per_hop=config.max_combinations_per_hop,
            final_props_num=config.final_n,
            alpha=config.alpha,
            max_no_qualified_rounds=max_no_qualified,
        )
        self.ablation_trace = {
            "expanded_combinations": 0,
            "retained_combinations": 0,
            "actual_hops": 1,
            "iteration_number": 1,
            "retrieval_mode": "dynamic",
            "stopped_early": False,
            "stop_reason": "",
        }

    def iter_retrieve(self):
        if not self.ablation_config.use_dynamic_update:
            return self._static_retrieve()
        return self._dynamic_retrieve()

    def _static_retrieve(self):
        """Static baseline: rank propositions once by initial query-proposition similarity only."""
        sorted_nodes = sorted(
            self.original_relevance_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        k = min(self.ablation_config.final_n, len(sorted_nodes))
        top_props = []
        for rank, (node_id, score) in enumerate(sorted_nodes[:k], start=1):
            top_props.append(
                {
                    "id": node_id,
                    "final_rank": rank,
                    "initial_rank": rank,
                    "rank_change": 0,
                    "final_score": float(score),
                    "initial_score": float(score),
                    "score_change": 0.0,
                    "text": self.graph.nodes[node_id]["text"],
                }
            )

        self.ablation_trace.update(
            {
                "expanded_combinations": 0,
                "retained_combinations": 0,
                "actual_hops": 1,
                "iteration_number": 1,
                "retrieval_mode": "static",
                "stopped_early": False,
                "stop_reason": "static_dense_retrieval",
            }
        )
        return top_props, self._final_stats(1, False, "static_dense_retrieval")

    def expand_combinations(
        self,
        current_combinations: List[Tuple[List[int], float]],
        prev_best_score: float,
    ) -> Tuple[List[Tuple[List[int], float]], List[Tuple[List[int], float]]]:
        qualified, candidates = self._expand_all_combinations(current_combinations, prev_best_score)
        total = len(qualified) + len(candidates)
        self.ablation_trace["expanded_combinations"] += total

        if not self.ablation_config.use_pruning:
            retained = qualified + candidates
            retained.sort(key=lambda item: item[1], reverse=True)
            cap = max(1, int(self.ablation_config.unpruned_safety_cap))
            retained = retained[:cap]
            retained_qualified = [item for item in retained if item in qualified]
            retained_candidates = [item for item in retained if item not in retained_qualified]
            self.ablation_trace["retained_combinations"] += len(retained)
            return retained_qualified, retained_candidates

        if total <= self.max_combinations_per_hop:
            self.ablation_trace["retained_combinations"] += total
            return qualified, candidates

        if len(qualified) >= self.max_combinations_per_hop:
            kept_qualified = qualified[: self.max_combinations_per_hop]
            self.ablation_trace["retained_combinations"] += len(kept_qualified)
            return kept_qualified, []

        remaining_slots = self.max_combinations_per_hop - len(qualified)
        kept_candidates = candidates[:remaining_slots]
        self.ablation_trace["retained_combinations"] += len(qualified) + len(kept_candidates)
        return qualified, kept_candidates

    def _expand_all_combinations(
        self,
        current_combinations: List[Tuple[List[int], float]],
        prev_best_score: float,
    ) -> Tuple[List[Tuple[List[int], float]], List[Tuple[List[int], float]]]:
        new_combinations = []
        combination_map = {}

        for combination, _ in current_combinations:
            for neighbor in self.get_valid_neighbors(combination):
                new_combination = combination + [neighbor]
                combination_set = frozenset(new_combination)
                if combination_set in self.qualified_cache or combination_set in self.candidate_cache:
                    continue
                idx = len(new_combinations)
                new_combinations.append(new_combination)
                combination_map[idx] = new_combination
                self.candidate_cache.add(combination_set)

        if not new_combinations:
            return [], []

        scores = self.compute_combinations_scores(new_combinations)
        qualified = []
        candidates = []
        for idx, score in enumerate(scores):
            combination = combination_map[idx]
            item = (combination, float(score))
            if score > prev_best_score:
                qualified.append(item)
                self.qualified_cache.add(frozenset(combination))
            else:
                candidates.append(item)

        qualified.sort(key=lambda item: item[1], reverse=True)
        candidates.sort(key=lambda item: item[1], reverse=True)
        return qualified, candidates

    def _dynamic_retrieve(self):
        current_combinations = self.get_initial_combinations()
        if not current_combinations:
            return [], self._final_stats(0, False, "empty_network")

        best_score = max(score for _, score in current_combinations)
        self.qualified_cache.clear()
        self.candidate_cache.clear()
        self.combinations_history = [(1, current_combinations, [])]
        for combination, _ in current_combinations:
            self.qualified_cache.add(frozenset(combination))

        no_qualified_rounds = 0
        stopped_early = False
        stop_reason = ""
        actual_hops = 1

        for hop in range(2, self.max_hops + 1):
            actual_hops = hop
            qualified, candidates = self.expand_combinations(current_combinations, best_score)

            if qualified:
                best_score = qualified[0][1]
                for combination, score in qualified:
                    self.update_relevance_scores(combination, score)
                no_qualified_rounds = 0
            else:
                no_qualified_rounds += 1

            current_combinations = qualified + candidates
            self.combinations_history.append((hop, qualified, candidates))

            if not current_combinations:
                stopped_early = True
                stop_reason = "no_combinations"
                break
            if (
                self.ablation_config.use_early_stop
                and no_qualified_rounds >= self.ablation_config.early_stop_rounds
            ):
                stopped_early = True
                stop_reason = f"no_qualified_rounds_{self.ablation_config.early_stop_rounds}"
                break

        iteration_number = len(self.combinations_history)
        self.ablation_trace.update(
            {
                "expanded_combinations": self.ablation_trace["expanded_combinations"],
                "retained_combinations": self.ablation_trace["retained_combinations"],
                "actual_hops": actual_hops,
                "iteration_number": iteration_number,
                "retrieval_mode": "dynamic",
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
                "no_qualified_rounds": no_qualified_rounds,
            }
        )
        return self._build_top_props(), self._final_stats(actual_hops, stopped_early, stop_reason)

    def retrieve_full_ranking(self) -> Tuple[List[Dict], Dict]:
        """Dynamic beam once; return all propositions ranked (ignore final_n)."""
        if not self.ablation_config.use_dynamic_update:
            return self._static_retrieve()
        saved_final = self.final_props_num
        self.final_props_num = -1
        try:
            return self._dynamic_retrieve()
        finally:
            self.final_props_num = saved_final

    def export_beam_cache_payload(self) -> Dict:
        """Save proposition scores + full ranking for reuse across dynamic top-n runs."""
        full_props, stats = self.retrieve_full_ranking()
        return {
            "full_props": full_props,
            "original_relevance_scores": {
                str(k): float(v) for k, v in self.original_relevance_scores.items()
            },
            "node_relevance_scores": {
                str(k): float(v) for k, v in self.node_relevance_scores.items()
            },
            "retrieval_stats": stats,
            "total_units": len(self.graph.nodes),
            "question": self.question,
        }

    def _build_top_props(self, limit: Optional[int] = None) -> List[Dict]:
        initial_ranks = {
            node_id: rank + 1
            for rank, (node_id, _) in enumerate(
                sorted(self.original_relevance_scores.items(), key=lambda item: item[1], reverse=True)
            )
        }
        sorted_props = sorted(self.node_relevance_scores.items(), key=lambda item: item[1], reverse=True)
        final_ranks = {node_id: rank + 1 for rank, (node_id, _) in enumerate(sorted_props)}

        top_props = []
        if limit is None:
            slice_end = None if self.final_props_num == -1 else self.final_props_num
        else:
            slice_end = None if limit == -1 else limit
        for node_id, score in sorted_props[:slice_end]:
            top_props.append(
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
        return top_props

    def _final_stats(self, actual_hops: int, stopped_early: bool, stop_reason: str) -> Dict:
        stats = self.get_improvement_statistics()
        stats.update(self.ablation_config.to_dict())
        stats.update(self.ablation_trace)
        stats["embedding_source"] = getattr(self, "embedding_source", "unknown")
        stats["num_precomputed_node_embeddings"] = len(self.node_embeddings)
        stats["question_embedding_dim"] = (
            int(self.question_embedding.shape[0]) if self.question_embedding is not None else 0
        )
        stats["actual_hops"] = actual_hops
        stats["iteration_number"] = self.ablation_trace.get("iteration_number", actual_hops)
        stats["stopped_early"] = stopped_early
        stats["stop_reason"] = stop_reason
        return stats
