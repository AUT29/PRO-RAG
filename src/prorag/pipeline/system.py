"""Complete PRO-RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from ..llm import LLMClient
from ..models import ShortTermMemory, SubQueryResult
from ..retrieval.bm25 import BM25RAGSystem
from ..retrieval.dense_rag import DenseRAGSystem
from ..retrieval.pnet import PNetRAGSystem
from ..usage import get_usage_summary, reset_usage_tracker
from .agents import RetrievalAgent
from .discussion import MeetingDiscussion
from .query_processor import QueryProcessor


@dataclass(frozen=True)
class ProRAGConfig:
    name: str = "pro_rag"
    retrievers: Tuple[str, ...] = ("pnet",)
    use_query_decomposition: bool = True
    use_information_summary: bool = True
    use_meeting_discussion: bool = True
    top_k: int = 20


class ProRAG:
    """PRO-RAG uses PNet by default; alternate retrievers are experiment options."""

    def __init__(
        self,
        pnet_file: str,
        config: ProRAGConfig | None = None,
        llm: LLMClient | None = None,
        agents: Dict[str, RetrievalAgent] | None = None,
    ):
        self.config = config or ProRAGConfig()
        self.llm = llm or LLMClient()
        self.memory = ShortTermMemory()
        self.query_processor = QueryProcessor(self.llm, self.memory)
        self.discussion = MeetingDiscussion(self.llm)
        self.agents = agents if agents is not None else self._build_agents(pnet_file)
        if set(self.agents) != set(self.config.retrievers):
            raise ValueError(
                "Configured retrievers and provided agents must have identical names"
            )

    def _build_agents(self, pnet_file: str) -> Dict[str, RetrievalAgent]:
        factories = {
            "bm25": BM25RAGSystem,
            "dense": DenseRAGSystem,
            "pnet": PNetRAGSystem,
        }
        agents = {}
        for name in self.config.retrievers:
            if name not in factories:
                raise ValueError(f"Unknown retriever: {name}")
            retriever = factories[name](top_k=self.config.top_k)
            retriever.build_index(pnet_file)
            agents[name] = RetrievalAgent(name, retriever, self.llm)
        return agents

    @staticmethod
    def _raw_evidence(documents: List[Dict]) -> str:
        return "\n".join(document.get("text", "") for document in documents)

    def run(self, question: str) -> Dict:
        reset_usage_tracker()
        subqueries = self.query_processor.decompose(
            question,
            enabled=self.config.use_query_decomposition,
        )
        results: List[SubQueryResult] = []
        retrieved_ids = set()

        pending = list(subqueries)
        completed = set()
        while pending:
            executable = [
                item for item in pending if all(dep in completed for dep in item.dependencies)
            ]
            if not executable:
                executable = [pending[0]]
            for subquery in executable:
                rewritten = self.query_processor.rewrite_with_dependencies(subquery)
                for name, agent in self.agents.items():
                    documents = agent.retrieve(rewritten)
                    for document in documents:
                        retrieved_ids.add(str(document.get("id", document.get("text", ""))))
                    summary = (
                        agent.summarize(rewritten, documents)
                        if self.config.use_information_summary
                        else self._raw_evidence(documents)
                    )
                    result = SubQueryResult(
                        subquery.query_id,
                        rewritten,
                        name,
                        documents,
                        summary,
                    )
                    results.append(result)
                    self.memory.add(result)
                completed.add(subquery.query_id)
                pending.remove(subquery)

        if self.config.use_meeting_discussion:
            discussion = self.discussion.synthesize(question, results)
        else:
            discussion = "\n".join(result.summary for result in results)
        answer = self.discussion.answer(question, discussion)
        return {
            "question": question,
            "answer": answer,
            "discussion": discussion,
            "subquery_results": [asdict(result) for result in results],
            "retrieved_proposition_ids": sorted(retrieved_ids),
            "used_units": len(retrieved_ids),
            "config": asdict(self.config),
            **get_usage_summary(),
        }
