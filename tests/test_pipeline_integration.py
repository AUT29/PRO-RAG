from prorag.experiments.full_pipeline_ablation import ABLATIONS
from prorag.pipeline.system import ProRAG


class FakeLLM:
    def __init__(self):
        self.complete_calls = []
        self.json_calls = 0

    def complete_json(self, messages, **kwargs):
        self.json_calls += 1
        return {
            "subqueries": [
                {"id": 1, "query": "first sub-query", "dependencies": []},
                {"id": 2, "query": "second sub-query", "dependencies": [1]},
            ]
        }

    def complete(self, messages, **kwargs):
        self.complete_calls.append(messages)
        system = messages[0]["content"]
        if system.startswith("Rewrite"):
            return "rewritten second sub-query"
        if system.startswith("Integrate"):
            return "meeting result"
        if system.startswith("Answer"):
            return "final answer"
        return "evidence summary"


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.retrieved = []
        self.summarized = []

    def retrieve(self, query):
        self.retrieved.append(query)
        return [{"id": f"{self.name}-{len(self.retrieved)}", "text": f"{self.name} evidence"}]

    def summarize(self, query, documents):
        self.summarized.append(query)
        return f"{self.name} summary"


def build_system(config_name):
    config = ABLATIONS[config_name]
    llm = FakeLLM()
    agents = {name: FakeAgent(name) for name in config.retrievers}
    return ProRAG("unused.json", config=config, llm=llm, agents=agents), llm, agents


def test_default_pipeline_uses_pnet_and_all_core_stages():
    system, llm, agents = build_system("pnet")
    result = system.run("question")

    assert list(agents) == ["pnet"]
    assert agents["pnet"].retrieved == ["first sub-query", "rewritten second sub-query"]
    assert agents["pnet"].summarized == ["first sub-query", "rewritten second sub-query"]
    assert result["discussion"] == "meeting result"
    assert result["answer"] == "final answer"
    assert result["used_units"] == 2
    assert llm.json_calls == 1


def test_no_decomposition_skips_decomposition():
    system, llm, agents = build_system("no_decomposition")
    system.run("whole question")

    assert llm.json_calls == 0
    assert agents["pnet"].retrieved == ["whole question"]


def test_no_summary_passes_raw_evidence_to_meeting():
    system, llm, agents = build_system("no_summary")
    result = system.run("question")

    assert agents["pnet"].summarized == []
    assert result["subquery_results"][0]["summary"] == "pnet evidence"


def test_no_meeting_skips_synthesis():
    system, llm, _ = build_system("no_meeting")
    result = system.run("question")

    assert result["discussion"] == "pnet summary\npnet summary"
    assert not any(
        call[0]["content"].startswith("Integrate") for call in llm.complete_calls
    )


def test_multi_retriever_ablation_runs_each_configured_retriever():
    system, _, agents = build_system("bm25_dense_pnet")
    result = system.run("question")

    assert set(agents) == {"bm25", "dense", "pnet"}
    assert all(len(agent.retrieved) == 2 for agent in agents.values())
    assert len(result["subquery_results"]) == 6
