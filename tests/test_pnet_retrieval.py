import json

import numpy as np

from prorag.retrieval.proposition_network import PropositionCombinationRetriever


class FakeDenseRetriever:
    @staticmethod
    def _get_embeddings(texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("alpha" in lowered or "beta" in lowered),
                    float("gamma" in lowered),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def test_small_pnet_can_be_loaded_and_retrieved(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            {
                "question": "alpha question",
                "nodes": [
                    {"id": 0, "text": "alpha fact", "entities": ["alpha"]},
                    {"id": 1, "text": "beta fact", "entities": ["alpha", "beta"]},
                    {"id": 2, "text": "gamma fact", "entities": ["gamma"]},
                ],
                "edges": [
                    {
                        "source": 0,
                        "target": 1,
                        "type": "entity",
                        "shared_entities": ["alpha"],
                    }
                ],
                "embeddings": {
                    "question": [1.0, 0.0],
                    "nodes": {
                        "0": [1.0, 0.0],
                        "1": [0.9, 0.1],
                        "2": [0.0, 1.0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    retriever = PropositionCombinationRetriever(
        FakeDenseRetriever(),
        max_hops=2,
        initial_props_num=2,
        expansion_size=2,
        max_combinations_per_hop=2,
        final_props_num=2,
    )

    retriever.load_from_json(str(path))
    retriever.set_current_subquery("alpha")
    results, _ = retriever.iter_retrieve()

    assert [result["id"] for result in results] == [0, 1]


def test_empty_pnet_returns_no_results(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(
            {
                "question": "question",
                "nodes": [],
                "edges": [],
                "embeddings": {"question": [1.0, 0.0], "nodes": {}},
            }
        ),
        encoding="utf-8",
    )
    retriever = PropositionCombinationRetriever(FakeDenseRetriever())
    retriever.load_from_json(str(path))

    results, _ = retriever.iter_retrieve()

    assert results == []
