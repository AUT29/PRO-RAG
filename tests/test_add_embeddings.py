import numpy as np

from prorag.preprocessing.add_embeddings import embed_network


class FakeDenseRetriever:
    @staticmethod
    def _get_embeddings(texts, max_batch_size=32):
        return np.asarray([[float(index), 1.0] for index, _ in enumerate(texts)])


def test_embedding_output_uses_actual_node_ids():
    data = {
        "question": "Question",
        "nodes": [
            {"id": 4, "text": "First"},
            {"id": 9, "text": "Second"},
        ],
    }

    result = embed_network(data, FakeDenseRetriever())

    assert result["embeddings"]["question"] == [0.0, 1.0]
    assert result["embeddings"]["nodes"] == {
        "4": [1.0, 1.0],
        "9": [2.0, 1.0],
    }
