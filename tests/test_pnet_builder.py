import json

from prorag.preprocessing.pnet_builder import (
    AlignedPropositionNetwork,
    process_directory,
)


def test_pnet_builder_preserves_document_ids():
    network = AlignedPropositionNetwork(
        {
            "id": "sample-1",
            "question": "Who is Alpha?",
            "answer": "Alpha",
            "question_entities": ["Alpha"],
            "nodes": [
                {
                    "id": 0,
                    "text": "Alpha knows Beta.",
                    "entities": ["Alpha", "Beta"],
                    "doc_id": 7,
                }
            ],
        }
    ).build()

    assert network["nodes"][0]["doc_id"] == 7
    assert network["id"] == "sample-1"
    assert network["answer"] == "Alpha"


def test_process_directory_preserves_nested_layout(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    nested = source / "dataset" / "sample.json"
    nested.parent.mkdir(parents=True)
    nested.write_text(
        json.dumps(
            {
                "id": "sample",
                "question": "Question",
                "question_entities": [],
                "nodes": [],
            }
        ),
        encoding="utf-8",
    )

    process_directory(source, destination, num_workers=1)

    assert (destination / "dataset" / "sample.json").exists()
