from prorag.preprocessing import prepare_dataset


def test_record_processing_never_passes_question_to_proposition_extractor(monkeypatch):
    seen_documents = []

    def fake_extract(document):
        seen_documents.append(document)
        return {"propositions": [{"text": "fact", "entities": ["Title"]}]}

    monkeypatch.setattr(prepare_dataset, "extract_propositions", fake_extract)
    monkeypatch.setattr(
        prepare_dataset,
        "extract_entities",
        lambda text: {"entities": ["Question Entity"]},
    )

    record = {
        "_id": "sample",
        "question": "SECRET QUESTION",
        "answer": "answer",
        "context": [["Title", ["Document body."]]],
    }
    result = prepare_dataset.process_record(record, "hotpot")

    assert seen_documents == ["Title\nDocument body."]
    assert "SECRET QUESTION" not in seen_documents[0]
    assert result["question"] == "SECRET QUESTION"
    assert result["nodes"][0]["doc_id"] == 0
