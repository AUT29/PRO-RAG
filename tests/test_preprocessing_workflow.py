from prorag.preprocessing import workflow


def test_prepare_pnet_runs_all_preprocessing_stages(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        workflow,
        "prepare_dataset",
        lambda *args, **kwargs: calls.append(("propositions", args, kwargs)),
    )
    monkeypatch.setattr(
        workflow,
        "process_directory",
        lambda *args, **kwargs: calls.append(("network", args, kwargs)),
    )
    monkeypatch.setattr(
        workflow,
        "add_embeddings_to_pnet",
        lambda *args, **kwargs: calls.append(("embeddings", args, kwargs)),
    )

    result = workflow.prepare_pnet("hotpot", tmp_path / "raw.json", tmp_path / "data")

    assert result == tmp_path / "data" / "pnet" / "hotpot"
    assert [name for name, _, _ in calls] == ["propositions", "network", "embeddings"]
