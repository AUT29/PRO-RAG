from prorag.experiments.full_pipeline_ablation import ABLATIONS


def test_default_pro_rag_uses_only_pnet():
    assert ABLATIONS["pnet"].retrievers == ("pnet",)


def test_required_full_pipeline_ablations_exist():
    assert set(ABLATIONS) == {
        "bm25",
        "dense",
        "pnet",
        "bm25_dense",
        "bm25_dense_pnet",
        "no_decomposition",
        "no_summary",
        "no_meeting",
    }


def test_no_direct_retriever_is_configured():
    assert all("direct" not in config.retrievers for config in ABLATIONS.values())
