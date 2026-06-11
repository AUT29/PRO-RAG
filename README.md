# PRO-RAG

PRO-RAG is a proposition-network retrieval augmented generation framework for
multi-hop question answering.

The default complete pipeline uses only the PNet retriever:

```text
document-only proposition extraction
-> proposition-network construction
-> sub-query decomposition
-> PNet retrieval
-> evidence summarization
-> meeting discussion
-> final answer generation
```

BM25 and dense retrieval are retained only as controlled full-pipeline
ablations.

## Project Layout

```text
src/prorag/
  preprocessing/   raw data -> document-only propositions -> embedded PNet
  retrieval/       BM25, dense, and PNet retrievers
  pipeline/        decomposition, retrieval, summary, meeting, final answer
  experiments/     full-pipeline, PNet-component, and unit-representation ablations
data/              local datasets and generated representations (README files only in Git)
outputs/           local experiment results (not committed)
tests/             pipeline, preprocessing, retrieval, and public-safety tests
```

## Install

```bash
python -m venv .venv
pip install -e .
```

Install the optional tokenizer dependency for knowledge-unit experiments:

```bash
pip install -e ".[units]"
```

Configure the environment variables shown in `.env.example`. PRO-RAG uses an
OpenAI-compatible API and does not contain provider names, model names, or API
keys in source code.

## Data Preparation

Place raw datasets under `data/raw/`. See [data/README.md](data/README.md) for
the expected structure.

The unified `prepare-pnet` entry point supports all three datasets, guarantees
that proposition extraction receives document text only, builds PNet files,
and adds embeddings:

```bash
prepare-pnet --dataset hotpot \
  --input data/raw/hotpot_dev_distractor_v1.json
```

The individual `prepare-data`, `build-pnet`, and `add-pnet-embeddings`
commands remain available for debugging or partial reruns.

Example:

```bash
prepare-data --dataset hotpot \
  --input data/raw/hotpot_dev_distractor_v1.json \
  --output data/propositions/hotpot
build-pnet --input data/propositions/hotpot --output data/pnet/hotpot
add-pnet-embeddings --pnet-dir data/pnet/hotpot
```

## Run PRO-RAG

```bash
prorag --pnet-file data/pnet/hotpot/sample.json \
  --question "Your multi-hop question"
```

## Full-Pipeline Ablations

```bash
prorag-ablation --name all --pnet-file data/pnet/hotpot/sample.json \
  --question "Your question"
```

Available configurations:

- `bm25`
- `dense`
- `pnet` (default PRO-RAG)
- `bm25_dense`
- `bm25_dense_pnet`
- `no_decomposition`
- `no_summary`
- `no_meeting`

## Additional Experiments

- `pnet-ablation`: dynamic scoring, alpha, early stopping, and pruning experiments
- `prepare-units`: prepare proposition, passage, chunk, and sentence representations
- `unit-experiment`: compare retrieval effectiveness across knowledge-unit representations

All experiment results are written under `outputs/` and are intentionally
excluded from version control.
