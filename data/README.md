# Data Directory

Dataset files and generated representations are not committed to the repository.

Expected structure:

```text
data/
  raw/
    hotpot_dev_distractor_v1.json
    2wiki_dev.json
    musique_ans_v1.0_dev.jsonl
  pnet/
    hotpot/
    2wiki/
    musique/
  propositions/
    hotpot/
    2wiki/
    musique/
  units/
    <dataset>_<representation>/
  pinned_subsets/
    <dataset>_n500_seed42.json
```

`raw/` contains downloaded datasets. `propositions/` contains document-level
extraction outputs. `pnet/` contains proposition-network JSON files with nodes,
edges, and embeddings. `units/` contains representations used by the
knowledge-unit experiment. `pinned_subsets/` contains reproducible sample ID
lists.

A proposition record stores `id`, `question`, `question_entities`, `answer`,
and `nodes`. Each node stores `id`, `text`, `entities`, and `doc_id`. A PNet
record preserves those fields and adds `edges`, `entity_groups`,
`network_stats`, and an `embeddings` block containing the question vector and
vectors keyed by proposition node ID.
