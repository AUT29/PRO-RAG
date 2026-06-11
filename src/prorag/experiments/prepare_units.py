#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build unit-representation datasets for proposition-validity experiments.

Uses the same fixed sample subset as the PNet ablation.
Writes prepared representations under data/units/.

  - {dataset}_proposition  : copy existing PNet JSON (nodes + precomputed embeddings)
  - {dataset}_passages     : one unit per doc; split only if >embed API limit (256/50)
  - {dataset}_chunk        : fine-grained token chunks per document
  - {dataset}_sentence     : sentence-level units

Usage:
  prepare-units --datasets hotpot --methods proposition --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ..retrieval.dense import DenseRetriever
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "units"
PINNED_DIR = PROJECT_ROOT / "data" / "pinned_subsets"

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SAMPLE_SEED = 42

# Document-level passages: split only when a single doc exceeds the embed limit.
PASSAGE_SPLIT_CHUNK_TOKENS = 256
PASSAGE_SPLIT_OVERLAP_TOKENS = 50
SMALL_CHUNK_TOKENS = 64

TOKENIZER_MODEL = os.getenv("PRORAG_TOKENIZER_MODEL", "")
TOKENIZER_MODEL_MAX_TOKENS = 512
# Keep chunks below the common 512-token embedding limit.
EMBED_API_MAX_TOKENS = 510
EMBED_API_BATCH_SIZE = 8

_TOKENIZER = None

DATASET_CONFIGS = {
    "hotpot": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json",
        "id_field": "_id",
        "format": "json",
    },
    "2wiki": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "2wiki_dev.json",
        "id_field": "_id",
        "format": "json",
    },
    "musique": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "musique_ans_v1.0_dev.jsonl",
        "id_field": "id",
        "format": "jsonl",
    },
}

METHOD_OUTPUT_DIR = {
    "proposition": "{dataset}_proposition",
    "passages": "{dataset}_passages",
    "chunk": "{dataset}_chunk",
    "sentence": "{dataset}_sentence",
}


def get_tokenizer():
    """Load the tokenizer configured for the embedding model."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        if not TOKENIZER_MODEL:
            raise RuntimeError("Set PRORAG_TOKENIZER_MODEL before preparing token-based units")
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    return _TOKENIZER


def count_tokens(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def encode_token_ids(text: str) -> List[int]:
    text = (text or "").strip()
    if not text:
        return []
    tokenizer = get_tokenizer()
    return tokenizer.encode(text, add_special_tokens=False)


def decode_token_ids(token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    tokenizer = get_tokenizer()
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into non-overlapping token windows."""
    token_ids = encode_token_ids(text)
    if not token_ids:
        return []
    if overlap > 0:
        step = max(1, chunk_size - overlap)
        chunks = []
        for start in range(0, len(token_ids), step):
            piece_ids = token_ids[start : start + chunk_size]
            if not piece_ids:
                continue
            chunks.append(decode_token_ids(piece_ids))
            if start + chunk_size >= len(token_ids):
                break
        return [c for c in chunks if c]
    return [
        decode_token_ids(token_ids[i : i + chunk_size])
        for i in range(0, len(token_ids), chunk_size)
        if token_ids[i : i + chunk_size]
    ]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def sample_id_from_pnet_path(path: Path) -> str:
    name = path.name
    if name.endswith("_pnet.json"):
        return name[: -len("_pnet.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def load_truth_index(dataset_name: str) -> Dict[str, Dict]:
    config = DATASET_CONFIGS[dataset_name]
    path = config["truth_file"]
    id_field = config["id_field"]
    records: List[Dict] = []
    if config["format"] == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    return {str(r.get(id_field, "")): r for r in records if r.get(id_field)}


def load_pinned_subset(dataset_name: str, sample_size: int, seed: int) -> Dict:
    pinned = PINNED_DIR / f"{dataset_name}_n{sample_size}_seed{seed}.json"
    if not pinned.exists():
        raise FileNotFoundError(
            f"Pinned subset not found: {pinned}\n"
                "Run the PNet ablation once to create pinned subsets, "
            f"or ensure {dataset_name}_n{sample_size}_seed{seed}.json exists."
        )
    with open(pinned, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_context_body(raw_body) -> str:
    if isinstance(raw_body, list):
        return " ".join(str(s).strip() for s in raw_body if str(s).strip())
    return str(raw_body or "").strip()


def record_documents(record: Dict, dataset_name: str) -> List[Dict]:
    """Extract every document paragraph in a sample (no silent skips except empty text)."""
    docs = []
    if dataset_name in {"hotpot", "2wiki"}:
        for idx, item in enumerate(record.get("context") or []):
            if not isinstance(item, (list, tuple)) or len(item) < 1:
                continue
            title = str(item[0]).strip()
            body = normalize_context_body(item[1]) if len(item) > 1 else ""
            if body:
                text = f"{title}. {body}".strip()
            else:
                text = title
            if not text:
                continue
            docs.append({"source_id": idx, "title": title, "text": text})
    elif dataset_name == "musique":
        for paragraph in record.get("paragraphs") or []:
            title = str(paragraph.get("title", "")).strip()
            body = str(paragraph.get("paragraph_text", "")).strip()
            source_id = paragraph.get("idx", len(docs))
            if body:
                text = f"{title}. {body}".strip() if title else body
            else:
                text = title
            if not text:
                continue
            docs.append({"source_id": source_id, "title": title, "text": text})
    return docs


def enforce_max_tokens(chunks: List[str], max_tokens: int) -> List[str]:
    """Split any chunk that still exceeds max_tokens (semantic merge edge cases)."""
    out: List[str] = []
    for chunk in chunks:
        ids = encode_token_ids(chunk)
        if len(ids) <= max_tokens:
            if chunk.strip():
                out.append(chunk)
            continue
        for start in range(0, len(ids), max_tokens):
            piece = decode_token_ids(ids[start : start + max_tokens])
            if piece.strip():
                out.append(piece)
    return out


def passage_units_for_document(text: str) -> List[str]:
    """One retrieval unit per candidate doc; split only if longer than embed API limit."""
    text = (text or "").strip()
    if not text:
        return []
    if count_tokens(text) <= EMBED_API_MAX_TOKENS:
        return [text]
    return chunk_text_by_tokens(
        text, PASSAGE_SPLIT_CHUNK_TOKENS, overlap=PASSAGE_SPLIT_OVERLAP_TOKENS
    )


def build_text_units(
    record: Dict,
    dataset_name: str,
    method: str,
    dense: Optional[DenseRetriever],
) -> List[Dict]:
    sid = str(record.get(DATASET_CONFIGS[dataset_name]["id_field"], ""))
    docs = record_documents(record, dataset_name)
    units: List[Dict] = []
    unit_id = 0

    docs_with_chunks = set()
    for doc in docs:
        text = doc["text"]
        if method == "passages":
            texts = passage_units_for_document(text)
        elif method == "chunk":
            texts = chunk_text_by_tokens(text, SMALL_CHUNK_TOKENS, overlap=0)
        elif method == "sentence":
            texts = enforce_max_tokens(split_sentences(text), EMBED_API_MAX_TOKENS)
        else:
            raise ValueError(f"Unknown unit method: {method}")

        if count_tokens(text) > 0 and not texts:
            if method == "passages":
                texts = passage_units_for_document(text)
            elif method == "chunk":
                texts = chunk_text_by_tokens(text, SMALL_CHUNK_TOKENS, overlap=0)
            else:
                texts = enforce_max_tokens(split_sentences(text), EMBED_API_MAX_TOKENS)

        for local_idx, chunk_text in enumerate(texts):
            clean = chunk_text.strip()
            if not clean:
                continue
            tok = count_tokens(clean)
            units.append(
                {
                    "id": unit_id,
                    "text": clean,
                    "token_count": tok,
                    "source_doc_id": doc["source_id"],
                    "source_title": doc.get("title", ""),
                    "local_index": local_idx,
                }
            )
            unit_id += 1
            docs_with_chunks.add(doc["source_id"])

        if count_tokens(text) > 0 and doc["source_id"] not in docs_with_chunks:
            raise RuntimeError(
                f"Document {doc['source_id']} produced no chunks for method={method} "
                f"(sample={sid})"
            )

    return units


def split_text_for_embed_api(
    text: str,
    chunk_size: int = PASSAGE_SPLIT_CHUNK_TOKENS,
    overlap: int = PASSAGE_SPLIT_OVERLAP_TOKENS,
) -> List[str]:
    """Split text into embedding-safe pieces with optional overlap."""
    text = (text or "").strip()
    if not text:
        return []
    pieces = chunk_text_by_tokens(text, chunk_size, overlap=overlap)
    return enforce_max_tokens(pieces, EMBED_API_MAX_TOKENS)


def expand_units_to_embeddable(units: List[Dict]) -> List[Dict]:
    """Proactively split units that exceed the embedding API token limit."""
    expanded: List[Dict] = []
    for unit in units:
        text = unit.get("text", "")
        if count_tokens(text) <= EMBED_API_MAX_TOKENS:
            expanded.append(dict(unit))
            continue
        pieces = split_text_for_embed_api(text)
        if not pieces:
            raise RuntimeError(f"Unit could not be split for embedding: id={unit.get('id')}")
        for local_idx, piece in enumerate(pieces):
            piece_unit = dict(unit)
            piece_unit["text"] = piece
            piece_unit["token_count"] = count_tokens(piece)
            piece_unit["local_index"] = local_idx
            piece_unit["split_reason"] = "exceeds_embed_api_max_tokens"
            expanded.append(piece_unit)
    for new_id, unit in enumerate(expanded):
        unit["id"] = new_id
    return expanded


class UnitNeedsSplitError(Exception):
    """Raised when a unit must be split into multiple retrieval units after an embed failure."""

    def __init__(self, unit_index: int):
        self.unit_index = unit_index
        super().__init__(f"Unit at index {unit_index} requires 256/50 split for embedding")


def prepare_text_for_embedding(text: str) -> str:
    """Sanitize and cap to EMBED_API_MAX_TOKENS before calling the embedding API."""
    text = (text or "").replace("\x00", " ").strip()
    if not text:
        return ""
    token_ids = encode_token_ids(text)
    if len(token_ids) <= EMBED_API_MAX_TOKENS:
        return text
    return decode_token_ids(token_ids[:EMBED_API_MAX_TOKENS])


def _request_embedding(text: str, dense: DenseRetriever) -> np.ndarray:
    prepared = prepare_text_for_embedding(text)
    if not prepared:
        raise ValueError("Empty text passed to embedding API after sanitization")
    result = dense._get_embeddings([prepared])
    return np.asarray(result[0], dtype=np.float32)


def embed_one_text_robust(
    text: str,
    dense: DenseRetriever,
    *,
    allow_unit_split: bool = False,
) -> np.ndarray:
    """Embed one string; on API failure fall back to 256/50 split before giving up."""
    try:
        return _request_embedding(text, dense)
    except Exception:
        pieces = split_text_for_embed_api(text)
        if not pieces:
            token_ids = encode_token_ids(text)[:EMBED_API_MAX_TOKENS]
            return _request_embedding(decode_token_ids(token_ids), dense)

        if len(pieces) > 1 and allow_unit_split:
            raise UnitNeedsSplitError(-1)

        last_error: Optional[Exception] = None
        for piece in pieces:
            try:
                return _request_embedding(piece, dense)
            except Exception as exc:
                last_error = exc
                continue

        token_ids = encode_token_ids(text)[:EMBED_API_MAX_TOKENS]
        try:
            return _request_embedding(decode_token_ids(token_ids), dense)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding failed after 256/50 split and truncation "
                f"(pieces={len(pieces)}, tokens={count_tokens(text)})"
            ) from (last_error or exc)


def embed_texts_safe(
    texts: List[str],
    dense: DenseRetriever,
    *,
    unit_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Embed texts; unit_indices maps text index -> unit list index for reactive split."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors: List[np.ndarray] = []
    for idx, text in enumerate(texts):
        is_unit = unit_indices is not None and idx in unit_indices
        try:
            vec = embed_one_text_robust(
                text, dense, allow_unit_split=is_unit
            )
        except UnitNeedsSplitError:
            if unit_indices is None or idx not in unit_indices:
                vec = embed_one_text_robust(text, dense, allow_unit_split=False)
            else:
                raise UnitNeedsSplitError(unit_indices[idx])
        vectors.append(vec)
    return np.vstack(vectors)


def embed_units_and_question(
    question: str,
    units: List[Dict],
    dense: DenseRetriever,
) -> Dict:
    """Embed question + units; reactively split any unit that still triggers API errors."""
    units = expand_units_to_embeddable(units)
    max_split_rounds = max(len(units) * 4, 8)

    for _ in range(max_split_rounds):
        texts = [question] + [u["text"] for u in units]
        unit_text_indices = {1 + i: i for i in range(len(units))}
        try:
            vectors = embed_texts_safe(
                texts, dense, unit_indices=unit_text_indices
            )
            break
        except UnitNeedsSplitError as err:
            unit_idx = err.unit_index
            if unit_idx < 0 or unit_idx >= len(units):
                raise
            bad_unit = units[unit_idx]
            pieces = split_text_for_embed_api(bad_unit["text"])
            if len(pieces) <= 1:
                token_ids = encode_token_ids(bad_unit["text"])[:EMBED_API_MAX_TOKENS]
                pieces = [decode_token_ids(token_ids)]
            replacement: List[Dict] = []
            for local_idx, piece in enumerate(pieces):
                new_unit = dict(bad_unit)
                new_unit["text"] = piece
                new_unit["token_count"] = count_tokens(piece)
                new_unit["local_index"] = local_idx
                new_unit["split_reason"] = "embed_api_fallback_256_50"
                replacement.append(new_unit)
            units = units[:unit_idx] + replacement + units[unit_idx + 1 :]
            for new_id, unit in enumerate(units):
                unit["id"] = new_id
    else:
        raise RuntimeError(
            f"Exceeded max embed split rounds ({max_split_rounds}) for sample"
        )

    if vectors.shape[0] != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: got {vectors.shape[0]} vectors for {len(texts)} texts"
        )
    q_vec = vectors[0]
    node_embeddings = {}
    for unit, vec in zip(units, vectors[1:]):
        node_embeddings[str(unit["id"])] = np.asarray(vec, dtype=np.float32).tolist()
    return {
        "question": np.asarray(q_vec, dtype=np.float32).tolist(),
        "nodes": node_embeddings,
    }


def output_filename_for_sample(sample_id: str, method: str, source_pnet: Path) -> str:
    if method == "proposition":
        return source_pnet.name
    return f"{sample_id}.json"


def copy_proposition_sample(source_pnet: Path, dest_dir: Path, skip_existing: bool) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_pnet.name
    if skip_existing and dest.exists():
        return dest
    shutil.copy2(source_pnet, dest)
    return dest


def build_chunk_sample(
    dataset_name: str,
    sample_id: str,
    record: Dict,
    method: str,
    source_pnet: Path,
    dense: DenseRetriever,
    skip_existing: bool,
    dest_dir: Path,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_name = output_filename_for_sample(sample_id, method, source_pnet)
    dest = dest_dir / out_name
    if skip_existing and dest.exists():
        return dest

    units = build_text_units(record, dataset_name, method, dense)
    units = expand_units_to_embeddable(units)
    docs = record_documents(record, dataset_name)
    question = record.get("question", "")
    answer = record.get("answer", "")
    embeddings = embed_units_and_question(question, units, dense)

    representation_by_method = {
        "passages": "document_passage",
        "chunk": "small_chunk",
        "sentence": "sentence",
    }
    chunk_config = {
        "tokenizer_model": TOKENIZER_MODEL,
        "token_count_metric": "configured_tokenizer",
        "tokenizer_model_max_tokens": TOKENIZER_MODEL_MAX_TOKENS,
        "embed_api_max_tokens": EMBED_API_MAX_TOKENS,
    }
    if method == "passages":
        chunk_config.update(
            {
                "unit_granularity": "one_passage_per_document",
                "split_when_tokens_exceed": EMBED_API_MAX_TOKENS,
                "split_chunk_tokens": PASSAGE_SPLIT_CHUNK_TOKENS,
                "split_overlap_tokens": PASSAGE_SPLIT_OVERLAP_TOKENS,
            }
        )
    elif method == "chunk":
        chunk_config["chunk_tokens"] = SMALL_CHUNK_TOKENS
    elif method == "sentence":
        chunk_config["max_tokens_per_unit"] = EMBED_API_MAX_TOKENS

    payload = {
        "sample_id": sample_id,
        "dataset": dataset_name,
        "question": question,
        "answer": answer,
        "method": method,
        "representation": representation_by_method.get(method, method),
        "source_pnet_path": str(source_pnet),
        "created_at": datetime.now().isoformat(),
        "chunk_config": chunk_config,
        "num_source_documents": len(docs),
        "source_document_ids": [d["source_id"] for d in docs],
        "unit_count": len(units),
        "nodes": units,
        "embeddings": embeddings,
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)
    return dest


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(args) -> None:
    datasets = parse_csv(args.datasets)
    methods = parse_csv(args.methods)
    if "all" in methods:
        methods = ["proposition", "passages", "chunk", "sentence"]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dense: Optional[DenseRetriever] = None
    needs_dense = any(m in methods for m in ("passages", "chunk", "sentence"))
    if needs_dense:
        dense = DenseRetriever()

    manifest = {
        "created_at": datetime.now().isoformat(),
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "pinned_dir": str(PINNED_DIR),
        "output_root": str(OUTPUT_ROOT),
        "methods": methods,
        "datasets": {},
    }

    for dataset_name in datasets:
        pinned = load_pinned_subset(dataset_name, args.sample_size, args.sample_seed)
        pnet_dir = PROJECT_ROOT / "data" / "pnet" / dataset_name
        file_paths = []
        for sample_id in pinned.get("sample_ids", []):
            candidates = [pnet_dir / f"{sample_id}.json", pnet_dir / f"{sample_id}_pnet.json"]
            match = next((candidate for candidate in candidates if candidate.exists()), None)
            if match is not None:
                file_paths.append(match)
        if args.limit > 0:
            file_paths = file_paths[: args.limit]

        truth_index = load_truth_index(dataset_name)
        manifest["datasets"][dataset_name] = {
            "pinned_file": str(PINNED_DIR / f"{dataset_name}_n{args.sample_size}_seed{args.sample_seed}.json"),
            "selected_count": len(file_paths),
            "sample_ids": [sample_id_from_pnet_path(p) for p in file_paths],
        }

        print(f"\n=== {dataset_name}: {len(file_paths)} samples (pinned seed={args.sample_seed}) ===")

        for method in methods:
            out_dir = OUTPUT_ROOT / METHOD_OUTPUT_DIR[method].format(dataset=dataset_name)
            print(f"  Method {method} -> {out_dir}")

            for pnet_path in tqdm(file_paths, desc=f"{dataset_name}/{method}", unit="sample"):
                sid = sample_id_from_pnet_path(pnet_path)
                if not pnet_path.exists():
                    raise FileNotFoundError(f"PNet file missing: {pnet_path}")
                record = truth_index.get(sid)
                if record is None:
                    raise KeyError(f"Truth record not found for sample_id={sid} in {dataset_name}")

                if method == "proposition":
                    copy_proposition_sample(pnet_path, out_dir, args.skip_existing)
                else:
                    build_chunk_sample(
                        dataset_name,
                        sid,
                        record,
                        method,
                        pnet_path,
                        dense,
                        args.skip_existing,
                        out_dir,
                    )

    manifest_path = OUTPUT_ROOT / "preprocess_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest), f, ensure_ascii=False, indent=2)
    print(f"\nSaved manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare knowledge-unit representations for the fixed experiment subset."
    )
    parser.add_argument("--datasets", default="hotpot,2wiki,musique")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated: proposition,passages,chunk,sentence or all",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--limit", type=int, default=0, help="Debug: cap samples per dataset.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sample if output file already exists (resume-friendly).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

