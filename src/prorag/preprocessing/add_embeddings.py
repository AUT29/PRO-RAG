"""Add question and proposition embeddings to PNet JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from tqdm import tqdm

from ..retrieval.dense import DenseRetriever


def embed_network(
    data: Dict,
    retriever: DenseRetriever,
    *,
    batch_size: int = 32,
) -> Dict:
    question = str(data.get("question", "")).strip()
    if not question:
        raise ValueError("PNet JSON is missing a question")

    nodes = data.get("nodes", [])
    texts = [question] + [str(node.get("text", "")).strip() for node in nodes]
    if any(not text for text in texts):
        raise ValueError("PNet JSON contains an empty proposition")

    vectors = retriever._get_embeddings(texts, max_batch_size=batch_size)
    data["embeddings"] = {
        "question": vectors[0].tolist(),
        "nodes": {
            str(node["id"]): vectors[index + 1].tolist()
            for index, node in enumerate(nodes)
        },
    }
    return data


def pnet_files(directory: Path) -> Iterable[Path]:
    return sorted(path for path in directory.rglob("*.json") if path.is_file())


def add_embeddings_to_pnet(
    pnet_dir: Path,
    output_dir: Path | None = None,
    *,
    batch_size: int = 32,
    overwrite: bool = False,
    retriever: DenseRetriever | None = None,
) -> None:
    source_root = pnet_dir.resolve()
    destination_root = output_dir.resolve() if output_dir else source_root
    destination_root.mkdir(parents=True, exist_ok=True)
    dense = retriever or DenseRetriever()

    for source in tqdm(list(pnet_files(source_root)), unit="file"):
        destination = destination_root / source.relative_to(source_root)
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("embeddings") and not overwrite:
            if destination != source and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            continue

        embedded = embed_network(data, dense, batch_size=batch_size)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(embedded, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add embeddings to PNet JSON files.")
    parser.add_argument("--pnet-dir", "--pnet_dir", dest="pnet_dir", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path)
    parser.add_argument("--batch-size", "--max_batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    add_embeddings_to_pnet(
        args.pnet_dir,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
