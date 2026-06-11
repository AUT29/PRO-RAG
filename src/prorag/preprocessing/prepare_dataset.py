"""Unified document-only proposition preprocessing for supported datasets."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

from .proposition_extractor import extract_entities, extract_propositions


def load_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return data


def sample_id(record: Dict, dataset: str) -> str:
    key = "id" if dataset == "musique" else "_id"
    value = record.get(key)
    if value is None:
        raise ValueError(f"Record is missing {key}")
    return str(value)


def documents(record: Dict, dataset: str) -> Iterable[Tuple[int, str, str]]:
    if dataset in {"hotpot", "2wiki"}:
        for index, item in enumerate(record.get("context", [])):
            title, body = item
            text = " ".join(body) if isinstance(body, list) else str(body)
            yield index, str(title), text
        return
    for index, paragraph in enumerate(record.get("paragraphs", [])):
        yield index, str(paragraph.get("title", "")), str(
            paragraph.get("paragraph_text", "")
        )


def process_record(record: Dict, dataset: str) -> Dict:
    nodes = []
    for doc_id, title, body in documents(record, dataset):
        document = f"{title}\n{body}".strip()
        if not document:
            continue
        result = extract_propositions(document)
        for proposition in result.get("propositions", []):
            nodes.append(
                {
                    "id": len(nodes),
                    "text": str(proposition.get("text", "")).strip(),
                    "entities": list(proposition.get("entities", [])),
                    "doc_id": doc_id,
                }
            )
    entity_result = extract_entities(str(record.get("question", "")))
    return {
        "id": sample_id(record, dataset),
        "question": record.get("question", ""),
        "question_entities": entity_result.get("entities", []),
        "answer": record.get("answer", ""),
        "nodes": nodes,
    }


def prepare_dataset(
    dataset: str,
    input_path: Path,
    output_dir: Path,
    *,
    limit: int = -1,
    workers: int = 1,
    skip_existing: bool = True,
) -> None:
    records = load_records(input_path)
    if limit > 0:
        records = records[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        record
        for record in records
        if not skip_existing
        or not (output_dir / f"{sample_id(record, dataset)}.json").exists()
    ]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(process_record, record, dataset): record for record in pending}
        for future in tqdm(as_completed(futures), total=len(futures), unit="sample"):
            result = future.result()
            destination = output_dir / f"{result['id']}.json"
            destination.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract document-only propositions from a supported dataset."
    )
    parser.add_argument("--dataset", choices=["hotpot", "2wiki", "musique"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_dataset(
        args.dataset,
        args.input,
        args.output,
        limit=args.limit,
        workers=args.workers,
        skip_existing=not args.overwrite,
    )


if __name__ == "__main__":
    main()
