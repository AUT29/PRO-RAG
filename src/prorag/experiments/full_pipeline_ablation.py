"""Ablations for the complete PRO-RAG pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from ..pipeline.system import ProRAG, ProRAGConfig


ABLATIONS: Dict[str, ProRAGConfig] = {
    "pnet": ProRAGConfig(name="pnet", retrievers=("pnet",)),
    "bm25": ProRAGConfig(name="bm25", retrievers=("bm25",)),
    "dense": ProRAGConfig(name="dense", retrievers=("dense",)),
    "bm25_dense": ProRAGConfig(name="bm25_dense", retrievers=("bm25", "dense")),
    "bm25_dense_pnet": ProRAGConfig(
        name="bm25_dense_pnet", retrievers=("bm25", "dense", "pnet")
    ),
    "no_decomposition": ProRAGConfig(
        name="no_decomposition",
        retrievers=("pnet",),
        use_query_decomposition=False,
    ),
    "no_summary": ProRAGConfig(
        name="no_summary",
        retrievers=("pnet",),
        use_information_summary=False,
    ),
    "no_meeting": ProRAGConfig(
        name="no_meeting",
        retrievers=("pnet",),
        use_meeting_discussion=False,
    ),
}


def run_ablation(name: str, pnet_file: str, question: str) -> Dict:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {name}")
    return ProRAG(pnet_file, config=ABLATIONS[name]).run(question)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["all", *sorted(ABLATIONS)], required=True)
    parser.add_argument("--pnet-file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", default="outputs/full_pipeline_ablation")
    args = parser.parse_args()
    output = Path(args.output)
    if args.name == "all":
        output.mkdir(parents=True, exist_ok=True)
        for name in ABLATIONS:
            result = run_ablation(name, args.pnet_file, args.question)
            (output / f"{name}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = output if output.suffix else output / f"{args.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(run_ablation(args.name, args.pnet_file, args.question), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
