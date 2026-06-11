"""Command-line entry points for PRO-RAG."""

from __future__ import annotations

import argparse
import json

from .experiments.full_pipeline_ablation import ABLATIONS
from .pipeline.system import ProRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete PRO-RAG pipeline.")
    parser.add_argument("--pnet-file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--config", choices=sorted(ABLATIONS), default="pnet")
    args = parser.parse_args()
    result = ProRAG(args.pnet_file, config=ABLATIONS[args.config]).run(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
