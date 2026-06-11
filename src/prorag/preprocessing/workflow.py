"""One-command raw dataset to embedded PNet workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .add_embeddings import add_embeddings_to_pnet
from .pnet_builder import process_directory
from .prepare_dataset import prepare_dataset


def prepare_pnet(
    dataset: str,
    input_path: Path,
    data_root: Path,
    *,
    limit: int = -1,
    workers: int = 1,
    batch_size: int = 32,
    overwrite: bool = False,
) -> Path:
    proposition_dir = data_root / "propositions" / dataset
    pnet_dir = data_root / "pnet" / dataset
    prepare_dataset(
        dataset,
        input_path,
        proposition_dir,
        limit=limit,
        workers=workers,
        skip_existing=not overwrite,
    )
    process_directory(
        proposition_dir,
        pnet_dir,
        workers,
        overwrite=overwrite,
    )
    add_embeddings_to_pnet(
        pnet_dir,
        batch_size=batch_size,
        overwrite=overwrite,
    )
    return pnet_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract propositions, build PNet files, and add embeddings."
    )
    parser.add_argument("--dataset", choices=["hotpot", "2wiki", "musique"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_pnet(
        args.dataset,
        args.input,
        args.data_root,
        limit=args.limit,
        workers=args.workers,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
