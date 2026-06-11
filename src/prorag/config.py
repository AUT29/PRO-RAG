"""Environment-based runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class APISettings:
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    rerank_model: str = ""

    @classmethod
    def from_env(cls) -> "APISettings":
        return cls(
            api_key=_required("PRORAG_API_KEY"),
            base_url=_required("PRORAG_BASE_URL").rstrip("/"),
            chat_model=_required("PRORAG_CHAT_MODEL"),
            embedding_model=_required("PRORAG_EMBEDDING_MODEL"),
            rerank_model=os.getenv("PRORAG_RERANK_MODEL", "").strip(),
        )
