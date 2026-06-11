"""Thread-safe LLM usage tracking for one pipeline run."""

from __future__ import annotations

import threading
from typing import Dict, List


_lock = threading.Lock()
_stats = {
    "llm_calls": 0,
    "llm_prompt_tokens": 0,
    "llm_completion_tokens": 0,
    "llm_total_tokens": 0,
    "llm_latency_seconds": 0.0,
}


def reset_usage_tracker() -> None:
    with _lock:
        for key in _stats:
            _stats[key] = 0.0 if key == "llm_latency_seconds" else 0


def _estimate_tokens_from_text(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def estimate_prompt_tokens(messages: List[Dict]) -> int:
    parts = []
    for message in messages or []:
        content = message.get("content", "")
        parts.append(content if isinstance(content, str) else str(content))
    return _estimate_tokens_from_text("\n".join(parts))


def record_llm_call(
    prompt_tokens: int,
    completion_tokens: int,
    latency_seconds: float,
    estimated: bool = False,
) -> None:
    del estimated
    prompt = max(0, int(prompt_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    with _lock:
        _stats["llm_calls"] += 1
        _stats["llm_prompt_tokens"] += prompt
        _stats["llm_completion_tokens"] += completion
        _stats["llm_total_tokens"] += prompt + completion
        _stats["llm_latency_seconds"] += max(0.0, float(latency_seconds or 0.0))


def get_usage_summary() -> Dict:
    with _lock:
        return {
            **_stats,
            "llm_latency_seconds": round(_stats["llm_latency_seconds"], 4),
        }
