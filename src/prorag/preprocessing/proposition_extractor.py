"""Document-only atomic proposition extraction."""

from __future__ import annotations

from typing import Dict

from ..llm import LLMClient


PROPOSITION_SYSTEM_PROMPT = """You extract atomic propositions from a document.

Requirements:
1. Resolve coreferences using only information present in the document.
2. Split compound statements into independently understandable atomic facts.
3. Preserve dates, numbers, entities, relations, negations, and qualifiers.
4. Cover the full document without selecting facts for any downstream question.
5. Return JSON only:
{"propositions": [{"text": "...", "entities": ["..."]}]}
"""


def extract_propositions(document: str, llm: LLMClient | None = None) -> Dict:
    """Extract propositions using only the supplied document."""
    client = llm or LLMClient()
    return client.complete_json(
        [
            {"role": "system", "content": PROPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Document:\n{document}"},
        ],
        max_tokens=3000,
    )


def extract_entities(text: str, llm: LLMClient | None = None) -> Dict:
    client = llm or LLMClient()
    return client.complete_json(
        [
            {
                "role": "system",
                "content": 'Extract named entities. Return JSON only: {"entities": ["..."]}',
            },
            {"role": "user", "content": text},
        ],
        max_tokens=500,
    )
