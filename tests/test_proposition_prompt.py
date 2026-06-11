import inspect

from prorag.preprocessing.proposition_extractor import (
    PROPOSITION_SYSTEM_PROMPT,
    extract_propositions,
)


def test_proposition_prompt_is_document_only():
    prompt = PROPOSITION_SYSTEM_PROMPT.lower()
    assert "downstream question" in prompt
    assert "auxiliary context" not in prompt


def test_proposition_extractor_does_not_accept_a_question():
    assert list(inspect.signature(extract_propositions).parameters) == ["document", "llm"]
