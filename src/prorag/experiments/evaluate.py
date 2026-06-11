"""Common exact-match and token-F1 evaluation helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def evaluate_records(records: Iterable[Dict]) -> Dict:
    rows: List[Dict] = []
    for record in records:
        prediction = str(record.get("prediction", record.get("answer", "")))
        reference = str(record.get("ground_truth", record.get("reference", "")))
        rows.append(
            {
                **record,
                "exact_match": exact_match(prediction, reference),
                "f1": token_f1(prediction, reference),
            }
        )
    count = len(rows)
    return {
        "count": count,
        "exact_match": sum(row["exact_match"] for row in rows) / count if count else 0,
        "f1": sum(row["f1"] for row in rows) / count if count else 0,
        "results": rows,
    }
