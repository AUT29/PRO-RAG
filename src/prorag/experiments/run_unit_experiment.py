#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Knowledge-unit effectiveness with BM25 and dense retrieval.

Uses the same fixed sample subset as the PNet component experiment.
Input: data/units/{dataset}_{representation}/
Output: outputs/unit_experiment/{dataset}/{representation}_{retriever}_{top_k}/

72 experiments per full run (3 datasets x 4 methods x 2 retrievers x 3 top-k).
The answer model is configured through environment variables.

Resume: re-run the same command; --skip-completed skips finished experiments, and
each experiment resumes from per-sample checkpoints in all_results.json (status=running
until all samples succeed). Failed samples are retried automatically.

Top-k: when unit_count < top_k, retrieval returns all available units (effective_top_k).

Usage:
    unit-experiment
    unit-experiment --datasets hotpot --methods proposition --limit 2
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import bm25s
import numpy as np
from tqdm import tqdm

try:
    import Stemmer
except ImportError:
    Stemmer = None

from ..llm import LargeModelLLM
from ..usage import get_usage_summary, reset_usage_tracker

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "units"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "unit_experiment"
PINNED_DIR = PROJECT_ROOT / "data" / "pinned_subsets"

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SAMPLE_SEED = 42

METHOD_FOLDER = {
    "proposition": "{dataset}_proposition",
    "passages": "{dataset}_passages",
    "chunk": "{dataset}_chunk",
    "sentence": "{dataset}_sentence",
}

METHOD_ORDER = ["proposition", "passages", "chunk", "sentence"]
RETRIEVER_ORDER = ["bm25", "dense"]

TOP_K_BY_METHOD = {
    "proposition": [20, 40, 60],
    "passages": [2, 4, 6],
    "chunk": [4, 8, 12],
    "sentence": [10, 20, 30],
}


# ---------------------------------------------------------------------------
# Metrics & IO
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\bthe\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------


def build_llm_prompt(question: str, units: List[Dict]) -> List[Dict]:
    evidence_lines = []
    for idx, unit in enumerate(units, start=1):
        uid = unit.get("id", idx - 1)
        evidence_lines.append(f"{idx}. [id={uid}] {unit.get('text', '')}")
    evidence = "\n".join(evidence_lines)

    user_prompt = f"""Based on the following retrieved proposition evidence, please generate a final answer to the original question.

Original Question: {question}

Retrieved Propositions (evidence):
{evidence}

Your Task:
1. Comprehend the question fully.
2. Analyze the related propositions thoroughly and apply logical reasoning to connect them with the question.
3. Synthesize the evidence to derive the most accurate answer.
4. If information is incomplete, make reasonable inferences based on available evidence and logical deduction.

Important Guidelines:
- When facing information gaps, use logical reasoning to make reasonable inferences (indicate confidence level)
- If multiple interpretations exist, present the most likely one while acknowledging alternatives
- Your goal is to provide the BEST POSSIBLE answer given available information, not just repeat what's known

Please generate a concise answer (usually a word or phrase) without extraneous explanation or irrelevant content.
A concise answer is usually a phrase or a word.

Please use the following format (a concise answer is usually a phrase or a word):
[ANSWER]
Your final answer here
[/ANSWER]

Typical examples (for format reference only):
Example 1 (Sufficient information - short answer):
Input Question: "In which year was Tesla founded?"
Output:
[ANSWER]
2003
[/ANSWER]

Example 2 (Sufficient information - phrase answer):
Input Question: "Who painted the Mona Lisa?"
Output:
[ANSWER]
Leonardo da Vinci
[/ANSWER]

Please begin:"""

    return [
        {
            "role": "system",
            "content": (
                "You are a professional Q&A assistant. Please answer questions in English "
                "with clear and concise language."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def extract_answer_from_llm_response(response: str) -> str:
    text = (response or "").strip()
    if "[ANSWER]" in text and "[/ANSWER]" in text:
        return text.split("[ANSWER]", 1)[1].split("[/ANSWER]", 1)[0].strip()
    if "[EVIDENCE_IDS]" in text:
        text = text.split("[EVIDENCE_IDS]", 1)[0].strip()
    return text


def create_ablation_llm() -> LargeModelLLM:
    return LargeModelLLM()


def ablation_llm_extra_body() -> Dict:
    return {}


def answer_with_llm(llm: Optional[LargeModelLLM], question: str, units: List[Dict]) -> str:
    if llm is None:
        return ""
    messages = build_llm_prompt(question, units)
    extra = ablation_llm_extra_body()
    raw = llm.call_api(
        messages,
        max_tokens=800,
        temperature=0.01,
        timeout=120.0,
        show_logs=False,
        extra_body=extra or None,
    )
    return extract_answer_from_llm_response(raw)


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------


@dataclass
class UnitExperimentConfig:
    experiment_name: str
    dataset: str
    method: str
    retriever: str
    top_k: int

    def to_dict(self) -> Dict:
        return asdict(self)


def experiment_name(method: str, retriever: str, top_k: int) -> str:
    return f"{method}_{retriever}_{top_k}"


def make_experiment_configs(args) -> List[UnitExperimentConfig]:
    datasets = parse_csv(args.datasets)
    methods = parse_csv(args.methods)
    retrievers = parse_csv(args.retrievers)
    configs: List[UnitExperimentConfig] = []

    for dataset in datasets:
        for method in methods:
            if method not in TOP_K_BY_METHOD:
                raise ValueError(f"Unknown method: {method}")
            top_k_values = (
                parse_csv_ints(args.top_k_override)
                if args.top_k_override
                else TOP_K_BY_METHOD[method]
            )
            for retriever in retrievers:
                for top_k in top_k_values:
                    configs.append(
                        UnitExperimentConfig(
                            experiment_name=experiment_name(method, retriever, top_k),
                            dataset=dataset,
                            method=method,
                            retriever=retriever,
                            top_k=top_k,
                        )
                    )
    return configs


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def sort_configs(configs: List[UnitExperimentConfig]) -> List[UnitExperimentConfig]:
    return sorted(
        configs,
        key=lambda c: (
            c.dataset,
            METHOD_ORDER.index(c.method),
            RETRIEVER_ORDER.index(c.retriever),
            c.top_k,
        ),
    )


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def pinned_subset_path(dataset: str, sample_size: int, seed: int) -> Path:
    return PINNED_DIR / f"{dataset}_n{sample_size}_seed{seed}.json"


def load_pinned_sample_ids(dataset: str, sample_size: int, seed: int) -> List[str]:
    pinned = pinned_subset_path(dataset, sample_size, seed)
    data = load_json(pinned)
    if not data:
        raise FileNotFoundError(f"Pinned subset missing: {pinned}")
    return [str(sid) for sid in data.get("sample_ids", [])]


def method_output_dir(dataset: str, method: str) -> Path:
    folder = METHOD_FOLDER[method].format(dataset=dataset)
    return OUTPUT_ROOT / folder


def proposition_filename_candidates(sample_id: str) -> List[str]:
    """Preprocess copies source PNet as-is: hotpot uses *_pnet.json, 2wiki/musique use {id}.json."""
    return [f"{sample_id}.json", f"{sample_id}_pnet.json"]


def resolve_sample_file_path(dataset: str, method: str, sample_id: str) -> Optional[Path]:
    out_dir = method_output_dir(dataset, method)
    if method == "proposition":
        for name in proposition_filename_candidates(sample_id):
            path = out_dir / name
            if path.exists():
                return path
        return None
    path = out_dir / f"{sample_id}.json"
    return path if path.exists() else None


def sample_file_path(dataset: str, method: str, sample_id: str) -> Path:
    """Preferred path for error messages when file is missing."""
    resolved = resolve_sample_file_path(dataset, method, sample_id)
    if resolved is not None:
        return resolved
    out_dir = method_output_dir(dataset, method)
    if method == "proposition":
        return out_dir / proposition_filename_candidates(sample_id)[0]
    return out_dir / f"{sample_id}.json"


def load_sample_units(path: Path, method: str, *, need_embeddings: bool = True) -> Dict:
    data = load_json(path)
    if not data:
        raise ValueError(f"Empty or missing sample file: {path}")

    if method == "proposition":
        sample_id = path.name[: -len("_pnet.json")] if path.name.endswith("_pnet.json") else path.stem
        raw_nodes = data.get("nodes", [])
        if isinstance(raw_nodes, dict):
            raw_nodes = list(raw_nodes.values())
        units = [
            {"id": node.get("id", idx), "text": node.get("text", "")}
            for idx, node in enumerate(raw_nodes)
            if (node.get("text") or "").strip()
        ]
        question = data.get("question", "")
        answer = ""
    else:
        sample_id = str(data.get("sample_id", path.stem))
        units = [
            {"id": node.get("id", idx), "text": node.get("text", "")}
            for idx, node in enumerate(data.get("nodes", []))
            if (node.get("text") or "").strip()
        ]
        question = data.get("question", "")
        answer = data.get("answer", "")

    unit_embeddings: List[np.ndarray] = []
    question_embedding = None
    if need_embeddings:
        embeddings = data.get("embeddings") or {}
        node_emb_map = embeddings.get("nodes") or {}
        for unit in units:
            key = str(unit["id"])
            vec = node_emb_map.get(key)
            if vec is None:
                raise KeyError(f"Missing embedding for unit id={key} in {path}")
            unit_embeddings.append(np.asarray(vec, dtype=np.float32))

        question_embedding = embeddings.get("question")
        if question_embedding is None:
            raise KeyError(f"Missing question embedding in {path}")
        question_embedding = np.asarray(question_embedding, dtype=np.float32)

    return {
        "sample_id": sample_id,
        "file_path": str(path),
        "question": question,
        "answer": answer,
        "method": method,
        "units": units,
        "unit_count": len(units),
        "unit_embeddings": unit_embeddings,
        "question_embedding": question_embedding,
    }


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class BM25UnitRetriever:
    def __init__(self, units: List[Dict], top_k: int):
        self.units = units
        self.top_k = top_k
        self.stemmer = Stemmer.Stemmer("english") if Stemmer is not None else None
        self.documents = [unit["text"] for unit in units]
        self.index = bm25s.BM25()
        if self.documents:
            corpus_tokens = bm25s.tokenize(
                self.documents,
                stopwords="en",
                stemmer=self.stemmer,
                show_progress=False,
            )
            self.index.index(corpus_tokens, show_progress=False)

    def retrieve(self, query: str) -> List[Dict]:
        if not self.documents:
            return []
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
        k = min(self.top_k, len(self.documents))
        indices, scores = self.index.retrieve(query_tokens, k=k, show_progress=False)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            unit = self.units[int(idx)].copy()
            unit["score"] = float(score)
            unit["rank"] = rank
            results.append(unit)
        return results


class PrecomputedDenseRetriever:
    def __init__(
        self,
        units: List[Dict],
        unit_embeddings: List[np.ndarray],
        question_embedding: np.ndarray,
        top_k: int,
    ):
        self.units = units
        self.top_k = top_k
        self.embeddings = np.stack(unit_embeddings) if unit_embeddings else np.zeros((0, 0))
        self.question_embedding = question_embedding

    def retrieve(self) -> List[Dict]:
        if len(self.units) == 0:
            return []
        query_norm = max(float(np.linalg.norm(self.question_embedding)), 1e-10)
        doc_norms = np.maximum(np.linalg.norm(self.embeddings, axis=1), 1e-10)
        scores = np.dot(self.embeddings, self.question_embedding) / (doc_norms * query_norm)
        k = min(self.top_k, len(self.units))
        order = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(order, start=1):
            unit = self.units[int(idx)].copy()
            unit["score"] = float(scores[int(idx)])
            unit["rank"] = rank
            results.append(unit)
        return results


def retrieve_units(sample: Dict, retriever_name: str, top_k: int) -> List[Dict]:
    units = sample["units"]
    if retriever_name == "bm25":
        return BM25UnitRetriever(units, top_k).retrieve(sample["question"])
    if retriever_name == "dense":
        return PrecomputedDenseRetriever(
            units,
            sample["unit_embeddings"],
            sample["question_embedding"],
            top_k,
        ).retrieve()
    raise ValueError(f"Unknown retriever: {retriever_name}")


# ---------------------------------------------------------------------------
# Results I/O & resume
# ---------------------------------------------------------------------------


def experiment_dir(dataset: str, experiment_name: str) -> Path:
    return RESULTS_ROOT / dataset / experiment_name


def sample_row_complete(row: Dict, *, require_prediction: bool = True) -> bool:
    if not row.get("success"):
        return False
    if not require_prediction:
        return True
    return bool((row.get("prediction") or "").strip())


def load_checkpoint_state(
    exp_dir: Path, *, require_prediction: bool = True
) -> Tuple[Dict, Dict[str, Dict], Dict[int, Dict]]:
    """Return (checkpoint_payload, completed_by_sample_id, all_rows_by_index)."""
    data = load_json(exp_dir / "all_results.json") or {}
    completed: Dict[str, Dict] = {}
    by_index: Dict[int, Dict] = {}
    for row in data.get("results", []):
        sid = row.get("sample_id")
        idx = row.get("index")
        if idx:
            by_index[int(idx)] = row
        if sid and sample_row_complete(row, require_prediction=require_prediction):
            completed[sid] = row
    return data, completed, by_index


def load_existing_results(exp_dir: Path, *, require_prediction: bool = True) -> Dict[str, Dict]:
    _, completed, _ = load_checkpoint_state(exp_dir, require_prediction=require_prediction)
    return completed


def effective_top_k(total_units: int, requested_top_k: int) -> int:
    if total_units <= 0:
        return 0
    return min(requested_top_k, total_units)


def experiment_is_done(
    exp_dir: Path, num_expected: int, *, require_prediction: bool = True
) -> bool:
    data = load_json(exp_dir / "all_results.json")
    if not data:
        return False
    if data.get("status") != "done":
        return False
    ready = [
        r
        for r in data.get("results", [])
        if sample_row_complete(r, require_prediction=require_prediction)
    ]
    return len(ready) >= num_expected


def summarize_results(results: List[Dict], config: UnitExperimentConfig) -> Dict:
    def avg(key: str) -> float:
        vals = [float(r.get(key, 0) or 0) for r in results]
        return mean(vals) if vals else 0.0

    total_units = sum(int(r.get("total_units", 0) or 0) for r in results)
    used_units = sum(int(r.get("used_units", 0) or 0) for r in results)
    return {
        "experiment_name": config.experiment_name,
        "dataset": config.dataset,
        "method": config.method,
        "retriever": config.retriever,
        "top_k": config.top_k,
        "config": config.to_dict(),
        "num_samples": len(results),
        "avg_em": avg("em"),
        "avg_f1": avg("f1"),
        "avg_icr": avg("icr"),
        "global_icr": (total_units / used_units) if used_units else 0.0,
        "avg_latency_seconds": avg("latency_seconds"),
        "avg_llm_calls": avg("llm_calls"),
        "avg_llm_total_tokens": avg("llm_total_tokens"),
        "total_units": total_units,
        "used_units": used_units,
    }


def flush_experiment(
    exp_dir: Path,
    config: UnitExperimentConfig,
    results: List[Dict],
    *,
    num_expected: int,
    status: str,
    started_at: str,
    elapsed_seconds: Optional[float] = None,
    require_prediction: bool = True,
) -> None:
    ready = [r for r in results if sample_row_complete(r, require_prediction=require_prediction)]
    payload = {
        "timestamp": datetime.now().isoformat(),
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "status": status,
        "dataset": config.dataset,
        "experiment_name": config.experiment_name,
        "config": config.to_dict(),
        "num_samples_expected": num_expected,
        "completed_samples": len(ready),
        "results": results,
    }
    if ready:
        summary = summarize_results(ready, config)
        summary["partial"] = status != "done" or len(ready) < num_expected
        summary["completed_samples"] = len(ready)
        payload["metrics_summary"] = summary
        save_json(exp_dir / "metrics_summary.json", summary)
    save_json(exp_dir / "all_results.json", payload)


def format_result_row(
    index: int,
    sample: Dict,
    config: UnitExperimentConfig,
    *,
    prediction: str = "",
    success: bool = True,
    error: str = "",
    retrieved_units: Optional[List[Dict]] = None,
    latency_seconds: float = 0.0,
    em: int = 0,
    f1: float = 0.0,
    llm_calls: int = 0,
    llm_total_tokens: int = 0,
) -> Dict:
    retrieved = retrieved_units or []
    total_units = int(sample.get("unit_count", 0) or 0)
    used_units = len(retrieved)
    eff_k = effective_top_k(total_units, config.top_k)
    icr = total_units / used_units if used_units else 0.0
    return {
        "index": index,
        "sample_id": sample["sample_id"],
        "file_path": sample["file_path"],
        "question": sample.get("question", ""),
        "ground_truth": sample.get("answer", ""),
        "prediction": prediction,
        "success": success,
        "error": error,
        "experiment_name": config.experiment_name,
        "method": config.method,
        "retriever": config.retriever,
        "top_k": config.top_k,
        "requested_top_k": config.top_k,
        "effective_top_k": eff_k,
        "top_k_capped": eff_k < config.top_k,
        "em": em,
        "f1": round(f1, 4),
        "icr": round(icr, 4),
        "total_units": total_units,
        "used_units": used_units,
        "retrieved_unit_ids": [u.get("id") for u in retrieved],
        "latency_seconds": round(latency_seconds, 4),
        "llm_calls": llm_calls,
        "llm_total_tokens": llm_total_tokens,
        "completed_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Per-sample & experiment loop
# ---------------------------------------------------------------------------


def _log_sample_stage(
    index: int,
    sample_id: str,
    config: UnitExperimentConfig,
    stage: str,
    *,
    extra: str = "",
) -> None:
    msg = f"    [{index}] {config.experiment_name} / {sample_id}: {stage}"
    if extra:
        msg = f"{msg} ({extra})"
    print(msg, flush=True)


def run_one_sample(
    index: int,
    sample_path: Path,
    config: UnitExperimentConfig,
    truth: Dict[str, Dict],
    llm: Optional[LargeModelLLM],
    *,
    retrieval_only: bool,
) -> Dict:
    t0 = time.perf_counter()
    sid_hint = sample_path.stem
    try:
        need_embeddings = config.retriever == "dense"
        _log_sample_stage(
            index,
            sid_hint,
            config,
            "loading units",
            extra="skip embeddings" if not need_embeddings else "with embeddings",
        )
        t_load = time.perf_counter()
        sample = load_sample_units(
            sample_path, config.method, need_embeddings=need_embeddings
        )
        sid = sample["sample_id"]
        gt = truth.get(sid, {})
        if gt.get("question"):
            sample["question"] = gt["question"]
        if gt.get("answer"):
            sample["answer"] = gt["answer"]
        _log_sample_stage(
            index,
            sid,
            config,
            f"loaded {sample['unit_count']} units",
            extra=f"{time.perf_counter() - t_load:.1f}s",
        )

        reset_usage_tracker()
        _log_sample_stage(index, sid, config, f"retrieving ({config.retriever}, top_k={config.top_k})")
        t_ret = time.perf_counter()
        retrieved = retrieve_units(sample, config.retriever, config.top_k)
        _log_sample_stage(
            index,
            sid,
            config,
            f"retrieved {len(retrieved)} units",
            extra=f"{time.perf_counter() - t_ret:.1f}s",
        )

        if retrieval_only:
            prediction = ""
        else:
            _log_sample_stage(index, sid, config, "LLM answer generation")
            t_llm = time.perf_counter()
            prediction = answer_with_llm(llm, sample["question"], retrieved)
            _log_sample_stage(
                index,
                sid,
                config,
                "LLM done",
                extra=f"{time.perf_counter() - t_llm:.1f}s",
            )
        usage = get_usage_summary()

        answer = sample.get("answer", "")
        em = exact_match(prediction, answer) if answer and prediction else 0
        f1 = f1_score(prediction, answer) if answer and prediction else 0.0
        return format_result_row(
            index,
            sample,
            config,
            prediction=prediction,
            success=True,
            retrieved_units=retrieved,
            latency_seconds=time.perf_counter() - t0,
            em=em,
            f1=f1,
            llm_calls=int(usage.get("llm_calls", 0) or 0),
            llm_total_tokens=int(usage.get("llm_total_tokens", 0) or 0),
        )
    except Exception as exc:
        sid = sample_path.stem
        if config.method == "proposition" and sid.endswith("_pnet"):
            sid = sid[: -len("_pnet")]
        gt = truth.get(sid, {})
        return format_result_row(
            index,
            {
                "sample_id": sid,
                "file_path": str(sample_path),
                "question": gt.get("question", ""),
                "answer": gt.get("answer", ""),
                "unit_count": 0,
            },
            config,
            success=False,
            error=str(exc),
            latency_seconds=time.perf_counter() - t0,
        )


DATASET_CONFIGS = {
    "hotpot": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json",
        "id_field": "_id",
    },
    "2wiki": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "2wiki_dev.json",
        "id_field": "_id",
    },
    "musique": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "musique_ans_v1.0_dev.jsonl",
        "id_field": "id",
    },
}


def load_truth_records(dataset: str) -> Dict[str, Dict]:
    cfg = DATASET_CONFIGS[dataset]
    records: List[Dict] = []
    if cfg["truth_file"].suffix.lower() == ".jsonl":
        with open(cfg["truth_file"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(cfg["truth_file"], "r", encoding="utf-8") as f:
            records = json.load(f)
    truth = {}
    for rec in records:
        sid = str(rec.get(cfg["id_field"], ""))
        if sid:
            truth[sid] = {"question": rec.get("question", ""), "answer": rec.get("answer", "")}
    return truth


def run_one_experiment(
    config: UnitExperimentConfig,
    sample_ids: List[str],
    truth: Dict[str, Dict],
    args,
) -> Dict:
    exp_dir = experiment_dir(config.dataset, config.experiment_name)
    num_expected = len(sample_ids)
    require_pred = not args.retrieval_only

    checkpoint, existing, results_by_index = load_checkpoint_state(
        exp_dir, require_prediction=require_pred
    )
    started_at = checkpoint.get("started_at") or datetime.now().isoformat()
    t0 = time.perf_counter()

    pending: List[Tuple[int, Path]] = []
    for idx, sid in enumerate(sample_ids, start=1):
        if sid in existing:
            row = dict(existing[sid])
            row["index"] = idx
            results_by_index[idx] = row
            continue
        sample_path = resolve_sample_file_path(config.dataset, config.method, sid)
        if sample_path is None:
            out_dir = method_output_dir(config.dataset, config.method)
            if config.method == "proposition":
                tried = [str(out_dir / name) for name in proposition_filename_candidates(sid)]
                err = f"Sample file not found for {sid}; tried: {', '.join(tried)}"
                missing_path = tried[0]
            else:
                missing_path = str(out_dir / f"{sid}.json")
                err = f"Sample file not found: {missing_path}"
            results_by_index[idx] = format_result_row(
                idx,
                {
                    "sample_id": sid,
                    "file_path": missing_path,
                    "question": truth.get(sid, {}).get("question", ""),
                    "answer": truth.get(sid, {}).get("answer", ""),
                    "unit_count": 0,
                },
                config,
                success=False,
                error=err,
            )
            continue
        pending.append((idx, sample_path))

    print(
        f"  {config.dataset}/{config.experiment_name}: "
        f"{len(existing)} done, {len(pending)} pending / {num_expected}"
    )

    llm = None if args.retrieval_only else create_ablation_llm()
    desc = f"{config.dataset}/{config.experiment_name}"
    with tqdm(total=len(pending), desc=desc, unit="sample") as pbar:
        for idx, sample_path in pending:
            row = run_one_sample(
                idx,
                sample_path,
                config,
                truth,
                llm,
                retrieval_only=args.retrieval_only,
            )
            results_by_index[row["index"]] = row
            ordered = [results_by_index[i] for i in sorted(results_by_index)]
            flush_experiment(
                exp_dir,
                config,
                ordered,
                num_expected=num_expected,
                status="running",
                started_at=started_at,
                require_prediction=require_pred,
            )
            pbar.update(1)

    ordered = [results_by_index[i] for i in sorted(results_by_index)]
    elapsed = time.perf_counter() - t0
    flush_experiment(
        exp_dir,
        config,
        ordered,
        num_expected=num_expected,
        status="done",
        started_at=started_at,
        elapsed_seconds=elapsed,
        require_prediction=require_pred,
    )
    summary = summarize_results(
        [r for r in ordered if sample_row_complete(r, require_prediction=require_pred)],
        config,
    )
    summary["elapsed_seconds"] = round(elapsed, 2)
    return summary


def run(args) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_sample_size = args.limit if args.limit > 0 else args.sample_size
    pinned_sample_size = args.sample_size
    configs = sort_configs(make_experiment_configs(args))

    print(f"Results root: {RESULTS_ROOT}")
    print(f"Experiments: {len(configs)} | Samples/dataset: {run_sample_size}")

    aggregate = {
        "updated_at": datetime.now().isoformat(),
        "sample_size": run_sample_size,
        "sample_seed": args.sample_seed,
        "output_root": str(OUTPUT_ROOT),
        "results_root": str(RESULTS_ROOT),
        "summaries": [],
    }

    datasets = sorted({c.dataset for c in configs})
    truth_by_dataset = {ds: load_truth_records(ds) for ds in datasets}
    sample_ids_by_dataset = {
        ds: load_pinned_sample_ids(ds, pinned_sample_size, args.sample_seed)[:run_sample_size]
        for ds in datasets
    }

    for config in configs:
        exp_dir = experiment_dir(config.dataset, config.experiment_name)
        require_pred = not args.retrieval_only

        if args.skip_completed and experiment_is_done(
            exp_dir, len(sample_ids_by_dataset[config.dataset]), require_prediction=require_pred
        ):
            print(f"Skip (done): {config.dataset}/{config.experiment_name}")
            data = load_json(exp_dir / "metrics_summary.json")
            if data:
                aggregate["summaries"].append(data)
            continue

        summary = run_one_experiment(
            config,
            sample_ids_by_dataset[config.dataset],
            truth_by_dataset[config.dataset],
            args,
        )
        aggregate["summaries"].append(summary)
        print(
            f"  Finished {config.experiment_name}: "
            f"F1={summary.get('avg_f1', 0):.4f} EM={summary.get('avg_em', 0):.4f}"
        )

    agg_path = RESULTS_ROOT / "aggregate_results.json"
    save_json(agg_path, aggregate)
    return agg_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge-unit effectiveness experiments."
    )
    parser.add_argument("--datasets", default="hotpot,2wiki,musique")
    parser.add_argument("--methods", default="proposition,passages,chunk,sentence")
    parser.add_argument("--retrievers", default="bm25,dense")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--limit", type=int, default=-1, help="If >0, overrides --sample-size.")
    parser.add_argument(
        "--top-k-override",
        default="",
        help="Comma-separated top-k values; overrides per-method defaults.",
    )
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip finished experiments; unfinished ones resume per-sample (default: true).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    path = run(parse_args())
    print(f"\nSaved aggregate: {path}")

