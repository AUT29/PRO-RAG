#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PNet internal ablation: simple RAG (retrieve propositions -> LLM answer) on 3 x 500 samples.

Output layout:
  outputs/pnet_ablation/<dataset>/<experiment_name>/all_results.json
  outputs/pnet_ablation/<dataset>/_beam_cache/<sample_id>.json
  outputs/pnet_ablation/<dataset>/_combination_score_cache/<sample_id>.json
  data/pinned_subsets/<dataset>_n500_seed42.json

The answer model is configured through environment variables.
Combination scores are cached per sample across all experiment configs (29 + optional pruning_m).

Resume: re-run the same command; completed experiments and finished samples are skipped.

Pruning-m ablation (optional): --experiments pruning_m
  sweeps max_combinations_per_hop m in {5,10,20,30,40,50}; other params match alpha/early_stop baseline.
  pruning_m_20 copies results from dynamic_on_topn_30 (identical config).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..llm import LargeModelLLM
from ..usage import get_usage_summary, reset_usage_tracker
from .combination_cache import CombinationScoreCache
from .pnet_ablation import (
    AblationPropositionRetriever,
    PNetAblationConfig,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "pnet_ablation"
PINNED_DIR = PROJECT_ROOT / "data" / "pinned_subsets"
BEAM_CACHE_DIRNAME = "_beam_cache"
COMBINATION_CACHE_DIRNAME = "_combination_score_cache"

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SAMPLE_SEED = 42
DEFAULT_TOP_N_VALUES = [10, 20, 30, 40, 50, 60]
DEFAULT_ALPHA_VALUES = [round(i * 0.1, 1) for i in range(11)]
DEFAULT_EARLY_STOP_VALUES = [0, 1, 2, 3, 4, 5]
DEFAULT_PRUNING_M_VALUES = [5, 10, 20, 30, 40, 50]

PAPER_INITIAL_N = 50
PAPER_ALPHA_EARLY_FINAL_N = 30
PAPER_MAX_HOPS = 6
PAPER_EXPANSION_SIZE = 20
PAPER_MAX_COMBINATIONS_PER_HOP = 20
PAPER_EARLY_STOP_ROUNDS = 2
PAPER_ALPHA = 0.5
DEFAULT_MAX_WORKERS = 1  # sequential by default (avoid LLM TPM limits)

GROUP_RUN_ORDER = ["dynamic", "alpha", "early_stop", "pruning_m"]

# Explicit copy map: target experiment -> source experiment (same retrieval+LLM behavior).
# Only these few pairs; copy all_results.json + metrics_summary.json from source folder.
EXPERIMENT_COPY_FROM: Dict[str, str] = {
    "alpha_0.5": "dynamic_on_topn_30",
    "early_stop_r_2": "dynamic_on_topn_30",
    "pruning_m_20": "dynamic_on_topn_30",
}

DATASET_CONFIGS = {
    "hotpot": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json",
        "id_field": "_id",
        "pnet_dirs": [PROJECT_ROOT / "data" / "pnet" / "hotpot"],
    },
    "2wiki": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "2wiki_dev.json",
        "id_field": "_id",
        "pnet_dirs": [PROJECT_ROOT / "data" / "pnet" / "2wiki"],
    },
    "musique": {
        "truth_file": PROJECT_ROOT / "data" / "raw" / "musique_ans_v1.0_dev.jsonl",
        "id_field": "id",
        "pnet_dirs": [PROJECT_ROOT / "data" / "pnet" / "musique"],
    },
}


# ---------------------------------------------------------------------------
# Metrics & IO helpers
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


def sample_id_from_pnet_path(path: Path) -> str:
    name = path.name
    if name.endswith("_pnet.json"):
        return name[: -len("_pnet.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_truth_records(dataset_name: str) -> Dict[str, Dict]:
    cfg = DATASET_CONFIGS[dataset_name]
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


def list_all_pnet_files(dataset_name: str) -> List[Path]:
    files: List[Path] = []
    for directory in DATASET_CONFIGS[dataset_name]["pnet_dirs"]:
        if directory.exists():
            files.extend(sorted(directory.glob("*.json")))
    return sorted(files)


def pinned_subset_path(dataset_name: str, sample_size: int, seed: int) -> Path:
    return PINNED_DIR / f"{dataset_name}_n{sample_size}_seed{seed}.json"


def load_or_create_sample_files(dataset_name: str, sample_size: int, seed: int) -> List[Path]:
    """Fixed 500-sample list shared across all experiments."""
    pinned = pinned_subset_path(dataset_name, sample_size, seed)
    if pinned.exists():
        data = load_json(pinned) or {}
        selected_ids = {str(value) for value in data.get("sample_ids", [])}
        return [
            path
            for path in list_all_pnet_files(dataset_name)
            if sample_id_from_pnet_path(path) in selected_ids
        ]

    all_files = list_all_pnet_files(dataset_name)
    if not all_files:
        raise FileNotFoundError(f"No PNet JSON files for dataset={dataset_name}")

    if sample_size <= 0 or sample_size >= len(all_files):
        chosen = all_files
    else:
        rng = random.Random(seed)
        chosen = sorted(rng.sample(all_files, sample_size), key=lambda p: p.name)

    manifest = {
        "dataset": dataset_name,
        "seed": seed,
        "sample_size": len(chosen),
        "total_available": len(all_files),
        "sample_ids": [sample_id_from_pnet_path(p) for p in chosen],
        "created_at": datetime.now().isoformat(),
    }
    PINNED_DIR.mkdir(parents=True, exist_ok=True)
    save_json(pinned, manifest)
    print(f"  Created pinned subset: {pinned} ({len(chosen)} samples)")
    return chosen


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def build_llm_prompt(question: str, propositions: List[Dict]) -> List[Dict]:
    """Build the final-answer prompt from retrieved propositions."""
    evidence_lines = []
    for idx, prop in enumerate(propositions, start=1):
        pid = prop.get("id", idx - 1)
        evidence_lines.append(f"{idx}. [id={pid}] {prop.get('text', '')}")
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


def ablation_llm_extra_body() -> Dict:
    return {}


def answer_with_llm(llm: Optional[LargeModelLLM], question: str, propositions: List[Dict]) -> str:
    if llm is None:
        return ""
    messages = build_llm_prompt(question, propositions)
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


def _log_sample_stage(
    index: int,
    sample_id: str,
    experiment_name: str,
    stage: str,
    *,
    extra: str = "",
) -> None:
    msg = f"    [{index}] {experiment_name} / {sample_id}: {stage}"
    if extra:
        msg = f"{msg} ({extra})"
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------


def make_experiment_configs(args) -> List[PNetAblationConfig]:
    configs: List[PNetAblationConfig] = []
    requested = {x.strip() for x in args.experiments.split(",") if x.strip()}

    if "dynamic" in requested:
        for top_n in args.top_n_values:
            for enabled in (True, False):
                configs.append(
                    PNetAblationConfig(
                        experiment_name=f"dynamic_{'on' if enabled else 'static'}_topn_{top_n}",
                        experiment_group="dynamic",
                        use_dynamic_update=enabled,
                        use_pruning=True,
                        use_early_stop=True,
                        initial_n=args.default_initial_n if enabled else top_n,
                        final_n=top_n,
                        top_n=top_n,
                        alpha=args.default_alpha,
                        max_hops=args.dynamic_max_hops,
                        expansion_size=args.expansion_size,
                        max_combinations_per_hop=args.max_combinations_per_hop,
                        early_stop_rounds=args.default_early_stop_rounds,
                    )
                )

    if "alpha" in requested or "pruning" in requested:
        for alpha in args.alpha_values:
            configs.append(
                PNetAblationConfig(
                    experiment_name=f"alpha_{alpha}",
                    experiment_group="alpha",
                    use_dynamic_update=True,
                    use_pruning=True,
                    use_early_stop=True,
                    initial_n=args.default_initial_n,
                    final_n=args.alpha_early_final_n,
                    top_n=args.alpha_early_final_n,
                    alpha=alpha,
                    max_hops=args.dynamic_max_hops,
                    expansion_size=args.expansion_size,
                    max_combinations_per_hop=args.max_combinations_per_hop,
                    early_stop_rounds=args.default_early_stop_rounds,
                )
            )

    if "early_stop" in requested:
        for rounds in args.early_stop_values:
            configs.append(
                PNetAblationConfig(
                    experiment_name=f"early_stop_r_{rounds}",
                    experiment_group="early_stop",
                    use_dynamic_update=True,
                    use_pruning=True,
                    use_early_stop=True,
                    initial_n=args.default_initial_n,
                    final_n=args.alpha_early_final_n,
                    top_n=args.alpha_early_final_n,
                    alpha=args.default_alpha,
                    max_hops=args.early_stop_max_hops,
                    expansion_size=args.expansion_size,
                    max_combinations_per_hop=args.max_combinations_per_hop,
                    early_stop_rounds=rounds,
                )
            )

    if "pruning_m" in requested:
        for m in args.pruning_m_values:
            configs.append(
                PNetAblationConfig(
                    experiment_name=f"pruning_m_{m}",
                    experiment_group="pruning_m",
                    use_dynamic_update=True,
                    use_pruning=True,
                    use_early_stop=True,
                    initial_n=args.default_initial_n,
                    final_n=args.alpha_early_final_n,
                    top_n=args.alpha_early_final_n,
                    alpha=args.default_alpha,
                    max_hops=args.dynamic_max_hops,
                    expansion_size=args.expansion_size,
                    max_combinations_per_hop=m,
                    early_stop_rounds=args.default_early_stop_rounds,
                )
            )

    return configs


def _config_sort_key(config: PNetAblationConfig) -> Tuple:
    order = {g: i for i, g in enumerate(GROUP_RUN_ORDER)}
    group_rank = order.get(config.experiment_group, 99)
    if config.experiment_group == "pruning_m":
        return group_rank, config.max_combinations_per_hop
    return group_rank, config.experiment_name


def sort_configs(configs: List[PNetAblationConfig]) -> List[PNetAblationConfig]:
    return sorted(configs, key=_config_sort_key)


def is_dynamic_on(config: PNetAblationConfig) -> bool:
    return config.experiment_group == "dynamic" and config.use_dynamic_update


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def experiment_dir(dataset_name: str, experiment_name: str) -> Path:
    return EXPERIMENT_ROOT / dataset_name / experiment_name


def beam_cache_dir(dataset_name: str) -> Path:
    return EXPERIMENT_ROOT / dataset_name / BEAM_CACHE_DIRNAME


def beam_cache_file(dataset_name: str, sample_id: str) -> Path:
    return beam_cache_dir(dataset_name) / f"{sample_id}.json"


def combination_cache_dir(dataset_name: str) -> Path:
    return EXPERIMENT_ROOT / dataset_name / COMBINATION_CACHE_DIRNAME


def combination_cache_file(dataset_name: str, sample_id: str) -> Path:
    return combination_cache_dir(dataset_name) / f"{sample_id}.json"


def attach_combination_score_cache(
    retriever: AblationPropositionRetriever,
    dataset_name: str,
    sample_id: str,
) -> CombinationScoreCache:
    combination_cache_dir(dataset_name).mkdir(parents=True, exist_ok=True)
    cache = CombinationScoreCache(
        combination_cache_file(dataset_name, sample_id),
        sample_id,
        target_key="question",
    )
    retriever.combination_score_cache = cache
    return cache


def flush_combination_score_cache(retriever: AblationPropositionRetriever) -> None:
    cache = getattr(retriever, "combination_score_cache", None)
    if cache is not None:
        cache.flush()


def create_ablation_llm() -> LargeModelLLM:
    return LargeModelLLM()


# ---------------------------------------------------------------------------
# Results I/O & resume
# ---------------------------------------------------------------------------


def sample_row_complete(row: Dict, *, require_prediction: bool = True) -> bool:
    if not row.get("success"):
        return False
    if not require_prediction:
        return True
    return bool((row.get("prediction") or "").strip())


def load_existing_results(exp_dir: Path, *, require_prediction: bool = True) -> Dict[str, Dict]:
    data = load_json(exp_dir / "all_results.json")
    if not data:
        return {}
    out = {}
    for row in data.get("results", []):
        sid = row.get("sample_id")
        if sid and sample_row_complete(row, require_prediction=require_prediction):
            out[sid] = row
    return out


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


def summarize_results(results: List[Dict], config: PNetAblationConfig) -> Dict:
    def avg(key: str) -> float:
        vals = [float(r.get(key, 0) or 0) for r in results]
        return mean(vals) if vals else 0.0

    total_units = sum(int(r.get("total_units", 0) or 0) for r in results)
    used_units = sum(int(r.get("used_units", 0) or 0) for r in results)
    return {
        "experiment_name": config.experiment_name,
        "experiment_group": config.experiment_group,
        "config": config.to_dict(),
        "num_samples": len(results),
        "avg_em": avg("em"),
        "avg_f1": avg("f1"),
        "avg_icr": avg("icr"),
        "avg_iteration_number": avg("iteration_number"),
        "global_icr": (total_units / used_units) if used_units else 0.0,
        "avg_latency_seconds": avg("latency_seconds"),
        "avg_llm_calls": avg("llm_calls"),
        "avg_llm_total_tokens": avg("llm_total_tokens"),
        "total_units": total_units,
        "used_units": used_units,
    }


def flush_experiment(
    exp_dir: Path,
    dataset_name: str,
    config: PNetAblationConfig,
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
        "dataset": dataset_name,
        "experiment_name": config.experiment_name,
        "experiment_group": config.experiment_group,
        "config": config.to_dict(),
        "num_samples_expected": num_expected,
        "completed_samples": len(ready),
        "results": results,
    }
    if ready:
        summary = summarize_results(ready, config)
        summary["dataset"] = dataset_name
        summary["partial"] = status != "done" or len(ready) < num_expected
        summary["completed_samples"] = len(ready)
        payload["metrics_summary"] = summary
        save_json(exp_dir / "metrics_summary.json", summary)
    save_json(exp_dir / "all_results.json", payload)


def format_result_row(
    index: int,
    pnet_path: Path,
    config: PNetAblationConfig,
    *,
    question: str,
    ground_truth: str,
    prediction: str = "",
    success: bool = True,
    error: str = "",
    total_units: int = 0,
    used_units: int = 0,
    icr: float = 0.0,
    iteration_number: int = 1,
    latency_seconds: float = 0.0,
    em: int = 0,
    f1: float = 0.0,
    llm_calls: int = 0,
    llm_total_tokens: int = 0,
    retrieval_from_cache: bool = False,
) -> Dict:
    return {
        "index": index,
        "sample_id": sample_id_from_pnet_path(pnet_path),
        "file_path": str(pnet_path),
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "success": success,
        "error": error,
        "experiment_name": config.experiment_name,
        "experiment_group": config.experiment_group,
        "em": em,
        "f1": round(f1, 4),
        "icr": round(icr, 4),
        "total_units": total_units,
        "used_units": used_units,
        "iteration_number": iteration_number,
        "latency_seconds": round(latency_seconds, 4),
        "llm_calls": llm_calls,
        "llm_total_tokens": llm_total_tokens,
        "retrieval_from_cache": retrieval_from_cache,
        "completed_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Per-sample pipelines
# ---------------------------------------------------------------------------


def load_or_run_beam_cache(
    dataset_name: str,
    pnet_path: Path,
    config: PNetAblationConfig,
    truth: Dict[str, Dict],
) -> Tuple[Dict, bool]:
    """Return beam cache dict; run beam once per sample if missing on disk."""
    sample_id = sample_id_from_pnet_path(pnet_path)
    cache_path = beam_cache_file(dataset_name, sample_id)
    cached = load_json(cache_path)
    if cached and cached.get("full_props") is not None:
        return cached, True

    gt = truth.get(sample_id, {})
    retriever = AblationPropositionRetriever(config)
    attach_combination_score_cache(retriever, dataset_name, sample_id)
    try:
        retriever.load_from_json(str(pnet_path))
        t0 = time.perf_counter()
        payload = retriever.export_beam_cache_payload()
    finally:
        flush_combination_score_cache(retriever)
    payload["sample_id"] = sample_id
    payload["file_path"] = str(pnet_path)
    payload["ground_truth"] = gt.get("answer", "")
    payload["question"] = gt.get("question") or payload.get("question") or retriever.question
    payload["retrieval_latency_seconds"] = round(time.perf_counter() - t0, 4)
    payload["cached_at"] = datetime.now().isoformat()
    save_json(cache_path, payload)
    return payload, False


def finish_from_beam_cache(
    index: int,
    dataset_name: str,
    pnet_path: Path,
    config: PNetAblationConfig,
    cache: Dict,
    *,
    retrieval_only: bool,
    from_disk_cache: bool,
) -> Dict:
    question = cache.get("question", "")
    answer = cache.get("ground_truth", "")
    full_props = cache.get("full_props") or []
    k = min(config.final_n, len(full_props))
    propositions = full_props[:k]
    t0 = time.perf_counter()

    try:
        reset_usage_tracker()
        llm = None if retrieval_only else create_ablation_llm()
        prediction = answer_with_llm(llm, question, propositions)
        usage = get_usage_summary()
        total_units = int(cache.get("total_units", 0) or 0)
        used_units = len(propositions)
        icr = total_units / used_units if used_units else 0.0
        stats = cache.get("retrieval_stats") or {}
        iteration_number = int(
            stats.get("iteration_number", stats.get("actual_hops", 1)) or 1
        )
        em = exact_match(prediction, answer) if answer and prediction else 0
        f1 = f1_score(prediction, answer) if answer and prediction else 0.0
        return format_result_row(
            index,
            pnet_path,
            config,
            question=question,
            ground_truth=answer,
            prediction=prediction,
            success=True,
            total_units=total_units,
            used_units=used_units,
            icr=icr,
            iteration_number=iteration_number,
            latency_seconds=time.perf_counter() - t0,
            em=em,
            f1=f1,
            llm_calls=int(usage.get("llm_calls", 0) or 0),
            llm_total_tokens=int(usage.get("llm_total_tokens", 0) or 0),
            retrieval_from_cache=from_disk_cache,
        )
    except Exception as exc:
        return format_result_row(
            index,
            pnet_path,
            config,
            question=question,
            ground_truth=answer,
            success=False,
            error=str(exc),
            latency_seconds=time.perf_counter() - t0,
            retrieval_from_cache=from_disk_cache,
        )


def run_standard_sample(
    index: int,
    dataset_name: str,
    pnet_path: Path,
    config: PNetAblationConfig,
    truth: Dict[str, Dict],
    *,
    retrieval_only: bool,
) -> Dict:
    sample_id = sample_id_from_pnet_path(pnet_path)
    gt = truth.get(sample_id, {})
    t0 = time.perf_counter()
    try:
        reset_usage_tracker()
        retriever = AblationPropositionRetriever(config)
        attach_combination_score_cache(retriever, dataset_name, sample_id)
        try:
            _log_sample_stage(index, sample_id, config.experiment_name, "loading PNet")
            retriever.load_from_json(str(pnet_path))
            question = gt.get("question") or retriever.question or ""
            answer = gt.get("answer", "")
            _log_sample_stage(index, sample_id, config.experiment_name, "retrieving (PNet beam/static)")
            t_ret = time.perf_counter()
            propositions, stats = retriever.iter_retrieve()
            _log_sample_stage(
                index,
                sample_id,
                config.experiment_name,
                f"retrieved {len(propositions)} props",
                extra=f"{time.perf_counter() - t_ret:.1f}s",
            )
            if retrieval_only:
                prediction = ""
            else:
                _log_sample_stage(index, sample_id, config.experiment_name, "LLM answer generation")
                t_llm = time.perf_counter()
                llm = create_ablation_llm()
                prediction = answer_with_llm(llm, question, propositions)
                _log_sample_stage(
                    index,
                    sample_id,
                    config.experiment_name,
                    "LLM done",
                    extra=f"{time.perf_counter() - t_llm:.1f}s",
                )
            usage = get_usage_summary()
            total_units = len(retriever.graph.nodes)
            used_units = len(propositions)
            icr = total_units / used_units if used_units else 0.0
            iteration_number = int(
                stats.get("iteration_number", stats.get("actual_hops", 1)) or 1
            )
            em = exact_match(prediction, answer) if answer and prediction else 0
            f1 = f1_score(prediction, answer) if answer and prediction else 0.0
            return format_result_row(
                index,
                pnet_path,
                config,
                question=question,
                ground_truth=answer,
                prediction=prediction,
                success=True,
                total_units=total_units,
                used_units=used_units,
                icr=icr,
                iteration_number=iteration_number,
                latency_seconds=time.perf_counter() - t0,
                em=em,
                f1=f1,
                llm_calls=int(usage.get("llm_calls", 0) or 0),
                llm_total_tokens=int(usage.get("llm_total_tokens", 0) or 0),
                retrieval_from_cache=False,
            )
        finally:
            flush_combination_score_cache(retriever)
    except Exception as exc:
        return format_result_row(
            index,
            pnet_path,
            config,
            question=gt.get("question", ""),
            ground_truth=gt.get("answer", ""),
            success=False,
            error=str(exc),
            latency_seconds=time.perf_counter() - t0,
        )


def run_dynamic_on_sample(
    index: int,
    dataset_name: str,
    pnet_path: Path,
    config: PNetAblationConfig,
    truth: Dict[str, Dict],
    *,
    retrieval_only: bool,
) -> Dict:
    cache, from_disk = load_or_run_beam_cache(dataset_name, pnet_path, config, truth)
    return finish_from_beam_cache(
        index,
        dataset_name,
        pnet_path,
        config,
        cache,
        retrieval_only=retrieval_only,
        from_disk_cache=from_disk,
    )


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------


def run_one_experiment(
    dataset_name: str,
    files: List[Path],
    truth: Dict[str, Dict],
    config: PNetAblationConfig,
    args,
) -> Dict:
    exp_dir = experiment_dir(dataset_name, config.experiment_name)
    num_expected = len(files)
    started_at = datetime.now().isoformat()
    t0 = time.perf_counter()

    require_pred = not args.retrieval_only
    existing = load_existing_results(exp_dir, require_prediction=require_pred)
    results_by_index: Dict[int, Dict] = {}
    for row in existing.values():
        if row.get("index"):
            results_by_index[int(row["index"])] = row

    pending = []
    for idx, pnet_path in enumerate(files, start=1):
        sid = sample_id_from_pnet_path(pnet_path)
        if sid in existing:
            results_by_index[idx] = existing[sid]
        else:
            pending.append((idx, pnet_path))

    print(
        f"  {dataset_name}/{config.experiment_name}: "
        f"{len(existing)} done, {len(pending)} pending / {num_expected}"
    )

    def process(idx: int, pnet_path: Path) -> Dict:
        if is_dynamic_on(config):
            return run_dynamic_on_sample(
                idx, dataset_name, pnet_path, config, truth, retrieval_only=args.retrieval_only
            )
        return run_standard_sample(
            idx, dataset_name, pnet_path, config, truth, retrieval_only=args.retrieval_only
        )

    if pending:
        desc = f"{dataset_name}/{config.experiment_name}"
        with tqdm(total=len(pending), desc=desc, unit="sample") as pbar:
            if args.max_workers <= 1:
                for idx, pnet_path in pending:
                    row = process(idx, pnet_path)
                    results_by_index[row["index"]] = row
                    ordered = [results_by_index[i] for i in sorted(results_by_index)]
                    flush_experiment(
                        exp_dir,
                        dataset_name,
                        config,
                        ordered,
                        num_expected=num_expected,
                        status="running",
                        started_at=started_at,
                        require_prediction=require_pred,
                    )
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                    futures = {pool.submit(process, idx, p): (idx, p) for idx, p in pending}
                    for fut in as_completed(futures):
                        row = fut.result()
                        results_by_index[row["index"]] = row
                        ordered = [results_by_index[i] for i in sorted(results_by_index)]
                        flush_experiment(
                            exp_dir,
                            dataset_name,
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
        dataset_name,
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
    summary["dataset"] = dataset_name
    summary["elapsed_seconds"] = round(elapsed, 2)
    return summary


def copy_experiment_results(
    dataset_name: str,
    target_name: str,
    source_name: str,
    config: PNetAblationConfig,
) -> Dict:
    """Copy result files from source experiment folder; patch experiment name in JSON."""
    src_dir = experiment_dir(dataset_name, source_name)
    dst_dir = experiment_dir(dataset_name, target_name)
    src_all = src_dir / "all_results.json"
    if not src_all.exists():
        raise FileNotFoundError(
            f"Cannot copy to {target_name}: source missing {src_all}"
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("all_results.json", "metrics_summary.json"):
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)

    all_data = load_json(dst_dir / "all_results.json") or {}
    all_data["experiment_name"] = target_name
    all_data["experiment_group"] = config.experiment_group
    all_data["reused_from"] = source_name
    all_data["status"] = "done"
    for row in all_data.get("results", []):
        row["experiment_name"] = target_name
        row["experiment_group"] = config.experiment_group
        row["reused_from"] = source_name
    save_json(dst_dir / "all_results.json", all_data)

    summary = load_json(dst_dir / "metrics_summary.json")
    if summary:
        summary["experiment_name"] = target_name
        summary["experiment_group"] = config.experiment_group
        summary["dataset"] = dataset_name
        summary["reused_from"] = source_name
        summary["partial"] = False
        save_json(dst_dir / "metrics_summary.json", summary)
    else:
        ready = [r for r in all_data.get("results", []) if sample_row_complete(r)]
        summary = summarize_results(ready, config)
        summary["dataset"] = dataset_name
        summary["reused_from"] = source_name
        save_json(dst_dir / "metrics_summary.json", summary)
    return summary


def run(args) -> Path:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_size = args.limit if args.limit > 0 else args.sample_size
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    configs = sort_configs(make_experiment_configs(args))

    if EXPERIMENT_COPY_FROM:
        print("\n【实验结果复制（显式列表）】")
        for target, source in EXPERIMENT_COPY_FROM.items():
            print(f"  {target} <- {source}")

    print(f"Output root: {EXPERIMENT_ROOT}")
    print(f"Configs: {len(configs)} | Datasets: {len(datasets)} | Samples/dataset: {sample_size}")

    aggregate = {
        "updated_at": datetime.now().isoformat(),
        "sample_size": sample_size,
        "sample_seed": args.sample_seed,
        "summaries": [],
    }

    for dataset_name in datasets:
        files = load_or_create_sample_files(dataset_name, sample_size, args.sample_seed)
        truth = load_truth_records(dataset_name)
        beam_cache_dir(dataset_name).mkdir(parents=True, exist_ok=True)
        print(f"\n=== {dataset_name}: {len(files)} samples ===")

        for config in configs:
            exp_dir = experiment_dir(dataset_name, config.experiment_name)
            require_pred = not args.retrieval_only

            if args.skip_completed and experiment_is_done(
                exp_dir, len(files), require_prediction=require_pred
            ):
                print(f"  Skip (done): {config.experiment_name}")
                data = load_json(exp_dir / "metrics_summary.json")
                if data:
                    aggregate["summaries"].append(data)
                continue

            copy_from = EXPERIMENT_COPY_FROM.get(config.experiment_name)
            if copy_from:
                print(f"  Copy folder {config.experiment_name} <- {copy_from}")
                summary = copy_experiment_results(
                    dataset_name, config.experiment_name, copy_from, config
                )
                aggregate["summaries"].append(summary)
                print(
                    f"  Copied {config.experiment_name}: "
                    f"F1={summary.get('avg_f1', 0):.4f} EM={summary.get('avg_em', 0):.4f}"
                )
                continue

            summary = run_one_experiment(dataset_name, files, truth, config, args)
            aggregate["summaries"].append(summary)
            print(
                f"  Finished {config.experiment_name}: "
                f"F1={summary.get('avg_f1', 0):.4f} EM={summary.get('avg_em', 0):.4f}"
            )

    agg_path = EXPERIMENT_ROOT / "aggregate_results.json"
    save_json(agg_path, aggregate)
    return agg_path


def parse_args():
    parser = argparse.ArgumentParser(description="PNet internal ablation (simple RAG, fixed output dirs).")
    parser.add_argument("--datasets", default="hotpot,2wiki,musique")
    parser.add_argument(
        "--experiments",
        default="dynamic,alpha,early_stop",
        help="Comma-separated groups: dynamic, alpha, early_stop, pruning_m "
        "(pruning_m = beam combination cap m ablation).",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--limit", type=int, default=-1, help="If >0, overrides --sample-size.")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--top-n-values", type=parse_csv_ints, default=DEFAULT_TOP_N_VALUES)
    parser.add_argument("--alpha-values", type=parse_csv_floats, default=DEFAULT_ALPHA_VALUES)
    parser.add_argument("--early-stop-values", type=parse_csv_ints, default=DEFAULT_EARLY_STOP_VALUES)
    parser.add_argument(
        "--pruning-m-values",
        type=parse_csv_ints,
        default=DEFAULT_PRUNING_M_VALUES,
        help="max_combinations_per_hop sweep for --experiments pruning_m.",
    )
    parser.add_argument("--default-initial-n", type=int, default=PAPER_INITIAL_N)
    parser.add_argument("--alpha-early-final-n", type=int, default=PAPER_ALPHA_EARLY_FINAL_N)
    parser.add_argument("--default-alpha", type=float, default=PAPER_ALPHA)
    parser.add_argument("--default-early-stop-rounds", type=int, default=PAPER_EARLY_STOP_ROUNDS)
    parser.add_argument("--dynamic-max-hops", type=int, default=PAPER_MAX_HOPS)
    parser.add_argument("--early-stop-max-hops", type=int, default=PAPER_MAX_HOPS)
    parser.add_argument("--expansion-size", type=int, default=PAPER_EXPANSION_SIZE)
    parser.add_argument("--max-combinations-per-hop", type=int, default=PAPER_MAX_COMBINATIONS_PER_HOP)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Sample parallelism (default 1 = sequential, for TPM limits).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Alias for --max-workers 1.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip experiments whose all_results.json is already done (default: true).",
    )
    args = parser.parse_args()
    if args.sequential:
        args.max_workers = 1
    return args


if __name__ == "__main__":
    path = run(parse_args())
    print(f"\nSaved aggregate: {path}")

