from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping

from selection_methods import (
    KMEANS_SELECTOR_N_CLUSTER,
    MEAN_KL_SELECTOR_NAME,
    UNCERTAINTY_BAND_SELECTOR_NAME,
    PooledCandidate,
    SelectionContext,
    list_methods,
    load_jsonl,
    normalize_candidate_rows,
    safe_hf_cache_dir,
    select_by_kmeans_score_indices,
    score_pooled_candidates_by_mean_kl,
    score_pooled_candidates_by_uncertainty_band,
    select_rows,
    write_jsonl,
)
from ssr_metrics import checkpoint_dir, infer_tasks, load_rouge_l


TRAIN_COLUMNS = {
    "prompt": "full_prompt",
    "query": "",
    "response": "output",
    "history": "",
}
REHEARSAL_COLUMNS = {
    "prompt": "inputs",
    "query": "",
    "response": "targets",
    "history": "",
}
ARTIFACT_KEY_MAP = {
    "raw": "raw_generation_file",
    "parsed": "parsed_generation_file",
    "refined": "refined_generation_file",
    "final": "rehearsal_file",
}
POOLED_SELECTORS = {MEAN_KL_SELECTOR_NAME, UNCERTAINTY_BAND_SELECTOR_NAME}
POOLED_SELECTION_MODES = ("global", "per_task_top_ratio", "per_task_top_count", "per_cluster")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def selector_output_name(selector: str, pooled_selection_mode: str) -> str:
    if selector not in POOLED_SELECTORS or pooled_selection_mode == "global":
        return selector
    return f"{selector}-{pooled_selection_mode}"


def repo_env(cuda: str) -> Dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = cuda
    env["LOCAL_RANK"] = "-1"
    env.pop("RANK", None)
    env.pop("WORLD_SIZE", None)
    safe_cache_root = safe_hf_cache_dir()
    hub_cache = os.path.join(safe_cache_root, "hub")
    transformers_cache = os.path.join(safe_cache_root, "transformers")
    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(transformers_cache, exist_ok=True)
    env["SMART_SSR_HF_CACHE"] = safe_cache_root
    env["HF_HOME"] = safe_cache_root
    env["HF_HUB_CACHE"] = hub_cache
    env["HUGGINGFACE_HUB_CACHE"] = hub_cache
    env["TRANSFORMERS_CACHE"] = transformers_cache
    return env


def run_command(command: List[str], env: Dict[str, str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def load_run_summary(base_run: Path) -> Dict[str, Dict[str, List[str]]]:
    summary_path = base_run / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Base run is missing {summary_path}. A paper/basic SSR run summary is required."
        )
    with summary_path.open("r") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected run summary format in {summary_path}.")
    return data


def resolve_target_index(tasks: List[str], iteration_task: str | None, iteration_index: int | None) -> int:
    if (iteration_task is None) == (iteration_index is None):
        raise ValueError("Specify exactly one of --iteration_task or --iteration_index.")

    if iteration_task is not None:
        if iteration_task not in tasks:
            raise ValueError(
                f"Unknown iteration task `{iteration_task}`. Available tasks: {', '.join(tasks)}"
            )
        return tasks.index(iteration_task)

    assert iteration_index is not None
    zero_based_index = iteration_index - 1
    if zero_based_index < 0 or zero_based_index >= len(tasks):
        raise ValueError(
            f"Iteration index {iteration_index} is out of range for tasks {tasks}."
        )
    return zero_based_index


def resolve_artifact_paths(
    run_summary: Mapping[str, Dict[str, List[str]]],
    task_name: str,
) -> Dict[str, Path]:
    if task_name not in run_summary:
        raise ValueError(f"Task `{task_name}` is missing from base run summary.")

    task_summary = run_summary[task_name]
    paths: Dict[str, Path] = {}
    for source_name, summary_key in ARTIFACT_KEY_MAP.items():
        values = task_summary.get(summary_key)
        if values:
            paths[source_name] = Path(values[0]).resolve()
    return paths


def make_dataset_entry(file_name: str, columns: Dict[str, str]) -> Dict[str, object]:
    return {
        "file_name": file_name,
        "columns": columns,
    }


def write_dataset_registry(
    dataset_dir: Path,
    tasks: List[str],
    rehearsal_files: Mapping[str, Path],
) -> Dict[str, str]:
    split_root = (repo_root() / "data" / "ni-cus0.12" / "split").resolve()
    dataset_info: Dict[str, Dict[str, object]] = {}
    rehearsal_keys: Dict[str, str] = {}

    for task_name in tasks:
        dataset_info[f"ni_c012_{task_name}_train"] = make_dataset_entry(
            str((split_root / f"{task_name}.train.json").resolve()),
            TRAIN_COLUMNS,
        )
        dataset_info[f"ni_c012_{task_name}_eval"] = make_dataset_entry(
            str((split_root / f"{task_name}.eval.json").resolve()),
            TRAIN_COLUMNS,
        )

    for source_task, rehearsal_path in rehearsal_files.items():
        dataset_key = f"proxy_rehearsal_{source_task}"
        rehearsal_keys[source_task] = dataset_key
        dataset_info[dataset_key] = make_dataset_entry(
            str(rehearsal_path.resolve()),
            REHEARSAL_COLUMNS,
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))
    return rehearsal_keys


def run_train(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    dataset_dir: Path,
    datasets: List[str],
    checkpoint: Path | None,
    output_dir: Path,
    max_source_length: int,
    max_target_length: int,
    num_train_epochs: float,
    max_samples: int | None,
    train_max_steps: int | None,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    command = [
        python_bin,
        "src/train_bash.py",
        "--stage", "sft",
        "--model_name_or_path", model_name_or_path,
        "--do_train", "True",
        "--overwrite_cache", "True",
        "--overwrite_output_dir", "True",
        "--finetuning_type", "lora",
        "--template", template,
        "--dataset_dir", str(dataset_dir),
        "--dataset", ",".join(datasets),
        "--max_source_length", str(max_source_length),
        "--max_target_length", str(max_target_length),
        "--learning_rate", "2e-4",
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--lr_scheduler_type", "cosine",
        "--max_grad_norm", "1.0",
        "--logging_steps", "5",
        "--save_strategy", "no",
        "--warmup_steps", "0",
        "--lora_rank", "8",
        "--lora_dropout", "0.1",
        "--lora_target", "q_proj,v_proj",
        "--resume_lora_training", "True",
        "--output_dir", str(output_dir),
        "--plot_loss", "True",
        "--bf16", "True",
    ]
    if checkpoint is not None and checkpoint.exists():
        command.extend(["--checkpoint_dir", str(checkpoint)])
    if train_max_steps is not None:
        command.extend(["--max_steps", str(train_max_steps)])
    else:
        command.extend(["--num_train_epochs", str(num_train_epochs)])
    if max_samples is not None:
        command.extend(["--max_samples", str(max_samples)])
    run_command(command, env)


def run_eval(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    dataset_dir: Path,
    checkpoint_dir_path: Path,
    eval_task: str,
    max_source_length: int,
    max_target_length: int,
    max_samples: int | None,
) -> None:
    output_dir = checkpoint_dir_path / f"ni_c012_{eval_task}_eval"
    command = [
        python_bin,
        "src/train_bash.py",
        "--stage", "sftrp",
        "--model_name_or_path", model_name_or_path,
        "--checkpoint_dir", str(checkpoint_dir_path),
        "--overwrite_cache", "True",
        "--overwrite_output_dir", "True",
        "--predict_with_generate", "True",
        "--finetuning_type", "lora",
        "--template", template,
        "--dataset_dir", str(dataset_dir),
        "--dataset", f"ni_c012_{eval_task}_eval",
        "--max_source_length", str(max_source_length),
        "--max_target_length", str(max_target_length),
        "--per_device_eval_batch_size", "1",
        "--output_dir", str(output_dir),
        "--do_predict", "True",
        "--do_sample", "False",
        "--bf16", "True",
    ]
    if max_samples is not None:
        command.extend(["--max_samples", str(max_samples)])
    run_command(command, env)


def load_stage_scores(stage_dir: Path, eval_tasks: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for eval_task in eval_tasks:
        result_path = stage_dir / f"ni_c012_{eval_task}_eval" / "all_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing evaluation result: {result_path}")
        scores[eval_task] = load_rouge_l(result_path)
    return scores


def average_score(scores: Mapping[str, float]) -> float | None:
    if not scores:
        return None
    return sum(scores.values()) / len(scores)


def has_stage_scores(stage_dir: Path, eval_tasks: List[str]) -> bool:
    return all(
        (stage_dir / f"ni_c012_{eval_task}_eval" / "all_results.json").exists()
        for eval_task in eval_tasks
    )


def available_selectors() -> List[str]:
    return sorted(set(list_methods()) | POOLED_SELECTORS)


def resolve_pooled_selection_mode(
    selector: str,
    mean_kl_selection_mode: str,
    uncertainty_selection_mode: str,
) -> str:
    if selector == MEAN_KL_SELECTOR_NAME:
        return mean_kl_selection_mode
    if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
        return uncertainty_selection_mode
    return "global"


def pooled_selection_mode_flag(selector: str) -> str:
    if selector == MEAN_KL_SELECTOR_NAME:
        return "--mean_kl_selection_mode"
    if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
        return "--uncertainty_selection_mode"
    raise ValueError(f"Selector `{selector}` is not a pooled selector.")


def pooled_selection_mode_field(selector: str) -> str:
    if selector == MEAN_KL_SELECTOR_NAME:
        return "mean_kl_selection_mode"
    if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
        return "uncertainty_selection_mode"
    raise ValueError(f"Selector `{selector}` is not a pooled selector.")


def pooled_score_field(selector: str) -> str:
    if selector == MEAN_KL_SELECTOR_NAME:
        return "mean_kl"
    if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
        return "uncertainty_band_score"
    raise ValueError(f"Selector `{selector}` is not a pooled selector.")


def resolve_global_selection_budget(
    num_candidates: int,
    num_previous_tasks: int,
    sample_memory: int,
    global_selection_count: int | None,
    global_selection_ratio: float | None,
) -> int:
    if global_selection_count is not None and global_selection_ratio is not None:
        raise ValueError(
            "Specify at most one of --global_selection_count or --global_selection_ratio."
        )

    if global_selection_ratio is not None:
        if not 0.0 <= global_selection_ratio <= 1.0:
            raise ValueError("--global_selection_ratio must be between 0.0 and 1.0.")
        budget = math.ceil(num_candidates * global_selection_ratio)
    elif global_selection_count is not None:
        if global_selection_count < 0:
            raise ValueError("--global_selection_count must be non-negative.")
        budget = global_selection_count
    else:
        budget = sample_memory * num_previous_tasks

    return min(num_candidates, budget)


def validate_selection_ratio(name: str, ratio: float | None) -> float:
    if ratio is None:
        raise ValueError(f"{name} must be provided.")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return ratio


def resolve_per_task_selection_budgets(
    candidate_counts: Mapping[str, int],
    selection_ratio: float,
) -> Dict[str, int]:
    return {
        source_task: min(count, math.ceil(count * selection_ratio))
        for source_task, count in candidate_counts.items()
    }


def resolve_per_task_selection_budgets_by_count(
    candidate_counts: Mapping[str, int],
    selection_count: int,
) -> Dict[str, int]:
    if selection_count < 0:
        raise ValueError("--per_task_selection_count must be non-negative.")
    return {
        source_task: min(count, selection_count)
        for source_task, count in candidate_counts.items()
    }


def run_pooled_selection(
    base_run: Path,
    selector: str,
    model_name_or_path: str,
    template: str,
    selection_dir: Path,
    output_root: Path,
    candidate_source: str,
    previous_tasks: List[str],
    tasks: List[str],
    history_checkpoints: List[Path],
    previous_checkpoint: Path,
    run_summary: Mapping[str, Dict[str, List[str]]],
    sample_memory: int,
    seed: int,
    encoder_model_name_or_path: str,
    global_selection_count: int | None,
    global_selection_ratio: float | None,
    pooled_selection_mode: str,
    per_task_selection_ratio: float | None,
    per_task_selection_count: int | None,
    max_source_length: int,
    max_target_length: int,
    kl_batch_size: int,
    uncertainty_batch_size: int,
    h_min: float,
    h_max: float,
) -> tuple[Dict[str, Path], Dict[str, Dict[str, object]], Dict[str, object]]:
    pooled_candidates: List[PooledCandidate] = []
    candidate_paths: Dict[str, Path] = {}
    candidate_counts: Dict[str, int] = {}
    candidate_rows_by_task: Dict[str, List[Dict[str, str]]] = {}

    for source_task in previous_tasks:
        source_task_index = tasks.index(source_task)
        source_paths = resolve_artifact_paths(run_summary, source_task)
        if candidate_source not in source_paths:
            raise ValueError(
                f"Task `{source_task}` in {base_run / 'run_summary.json'} does not expose "
                f"a `{candidate_source}` artifact."
            )

        candidate_path = source_paths[candidate_source]
        candidate_raw_rows = load_jsonl(candidate_path)
        candidate_rows = normalize_candidate_rows(candidate_raw_rows)
        candidate_paths[source_task] = candidate_path
        candidate_counts[source_task] = len(candidate_rows)
        candidate_rows_by_task[source_task] = candidate_rows

        for candidate_index, (row, raw_row) in enumerate(zip(candidate_rows, candidate_raw_rows)):
            pooled_candidates.append(
                PooledCandidate(
                    source_task=source_task,
                    source_task_index=source_task_index,
                    source_checkpoint=checkpoint_dir(base_run, source_task_index, source_task).resolve(),
                    candidate_index=candidate_index,
                    row=row,
                    raw_row=raw_row,
                )
            )

    selection_ratio_applied: float | None = None
    selection_count_applied: int | None = None
    per_task_budgets: Dict[str, int] | None = None
    if pooled_selection_mode == "global":
        budget = resolve_global_selection_budget(
            num_candidates=len(pooled_candidates),
            num_previous_tasks=len(previous_tasks),
            sample_memory=sample_memory,
            global_selection_count=global_selection_count,
            global_selection_ratio=global_selection_ratio,
        )
    elif pooled_selection_mode == "per_task_top_ratio":
        if global_selection_count is not None or global_selection_ratio is not None:
            raise ValueError(
                "--global_selection_count/--global_selection_ratio cannot be used with "
                f"{pooled_selection_mode_flag(selector)} per_task_top_ratio."
            )
        selection_ratio_applied = validate_selection_ratio(
            "--per_task_selection_ratio",
            per_task_selection_ratio,
        )
        per_task_budgets = resolve_per_task_selection_budgets(
            candidate_counts=candidate_counts,
            selection_ratio=selection_ratio_applied,
        )
        budget = sum(per_task_budgets.values())
    elif pooled_selection_mode == "per_task_top_count":
        if (
            global_selection_count is not None
            or global_selection_ratio is not None
            or per_task_selection_ratio is not None
        ):
            raise ValueError(
                "--global_selection_count/--global_selection_ratio/--per_task_selection_ratio "
                f"cannot be used with {pooled_selection_mode_flag(selector)} per_task_top_count."
            )
        if per_task_selection_count is None:
            raise ValueError("--per_task_selection_count must be provided.")
        selection_count_applied = per_task_selection_count
        per_task_budgets = resolve_per_task_selection_budgets_by_count(
            candidate_counts=candidate_counts,
            selection_count=selection_count_applied,
        )
        budget = sum(per_task_budgets.values())
    elif pooled_selection_mode == "per_cluster":
        if (
            global_selection_count is not None
            or global_selection_ratio is not None
            or per_task_selection_ratio is not None
            or per_task_selection_count is not None
        ):
            raise ValueError(
                "--global_selection_count/--global_selection_ratio/--per_task_selection_ratio/"
                "--per_task_selection_count cannot be used with "
                f"{pooled_selection_mode_flag(selector)} per_cluster."
            )
        per_task_budgets = {
            source_task: min(count, sample_memory)
            for source_task, count in candidate_counts.items()
        }
        budget = sum(per_task_budgets.values())
    else:
        raise ValueError(
            f"Unknown pooled selection mode `{pooled_selection_mode}` for selector `{selector}`. "
            f"Available modes: {', '.join(POOLED_SELECTION_MODES)}"
        )

    score_field = pooled_score_field(selector)
    uncertainty_band_by_task: Dict[str, Dict[str, float]] = {}
    scoring_metadata: Dict[str, object]
    if selector == MEAN_KL_SELECTOR_NAME:
        scored_candidates = score_pooled_candidates_by_mean_kl(
            model_name_or_path=model_name_or_path,
            current_checkpoint=previous_checkpoint,
            pooled_candidates=pooled_candidates,
            template_name=template,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            batch_size=kl_batch_size,
        )
        scoring_metadata = {
            "kl_batch_size": kl_batch_size,
        }
    elif selector == UNCERTAINTY_BAND_SELECTOR_NAME:
        uncertainty_result = score_pooled_candidates_by_uncertainty_band(
            model_name_or_path=model_name_or_path,
            current_checkpoint=previous_checkpoint,
            pooled_candidates=pooled_candidates,
            template_name=template,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            selection_mode=pooled_selection_mode,
            h_min=h_min,
            h_max=h_max,
            batch_size=uncertainty_batch_size,
        )
        scored_candidates = uncertainty_result.scored_candidates
        uncertainty_band_by_task = {
            source_task: dict(task_metadata)
            for source_task, task_metadata in uncertainty_result.band_by_source_task.items()
        }
        scoring_metadata = {
            "uncertainty_batch_size": uncertainty_batch_size,
            "h_min": h_min,
            "h_max": h_max,
            "mean_entropy_min": uncertainty_result.entropy_min,
            "mean_entropy_max": uncertainty_result.entropy_max,
            "entropy_band_min": uncertainty_result.band_min_entropy,
            "entropy_band_max": uncertainty_result.band_max_entropy,
            "entropy_bands_by_task": uncertainty_band_by_task,
        }
    else:
        raise ValueError(f"Selector `{selector}` is not a pooled selector.")

    ranked_score_path = output_root / f"{selector}_scores.{candidate_source}.jsonl"
    ranked_rows: List[Dict[str, object]] = []
    all_scores_by_task: Dict[str, List[float]] = defaultdict(list)
    selected_scores_by_task: Dict[str, List[float]] = defaultdict(list)
    selected_entropies_by_task: Dict[str, List[float]] = defaultdict(list)
    selected_rows_by_task: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    selected_counts_by_task: Dict[str, int] = defaultdict(int)
    cluster_selection_by_task: Dict[str, Dict[str, object]] = {}
    selected_candidate_keys: set[tuple[str, int]] = set()

    if pooled_selection_mode == "per_cluster":
        scored_candidates_by_task: Dict[str, Dict[int, object]] = defaultdict(dict)
        for scored_candidate in scored_candidates:
            candidate = scored_candidate.candidate
            scored_candidates_by_task[candidate.source_task][candidate.candidate_index] = scored_candidate

        for source_task in previous_tasks:
            task_rows = candidate_rows_by_task[source_task]
            task_scored_candidates = scored_candidates_by_task[source_task]
            if len(task_scored_candidates) != len(task_rows):
                raise ValueError(
                    f"Per-cluster selection expected one score per candidate for task `{source_task}`."
                )

            task_scores = [
                float(task_scored_candidates[candidate_index].score)
                for candidate_index in range(len(task_rows))
            ]
            selected_indices, cluster_state = select_by_kmeans_score_indices(
                rows=task_rows,
                ranking_scores=task_scores,
                encoder_model_name_or_path=encoder_model_name_or_path,
                sample_size=sample_memory,
                seed=seed,
                n_cluster=KMEANS_SELECTOR_N_CLUSTER,
                descending=True,
            )
            selected_rows_by_task[source_task] = [task_rows[idx] for idx in selected_indices]
            selected_scores_by_task[source_task] = [task_scores[idx] for idx in selected_indices]
            selected_counts_by_task[source_task] = len(selected_indices)
            selected_candidate_keys.update((source_task, idx) for idx in selected_indices)
            cluster_selection_by_task[source_task] = {
                "kmeans_clustered": cluster_state is not None,
                "kmeans_n_cluster": min(KMEANS_SELECTOR_N_CLUSTER, len(task_rows)),
                "kmeans_cluster_counts": (
                    None
                    if cluster_state is None
                    else [int(count) for count in cluster_state.counts.tolist()]
                ),
                "kmeans_cluster_allocations": (
                    None
                    if cluster_state is None
                    else [int(allocation) for allocation in cluster_state.allocations.tolist()]
                ),
            }

            if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
                selected_entropies: List[float] = []
                for idx in selected_indices:
                    mean_entropy = task_scored_candidates[idx].mean_entropy
                    if mean_entropy is None:
                        raise ValueError("Uncertainty scoring must populate mean entropy.")
                    selected_entropies.append(mean_entropy)
                selected_entropies_by_task[source_task] = selected_entropies

    for rank, scored_candidate in enumerate(scored_candidates, start=1):
        candidate = scored_candidate.candidate
        score = scored_candidate.score
        all_scores_by_task[candidate.source_task].append(score)
        ranked_row = {
            "rank": rank,
            "source_task": candidate.source_task,
            "source_task_index": candidate.source_task_index,
            "source_checkpoint": str(candidate.source_checkpoint),
            "candidate_index": candidate.candidate_index,
            score_field: score,
            "inputs": candidate.row["inputs"],
            "targets": candidate.row["targets"],
        }
        if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
            if scored_candidate.mean_entropy is None:
                raise ValueError("Uncertainty scoring must populate mean entropy.")
            ranked_row["mean_entropy"] = scored_candidate.mean_entropy
            task_band = uncertainty_band_by_task.get(candidate.source_task)
            if task_band is not None:
                ranked_row["entropy_band_min"] = task_band["band_min_entropy"]
                ranked_row["entropy_band_max"] = task_band["band_max_entropy"]
        if pooled_selection_mode == "per_cluster":
            ranked_row["selected"] = (
                candidate.source_task,
                candidate.candidate_index,
            ) in selected_candidate_keys
            ranked_rows.append(ranked_row)
            continue
        ranked_rows.append(ranked_row)
        should_select = False
        if pooled_selection_mode == "global":
            should_select = rank <= budget
        else:
            assert per_task_budgets is not None
            should_select = selected_counts_by_task[candidate.source_task] < per_task_budgets[candidate.source_task]

        if should_select:
            selected_rows_by_task[candidate.source_task].append(candidate.row)
            selected_scores_by_task[candidate.source_task].append(score)
            selected_counts_by_task[candidate.source_task] += 1
            if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
                assert scored_candidate.mean_entropy is not None
                selected_entropies_by_task[candidate.source_task].append(scored_candidate.mean_entropy)

    write_jsonl(ranked_score_path, ranked_rows)

    rehearsal_files: Dict[str, Path] = {}
    selection_metadata: Dict[str, Dict[str, object]] = {}
    for source_task in previous_tasks:
        selected_rows = selected_rows_by_task.get(source_task, [])
        selected_path: str | None = None
        if selected_rows:
            selection_path = selection_dir / f"{source_task}.{selector}.{candidate_source}.jsonl"
            write_jsonl(selection_path, selected_rows)
            rehearsal_files[source_task] = selection_path
            selected_path = str(selection_path)

        source_scores = all_scores_by_task.get(source_task, [])
        chosen_scores = selected_scores_by_task.get(source_task, [])
        selection_metadata[source_task] = {
            "candidate_path": str(candidate_paths[source_task]),
            "selected_path": selected_path,
            "candidate_source": candidate_source,
            "history_checkpoints": [str(path) for path in history_checkpoints],
            "selector": selector,
            pooled_selection_mode_field(selector): pooled_selection_mode,
            "current_checkpoint": str(previous_checkpoint),
            "num_candidates": candidate_counts[source_task],
            "selected_count": len(selected_rows),
            "selection_ratio": selection_ratio_applied,
            "selection_count": selection_count_applied,
            "selection_budget": (
                None
                if per_task_budgets is None
                else per_task_budgets[source_task]
            ),
            f"{score_field}_min": min(source_scores) if source_scores else None,
            f"{score_field}_max": max(source_scores) if source_scores else None,
            f"selected_{score_field}_min": min(chosen_scores) if chosen_scores else None,
            f"selected_{score_field}_max": max(chosen_scores) if chosen_scores else None,
        }
        if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
            task_band = uncertainty_band_by_task.get(source_task, {})
            chosen_entropies = selected_entropies_by_task.get(source_task, [])
            selection_metadata[source_task].update(
                {
                    "h_min": h_min,
                    "h_max": h_max,
                    "mean_entropy_min": task_band.get("entropy_min"),
                    "mean_entropy_max": task_band.get("entropy_max"),
                    "selected_mean_entropy_min": (
                        min(chosen_entropies) if chosen_entropies else None
                    ),
                    "selected_mean_entropy_max": (
                        max(chosen_entropies) if chosen_entropies else None
                    ),
                    "entropy_band_min": task_band.get("band_min_entropy"),
                    "entropy_band_max": task_band.get("band_max_entropy"),
                }
            )
        if pooled_selection_mode == "per_cluster":
            selection_metadata[source_task].update(cluster_selection_by_task.get(source_task, {}))

    selection_pool_metadata = {
        "selector": selector,
        "candidate_source": candidate_source,
        pooled_selection_mode_field(selector): pooled_selection_mode,
        "current_checkpoint": str(previous_checkpoint),
        "num_candidates_total": len(pooled_candidates),
        "selected_count_total": budget,
        "default_total_budget": sample_memory * len(previous_tasks),
        "global_selection_count": global_selection_count,
        "global_selection_ratio": global_selection_ratio,
        "per_task_selection_ratio": selection_ratio_applied,
        "per_task_selection_count": selection_count_applied,
        "per_task_selection_budgets": per_task_budgets,
        "per_task_cluster_selection": cluster_selection_by_task or None,
        "score_path": str(ranked_score_path),
        **scoring_metadata,
    }
    return rehearsal_files, selection_metadata, selection_pool_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--template", default="llama2")
    parser.add_argument("--iteration_task")
    parser.add_argument("--iteration_index", type=int)
    parser.add_argument("--selector", choices=available_selectors(), default="kmeans")
    parser.add_argument("--candidate_source", choices=sorted(ARTIFACT_KEY_MAP), default="refined")
    parser.add_argument(
        "--encoder_model_name_or_path",
        default="princeton-nlp/sup-simcse-roberta-base",
    )
    parser.add_argument("--output_root")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--sample_memory", type=int, default=200)
    parser.add_argument("--global_selection_count", type=int)
    parser.add_argument("--global_selection_ratio", type=float)
    parser.add_argument(
        "--mean_kl_selection_mode",
        choices=POOLED_SELECTION_MODES,
        default="global",
    )
    parser.add_argument(
        "--uncertainty_selection_mode",
        choices=POOLED_SELECTION_MODES,
        default="global",
    )
    parser.add_argument("--per_task_selection_ratio", type=float)
    parser.add_argument("--per_task_selection_count", type=int)
    parser.add_argument("--kl_batch_size", type=int, default=1)
    parser.add_argument("--uncertainty_batch_size", type=int, default=1)
    parser.add_argument("--h_min", "--uncertainty_h_min", dest="h_min", type=float, default=0.25)
    parser.add_argument("--h_max", "--uncertainty_h_max", dest="h_max", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--train_max_steps", type=int)
    parser.add_argument("--eval_max_samples", type=int, default=100000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_all_tasks", action="store_true", default=False)
    parser.add_argument("--skip_train", action="store_true", default=False)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_run = Path(args.base_run).expanduser().resolve()
    if not base_run.exists():
        raise FileNotFoundError(f"Base run path does not exist: {base_run}")

    tasks = infer_tasks(base_run)
    if not tasks:
        raise ValueError(f"No stage directories found under {base_run}.")

    target_index = resolve_target_index(tasks, args.iteration_task, args.iteration_index)
    target_task = tasks[target_index]
    previous_tasks = tasks[:target_index]
    history_checkpoints = [
        checkpoint_dir(base_run, idx, task_name).resolve()
        for idx, task_name in enumerate(previous_tasks)
    ]
    previous_checkpoint = history_checkpoints[-1] if history_checkpoints else None
    pooled_selection_mode = resolve_pooled_selection_mode(
        args.selector,
        args.mean_kl_selection_mode,
        args.uncertainty_selection_mode,
    )

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (
            repo_root()
            / "saves"
            / (
                f"selection-proxy-{base_run.name}-"
                f"{selector_output_name(args.selector, pooled_selection_mode)}-"
                f"{args.candidate_source}-{target_index + 1:02d}_{target_task}"
            )
        ).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / "proxy_data"
    selection_dir = dataset_dir / "rehearsal"
    selection_dir.mkdir(parents=True, exist_ok=True)

    run_summary = load_run_summary(base_run)

    rehearsal_files: Dict[str, Path] = {}
    selection_metadata: Dict[str, Dict[str, object]] = {}
    selection_pool_metadata: Dict[str, object] | None = None

    if args.selector in POOLED_SELECTORS and previous_tasks:
        if previous_checkpoint is None:
            raise ValueError(
                f"Selector `{args.selector}` requires a current checkpoint, but no previous stage exists."
            )
        rehearsal_files, selection_metadata, selection_pool_metadata = run_pooled_selection(
            base_run=base_run,
            selector=args.selector,
            model_name_or_path=args.model_name_or_path,
            template=args.template,
            selection_dir=selection_dir,
            output_root=output_root,
            candidate_source=args.candidate_source,
            previous_tasks=previous_tasks,
            tasks=tasks,
            history_checkpoints=history_checkpoints,
            previous_checkpoint=previous_checkpoint,
            run_summary=run_summary,
            sample_memory=args.sample_memory,
            seed=args.seed,
            encoder_model_name_or_path=args.encoder_model_name_or_path,
            global_selection_count=args.global_selection_count,
            global_selection_ratio=args.global_selection_ratio,
            pooled_selection_mode=pooled_selection_mode,
            per_task_selection_ratio=args.per_task_selection_ratio,
            per_task_selection_count=args.per_task_selection_count,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            kl_batch_size=args.kl_batch_size,
            uncertainty_batch_size=args.uncertainty_batch_size,
            h_min=args.h_min,
            h_max=args.h_max,
        )
    else:
        for source_task in previous_tasks:
            source_task_index = tasks.index(source_task)
            source_paths = resolve_artifact_paths(run_summary, source_task)
            if args.candidate_source not in source_paths:
                raise ValueError(
                    f"Task `{source_task}` in {base_run / 'run_summary.json'} does not expose "
                    f"a `{args.candidate_source}` artifact."
                )

            candidate_path = source_paths[args.candidate_source]
            candidate_raw_rows = load_jsonl(candidate_path)
            candidate_rows = normalize_candidate_rows(candidate_raw_rows)
            selection_path = selection_dir / f"{source_task}.{args.selector}.{args.candidate_source}.jsonl"

            context = SelectionContext(
                method_name=args.selector,
                source_task=source_task,
                source_task_index=source_task_index,
                source_checkpoint=checkpoint_dir(base_run, source_task_index, source_task).resolve(),
                target_task=target_task,
                target_index=target_index,
                sample_memory=args.sample_memory,
                seed=args.seed,
                encoder_model_name_or_path=args.encoder_model_name_or_path,
                base_run_dir=base_run,
                candidate_source_name=args.candidate_source,
                candidate_path=candidate_path,
                candidate_rows=candidate_rows,
                candidate_raw_rows=candidate_raw_rows,
                source_paths=source_paths,
                history_tasks=list(previous_tasks),
                history_checkpoints=list(history_checkpoints),
                work_dir=output_root,
            )
            result = select_rows(args.selector, context)
            write_jsonl(selection_path, result.rows)

            rehearsal_files[source_task] = selection_path
            selection_metadata[source_task] = {
                "candidate_path": str(candidate_path),
                "selected_path": str(selection_path),
                "candidate_source": args.candidate_source,
                "history_checkpoints": [str(path) for path in history_checkpoints],
                **result.metadata,
            }

    rehearsal_dataset_keys = write_dataset_registry(dataset_dir, tasks, rehearsal_files)

    train_datasets = [f"ni_c012_{target_task}_train"]
    selected_source_tasks = [source_task for source_task in previous_tasks if source_task in rehearsal_dataset_keys]
    train_datasets.extend(rehearsal_dataset_keys[source_task] for source_task in selected_source_tasks)

    current_stage_dir = checkpoint_dir(output_root, target_index, target_task)
    current_stage_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    env = repo_env(args.cuda)

    if not args.skip_train:
        run_train(
            python_bin=python_bin,
            env=env,
            model_name_or_path=args.model_name_or_path,
            template=args.template,
            dataset_dir=dataset_dir,
            datasets=train_datasets,
            checkpoint=previous_checkpoint,
            output_dir=current_stage_dir,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            num_train_epochs=args.num_train_epochs,
            max_samples=args.max_samples,
            train_max_steps=args.train_max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

    eval_tasks = tasks if args.eval_all_tasks else tasks[:target_index + 1]
    if not args.skip_eval:
        for eval_task in eval_tasks:
            run_eval(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=args.template,
                dataset_dir=dataset_dir,
                checkpoint_dir_path=current_stage_dir,
                eval_task=eval_task,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                max_samples=args.eval_max_samples,
            )

    proxy_metrics_path: str | None = None
    if has_stage_scores(current_stage_dir, eval_tasks):
        base_stage_dir = checkpoint_dir(base_run, target_index, target_task)
        base_scores = load_stage_scores(base_stage_dir, eval_tasks)
        replay_scores = load_stage_scores(current_stage_dir, eval_tasks)
        delta_scores = {
            eval_task: replay_scores[eval_task] - base_scores[eval_task]
            for eval_task in eval_tasks
        }
        base_seen_average = average_score(base_scores)
        replay_seen_average = average_score(replay_scores)

        proxy_metrics = {
            "base_run": str(base_run),
            "output_root": str(output_root),
            "target_index": target_index + 1,
            "target_task": target_task,
            "selector": args.selector,
            "candidate_source": args.candidate_source,
            "train_datasets": train_datasets,
            "eval_tasks": eval_tasks,
            "base_scores_rouge_l": base_scores,
            "replay_scores_rouge_l": replay_scores,
            "delta_scores_rouge_l": delta_scores,
            "base_seen_average_rouge_l": base_seen_average,
            "replay_seen_average_rouge_l": replay_seen_average,
            "delta_seen_average_rouge_l": (
                None
                if base_seen_average is None or replay_seen_average is None
                else replay_seen_average - base_seen_average
            ),
        }
        proxy_metrics_path = str(output_root / "proxy_metrics.json")
        (output_root / "proxy_metrics.json").write_text(json.dumps(proxy_metrics, indent=2))

    summary = {
        "base_run": str(base_run),
        "tasks": tasks,
        "target_index": target_index + 1,
        "target_task": target_task,
        "previous_tasks": previous_tasks,
        "selected_source_tasks": selected_source_tasks,
        "previous_checkpoint": str(previous_checkpoint) if previous_checkpoint else None,
        "selector": args.selector,
        "candidate_source": args.candidate_source,
        "selection_pool": selection_pool_metadata,
        "selection": selection_metadata,
        "dataset_dir": str(dataset_dir),
        "stage_dir": str(current_stage_dir),
        "train_datasets": train_datasets,
        "eval_tasks": eval_tasks,
        "proxy_metrics_path": proxy_metrics_path,
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved replay summary to {output_root / 'run_summary.json'}")
    if proxy_metrics_path is not None:
        print(f"Saved proxy metrics to {output_root / 'proxy_metrics.json'}")


if __name__ == "__main__":
    main()
