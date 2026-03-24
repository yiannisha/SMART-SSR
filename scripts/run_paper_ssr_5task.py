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
    MEAN_KL_SELECTOR_NAME,
    UNCERTAINTY_BAND_SELECTOR_NAME,
    PooledCandidate,
    SelectionContext,
    list_methods,
    load_jsonl,
    normalize_candidate_rows,
    safe_hf_cache_dir,
    score_pooled_candidates_by_mean_kl,
    score_pooled_candidates_by_uncertainty_band,
    select_rows,
    write_jsonl,
)


TASKS = ["qa", "qg", "sa", "sum", "trans"]
MODEL_SPECS = {
    "tinyllama": {
        "train_template": "llama2",
        "gen_template": "vanilla",
        "model_tag": "tinyllama",
        "dataset_key": lambda task: f"ni_c012_icl_gen_km20_self_cl_queue_tinyllama_{task}",
    },
    "llama2-7b-chat": {
        "train_template": "llama2",
        "gen_template": "vanilla",
        "model_tag": "llama2-7b-chat",
        "dataset_key": lambda task: f"ni_c012_icl_gen_km20_self_cl_queue_llama2_7b_chat_{task}",
    },
    "llama2-7b": {
        "train_template": "vanilla",
        "gen_template": "vanilla",
        "model_tag": "llama2-7b",
        "dataset_key": lambda task: f"ni_c012_icl_gen_km20_self_cl_queue_{task}",
    }
}
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
POOLED_SELECTION_MODES = ("global", "per_task_top_ratio")


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


def checkpoint_dir(output_root: Path, task_index: int, task_name: str) -> Path:
    return output_root / f"{task_index + 1:02d}_{task_name}"


def build_paths(output_root: Path, model_tag: str, task_name: str) -> Dict[str, Path]:
    raw_root = Path(f"data/ni-cus0.12/genearated-icl-naive/{model_tag}/ori-van")
    parsed_root = Path(f"data/ni-cus0.12/genearated-icl-naive-parsed-filtered/{model_tag}/ori-van")
    artifact_root = output_root / "artifacts"
    refined_root = artifact_root / "refined"
    final_root = artifact_root / "cl_queue"
    suffix = f"{task_name}.train.smp001.2shot.smp3.rp1.2.json"
    return {
        "raw": raw_root / suffix,
        "parsed": parsed_root / suffix,
        "refined": refined_root / suffix,
        "final": final_root / suffix,
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_candidate_source(selector: str, candidate_source: str | None) -> str:
    if candidate_source is not None:
        return candidate_source
    if selector == "kmeans":
        return "final"
    return "refined"


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
        dataset_key = f"stage_rehearsal_{source_task}"
        rehearsal_keys[source_task] = dataset_key
        dataset_info[dataset_key] = make_dataset_entry(
            str(rehearsal_path.resolve()),
            REHEARSAL_COLUMNS,
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))
    return rehearsal_keys


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


def stage_data_dir(output_root: Path, task_index: int, task_name: str) -> Path:
    return output_root / "stage_data" / f"{task_index + 1:02d}_{task_name}"


def stage_selection_dir(output_root: Path, task_index: int, task_name: str) -> Path:
    return stage_data_dir(output_root, task_index, task_name) / "rehearsal"


def stage_score_path(
    output_root: Path,
    task_index: int,
    task_name: str,
    selector: str,
    candidate_source: str,
) -> Path:
    return (
        stage_data_dir(output_root, task_index, task_name)
        / f"{selector}_scores.{candidate_source}.jsonl"
    )


def run_single_task_train(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    dataset_dir: Path,
    task_name: str,
    output_dir: Path,
    max_source_length: int,
    max_target_length: int,
    num_train_epochs: float,
    max_samples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int
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
        "--dataset", f"ni_c012_{task_name}_train",
        "--max_source_length", str(max_source_length),
        "--max_target_length", str(max_target_length),
        "--learning_rate", "2e-4",
        "--num_train_epochs", str(num_train_epochs),
        "--max_samples", str(max_samples),
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
        "--bf16", "True"
    ]
    run_command(command, env)


def run_cl_train(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    dataset_dir: Path,
    datasets: List[str],
    checkpoint: Path,
    output_dir: Path,
    max_source_length: int,
    max_target_length: int,
    num_train_epochs: float,
    max_samples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int
) -> None:
    command = [
        python_bin,
        "src/train_bash.py",
        "--stage", "sft",
        "--model_name_or_path", model_name_or_path,
        "--checkpoint_dir", str(checkpoint),
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
        "--num_train_epochs", str(num_train_epochs),
        "--max_samples", str(max_samples),
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
        "--bf16", "True"
    ]
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
    max_samples: int
) -> None:
    output_dir = checkpoint_dir_path / f"ni_c012_{eval_task}_eval"
    command = [
        python_bin,
        "src/train_bash.py",
        "--stage", "sftrp",
        "--model_name_or_path", model_name_or_path,
        "--checkpoint_dir", str(checkpoint_dir_path),
        "--overwrite_cache", "True",
        "--predict_with_generate", "True",
        "--finetuning_type", "lora",
        "--template", template,
        "--dataset_dir", str(dataset_dir),
        "--dataset", f"ni_c012_{eval_task}_eval",
        "--max_source_length", str(max_source_length),
        "--max_target_length", str(max_target_length),
        "--max_samples", str(max_samples),
        "--per_device_eval_batch_size", "1",
        "--output_dir", str(output_dir),
        "--do_predict", "True",
        "--do_sample", "False",
        "--bf16", "True"
    ]
    run_command(command, env)


def run_raw_generation(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    task_name: str,
    output_path: Path
) -> None:
    command = [
        python_bin,
        "custom/icl_gen/complete_param_nic010_cate.py",
        "--model_name_or_path", model_name_or_path,
        "--input_path", f"data/ni-cus0.12/split/{task_name}.train.smp001.json",
        "--output_path", str(output_path),
        "--do_sample", "True",
        "--do_sample_retries", "3",
        "--top_p", "0.6",
        "--temperature", "0.9",
        "--repetition_penalty", "1.2",
        "--max_length", "2048",
        "--num_beams", "1",
        "--n_shots", "2",
        "--template", template,
        "--cate_task_style", "False",
        "--resume", "True"
    ]
    run_command(command, env)


def run_parse_filter(
    python_bin: str,
    env: Dict[str, str],
    tokenizer_name_or_path: str,
    input_path: Path,
    output_path: Path
) -> None:
    command = [
        python_bin,
        "custom/icl_gen/parse_filter_generated.py",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--tokenizer_name_or_path", tokenizer_name_or_path,
    ]
    run_command(command, env)


def run_kmeans_selection(
    python_bin: str,
    env: Dict[str, str],
    input_path: Path,
    output_path: Path,
    sample_memory: int
) -> None:
    if output_path.exists():
        output_path.unlink()
    command = [
        python_bin,
        "custom/icl_gen/select_kmeans_examples.py",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--sample_memory", str(sample_memory),
        "--n_cluster", "20",
    ]
    run_command(command, env)


def run_refinement(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    checkpoint_dir_path: Path,
    input_path: Path,
    output_path: Path
) -> None:
    if output_path.exists():
        output_path.unlink()
    command = [
        python_bin,
        "custom/icl_gen/label_param.py",
        "--model_name_or_path", model_name_or_path,
        "--ckpt_dir", str(checkpoint_dir_path),
        "--finetuning_type", "lora",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--do_sample", "False",
        "--max_length", "2048",
        "--template", template
    ]
    run_command(command, env)


def select_stage_rehearsal_files(
    output_root: Path,
    spec: Dict[str, object],
    tasks: List[str],
    target_index: int,
    target_task: str,
    selector: str,
    candidate_source: str,
    model_name_or_path: str,
    template: str,
    sample_memory: int,
    seed: int,
    encoder_model_name_or_path: str,
    global_selection_count: int | None,
    global_selection_ratio: float | None,
    pooled_selection_mode: str,
    per_task_selection_ratio: float | None,
    max_source_length: int,
    max_target_length: int,
    kl_batch_size: int,
    uncertainty_batch_size: int,
    h_min: float,
    h_max: float,
) -> tuple[Dict[str, Path], Dict[str, Dict[str, object]], Dict[str, object] | None]:
    previous_tasks = tasks[:target_index]
    if not previous_tasks:
        return {}, {}, None

    history_checkpoints = [
        checkpoint_dir(output_root, idx, task_name).resolve()
        for idx, task_name in enumerate(previous_tasks)
    ]
    previous_checkpoint = history_checkpoints[-1]
    selection_dir = stage_selection_dir(output_root, target_index, target_task)
    selection_dir.mkdir(parents=True, exist_ok=True)
    work_dir = stage_data_dir(output_root, target_index, target_task)

    if selector in POOLED_SELECTORS:
        pooled_candidates: List[PooledCandidate] = []
        candidate_paths: Dict[str, Path] = {}
        candidate_counts: Dict[str, int] = {}

        for source_task in previous_tasks:
            source_task_index = tasks.index(source_task)
            source_paths = build_paths(output_root, spec["model_tag"], source_task)
            candidate_path = source_paths[candidate_source]
            if not candidate_path.exists():
                raise FileNotFoundError(
                    f"Candidate artifact does not exist for task `{source_task}`: {candidate_path}"
                )
            candidate_raw_rows = load_jsonl(candidate_path)
            candidate_rows = normalize_candidate_rows(candidate_raw_rows)
            candidate_paths[source_task] = candidate_path
            candidate_counts[source_task] = len(candidate_rows)

            for candidate_index, (row, raw_row) in enumerate(zip(candidate_rows, candidate_raw_rows)):
                pooled_candidates.append(
                    PooledCandidate(
                        source_task=source_task,
                        source_task_index=source_task_index,
                        source_checkpoint=checkpoint_dir(
                            output_root,
                            source_task_index,
                            source_task,
                        ).resolve(),
                        candidate_index=candidate_index,
                        row=row,
                        raw_row=raw_row,
                    )
                )

        selection_ratio_applied: float | None = None
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

        ranked_score_path = stage_score_path(
            output_root,
            target_index,
            target_task,
            selector,
            candidate_source,
        )
        ranked_rows: List[Dict[str, object]] = []
        all_scores_by_task: Dict[str, List[float]] = defaultdict(list)
        selected_scores_by_task: Dict[str, List[float]] = defaultdict(list)
        selected_entropies_by_task: Dict[str, List[float]] = defaultdict(list)
        selected_rows_by_task: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        selected_counts_by_task: Dict[str, int] = defaultdict(int)

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
            ranked_rows.append(ranked_row)

            should_select = False
            if pooled_selection_mode == "global":
                should_select = rank <= budget
            else:
                assert per_task_budgets is not None
                should_select = (
                    selected_counts_by_task[candidate.source_task]
                    < per_task_budgets[candidate.source_task]
                )

            if should_select:
                selected_rows_by_task[candidate.source_task].append(candidate.row)
                selected_scores_by_task[candidate.source_task].append(score)
                selected_counts_by_task[candidate.source_task] += 1
                if selector == UNCERTAINTY_BAND_SELECTOR_NAME:
                    assert scored_candidate.mean_entropy is not None
                    selected_entropies_by_task[candidate.source_task].append(
                        scored_candidate.mean_entropy
                    )

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
            "per_task_selection_budgets": per_task_budgets,
            "score_path": str(ranked_score_path),
            **scoring_metadata,
        }
        return rehearsal_files, selection_metadata, selection_pool_metadata

    rehearsal_files = {}
    selection_metadata = {}
    for source_task in previous_tasks:
        source_task_index = tasks.index(source_task)
        source_paths = build_paths(output_root, spec["model_tag"], source_task)
        candidate_path = source_paths[candidate_source]
        if not candidate_path.exists():
            raise FileNotFoundError(
                f"Candidate artifact does not exist for task `{source_task}`: {candidate_path}"
            )

        candidate_raw_rows = load_jsonl(candidate_path)
        candidate_rows = normalize_candidate_rows(candidate_raw_rows)
        selection_path = selection_dir / f"{source_task}.{selector}.{candidate_source}.jsonl"
        context = SelectionContext(
            method_name=selector,
            source_task=source_task,
            source_task_index=source_task_index,
            source_checkpoint=checkpoint_dir(output_root, source_task_index, source_task).resolve(),
            target_task=target_task,
            target_index=target_index,
            sample_memory=sample_memory,
            seed=seed,
            encoder_model_name_or_path=encoder_model_name_or_path,
            base_run_dir=output_root,
            candidate_source_name=candidate_source,
            candidate_path=candidate_path,
            candidate_rows=candidate_rows,
            candidate_raw_rows=candidate_raw_rows,
            source_paths=source_paths,
            history_tasks=list(previous_tasks),
            history_checkpoints=list(history_checkpoints),
            work_dir=work_dir,
        )
        result = select_rows(selector, context)

        selected_path: str | None = None
        if result.rows:
            write_jsonl(selection_path, result.rows)
            rehearsal_files[source_task] = selection_path
            selected_path = str(selection_path)

        selection_metadata[source_task] = {
            "candidate_path": str(candidate_path),
            "selected_path": selected_path,
            "candidate_source": candidate_source,
            "history_checkpoints": [str(path) for path in history_checkpoints],
            **result.metadata,
        }

    return rehearsal_files, selection_metadata, None


def load_rouge_l(path: Path) -> float:
    with path.open("r") as handle:
        return float(json.load(handle)["predict_rouge-l"])


def save_aggregate_metrics(output_root: Path, tasks: List[str]) -> None:
    score_matrix: Dict[str, Dict[str, float]] = {}
    for idx, task_name in enumerate(tasks):
        stage_dir = checkpoint_dir(output_root, idx, task_name)
        score_matrix[task_name] = {}
        for eval_task in tasks:
            result_path = stage_dir / f"ni_c012_{eval_task}_eval" / "all_results.json"
            if result_path.exists():
                score_matrix[task_name][eval_task] = load_rouge_l(result_path)

    final_task = tasks[-1]
    final_scores = [score_matrix[final_task][task] for task in tasks]
    average_rouge_l = sum(final_scores) / len(final_scores)

    bwt_terms = []
    for idx, task_name in enumerate(tasks[:-1]):
        bwt_terms.append(score_matrix[final_task][task_name] - score_matrix[task_name][task_name])
    bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else 0.0

    fwt_terms = []
    for idx in range(1, len(tasks)):
        previous_stage = tasks[idx - 1]
        current_task = tasks[idx]
        fwt_terms.append(score_matrix[previous_stage][current_task])
    fwt = sum(fwt_terms) / len(fwt_terms) if fwt_terms else 0.0

    aggregate = {
        "average_rouge_l": average_rouge_l,
        "bwt": bwt,
        "fwt": fwt,
        "score_matrix_rouge_l": score_matrix
    }
    (output_root / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_family", choices=sorted(MODEL_SPECS), default="llama2-7b-chat")
    parser.add_argument("--output_root", default="saves/paper-ssr-5task")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--selector", choices=available_selectors(), default="kmeans")
    parser.add_argument("--candidate_source", choices=sorted(ARTIFACT_KEY_MAP), default=None)
    parser.add_argument(
        "--encoder_model_name_or_path",
        default="princeton-nlp/sup-simcse-roberta-base",
    )
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
    parser.add_argument("--kl_batch_size", type=int, default=1)
    parser.add_argument("--uncertainty_batch_size", type=int, default=1)
    parser.add_argument("--h_min", "--uncertainty_h_min", dest="h_min", type=float, default=0.25)
    parser.add_argument("--h_max", "--uncertainty_h_max", dest="h_max", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_all_tasks", action="store_true", default=True)
    parser.add_argument("--skip_completed", action="store_true", default=False)
    args = parser.parse_args()

    python_bin = sys.executable
    env = repo_env(args.cuda)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[args.model_family]
    candidate_source = resolve_candidate_source(args.selector, args.candidate_source)
    pooled_selection_mode = resolve_pooled_selection_mode(
        args.selector,
        args.mean_kl_selection_mode,
        args.uncertainty_selection_mode,
    )

    summary: Dict[str, Dict[str, object]] = {}

    for idx, task_name in enumerate(args.tasks):
        paths = build_paths(output_root, spec["model_tag"], task_name)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not paths["raw"].exists():
            run_raw_generation(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=spec["gen_template"],
                task_name=task_name,
                output_path=paths["raw"]
            )

        if not paths["parsed"].exists():
            run_parse_filter(
                python_bin=python_bin,
                env=env,
                tokenizer_name_or_path=args.model_name_or_path,
                input_path=paths["raw"],
                output_path=paths["parsed"]
            )

        current_checkpoint = checkpoint_dir(output_root, idx, task_name)
        previous_checkpoint = checkpoint_dir(output_root, idx - 1, args.tasks[idx - 1]) if idx > 0 else None
        current_stage_dataset_dir = stage_data_dir(output_root, idx, task_name)
        should_train_stage = not args.skip_completed or not current_checkpoint.exists()

        rehearsal_files: Dict[str, Path] = {}
        selection_metadata: Dict[str, Dict[str, object]] = {}
        selection_pool_metadata: Dict[str, object] | None = None
        if idx > 0:
            rehearsal_files, selection_metadata, selection_pool_metadata = select_stage_rehearsal_files(
                output_root=output_root,
                spec=spec,
                tasks=args.tasks,
                target_index=idx,
                target_task=task_name,
                selector=args.selector,
                candidate_source=candidate_source,
                model_name_or_path=args.model_name_or_path,
                template=spec["train_template"],
                sample_memory=args.sample_memory,
                seed=args.seed,
                encoder_model_name_or_path=args.encoder_model_name_or_path,
                global_selection_count=args.global_selection_count,
                global_selection_ratio=args.global_selection_ratio,
                pooled_selection_mode=pooled_selection_mode,
                per_task_selection_ratio=args.per_task_selection_ratio,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                kl_batch_size=args.kl_batch_size,
                uncertainty_batch_size=args.uncertainty_batch_size,
                h_min=args.h_min,
                h_max=args.h_max,
            )
        rehearsal_dataset_keys = write_dataset_registry(
            current_stage_dataset_dir,
            args.tasks,
            rehearsal_files,
        )
        selected_source_tasks = [
            source_task
            for source_task in args.tasks[:idx]
            if source_task in rehearsal_dataset_keys
        ]
        train_datasets = [f"ni_c012_{task_name}_train"]
        train_datasets.extend(
            rehearsal_dataset_keys[source_task]
            for source_task in selected_source_tasks
        )

        if should_train_stage:
            if idx == 0:
                run_single_task_train(
                    python_bin=python_bin,
                    env=env,
                    model_name_or_path=args.model_name_or_path,
                    template=spec["train_template"],
                    dataset_dir=current_stage_dataset_dir,
                    task_name=task_name,
                    output_dir=current_checkpoint,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                    num_train_epochs=args.num_train_epochs,
                    max_samples=args.max_samples,
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps
                )
            else:
                run_cl_train(
                    python_bin=python_bin,
                    env=env,
                    model_name_or_path=args.model_name_or_path,
                    template=spec["train_template"],
                    dataset_dir=current_stage_dataset_dir,
                    datasets=train_datasets,
                    checkpoint=previous_checkpoint,
                    output_dir=current_checkpoint,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                    num_train_epochs=args.num_train_epochs,
                    max_samples=args.max_samples,
                    per_device_train_batch_size=args.per_device_train_batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps
                    )

        if should_train_stage or not paths["refined"].exists():
            run_refinement(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=spec["train_template"],
                checkpoint_dir_path=current_checkpoint,
                input_path=paths["parsed"],
                output_path=paths["refined"]
            )

        if should_train_stage or not paths["final"].exists():
            run_kmeans_selection(
                python_bin=python_bin,
                env=env,
                input_path=paths["refined"],
                output_path=paths["final"],
                sample_memory=args.sample_memory
            )

        eval_tasks = args.tasks if args.eval_all_tasks else args.tasks[:idx + 1]
        for eval_task in eval_tasks:
            run_eval(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=spec["train_template"],
                dataset_dir=current_stage_dataset_dir,
                checkpoint_dir_path=current_checkpoint,
                eval_task=eval_task,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                max_samples=args.max_samples
            )

        summary[task_name] = {
            "checkpoint_dir": [str(current_checkpoint)],
            "stage_dataset_dir": [str(current_stage_dataset_dir)],
            "raw_generation_file": [str(paths["raw"])],
            "parsed_generation_file": [str(paths["parsed"])],
            "refined_generation_file": [str(paths["refined"])],
            "kmeans_selection_file": [str(paths["final"])],
            "rehearsal_file": [str(paths["final"])],
            "selector": args.selector,
            "candidate_source": candidate_source,
            "artifact_scope": {
                "raw": "shared_model_family",
                "parsed": "shared_model_family",
                "refined": "run_local",
                "final": "run_local",
            },
            "selection_pool": selection_pool_metadata,
            "selection": selection_metadata,
            "train_datasets": train_datasets,
            "selected_source_tasks": selected_source_tasks,
            "evaluated_tasks": eval_tasks
        }

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    save_aggregate_metrics(output_root, args.tasks)
    print(f"Saved run summary to {summary_path}.")
    print(f"Saved aggregate metrics to {output_root / 'aggregate_metrics.json'}.")


if __name__ == "__main__":
    main()
