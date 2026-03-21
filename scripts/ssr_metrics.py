from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence


STAGE_DIR_PATTERN = re.compile(r"^(?P<index>\d+)_(?P<task>.+)$")
ROUGE_L_KEY = "predict_rouge-l"


def infer_tasks(output_root: Path) -> List[str]:
    stages = []
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        match = STAGE_DIR_PATTERN.match(path.name)
        if match is None:
            continue
        stages.append((int(match.group("index")), match.group("task")))
    return [task for _, task in sorted(stages)]


def checkpoint_dir(output_root: Path, task_index: int, task_name: str) -> Path:
    return output_root / f"{task_index + 1:02d}_{task_name}"


def load_rouge_l(path: Path) -> float:
    with path.open("r") as handle:
        return float(json.load(handle)[ROUGE_L_KEY])


def build_score_matrix(output_root: Path, tasks: Sequence[str]) -> Dict[str, Dict[str, float]]:
    score_matrix: Dict[str, Dict[str, float]] = {}
    for idx, stage_task in enumerate(tasks):
        stage_dir = checkpoint_dir(output_root, idx, stage_task)
        stage_scores: Dict[str, float] = {}
        for eval_task in tasks:
            result_path = stage_dir / f"ni_c012_{eval_task}_eval" / "all_results.json"
            if result_path.exists():
                stage_scores[eval_task] = load_rouge_l(result_path)
        score_matrix[stage_task] = stage_scores
    return score_matrix


def average_for_tasks(stage_scores: Dict[str, float], tasks: Sequence[str]) -> float | None:
    if not tasks or any(task not in stage_scores for task in tasks):
        return None
    return sum(stage_scores[task] for task in tasks) / len(tasks)


def compute_aggregate_metrics(
    tasks: Sequence[str],
    score_matrix: Dict[str, Dict[str, float]]
) -> Dict[str, object]:
    if not tasks:
        raise ValueError("No tasks found in output root.")

    final_task = tasks[-1]
    final_scores = score_matrix.get(final_task, {})
    average_rouge_l = average_for_tasks(final_scores, tasks)

    bwt_terms: List[float] = []
    for task_name in tasks[:-1]:
        final_score = final_scores.get(task_name)
        diagonal_score = score_matrix.get(task_name, {}).get(task_name)
        if final_score is None or diagonal_score is None:
            bwt_terms = []
            break
        bwt_terms.append(final_score - diagonal_score)
    bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else None

    fwt_terms: List[float] = []
    for idx in range(1, len(tasks)):
        previous_stage = tasks[idx - 1]
        current_task = tasks[idx]
        score = score_matrix.get(previous_stage, {}).get(current_task)
        if score is None:
            fwt_terms = []
            break
        fwt_terms.append(score)
    fwt = sum(fwt_terms) / len(fwt_terms) if fwt_terms else None

    per_stage_seen_average: Dict[str, float | None] = {}
    per_stage_all_average: Dict[str, float | None] = {}
    for idx, task_name in enumerate(tasks):
        stage_scores = score_matrix.get(task_name, {})
        per_stage_seen_average[task_name] = average_for_tasks(stage_scores, tasks[:idx + 1])
        per_stage_all_average[task_name] = average_for_tasks(stage_scores, tasks)

    return {
        "tasks": list(tasks),
        "average_rouge_l": average_rouge_l,
        "bwt": bwt,
        "fwt": fwt,
        "score_matrix_rouge_l": score_matrix,
        "per_stage_average_rouge_l_seen_tasks": per_stage_seen_average,
        "per_stage_average_rouge_l_all_tasks": per_stage_all_average,
    }


def collect_run_metrics(output_root: Path, tasks: Sequence[str] | None = None) -> Dict[str, object]:
    ordered_tasks = list(tasks) if tasks is not None else infer_tasks(output_root)
    score_matrix = build_score_matrix(output_root, ordered_tasks)
    return compute_aggregate_metrics(ordered_tasks, score_matrix)


def save_aggregate_metrics(output_root: Path, tasks: Sequence[str] | None = None) -> Dict[str, object]:
    aggregate = collect_run_metrics(output_root, tasks)
    output_path = output_root / "aggregate_metrics.json"
    output_path.write_text(json.dumps(aggregate, indent=2))
    return aggregate
