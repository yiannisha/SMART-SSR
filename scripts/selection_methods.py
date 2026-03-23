from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping

import jsonlines
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


SelectorFn = Callable[["SelectionContext"], "SelectionResult"]
SELECTORS: Dict[str, SelectorFn] = {}


@dataclass(frozen=True)
class SelectionContext:
    method_name: str
    source_task: str
    source_task_index: int
    source_checkpoint: Path | None
    target_task: str
    target_index: int
    sample_memory: int
    seed: int
    encoder_model_name_or_path: str
    base_run_dir: Path
    candidate_source_name: str
    candidate_path: Path
    candidate_rows: List[Dict[str, str]]
    candidate_raw_rows: List[Dict]
    source_paths: Mapping[str, Path]
    history_tasks: List[str]
    history_checkpoints: List[Path]
    work_dir: Path


@dataclass
class SelectionResult:
    rows: List[Dict[str, str]]
    metadata: Dict[str, object] = field(default_factory=dict)


def register_selector(name: str) -> Callable[[SelectorFn], SelectorFn]:
    def decorator(func: SelectorFn) -> SelectorFn:
        SELECTORS[name] = func
        return func

    return decorator


def list_methods() -> List[str]:
    return sorted(SELECTORS)


def load_jsonl(path: Path) -> List[Dict]:
    with jsonlines.open(path, "r") as reader:
        return [row for row in reader]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(rows)


def normalize_candidate_rows(rows: List[Dict]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for row in rows:
        if "inputs" in row:
            prompt = row["inputs"]
        elif "full_prompt" in row:
            prompt = row["full_prompt"]
        else:
            raise ValueError("Candidate row is missing `inputs`/`full_prompt`.")

        if "targets" in row:
            target = row["targets"]
        elif "outputs" in row:
            target = row["outputs"]
        elif "output" in row:
            target = row["output"]
        else:
            raise ValueError("Candidate row is missing `targets`/`outputs`/`output`.")

        normalized.append({"inputs": prompt, "targets": target})

    return normalized


def pooled_embeddings(model_output) -> torch.Tensor:
    if getattr(model_output, "pooler_output", None) is not None:
        return model_output.pooler_output
    return model_output.last_hidden_state[:, 0]


def encode_texts(texts: List[str], model_name_or_path: str, batch_size: int = 32) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    embeddings: List[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                embeddings.append(pooled_embeddings(outputs).cpu().numpy())
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)


def select_by_kmeans_indices(
    embeddings: np.ndarray,
    sample_size: int,
    n_cluster: int,
    seed: int
) -> List[int]:
    if len(embeddings) <= sample_size:
        return list(range(len(embeddings)))

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

    n_cluster = min(n_cluster, len(embeddings))
    labels, centers = fit_kmeans(embeddings, n_cluster=n_cluster, seed=seed)
    distances = np.linalg.norm(embeddings - centers[labels], axis=-1)

    counts = np.bincount(labels, minlength=n_cluster)
    raw_allocations = counts * float(sample_size) / float(len(embeddings))
    allocations = np.floor(raw_allocations).astype(int)
    remainder = sample_size - int(allocations.sum())
    if remainder > 0:
        order = np.argsort(-(raw_allocations - allocations))
        for idx in order[:remainder]:
            allocations[idx] += 1

    selected_indices: List[int] = []
    for cluster_id in range(n_cluster):
        cluster_indices = np.where(labels == cluster_id)[0]
        if cluster_indices.size == 0 or allocations[cluster_id] == 0:
            continue
        cluster_distances = distances[cluster_indices]
        ordered_indices = cluster_indices[np.argsort(cluster_distances)]
        selected_indices.extend(ordered_indices[:allocations[cluster_id]].tolist())

    if len(selected_indices) < sample_size:
        seen = set(selected_indices)
        fallback_order = np.argsort(distances)
        for idx in fallback_order:
            if idx in seen:
                continue
            selected_indices.append(int(idx))
            if len(selected_indices) >= sample_size:
                break

    return selected_indices[:sample_size]


def init_kmeans_plus_plus(
    embeddings: np.ndarray,
    n_cluster: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_samples = len(embeddings)
    centers = np.empty((n_cluster, embeddings.shape[1]), dtype=embeddings.dtype)

    first_idx = int(rng.integers(0, n_samples))
    centers[0] = embeddings[first_idx]

    closest_dist_sq = np.sum((embeddings - centers[0]) ** 2, axis=1)
    for cluster_idx in range(1, n_cluster):
        total = float(closest_dist_sq.sum())
        if total <= 0:
            next_idx = int(rng.integers(0, n_samples))
        else:
            probs = closest_dist_sq / total
            next_idx = int(rng.choice(n_samples, p=probs))
        centers[cluster_idx] = embeddings[next_idx]
        dist_sq = np.sum((embeddings - centers[cluster_idx]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)

    return centers


def fit_kmeans(
    embeddings: np.ndarray,
    n_cluster: int,
    seed: int,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = init_kmeans_plus_plus(embeddings, n_cluster=n_cluster, rng=rng)
    labels = np.full(len(embeddings), -1, dtype=np.int64)

    for _ in range(max_iter):
        distances = np.sum((embeddings[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for cluster_idx in range(n_cluster):
            cluster_points = embeddings[labels == cluster_idx]
            if len(cluster_points) == 0:
                centers[cluster_idx] = embeddings[int(rng.integers(0, len(embeddings)))]
                continue
            centers[cluster_idx] = cluster_points.mean(axis=0)

    distances = np.sum((embeddings[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, centers


def select_rows(method_name: str, context: SelectionContext) -> SelectionResult:
    if method_name not in SELECTORS:
        raise ValueError(
            f"Unknown selector `{method_name}`. Available methods: {', '.join(list_methods())}"
        )
    return SELECTORS[method_name](context)


@register_selector("head")
def head_selector(context: SelectionContext) -> SelectionResult:
    rows = context.candidate_rows[:context.sample_memory]
    return SelectionResult(
        rows=rows,
        metadata={
            "selector": "head",
            "num_candidates": len(context.candidate_rows),
            "selected_count": len(rows),
        },
    )


@register_selector("random")
def random_selector(context: SelectionContext) -> SelectionResult:
    if len(context.candidate_rows) <= context.sample_memory:
        rows = list(context.candidate_rows)
    else:
        rng = random.Random(context.seed)
        rows = rng.sample(context.candidate_rows, context.sample_memory)

    return SelectionResult(
        rows=rows,
        metadata={
            "selector": "random",
            "num_candidates": len(context.candidate_rows),
            "selected_count": len(rows),
        },
    )


@register_selector("kmeans")
def kmeans_selector(context: SelectionContext) -> SelectionResult:
    if len(context.candidate_rows) <= context.sample_memory:
        rows = list(context.candidate_rows)
        return SelectionResult(
            rows=rows,
            metadata={
                "selector": "kmeans",
                "encoder_model_name_or_path": context.encoder_model_name_or_path,
                "num_candidates": len(context.candidate_rows),
                "selected_count": len(rows),
                "n_cluster": min(20, len(rows)),
            },
        )

    texts = [row["inputs"] for row in context.candidate_rows]
    embeddings = encode_texts(texts, context.encoder_model_name_or_path)
    selected_indices = select_by_kmeans_indices(
        embeddings=embeddings,
        sample_size=context.sample_memory,
        n_cluster=20,
        seed=context.seed,
    )
    rows = [context.candidate_rows[idx] for idx in selected_indices]
    return SelectionResult(
        rows=rows,
        metadata={
            "selector": "kmeans",
            "encoder_model_name_or_path": context.encoder_model_name_or_path,
            "num_candidates": len(context.candidate_rows),
            "selected_count": len(rows),
            "n_cluster": min(20, len(context.candidate_rows)),
        },
    )
