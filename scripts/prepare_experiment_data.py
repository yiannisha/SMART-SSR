#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import jsonlines
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer


DEFAULT_TASKS = ["qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"]


def read_jsonl(path: Path) -> List[Dict]:
    with jsonlines.open(path, "r") as reader:
        return [row for row in reader]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(list(rows))


def pooled_embeddings(model_output) -> torch.Tensor:
    if getattr(model_output, "pooler_output", None) is not None:
        return model_output.pooler_output
    return model_output.last_hidden_state[:, 0]


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            embeddings.append(pooled_embeddings(outputs).cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def select_by_kmeans(embeddings: np.ndarray, sample_size: int, n_cluster: int, seed: int) -> List[int]:
    if len(embeddings) <= sample_size:
        return list(range(len(embeddings)))

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

    n_cluster = min(n_cluster, len(embeddings))
    kmeans = KMeans(n_clusters=n_cluster, n_init="auto", random_state=seed)
    labels = kmeans.fit_predict(embeddings)
    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[labels], axis=-1)

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


def prepare_kmeans_subset(
    task_name: str,
    source_dir: Path,
    output_dir: Path,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    sample_size: int,
    n_cluster: int,
    seed: int,
    force: bool,
) -> str:
    source_path = source_dir / f"{task_name}.train.json"
    output_path = output_dir / f"{task_name}.train.smp01.json"

    if output_path.exists() and not force:
        return f"{task_name}: kept existing {output_path}"

    rows = read_jsonl(source_path)
    texts = [row["input"] for row in rows]
    embeddings = encode_texts(texts, tokenizer, model, device, batch_size)
    selected_indices = select_by_kmeans(embeddings, sample_size, n_cluster, seed)
    selected_rows = [rows[idx] for idx in selected_indices]
    write_jsonl(output_path, selected_rows)
    return f"{task_name}: wrote {len(selected_rows)} rows to {output_path}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/ni-cus0.12/split")
    parser.add_argument("--output-dir", default="data/ni-cus0.12/split-kmeans20")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--encoder-model-name-or-path",
        default="princeton-nlp/sup-simcse-roberta-base"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--n-cluster", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_sources = [task for task in args.tasks if not (source_dir / f"{task}.train.json").exists()]
    if missing_sources:
        raise FileNotFoundError(
            f"Missing training splits for tasks: {', '.join(missing_sources)} under {source_dir}."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name_or_path)
    model = AutoModel.from_pretrained(args.encoder_model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    for task_name in args.tasks:
        message = prepare_kmeans_subset(
            task_name=task_name,
            source_dir=source_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            n_cluster=args.n_cluster,
            seed=args.seed,
            force=args.force,
        )
        print(message)


if __name__ == "__main__":
    main()
