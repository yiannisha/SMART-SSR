import argparse
from pathlib import Path
from typing import Dict, List

import jsonlines
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> List[Dict]:
    with jsonlines.open(path, "r") as reader:
        return [row for row in reader]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(rows)


def encode_texts(texts: List[str], model_name_or_path: str, batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            pooled = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--encoder_model_name_or_path",
        default="princeton-nlp/sup-simcse-roberta-base"
    )
    parser.add_argument("--sample_memory", type=int, default=200)
    parser.add_argument("--n_cluster", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_path))
    if len(rows) <= args.sample_memory:
        write_jsonl(Path(args.output_path), rows)
        print(f"Input only has {len(rows)} rows; copied all rows to {args.output_path}.")
        return

    texts = [row["inputs"] for row in rows]
    embeddings = encode_texts(texts, args.encoder_model_name_or_path, args.batch_size)
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

    np.random.seed(args.seed)
    kmeans = KMeans(n_clusters=args.n_cluster, n_init="auto", random_state=args.seed)
    labels = kmeans.fit_predict(embeddings)
    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[labels], axis=-1)

    counts = np.bincount(labels, minlength=args.n_cluster)
    raw_allocations = counts * float(args.sample_memory) / float(len(rows))
    allocations = np.floor(raw_allocations).astype(int)
    remainder = args.sample_memory - int(allocations.sum())
    if remainder > 0:
        order = np.argsort(-(raw_allocations - allocations))
        for idx in order[:remainder]:
            allocations[idx] += 1

    selected_indices: List[int] = []
    for cluster_id in range(args.n_cluster):
        cluster_indices = np.where(labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue
        cluster_distances = distances[cluster_indices]
        cluster_order = cluster_indices[np.argsort(cluster_distances)]
        selected_indices.extend(cluster_order[:allocations[cluster_id]].tolist())

    selected_indices = selected_indices[:args.sample_memory]
    selected_rows = [rows[idx] for idx in selected_indices]
    write_jsonl(Path(args.output_path), selected_rows)
    print(f"Saved {len(selected_rows)} selected rows to {args.output_path}.")


if __name__ == "__main__":
    main()
