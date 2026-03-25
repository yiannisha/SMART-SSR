from __future__ import annotations

import errno
import gc
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping

import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModelForCausalLM
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llmtuner.extras.template import get_template_and_fix_tokenizer


SelectorFn = Callable[["SelectionContext"], "SelectionResult"]
SELECTORS: Dict[str, SelectorFn] = {}
MEAN_KL_SELECTOR_NAME = "mean_kl"
UNCERTAINTY_BAND_SELECTOR_NAME = "uncertainty_band"
KMEANS_SELECTOR_N_CLUSTER = 20


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


@dataclass(frozen=True)
class PooledCandidate:
    source_task: str
    source_task_index: int
    source_checkpoint: Path
    candidate_index: int
    row: Dict[str, str]
    raw_row: Dict


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: PooledCandidate
    score: float
    mean_kl: float | None = None
    mean_entropy: float | None = None


@dataclass(frozen=True)
class CandidateEntropy:
    candidate: PooledCandidate
    mean_entropy: float


@dataclass(frozen=True)
class UncertaintyBandScoringResult:
    scored_candidates: List[ScoredCandidate]
    entropy_min: float | None
    entropy_max: float | None
    band_min_entropy: float | None
    band_max_entropy: float | None
    band_by_source_task: Mapping[str, Dict[str, float]]


@dataclass(frozen=True)
class KMeansSelectionState:
    n_cluster: int
    labels: np.ndarray
    centers: np.ndarray
    centroid_distances: np.ndarray
    counts: np.ndarray
    allocations: np.ndarray


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


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def is_stale_file_handle_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno == errno.ESTALE:
        return True
    return "Stale file handle" in str(exc)


def safe_hf_cache_dir() -> str:
    return os.environ.get(
        "SMART_SSR_HF_CACHE",
        os.path.join(tempfile.gettempdir(), "smart-ssr-hf-cache"),
    )


def ensure_safe_hf_cache_dir() -> str:
    cache_dir = safe_hf_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def from_pretrained_with_safe_hf_cache(factory, model_name_or_path: str, **kwargs):
    cache_dir = ensure_safe_hf_cache_dir()
    kwargs.setdefault("cache_dir", cache_dir)

    try:
        return factory.from_pretrained(model_name_or_path, **kwargs)
    except OSError as exc:
        if not is_stale_file_handle_error(exc):
            raise

        raise RuntimeError(
            "Hugging Face download failed with a stale file handle while writing to "
            f"cache directory {cache_dir!r}. Set SMART_SSR_HF_CACHE to a stable local "
            "path such as /tmp/smart-ssr-hf-cache and rerun."
        ) from exc


def get_model_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def model_input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def build_source_adapter_name(source_task: str) -> str:
    sanitized = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in source_task
    )
    return f"source_{sanitized}"


def build_scoring_batch(
    rows: List[Dict[str, str]],
    tokenizer,
    template,
    max_source_length: int,
    max_target_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_id_rows: List[List[int]] = []
    target_mask_rows: List[List[int]] = []

    for row in rows:
        source_ids, target_ids = template.encode_oneturn(
            tokenizer,
            row["inputs"],
            row["targets"],
            history=None,
            system=None,
        )
        if len(source_ids) > max_source_length:
            source_ids = source_ids[:max_source_length]
        if len(target_ids) > max_target_length:
            target_ids = target_ids[:max_target_length]
        if template.efficient_eos:
            target_ids = target_ids + [tokenizer.eos_token_id]
        if not target_ids:
            raise ValueError("Encountered a candidate row with an empty target after truncation.")

        input_ids = source_ids + target_ids
        target_mask = [0] * len(source_ids) + [1] * len(target_ids)
        input_id_rows.append(input_ids)
        target_mask_rows.append(target_mask)

    max_length = max(len(row) for row in input_id_rows)
    pad_token_id = tokenizer.pad_token_id
    padded_inputs: List[List[int]] = []
    padded_attention_masks: List[List[int]] = []
    padded_target_masks: List[List[int]] = []

    for input_ids, target_mask in zip(input_id_rows, target_mask_rows):
        pad_len = max_length - len(input_ids)
        padded_inputs.append(input_ids + [pad_token_id] * pad_len)
        padded_attention_masks.append([1] * len(input_ids) + [0] * pad_len)
        padded_target_masks.append(target_mask + [0] * pad_len)

    return (
        torch.tensor(padded_inputs, dtype=torch.long, device=device),
        torch.tensor(padded_target_masks, dtype=torch.bool, device=device),
        torch.tensor(padded_attention_masks, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def mean_distribution_shift(
    model,
    snapshot_adapter_name: str,
    current_adapter_name: str,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if snapshot_adapter_name == current_adapter_name:
        return torch.zeros(input_ids.size(0), dtype=torch.float32, device=input_ids.device)

    model.set_adapter(snapshot_adapter_name)
    snapshot_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, :-1, :]

    model.set_adapter(current_adapter_name)
    current_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, :-1, :]

    log_p = F.log_softmax(snapshot_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(current_logits, dim=-1, dtype=torch.float32)
    kl_per_pos = F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(dim=-1)

    valid_mask = target_mask[:, 1:].to(kl_per_pos.dtype)
    if attention_mask is not None:
        valid_mask = valid_mask * attention_mask[:, 1:].to(kl_per_pos.dtype)

    denom = valid_mask.sum(dim=-1).clamp_min(1.0)
    mean_kl = (kl_per_pos * valid_mask).sum(dim=-1) / denom
    return mean_kl


@torch.inference_mode()
def mean_predictive_entropy(
    model,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, :-1, :]

    log_p = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    p = log_p.exp()
    entropy_per_pos = -(p * log_p).sum(dim=-1)

    valid_mask = target_mask[:, 1:].to(entropy_per_pos.dtype)
    if attention_mask is not None:
        valid_mask = valid_mask * attention_mask[:, 1:].to(entropy_per_pos.dtype)

    denom = valid_mask.sum(dim=-1).clamp_min(1.0)
    mean_entropy = (entropy_per_pos * valid_mask).sum(dim=-1) / denom
    return mean_entropy


def validate_percentage(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return value


def resolve_entropy_band_bounds(
    mean_entropies: List[float] | np.ndarray | torch.Tensor,
    h_min: float,
    h_max: float,
) -> tuple[float, float]:
    h_min = validate_percentage("h_min", h_min)
    h_max = validate_percentage("h_max", h_max)
    if h_min > h_max:
        raise ValueError("h_min must be less than or equal to h_max.")

    if isinstance(mean_entropies, torch.Tensor):
        entropies = mean_entropies.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        entropies = np.asarray(mean_entropies, dtype=np.float64)

    if entropies.size == 0:
        raise ValueError("At least one mean entropy value is required.")

    return (
        float(np.quantile(entropies, h_min)),
        float(np.quantile(entropies, h_max)),
    )


def uncertainty_band_score(
    mean_entropy: torch.Tensor,
    H_min: float,
    H_max: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    H_mid = 0.5 * (H_min + H_max)
    half_width = 0.5 * (H_max - H_min)

    inside = (mean_entropy >= H_min) & (mean_entropy <= H_max)
    score_inside = 1.0 - torch.abs(mean_entropy - H_mid) / (half_width + eps)
    score = torch.where(inside, score_inside, torch.zeros_like(mean_entropy))
    return torch.clamp(score, min=0.0, max=1.0)


def score_pooled_candidates_by_mean_kl(
    model_name_or_path: str,
    current_checkpoint: Path,
    pooled_candidates: List[PooledCandidate],
    template_name: str,
    max_source_length: int,
    max_target_length: int,
    batch_size: int = 1,
) -> List[ScoredCandidate]:
    if batch_size <= 0:
        raise ValueError("KL batch size must be positive.")
    if not pooled_candidates:
        return []
    if not current_checkpoint.exists():
        raise FileNotFoundError(f"Current checkpoint does not exist: {current_checkpoint}")

    tokenizer = from_pretrained_with_safe_hf_cache(AutoTokenizer, model_name_or_path)
    template = get_template_and_fix_tokenizer(template_name, tokenizer)

    model = from_pretrained_with_safe_hf_cache(
        AutoModelForCausalLM,
        model_name_or_path,
        torch_dtype=get_model_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model = PeftModelForCausalLM.from_pretrained(
        model,
        str(current_checkpoint),
        adapter_name="current",
        is_trainable=False,
    )
    model.eval()

    device = model_input_device(model)
    candidates_by_task: Dict[str, List[PooledCandidate]] = {}
    source_checkpoints: Dict[str, Path] = {}
    for candidate in pooled_candidates:
        candidates_by_task.setdefault(candidate.source_task, []).append(candidate)
        source_checkpoints[candidate.source_task] = candidate.source_checkpoint

    scored_candidates: List[ScoredCandidate] = []
    current_checkpoint_resolved = current_checkpoint.resolve()

    try:
        for source_task, task_candidates in candidates_by_task.items():
            source_checkpoint = source_checkpoints[source_task].resolve()
            source_adapter_name = "current"
            if source_checkpoint != current_checkpoint_resolved:
                source_adapter_name = build_source_adapter_name(source_task)
                model.load_adapter(
                    str(source_checkpoint),
                    adapter_name=source_adapter_name,
                    is_trainable=False,
                )

            for start in range(0, len(task_candidates), batch_size):
                batch_candidates = task_candidates[start:start + batch_size]
                input_ids, target_mask, attention_mask = build_scoring_batch(
                    rows=[candidate.row for candidate in batch_candidates],
                    tokenizer=tokenizer,
                    template=template,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    device=device,
                )
                batch_scores = mean_distribution_shift(
                    model=model,
                    snapshot_adapter_name=source_adapter_name,
                    current_adapter_name="current",
                    input_ids=input_ids,
                    target_mask=target_mask,
                    attention_mask=attention_mask,
                )
                for candidate, score in zip(batch_candidates, batch_scores.tolist()):
                    scored_candidates.append(
                        ScoredCandidate(
                            candidate=candidate,
                            score=float(score),
                            mean_kl=float(score),
                        )
                    )

                del input_ids, target_mask, attention_mask, batch_scores

            cleanup_cuda_memory()
    finally:
        del model
        cleanup_cuda_memory()

    return sorted(scored_candidates, key=lambda candidate: candidate.score, reverse=True)


def score_pooled_candidates_by_uncertainty_band(
    model_name_or_path: str,
    current_checkpoint: Path,
    pooled_candidates: List[PooledCandidate],
    template_name: str,
    max_source_length: int,
    max_target_length: int,
    selection_mode: str,
    h_min: float,
    h_max: float,
    batch_size: int = 1,
) -> UncertaintyBandScoringResult:
    if batch_size <= 0:
        raise ValueError("Uncertainty batch size must be positive.")
    if not pooled_candidates:
        return UncertaintyBandScoringResult(
            scored_candidates=[],
            entropy_min=None,
            entropy_max=None,
            band_min_entropy=None,
            band_max_entropy=None,
            band_by_source_task={},
        )
    if not current_checkpoint.exists():
        raise FileNotFoundError(f"Current checkpoint does not exist: {current_checkpoint}")

    tokenizer = from_pretrained_with_safe_hf_cache(AutoTokenizer, model_name_or_path)
    template = get_template_and_fix_tokenizer(template_name, tokenizer)

    model = from_pretrained_with_safe_hf_cache(
        AutoModelForCausalLM,
        model_name_or_path,
        torch_dtype=get_model_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model = PeftModelForCausalLM.from_pretrained(
        model,
        str(current_checkpoint),
        adapter_name="current",
        is_trainable=False,
    )
    if hasattr(model, "set_adapter"):
        model.set_adapter("current")
    model.eval()

    device = model_input_device(model)
    candidate_entropies: List[CandidateEntropy] = []

    try:
        for start in range(0, len(pooled_candidates), batch_size):
            batch_candidates = pooled_candidates[start:start + batch_size]
            input_ids, target_mask, attention_mask = build_scoring_batch(
                rows=[candidate.row for candidate in batch_candidates],
                tokenizer=tokenizer,
                template=template,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
                device=device,
            )
            batch_entropies = mean_predictive_entropy(
                model=model,
                input_ids=input_ids,
                target_mask=target_mask,
                attention_mask=attention_mask,
            )
            for candidate, mean_entropy in zip(batch_candidates, batch_entropies.tolist()):
                candidate_entropies.append(
                    CandidateEntropy(
                        candidate=candidate,
                        mean_entropy=float(mean_entropy),
                    )
                )

            del input_ids, target_mask, attention_mask, batch_entropies

        cleanup_cuda_memory()
    finally:
        del model
        cleanup_cuda_memory()

    entropies_by_task: Dict[str, List[CandidateEntropy]] = {}
    for candidate_entropy in candidate_entropies:
        entropies_by_task.setdefault(candidate_entropy.candidate.source_task, []).append(candidate_entropy)

    all_mean_entropies = [candidate_entropy.mean_entropy for candidate_entropy in candidate_entropies]
    entropy_min = min(all_mean_entropies)
    entropy_max = max(all_mean_entropies)
    scored_candidates: List[ScoredCandidate] = []
    band_by_source_task: Dict[str, Dict[str, float]] = {}
    band_min_entropy: float | None = None
    band_max_entropy: float | None = None

    if selection_mode == "global":
        band_min_entropy, band_max_entropy = resolve_entropy_band_bounds(
            all_mean_entropies,
            h_min=h_min,
            h_max=h_max,
        )
        all_scores = uncertainty_band_score(
            torch.tensor(all_mean_entropies, dtype=torch.float32),
            H_min=band_min_entropy,
            H_max=band_max_entropy,
        ).tolist()
        for candidate_entropy, score in zip(candidate_entropies, all_scores):
            scored_candidates.append(
                ScoredCandidate(
                    candidate=candidate_entropy.candidate,
                    score=float(score),
                    mean_entropy=candidate_entropy.mean_entropy,
                )
            )
        for source_task, task_candidate_entropies in entropies_by_task.items():
            task_mean_entropies = [item.mean_entropy for item in task_candidate_entropies]
            band_by_source_task[source_task] = {
                "entropy_min": min(task_mean_entropies),
                "entropy_max": max(task_mean_entropies),
                "band_min_entropy": band_min_entropy,
                "band_max_entropy": band_max_entropy,
            }
    elif selection_mode in {"per_task_top_ratio", "per_task_top_count", "per_cluster"}:
        for source_task, task_candidate_entropies in entropies_by_task.items():
            task_mean_entropies = [item.mean_entropy for item in task_candidate_entropies]
            task_band_min_entropy, task_band_max_entropy = resolve_entropy_band_bounds(
                task_mean_entropies,
                h_min=h_min,
                h_max=h_max,
            )
            task_scores = uncertainty_band_score(
                torch.tensor(task_mean_entropies, dtype=torch.float32),
                H_min=task_band_min_entropy,
                H_max=task_band_max_entropy,
            ).tolist()
            band_by_source_task[source_task] = {
                "entropy_min": min(task_mean_entropies),
                "entropy_max": max(task_mean_entropies),
                "band_min_entropy": task_band_min_entropy,
                "band_max_entropy": task_band_max_entropy,
            }
            for candidate_entropy, score in zip(task_candidate_entropies, task_scores):
                scored_candidates.append(
                    ScoredCandidate(
                        candidate=candidate_entropy.candidate,
                        score=float(score),
                        mean_entropy=candidate_entropy.mean_entropy,
                    )
                )
    else:
        raise ValueError(
            f"Unknown uncertainty selection mode `{selection_mode}`. "
            "Available modes: global, per_task_top_ratio, per_task_top_count, per_cluster"
        )

    return UncertaintyBandScoringResult(
        scored_candidates=sorted(scored_candidates, key=lambda candidate: candidate.score, reverse=True),
        entropy_min=entropy_min,
        entropy_max=entropy_max,
        band_min_entropy=band_min_entropy,
        band_max_entropy=band_max_entropy,
        band_by_source_task=band_by_source_task,
    )


def pooled_embeddings(model_output) -> torch.Tensor:
    if getattr(model_output, "pooler_output", None) is not None:
        return model_output.pooler_output
    return model_output.last_hidden_state[:, 0]


def encode_texts(texts: List[str], model_name_or_path: str, batch_size: int = 32) -> np.ndarray:
    tokenizer = from_pretrained_with_safe_hf_cache(AutoTokenizer, model_name_or_path)
    model = from_pretrained_with_safe_hf_cache(AutoModel, model_name_or_path)
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


def allocate_cluster_samples(
    counts: np.ndarray,
    sample_size: int,
    total_count: int,
) -> np.ndarray:
    raw_allocations = counts * float(sample_size) / float(total_count)
    allocations = np.floor(raw_allocations).astype(int)
    remainder = sample_size - int(allocations.sum())
    if remainder > 0:
        order = np.argsort(-(raw_allocations - allocations))
        for idx in order[:remainder]:
            allocations[idx] += 1
    return allocations


def build_kmeans_selection_state(
    embeddings: np.ndarray,
    sample_size: int,
    n_cluster: int,
    seed: int,
) -> KMeansSelectionState | None:
    if len(embeddings) <= sample_size:
        return None

    normalized_embeddings = normalize_embeddings(embeddings)
    n_cluster = min(n_cluster, len(normalized_embeddings))
    labels, centers = fit_kmeans(normalized_embeddings, n_cluster=n_cluster, seed=seed)
    centroid_distances = np.linalg.norm(normalized_embeddings - centers[labels], axis=-1)
    counts = np.bincount(labels, minlength=n_cluster)
    allocations = allocate_cluster_samples(counts, sample_size, len(normalized_embeddings))
    return KMeansSelectionState(
        n_cluster=n_cluster,
        labels=labels,
        centers=centers,
        centroid_distances=centroid_distances,
        counts=counts,
        allocations=allocations,
    )


def order_indices_by_scores(
    indices: np.ndarray,
    scores: np.ndarray,
    descending: bool,
) -> np.ndarray:
    ordered = np.argsort(-scores[indices] if descending else scores[indices])
    return indices[ordered]


def select_indices_by_cluster_scores(
    state: KMeansSelectionState,
    ranking_scores: np.ndarray,
    sample_size: int,
    descending: bool,
    fallback_scores: np.ndarray | None = None,
    fallback_descending: bool | None = None,
) -> List[int]:
    ranking_scores = np.asarray(ranking_scores)
    if ranking_scores.shape[0] != state.labels.shape[0]:
        raise ValueError("Ranking scores must align with k-means labels.")

    selected_indices: List[int] = []
    for cluster_id in range(state.n_cluster):
        cluster_indices = np.where(state.labels == cluster_id)[0]
        if cluster_indices.size == 0 or state.allocations[cluster_id] == 0:
            continue
        ordered_indices = order_indices_by_scores(
            cluster_indices,
            ranking_scores,
            descending=descending,
        )
        selected_indices.extend(ordered_indices[:state.allocations[cluster_id]].tolist())

    if len(selected_indices) < sample_size:
        seen = set(selected_indices)
        fallback_scores = ranking_scores if fallback_scores is None else np.asarray(fallback_scores)
        fallback_descending = descending if fallback_descending is None else fallback_descending
        fallback_indices = order_indices_by_scores(
            np.arange(len(state.labels)),
            fallback_scores,
            descending=fallback_descending,
        )
        for idx in fallback_indices:
            if int(idx) in seen:
                continue
            selected_indices.append(int(idx))
            if len(selected_indices) >= sample_size:
                break

    return selected_indices[:sample_size]


def select_by_kmeans_score_indices(
    rows: List[Dict[str, str]],
    ranking_scores: np.ndarray,
    encoder_model_name_or_path: str,
    sample_size: int,
    seed: int,
    n_cluster: int = KMEANS_SELECTOR_N_CLUSTER,
    descending: bool = True,
    fallback_scores: np.ndarray | None = None,
    fallback_descending: bool | None = None,
) -> tuple[List[int], KMeansSelectionState | None]:
    if len(rows) <= sample_size:
        return list(range(len(rows))), None

    texts = [row["inputs"] for row in rows]
    embeddings = encode_texts(texts, encoder_model_name_or_path)
    state = build_kmeans_selection_state(
        embeddings=embeddings,
        sample_size=sample_size,
        n_cluster=n_cluster,
        seed=seed,
    )
    if state is None:
        return list(range(len(rows))), None

    selected_indices = select_indices_by_cluster_scores(
        state=state,
        ranking_scores=ranking_scores,
        sample_size=sample_size,
        descending=descending,
        fallback_scores=fallback_scores,
        fallback_descending=fallback_descending,
    )
    return selected_indices, state


def select_by_kmeans_indices(
    embeddings: np.ndarray,
    sample_size: int,
    n_cluster: int,
    seed: int
) -> List[int]:
    if len(embeddings) <= sample_size:
        return list(range(len(embeddings)))

    state = build_kmeans_selection_state(
        embeddings=embeddings,
        sample_size=sample_size,
        n_cluster=n_cluster,
        seed=seed,
    )
    if state is None:
        return list(range(len(embeddings)))

    return select_indices_by_cluster_scores(
        state=state,
        ranking_scores=state.centroid_distances,
        sample_size=sample_size,
        descending=False,
    )


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
                "n_cluster": min(KMEANS_SELECTOR_N_CLUSTER, len(rows)),
            },
        )

    texts = [row["inputs"] for row in context.candidate_rows]
    embeddings = encode_texts(texts, context.encoder_model_name_or_path)
    selected_indices = select_by_kmeans_indices(
        embeddings=embeddings,
        sample_size=context.sample_memory,
        n_cluster=KMEANS_SELECTOR_N_CLUSTER,
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
            "n_cluster": min(KMEANS_SELECTOR_N_CLUSTER, len(context.candidate_rows)),
        },
    )
