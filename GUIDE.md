# SMART-SSR Fresh Setup Guide

This guide is for the repository in its current state and focuses on the maintained path for a full 5-task SSR run with `meta-llama/Llama-2-7b-chat-hf` and the `hybrid_cluster` selector.

Use the maintained runners in `scripts/`. Do not start from the historical shell scripts under `src/scripts-ni-c012/`; many of those still assume old absolute paths.

## What this guide covers

- fresh environment setup from the repo root
- gated Hugging Face access for `meta-llama/Llama-2-7b-chat-hf`
- the maintained paper-style 5-task SSR pipeline
- hybrid rehearsal selection with explicit weight control
- resume behavior and where outputs land

## Repo assumptions

The maintained paper runner uses the default 5-task queue:

```text
qa qg sa sum trans
```

For each stage it does the following:

1. generate raw synthetic candidates
2. parse and filter them
3. train the current task, with rehearsal after stage 1
4. refine that task's synthetic outputs with the current checkpoint
5. write a compatibility k-means rehearsal artifact
6. evaluate the current checkpoint on all tasks

Important details from the current code:

- `scripts/run_paper_ssr_5task_llama2_7b_chat.sh` is the simplest entrypoint for `llama2-7b-chat`
- that wrapper auto-sources `keys.sh` if present
- it auto-runs `scripts/bootstrap_env.sh`
- the bootstrap creates `.venv`, installs a CUDA 12.1 PyTorch build, installs `requirements.txt`, and installs the repo editable
- training and evaluation currently pass `--bf16 True`, so the maintained path assumes a BF16-capable GPU

## 1. Fresh setup

Start from the repository root:

```bash
cd /path/to/SMART-SSR
```

### 1.1 Optional: provide a Hugging Face token

`meta-llama/Llama-2-7b-chat-hf` is gated. You need approval for that model on Hugging Face.

The maintained shell wrapper looks for `keys.sh` and maps `HUGGING_FACE_ACCESS_TOKEN` into `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`.

Minimal `keys.sh`:

```bash
export HUGGING_FACE_ACCESS_TOKEN=hf_xxx
```

If you do not want a `keys.sh`, exporting the token in your shell is also fine:

```bash
export HUGGING_FACE_ACCESS_TOKEN=hf_xxx
```

### 1.2 Bootstrap the environment

```bash
bash scripts/bootstrap_env.sh
source .venv/bin/activate
```

What this does now:

- creates `.venv` if it does not exist
- installs `torch==2.4.1` from the CUDA 12.1 index
- installs the repo dependencies
- installs the package with `pip install -e .`

### 1.3 Optional sanity checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python -c "import transformers, peft, trl; print(transformers.__version__)"
```

If `torch.cuda.is_bf16_supported()` prints `False`, the maintained paper runner will need code changes before it can run on that machine because the train/eval commands are hardcoded to BF16.

## 2. The maintained 5-task runner

The entrypoints you should use are:

- shell wrapper: `scripts/run_paper_ssr_5task_llama2_7b_chat.sh`
- Python driver: `scripts/run_paper_ssr_5task.py`

The wrapper expands to:

```bash
bash scripts/run_paper_ssr_5task.sh \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --model_family llama2-7b-chat \
  --output_root saves/paper-ssr-5task-llama2-7b-chat \
  ...
```

For a new experiment, always give a fresh `--output_root`.

## 3. Full 5-task SSR run with `hybrid_cluster`

### 3.1 Recommended command

This is the cleanest full run command for the current repo:

```bash
cd /path/to/SMART-SSR

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh \
  --cuda 0 \
  --selector hybrid_cluster \
  --candidate_source refined \
  --sample_memory 200 \
  --hybrid_cluster_diversity_weight 1.0 \
  --hybrid_cluster_mean_kl_weight 1.0 \
  --hybrid_cluster_uncertainty_weight 1.0 \
  --output_root saves/paper-ssr-5task-llama2-7b-chat-hybrid-cluster-eq
```

That runs the full default task order:

```text
qa -> qg -> sa -> sum -> trans
```

### 3.2 Example with custom weights

If you want the weighted mix used in this repo's saved experiments:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh \
  --cuda 0 \
  --selector hybrid_cluster \
  --candidate_source refined \
  --sample_memory 200 \
  --hybrid_cluster_diversity_weight 2.0 \
  --hybrid_cluster_mean_kl_weight 6.0 \
  --hybrid_cluster_uncertainty_weight 2.0 \
  --output_root saves/paper-ssr-5task-llama2-7b-chat-hybrid-cluster-2-6-2
```

Another valid variant:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh \
  --cuda 0 \
  --selector hybrid_cluster \
  --candidate_source refined \
  --sample_memory 200 \
  --hybrid_cluster_diversity_weight 2.0 \
  --hybrid_cluster_mean_kl_weight 2.0 \
  --hybrid_cluster_uncertainty_weight 6.0 \
  --output_root saves/paper-ssr-5task-llama2-7b-chat-hybrid-cluster-2-2-6
```

## 4. How the hybrid selector works in this repo

`hybrid_cluster` is a pooled selector with a fixed selection mode:

```text
--hybrid_cluster_selection_mode per_cluster
```

You do not need to pass that flag manually because it is the only supported mode for `hybrid_cluster`.

The current implementation:

- clusters each prior task's candidate pool with the maintained k-means path
- computes three signals for each candidate:
  - diversity
  - `mean_kl`
  - `uncertainty_band_score`
- min-max normalizes those signals within each cluster
- ranks candidates by the weighted sum of those normalized scores
- keeps the same per-cluster quota structure as the maintained k-means selection path

The weight flags are:

- `--hybrid_cluster_diversity_weight`
- `--hybrid_cluster_mean_kl_weight`
- `--hybrid_cluster_uncertainty_weight`

### Budget semantics

For `hybrid_cluster`, `sample_memory` is the important budget flag.

With:

```bash
--sample_memory 200
```

the runner selects up to 200 rehearsal examples per previous task at each stage, subject to candidate availability. Because `hybrid_cluster` is fixed to `per_cluster`, these flags do not apply here:

- `--global_selection_count`
- `--global_selection_ratio`
- `--per_task_selection_ratio`
- `--per_task_selection_count`

### Candidate source

For `hybrid_cluster`, use:

```bash
--candidate_source refined
```

That makes the selector score the checkpoint-refined candidate pool. In the current runner, `hybrid_cluster` defaults to `refined` if `--candidate_source` is omitted, but it is better to pass it explicitly.

## 5. Outputs to expect

For an output root like:

```text
saves/paper-ssr-5task-llama2-7b-chat-hybrid-cluster-2-6-2
```

you should expect:

- checkpoints:
  - `.../01_qa`
  - `.../02_qg`
  - `.../03_sa`
  - `.../04_sum`
  - `.../05_trans`
- stage-local dataset registries:
  - `.../stage_data/01_qa/dataset_info.json`
  - `.../stage_data/02_qg/dataset_info.json`
  - etc.
- selected rehearsal subsets per stage:
  - `.../stage_data/02_qg/rehearsal/*.jsonl`
  - `.../stage_data/03_sa/rehearsal/*.jsonl`
  - etc.
- ranked hybrid scores per stage:
  - `.../stage_data/02_qg/hybrid_cluster_scores.refined.jsonl`
  - `.../stage_data/03_sa/hybrid_cluster_scores.refined.jsonl`
  - etc.
- run-local refined artifacts:
  - `.../artifacts/refined/*.json`
- run-local compatibility k-means artifacts:
  - `.../artifacts/cl_queue/*.json`
- run summaries:
  - `.../run_summary.json`
  - `.../aggregate_metrics.json`

Shared raw and parsed generation artifacts are stored under:

- `data/ni-cus0.12/genearated-icl-naive/llama2-7b-chat/ori-van/`
- `data/ni-cus0.12/genearated-icl-naive-parsed-filtered/llama2-7b-chat/ori-van/`

Refined and final artifacts are run-local under the selected `output_root`.

## 6. Resume an interrupted run

Use the exact same command and add `--skip_completed`:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh \
  --cuda 0 \
  --selector hybrid_cluster \
  --candidate_source refined \
  --sample_memory 200 \
  --hybrid_cluster_diversity_weight 2.0 \
  --hybrid_cluster_mean_kl_weight 6.0 \
  --hybrid_cluster_uncertainty_weight 2.0 \
  --output_root saves/paper-ssr-5task-llama2-7b-chat-hybrid-cluster-2-6-2 \
  --skip_completed
```

`--skip_completed` skips stages whose checkpoint directory already exists.

## 7. Optional: direct Python invocation

If you want the same run without the wrapper:

```bash
source .venv/bin/activate

python scripts/run_paper_ssr_5task.py \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --model_family llama2-7b-chat \
  --cuda 0 \
  --selector hybrid_cluster \
  --candidate_source refined \
  --sample_memory 200 \
  --hybrid_cluster_diversity_weight 2.0 \
  --hybrid_cluster_mean_kl_weight 6.0 \
  --hybrid_cluster_uncertainty_weight 2.0 \
  --output_root saves/paper-ssr-5task-llama2-7b-chat-hybrid-cluster-2-6-2
```

The shell wrapper is still the safer default because it handles token loading and environment bootstrap.

## 8. Optional data-prep note

The paper-style runner does not require `PREPARE_EXPERIMENT_DATA=1` for the normal 5-task hybrid run.

That switch exists only if you want the wrapper to regenerate the `split-kmeans20` helper subsets through `scripts/prepare_experiment_data.py`.

## 9. Troubleshooting

### Hugging Face permission error

If the run fails while resolving `meta-llama/Llama-2-7b-chat-hf`:

- confirm the account has accepted the model license
- confirm `HUGGING_FACE_ACCESS_TOKEN` is exported before starting the wrapper

### BF16 / GPU error

The maintained runner currently uses BF16 in train and eval commands. If the machine cannot run BF16, the maintained path will fail until those calls are patched.

### Reusing an old output root

If you reuse an existing `--output_root` without `--skip_completed`, the runner may retrain stages and refresh run-local `refined` and `final` artifacts. Use a new output root for clean experiments.

### Historical scripts vs maintained runner

If a script under `src/scripts-ni-c012/` references old absolute paths, ignore it for this workflow. The maintained path for this repo is the `scripts/run_paper_ssr_5task*.sh` and `scripts/run_paper_ssr_5task.py` stack.
