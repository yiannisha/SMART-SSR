# SMART-SSR-CL Guide

This workspace contains the SSR research code at the repository root.
It is a fork/extension of LLaMA-Efficient-Tuning (LLaMA-Factory style) for continual learning experiments from the ACL 2024 SSR paper.

## 1) What this codebase is

Core training runner:
- `src/train_bash.py` -> calls `llmtuner.run_exp()`
- `src/llmtuner/tuner/tune.py` dispatches by `--stage`

Supported stages in this fork:
- `pt` (pretraining)
- `sft` (supervised fine-tuning)
- `sftrp` (SFT-style predict/eval used heavily for CL evaluation)
- `sftreg` (regularization CL: EWC/L2)
- `rm`, `ppo`, `dpo` (inherited from upstream framework)

SSR-specific additions live in:
- `custom/icl_gen/` (synthetic instance generation + labeling)
- `custom/niv2-c012/` (SuperNI filtering/splitting/KMeans subset creation)
- `src/scripts-ni-c012/` (original experiment shell scripts)
- `scripts/run_basic_ssr_5task.py` and `scripts/run_basic_ssr_5task.sh` (working 5-task SSR runner in this workspace)
- `scripts/run_paper_ssr_5task.py` (paper-style artifact pipeline)
- `scripts/selection_methods.py` (selector registry for replay experiments)
- `scripts/run_selection_proxy_iteration.py` (single-iteration selector replay)

## 2) What experiments are supported

### A. Continual learning on SuperNI categories (`ni-cus0.12`)
Task categories used in scripts:
- 10-task setup: `qa qg sa sum trans dsg expl para pe pos`
- 5-task setup: first five only (`qa qg sa sum trans`)

Experiment families (main paper-style):
- Single-task start (stage-1 CL init): `lora/sing/...`
- Non-rehearsal CL: `lora/cl...` with `cl_queue`
- RandSel rehearsal: scripts with `_rp`
- KMeansSel rehearsal: scripts with `_km20_rp`
- SSR rehearsal (self-synthesized + refined): scripts with `_iclgen_self`
- Queue variants: `cl`, `cl2`, `cl3` (different task orders / queues)

Representative script paths:
- Non-rehearsal: `src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue.3ep.bs32x1x1.lr2e-04.bf16.sh`
- RandSel: `src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_rp.3ep.bs32x1x1.lr2e-04.bf16.sh`
- KMeansSel: `src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_km20_rp.3ep.bs32x1x1.lr2e-04.bf16.sh`
- SSR: `src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh`

### B. Multi-task (joint) baselines
- `lora/all/...all...sh` (all categories)
- `lora/all/...all_5...sh` (5-category subset)

### C. Regularization-based CL baselines
- `--stage sftreg` with `--reg_cl_method ewc|l2`
- Example scripts in:
  - `src/scripts-ni-c012/lora/cl/llama2-7b-reg/`

### D. Synthetic data generation pipeline (SSR data creation)
Pipeline in README and custom scripts:
1. Generate ICL synthetic instances:
   - `custom/icl_gen/complete_param_nic010_cate.py`
2. Parse/filter generated outputs:
   - `custom/icl_gen/parse_filter_generated.py` (or alpaca variant scripts)
3. Refine synthetic outputs with current checkpoint:
   - `custom/icl_gen/label_param.py`
4. Select rehearsal subset from the refined outputs:
   - Legacy random selector: `custom/icl_gen/random_select.py`
   - Legacy KMeans selector: `custom/icl_gen/kmeans_self.py`
   - Replay/dev selector path: `scripts/selection_methods.py` plus `scripts/run_selection_proxy_iteration.py`

The maintained paper-style runner shares only `raw` and `parsed` artifacts across runs. Checkpoint-dependent `refined` and compatibility `cl_queue` artifacts are run-local, so selector comparisons do not silently reuse another run's refined data.

### E. Additional evaluation workflows
- CL eval in main scripts uses `--stage sftrp` and writes:
  - `all_results.json`
  - `generated_predictions.jsonl`
- MMLU:
  - `mmlu_test/evaluate_causal.py`
  - demo: `mmlu_test/mmlu_demo.sh`
- AlpacaEval helper:
  - `custom/alpaca_eval/alpaca_demo.sh`

## 3) Before running: critical path fixes

Most provided `.sh` scripts are hardcoded to old absolute paths like `/home/hjh/data/public/SSR`.
You must edit at least these variables in each script you use:
- `REPO_ROOT_DIR`
- `SRC_DIR`
- `MODEL_DIR`
- any hardcoded data output paths

For this workspace, prefer the maintained wrapper instead of patching every historical script:
- `bash scripts/run_basic_ssr_5task.sh ...`
- `python scripts/run_paper_ssr_5task.py ...`
- `python scripts/run_selection_proxy_iteration.py ...` for selector replay against existing artifacts

The working wrapper already handles:
- sourcing `keys.sh` if present
- bootstrapping `.venv` through `scripts/bootstrap_env.sh`
- training the 5-task queue `qa qg sa sum trans`
- building synthetic rehearsal files for each completed task
- evaluating each checkpoint on all tasks seen so far

Compatibility fixes already applied in this repo for the maintained path:
- lazy imports in `src/llmtuner/__init__.py` so Python 3.12 can train without importing the old FastAPI stack
- local JSON dataset loading fix in `src/llmtuner/dsets/loader.py`
- `accelerate<1.0.0` pin in `requirements.txt` for compatibility with `transformers==4.36.0`
- CUDA 12.1 torch bootstrap in `scripts/bootstrap_env.sh` so this workspace uses a driver-compatible GPU build

## 4) Environment setup

```bash
cd /workspace/SMART-SSR
python -m venv .venv
source .venv/bin/activate
source keys.sh
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Notes:
- Default scripts use `--bf16 True`; you need BF16-capable GPU.
- If BF16 unsupported, switch scripts/commands to `--fp16 True` and remove `--bf16 True`.

## 5) Fastest working end-to-end run in this workspace

```bash
cd /workspace/SMART-SSR
source .venv/bin/activate
source keys.sh

bash scripts/run_basic_ssr_5task.sh \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_root saves/basic-ssr-5task \
  --train_max_steps 2 \
  --train_max_samples 10 \
  --eval_max_samples 2 \
  --prompts_per_group 2 \
  --max_candidates 5 \
  --synthesis_max_new_tokens 64 \
  --refine_max_new_tokens 64
```

Outputs:
- checkpoints: `saves/basic-ssr-5task/01_qa` ... `saves/basic-ssr-5task/05_trans`
- final summary: `saves/basic-ssr-5task/run_summary.json`
- rehearsal files for the default TinyLlama example: `data/ni-cus0.12/genearated-icl-naive-kmeans20-self/tinyllama/cl_queue/*.json`

To resume after interruption:

```bash
bash scripts/run_basic_ssr_5task.sh \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_root saves/basic-ssr-5task \
  --train_max_steps 2 \
  --train_max_samples 10 \
  --eval_max_samples 2 \
  --prompts_per_group 2 \
  --max_candidates 5 \
  --synthesis_max_new_tokens 64 \
  --refine_max_new_tokens 64 \
  --skip_completed
```

### Paper-style full run with selector control

Use `scripts/run_paper_ssr_5task.py` when you want the maintained artifact pipeline (`raw -> parsed -> refined`) plus selector-controlled rehearsal for training.

Example full run for `llama2-7b-chat` with pooled KL selection:

```bash
cd /workspace/SMART-SSR
source .venv/bin/activate
source keys.sh

python scripts/run_paper_ssr_5task.py \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --model_family llama2-7b-chat \
  --selector mean_kl \
  --candidate_source refined \
  --output_root saves/paper-ssr-5task-llama2-7b-chat-mean-kl \
  --cuda 0
```

The shortest paper-style command for this repo is now:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh
```

Use the same launcher with extra flags to change selector or resume:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh \
  --selector mean_kl \
  --candidate_source refined \
  --skip_completed
```

Outputs:
- checkpoints: `<output_root>/01_qa` ... `<output_root>/05_trans`
- per-stage dataset registries and selected rehearsal files: `<output_root>/stage_data/<stage>/`
- run-local checkpoint-dependent artifacts: `<output_root>/artifacts/refined/` and `<output_root>/artifacts/cl_queue/`
- run summary: `<output_root>/run_summary.json`
- aggregate metrics: `<output_root>/aggregate_metrics.json`
- shared reusable generation artifacts: `data/ni-cus0.12/genearated-icl-naive/{model_family}/ori-van/*.json` and `data/ni-cus0.12/genearated-icl-naive-parsed-filtered/{model_family}/ori-van/*.json`

### Selector development / single-iteration replay

Use `scripts/run_selection_proxy_iteration.py` when you already have a base SSR run with checkpoints, generated artifacts and `run_summary.json`, and you want to test a different selector on one intermediate stage without regenerating all data.

Built-in selectors exposed by the maintained runners:
- `head`
- `random`
- `kmeans`
- `mean_kl`
- `uncertainty_band`

Selector context includes:
- source task and target task
- source task index and source checkpoint
- checkpoint history before the replay target stage
- candidate artifact path plus normalized candidate rows

Example:

```bash
cd /workspace/SMART-SSR
source .venv/bin/activate

python scripts/run_selection_proxy_iteration.py \
  --base_run saves/paper-ssr-5task-tinyllama \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --template llama2 \
  --iteration_task qg \
  --selector random \
  --candidate_source refined \
  --output_root saves/selection-proxy-tinyllama-qg-random \
  --train_max_steps 50 \
  --eval_max_samples 50
```

Replay outputs:
- selected rehearsal subsets in `<output_root>/proxy_data/rehearsal/*.jsonl`
- experiment-local dataset registry in `<output_root>/proxy_data/dataset_info.json`
- replayed checkpoint and eval outputs under `<output_root>/<stage_name>/`
- `<output_root>/run_summary.json`
- `<output_root>/proxy_metrics.json`

Notes:
- `--candidate_source` accepts `raw`, `parsed`, `refined` or `final`.
- `--skip_train --skip_eval` materializes only the selected rehearsal subsets.
- The replay runner uses an experiment-local `dataset_info.json` and does not modify `data/dataset_info.json`.
- `mean_kl` supports `--mean_kl_selection_mode global`, `--mean_kl_selection_mode per_task_top_ratio`, and `--mean_kl_selection_mode per_task_top_count`.
- `uncertainty_band` supports `--uncertainty_selection_mode global`, `--uncertainty_selection_mode per_task_top_ratio`, and `--uncertainty_selection_mode per_task_top_count`.
- `--h_min` and `--h_max` are percentile cutoffs in `[0.0, 1.0]`, not raw entropy values. `0.25` / `0.75` means the middle 50% entropy band of the scored candidate set, or of each source task in `per_task_top_ratio` / `per_task_top_count` mode.
- `--per_task_selection_count` applies a fixed cap per prior task when you use a `per_task_top_count` mode. This is the fair-budget option if you want pooled selectors to match the legacy 200-per-task rehearsal budget.
- Use the replay path for fast selector iteration; the full paper-style runner now accepts `--selector` and writes stage-local training inputs under `<output_root>/stage_data/<stage>/`.
- If you rerun the same paper-style `output_root` without `--skip_completed`, the runner refreshes `refined` and `final` for any retrained stage.

## 6) Minimal way to run core CL experiments manually

### Step 1: train the first task checkpoint (required)

```bash
cd /workspace/SMART-SSR
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sft \
  --model_name_or_path /path/to/base/model \
  --do_train True \
  --overwrite_cache True \
  --finetuning_type lora \
  --template llama2 \
  --dataset_dir data \
  --dataset ni_c012_qa_train \
  --max_source_length 1024 \
  --max_target_length 512 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --lora_rank 8 \
  --lora_dropout 0.1 \
  --lora_target q_proj,v_proj \
  --output_dir saves/ni-c012/LLAMA2-7B-Chat/lora/qa/bs32x1x1-3ep-bf16 \
  --plot_loss True \
  --bf16 True
```

### Step 2: run a CL method for next task (`qg` example)

Use previous checkpoint from Step 1 via `--checkpoint_dir`.

Non-rehearsal:
```bash
--dataset ni_c012_qg_train
```

RandSel rehearsal:
```bash
--dataset ni_c012_qg_train,ni_c012_qa_train_smp01
```

KMeansSel rehearsal:
```bash
--dataset ni_c012_qg_train,ni_c012_qa_train_km20_smp01
```

SSR rehearsal:
```bash
--dataset ni_c012_qg_train,ni_c012_icl_gen_km20_self_cl_queue_llama2_7b_chat_qa
```

All of the above use the same `train_bash.py` pattern as Step 1, but add:
```bash
--checkpoint_dir <previous_task_ckpt>
```

### Step 3: evaluate each checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sftrp \
  --model_name_or_path /path/to/base/model \
  --checkpoint_dir <ckpt_dir> \
  --overwrite_cache True \
  --predict_with_generate True \
  --finetuning_type lora \
  --template llama2 \
  --dataset_dir data \
  --dataset ni_c012_qg_eval \
  --max_source_length 1024 \
  --max_target_length 512 \
  --per_device_eval_batch_size 1 \
  --output_dir <ckpt_dir>/ni_c012_qg_eval \
  --do_predict True \
  --do_sample False \
  --bf16 True
```

Metrics are saved in:
- `<output_dir>/all_results.json`
- `<output_dir>/generated_predictions.jsonl`

## 7) Running the provided historical scripts

After editing path variables, run directly:

- Single-task init:
```bash
bash src/scripts-ni-c012/lora/sing/llama2-7b-chat/llama2-7b-chat.lora.single.3ep.bs32x1x1.bf16.sh qa 0
```

- Non-rehearsal CL sweep:
```bash
bash src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue.3ep.bs32x1x1.lr2e-04.bf16.sh 0
```

- RandSel CL sweep:
```bash
bash src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_rp.3ep.bs32x1x1.lr2e-04.bf16.sh 0 01
```

- KMeansSel CL sweep:
```bash
bash src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_km20_rp.3ep.bs32x1x1.lr2e-04.bf16.sh 0 01
```

- SSR CL sweep:
```bash
bash src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh 0
```

- Queue variants:
```bash
bash src/scripts-ni-c012/lora/cl2/llama2-7b-chat/llama2-7b-chat.lora.cl_queue2_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh 0
bash src/scripts-ni-c012/lora/cl3/llama2-7b-chat/llama2-7b-chat.lora.cl_queue3_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh 0
```

These scripts are not the fastest path to a verified working run in this workspace; use `scripts/run_basic_ssr_5task.sh` if you want the maintained path.

## 8) How to run SSR synthetic data generation

Example flow (llama2-chat family):

1. Generate synthetic instances:
```bash
bash custom/icl_gen/scripts-ni-c012/llama2-7b-chat/ori-van.sh 0 "qa qg sa sum trans" 2 3 1.2
```

2. Parse/filter generated files:
- adapt and run `custom/icl_gen/parse_filter_generated.py`

3. Refine synthetic outputs using current model:
```bash
bash custom/icl_gen/scripts-ni-c012/llama2-7b-chat/label-self.sh qa 0 "qg sa sum trans"
```

4. Select subset from the refined outputs:
- random: `custom/icl_gen/random_select.py`
- kmeans: compute embeddings first, then `custom/icl_gen/kmeans_self.py`
- selector replay/dev path: `scripts/selection_methods.py` with `scripts/run_selection_proxy_iteration.py`

Important:
- These scripts also contain hardcoded paths and may need edits before use.
- Dataset keys referenced by CL scripts are pre-registered in `data/dataset_info.json`.

## 9) Data preprocessing from raw NI (if rebuilding dataset)

The `ni-cus0.12` split files are already present. If you need to rebuild:

1. Length filter raw NI task JSON:
- `custom/niv2-c012/1_length_fiiter.py`
2. Train/eval/extra split + sampled subsets:
- `custom/niv2-c012/2_split_and_random_selection.py`
3. (Optional) embedding + KMeans subset construction:
- `custom/niv2-c012/text2emb.py`
- `custom/niv2-c012/kmeans_selection.py`

## 10) Useful outputs to inspect

- Training logs/metrics per run: inside each `--output_dir`
- Evaluation metrics: `<output_dir>/all_results.json`
- Predictions: `<output_dir>/generated_predictions.jsonl`

## 11) Common gotchas

- Hardcoded paths in many research scripts are the #1 failure point.
- `dataset_info.json` key names must match `--dataset` exactly.
- The replay runner writes its own dataset registry under `<output_root>/proxy_data/dataset_info.json`; do not edit the global registry just to test a selector.
- `--predict_with_generate True` is required for `sftrp` prediction output.
- If output directory already exists and is non-empty, set unique `--output_dir` or `--overwrite_output_dir` as needed.
