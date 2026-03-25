# Self-Synthesized Rehearsal (SSR)

🎉 Welcome to the repository for "Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal" (ACL2024, [📃arXiv Paper](https://arxiv.org/abs/2403.01244)).

![](./framework.png)

## 🧱 Codebase Structure

This codebase is built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework.

To get started with SSR, please refer to the following directory tree structure of the codebase:

```shell
├── custom
│   ├── alpaca_eval
│   ├── icl_gen                     # Instance synthesis
│   └── niv2-c012                   # SuperNI data preprocessing
├── data                            # datasets
├── mmlu_test
├── scripts                         # maintained runners, metrics and replay tools
├── saves
└── src
    ├── llmtuner
    └── scripts-ni-c012             # Examples of run scripts
```

## 📲 Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

If you need gated Hugging Face access, source `keys.sh` first:

```bash
source keys.sh
```

The maintained shell wrappers now bootstrap `.venv` through `scripts/bootstrap_env.sh`.
That bootstrap pins a CUDA 12.1 PyTorch build (`torch==2.4.1`) so the repo runs cleanly on the current A100/driver setup in this workspace.

## 🚀 Working 5-Task SSR Quickstart

The original paper scripts under `src/scripts-ni-c012/` are useful references, but many of them assume old absolute paths and prebuilt synthetic rehearsal files.

This repository now includes maintained runners for the main SSR workflows in this workspace:

- tasks: `qa qg sa sum trans`
- entrypoint: `scripts/run_basic_ssr_5task.sh`
- Python driver: `scripts/run_basic_ssr_5task.py`
- paper-style artifact pipeline: `scripts/run_paper_ssr_5task.py`
- synthetic rehearsal builder: `custom/icl_gen/build_ssr_dataset.py`
- selector registry and scoring helpers: `scripts/selection_methods.py`
- single-iteration selector replay: `scripts/run_selection_proxy_iteration.py`

Example:

```bash
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

Notes:
- The default/tested base model for the working quickstart is `TinyLlama/TinyLlama-1.1B-Chat-v1.0` because it is fast enough to validate the full pipeline in one workspace session.
- The generated rehearsal JSON files are written under `data/ni-cus0.12/genearated-icl-naive-kmeans20-self/<model_family>/cl_queue/`. For the default quickstart, that means `data/ni-cus0.12/genearated-icl-naive-kmeans20-self/tinyllama/cl_queue/`.
- Final checkpoints and eval outputs are written under `saves/basic-ssr-5task/`.
- To resume after an interruption, rerun the same command and add `--skip_completed`.

## 🔬 Selector Development And Replay

For fast selection-method iteration, use `scripts/run_selection_proxy_iteration.py` against an existing SSR run that already has:

- stage checkpoints
- generated artifacts (`raw`, `parsed`, `refined`, `final`)
- `run_summary.json`

Built-in selectors exposed by the maintained runners:

- `head`
- `random`
- `kmeans`
- `mean_kl`
- `uncertainty_band`

Each selector receives a `SelectionContext` with:

- source and target task names
- source task index and source checkpoint
- full checkpoint history before the replayed target stage
- candidate artifact path and normalized candidate rows
- sample budget, seed, working directory and encoder config

Example replay:

```bash
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

Full paper-style run with `mean_kl`:

```bash
source .venv/bin/activate

python scripts/run_paper_ssr_5task.py \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --model_family llama2-7b-chat \
  --selector mean_kl \
  --candidate_source refined \
  --output_root saves/paper-ssr-5task-llama2-7b-chat-mean-kl \
  --cuda 0
```

Default paper-style `llama2-7b-chat` launcher:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh
```

Override the selector or resume an interrupted run:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh \
  --selector mean_kl \
  --candidate_source refined \
  --skip_completed
```

Replay outputs:

- selected rehearsal files under `<output_root>/proxy_data/rehearsal/`
- an experiment-local dataset registry at `<output_root>/proxy_data/dataset_info.json`
- the replayed stage checkpoint and eval outputs under `<output_root>/<stage_name>/`
- `run_summary.json`
- `proxy_metrics.json` with base-vs-replay Rouge-L deltas

Notes:

- `--candidate_source` can be `raw`, `parsed`, `refined` or `final`.
- `--skip_train --skip_eval` builds only the selected rehearsal subsets.
- The replay runner does not modify `data/dataset_info.json`.
- `mean_kl` supports `--mean_kl_selection_mode global`, `--mean_kl_selection_mode per_task_top_ratio`, and `--mean_kl_selection_mode per_task_top_count`.
- `uncertainty_band` supports `--uncertainty_selection_mode global`, `--uncertainty_selection_mode per_task_top_ratio`, and `--uncertainty_selection_mode per_task_top_count`.
- `--h_min` and `--h_max` are percentile cutoffs in `[0.0, 1.0]`, not raw entropy values. `0.25` / `0.75` means the middle 50% entropy band of the scored candidate set, or of each source task in `per_task_top_ratio` mode.
- `--per_task_selection_count` applies a fixed cap per prior task when you use a `per_task_top_count` mode. This is the fair-budget option if you want pooled selectors to match the legacy 200-per-task rehearsal budget.
- The maintained paper-style runner now accepts `--selector` and writes stage-local dataset registries under `<output_root>/stage_data/<stage>/dataset_info.json`.
- The paper-style runner shares only `raw` and `parsed` artifacts across runs. Checkpoint-dependent `refined` and compatibility `final/cl_queue` artifacts are written under `<output_root>/artifacts/`.
- If you rerun the same `output_root` without `--skip_completed`, the runner refreshes `refined` and `final` for any retrained stage.

## 🛢 Pipeline

### Step 1: In-Context Learning Based Instance Synthesis

1. use `custom/icl_gen/complete_param_nic010_cate.py` for generation 
2. use `custom/icl_gen/parse_filter_generated.py` for post-processing and filtering

### Step 2: Synthetic Output Refinement

1. use `custom/icl_gen/label_param.py` (integrated in the following scripts)

### Step 3: Rehearsal with Selected Synthetic Instances

1. legacy selection scripts are still available in `custom/icl_gen/random_select.py` and `custom/icl_gen/select_kmeans_examples.py`
2. selector development now lives in `scripts/selection_methods.py`
3. fast proxy evaluation of a selector against an existing run lives in `scripts/run_selection_proxy_iteration.py`
4. the maintained paper-style runner keeps the artifact order `synthesis -> refinement -> compatibility kmeans artifact`, but training now uses stage-local selector outputs built from the configured `--candidate_source`

- **multi-task learning (MTL)**: `src/scripts-ni-c012/lora/all/[model_name]/[model_name].lora.[all|all_5].3ep.bs32x1x1.bf16.sh`
- **single task (& Stage 1 in continual learning)**: `src/scripts-ni-c012/lora/sing/[model_name]/[model_name].lora.single.3ep.bs32x1x1.bf16.sh`
- **Non-rehearsal**: `src/scripts-ni-c012/lora/[cl|cl2|cl3]/[model_name]/[model_name].lora.[cl_queue|cl_queue2|cl_queue3].3ep.bs32x1x1.lr2e-04.bf16.sh`
- **RandSel**: `src/scripts-ni-c012/lora/[cl|cl2|cl3]/[model_name]/[model_name].lora.[cl_queue|cl_queue2|cl_queue3]_rp.3ep.bs32x1x1.lr2e-04.bf16.sh`
- **KMeansSel**: `src/scripts-ni-c012/lora/[cl|cl2|cl3]/[model_name]/[model_name].lora.[cl_queue|cl_queue2|cl_queue3]_km20_rp.3ep.bs32x1x1.lr2e-04.bf16.sh`
- **SSR**: `src/scripts-ni-c012/lora/[cl|cl2|cl3]/[model_name]/[model_name].lora.[cl_queue|cl_queue2|cl_queue3]_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh`

**NOTE**: You should train **the first task of continual learning** using the `single task` script before executing `SSR`/`RandSel`/`KMeansSel`/`Non-rehearsal` scripts.

## 📝 Citation

If you find this useful in your research, please consider citing:

``` bibtex
@inproceedings{huang-etal-2024-mitigating,
    title = "Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal",
    author = "Huang, Jianheng  and
      Cui, Leyang  and
      Wang, Ante  and
      Yang, Chengyi  and
      Liao, Xinting  and
      Song, Linfeng  and
      Yao, Junfeng  and
      Su, Jinsong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.77",
    pages = "1416--1428",
    abstract = "Large language models (LLMs) suffer from catastrophic forgetting during continual learning. Conventional rehearsal-based methods rely on previous training data to retain the model{'}s ability, which may not be feasible in real-world applications. When conducting continual learning based on a publicly-released LLM checkpoint, the availability of the original training data may be non-existent. To address this challenge, we propose a framework called Self-Synthesized Rehearsal (SSR) that uses the LLM to generate synthetic instances for rehearsal. Concretely, we first employ the base LLM for in-context learning to generate synthetic instances. Subsequently, we utilize the latest LLM to refine the instance outputs based on the synthetic inputs, preserving its acquired ability. Finally, we select diverse high-quality synthetic instances for rehearsal in future stages. Experimental results demonstrate that SSR achieves superior or comparable performance compared to conventional rehearsal-based approaches while being more data-efficient. Besides, SSR effectively preserves the generalization capabilities of LLMs in general domains.",
}
```

or

``` bibtex
@misc{huang2024mitigating,
    title={Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal}, 
    author={Jianheng Huang and Leyang Cui and Ante Wang and Chengyi Yang and Xinting Liao and Linfeng Song and Junfeng Yao and Jinsong Su},
    year={2024},
    eprint={2403.01244},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
