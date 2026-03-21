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

## 🚀 Working 5-Task SSR Quickstart

The original paper scripts under `src/scripts-ni-c012/` are useful references, but many of them assume old absolute paths and prebuilt synthetic rehearsal files.

This repository now includes a working end-to-end runner for the basic 5-task SSR queue:

- tasks: `qa qg sa sum trans`
- entrypoint: `scripts/run_basic_ssr_5task.sh`
- Python driver: `scripts/run_basic_ssr_5task.py`
- synthetic rehearsal builder: `custom/icl_gen/build_ssr_dataset.py`

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
- The generated rehearsal JSON files are written under `data/ni-cus0.12/genearated-icl-naive-kmeans20-self/llama2-7b-chat/cl_queue/` so they reuse the dataset keys already registered in `data/dataset_info.json`.
- Final checkpoints and eval outputs are written under `saves/basic-ssr-5task/`.
- To resume after an interruption, rerun the same command and add `--skip_completed`.
```

## 🛢 Pipeline

### Step 1: In-Context Learning Based Instance Synthesis

1. use `custom/icl_gen/complete_param_nic010_cate.py` for generation 
2. use `custom/icl_gen/parse_filter_generated.py` for post-processing and filtering

### Step 2: Synthetic Output Refinement

1. use `custom/icl_gen/label_param.py` (integrated in the following scripts)

### Step 3: Rehearsal with Selected Synthetic Instances

1. select refined synthetic instances for rehearsal with `custom/icl_gen/random_select.py` or `custom/icl_gen/select_kmeans_examples.py`
2. for KMeans-based selection, the current implementation clusters using synthetic instance inputs; this reproduces the paper results in this repo, while the pipeline order remains `synthesis -> refinement -> selection`

- **multi-task learning (MTL)**: `src/scripts-ni-c012/lora/all/[model_name]/[model_name].lora.[all|all_5].3ep.bs32x1x1.bf16.sh`
- **single task (& Stage 1 in continual learing)**: `src/scripts-ni-c012/lora/sing/[model_name]/[model_name].lora.single.3ep.bs32x1x1.bf16.sh`
- **Non-rehearsal**: `src/scripts-ni-c012/lora/[cl|cl2|cl3]/[model_name]/[model_name].lora.[cl_queue|cl_queue2|cl_queue3].3ep.bs32x1x1.lr2e-04.bf16.sh`
- **RandSel**: `src/scripts-ni-c012/lora/[cl|cl2|cl3]/[model_name]/[model_name].lora.[cl_queue|cl_queue2|cl_queue3]_rp.3ep.bs32x1x1.lr2e-04.bf16.sh`
- **KMeansSel**: `src/scripts-ni-c012/lora/[cl|cl2|cl3]/[model_name]/[model_name].lora.[cl_queue|cl_queue2|cl_queue3]_km20_rp.3ep.bs32x1x1.lr2e-04.bf16.sh`
- **SSR**: `src/scripts-ni-c012/lora/[cl|cl2|cl3]/[model_name]/[model_name].lora.[cl_queue|cl_queue2|cl_queue3]_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh`

**NOTE**: You should train **the first task of contiunal learning** using the `single task` script before executing `SSR`/`RandSel`/`KMeansSel`/`Non-rehearsal` scripts.

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
