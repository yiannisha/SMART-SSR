# Experimental Setup

This file tracks the hybrid-selector SSR runs that are already completed and the ones still pending.

Assumption used here:

- order A = `qa -> qg -> sa -> sum -> trans`
- order B = `trans -> sa -> qa -> sum -> qg`
- order C = `sum -> qg -> trans -> qa -> sa`

These order definitions match the historical `cl_queue`, `cl_queue2`, and `cl_queue3` task lists in `src/scripts-ni-c012/`.

The hybrid weight vector is listed as:

```text
[diversity, mean_kl, uncertainty]
```

## Weight Legend

| ID | Weight vector | Notes |
| --- | --- | --- |
| W1 | `[0, 0, 1]` | uncertainty only |
| W2 | `[0, 1, 0]` | mean-KL only |
| W3 | `[1, 0, 0]` | diversity only |
| W4 | `[0.33, 0.33, 0.33]` | near-equal weighting |
| W5 | `[0.2, 0.2, 0.6]` | uncertainty heavy |
| W5 | `[0.2, 0.6, 0.2]` | mean-KL heavy |
| W7 | `[0.6, 0.2, 0.2]` | diversity heavy |
| W8 | `[0.3, 0.3, 0.4]` |
| W9 | `[0.3, 0.4, 0.3]` |
| W10 | `[0.4, 0.3, 0.3]` |

## Runnable Today

With the current maintained runner, the runnable model families are:

- `llama2-7b-chat`
- `llama2-7b`

The following requested model families are not yet supported by `scripts/run_paper_ssr_5task.py` today:

- `qwen3`
- `llama3.3-7b-chat`

So the command list below only covers the currently possible pending runs.

## Experiment Matrix

| Model | Order | Task order | Weights | Status | Run count |
| --- | --- | --- | --- | --- | --- |
| `llama2-7b-chat` | A | `qa -> qg -> sa -> sum -> trans` | `W1, W2, W3, W4, W5, W6` | Done | 6 |
| `llama2-7b-chat` | B | `trans -> sa -> qa -> sum -> qg` | `W1, W2, W3, W4, W5, W6, W7` | To do | 7 |
| `llama2-7b-chat` | C | `sum -> qg -> trans -> qa -> sa` | `W1, W2, W3, W4, W5, W6, W7` | To do | 7 |
| `llama2-7b` | B | `trans -> sa -> qa -> sum -> qg` | `W1, W2, W3, W4, W5, W6, W7` | To do | 7 |
| `llama2-7b` | C | `sum -> qg -> trans -> qa -> sa` | `W1, W2, W3, W4, W5, W6, W7` | To do | 7 |
| `qwen3` | B | `trans -> sa -> qa -> sum -> qg` | `W1, W2, W3, W4, W5, W6, W7` | To do | 7 |
| `qwen3` | C | `sum -> qg -> trans -> qa -> sa` | `W1, W2, W3, W4, W5, W6, W7` | To do | 7 |
| `llama3.3-7b-chat` | B | `trans -> sa -> qa -> sum -> qg` | `W1, W2, W3, W4, W5, W6, W7` | To do | 7 |
| `llama3.3-7b-chat` | C | `sum -> qg -> trans -> qa -> sa` | `W1, W2, W3, W4, W5, W6, W7` | To do | 7 |

## Totals

| Category | Count |
| --- | --- |
| Completed runs | 6 |
| Pending runs | 56 |
| Total runs in this plan | 62 |

## Explicit Completed Runs

| Model | Order | Weight vector | Status |
| --- | --- | --- | --- |
| `llama2-7b-chat` | A | `[0, 0, 1]` | Done |
| `llama2-7b-chat` | A | `[0, 1, 0]` | Done |
| `llama2-7b-chat` | A | `[1, 0, 0]` | Done |
| `llama2-7b-chat` | A | `[0.2, 0.6, 0.2]` | Done |
| `llama2-7b-chat` | A | `[0.2, 0.2, 0.6]` | Done |
| `llama2-7b-chat` | A | `[0.33, 0.33, 0.33]` | Done |

## Explicit Pending Runs

| Model | Orders still needed | Weight vectors still needed | Run count |
| --- | --- | --- | --- |
| `llama2-7b-chat` | B, C | `[0, 0, 1]`, `[0, 1, 0]`, `[1, 0, 0]`, `[0.2, 0.6, 0.2]`, `[0.2, 0.2, 0.6]`, `[0.33, 0.33, 0.33]`, `[0.6, 0.2, 0.2]` | 14 |
| `llama2-7b` | B, C | `[0, 0, 1]`, `[0, 1, 0]`, `[1, 0, 0]`, `[0.2, 0.6, 0.2]`, `[0.2, 0.2, 0.6]`, `[0.33, 0.33, 0.33]`, `[0.6, 0.2, 0.2]` | 14 |
| `qwen3` | B, C | `[0, 0, 1]`, `[0, 1, 0]`, `[1, 0, 0]`, `[0.2, 0.6, 0.2]`, `[0.2, 0.2, 0.6]`, `[0.33, 0.33, 0.33]`, `[0.6, 0.2, 0.2]` | 14 |
| `llama3.3-7b-chat` | B, C | `[0, 0, 1]`, `[0, 1, 0]`, `[1, 0, 0]`, `[0.2, 0.6, 0.2]`, `[0.2, 0.2, 0.6]`, `[0.33, 0.33, 0.33]`, `[0.6, 0.2, 0.2]` | 14 |

## Commands For Currently Possible Pending Runs

All commands below use:

- selector: `hybrid_cluster`
- candidate source: `refined`
- sample memory: `200`
- CUDA device: `0`

Order B:

```text
trans -> sa -> qa -> sum -> qg
```

Order C:

```text
sum -> qg -> trans -> qa -> sa
```

If you are resuming an interrupted run, add:

```bash
--skip_completed
```

### `llama2-7b-chat` pending runs

Order B:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.0 --hybrid_cluster_mean_kl_weight 0.0 --hybrid_cluster_uncertainty_weight 1.0 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-b-hybrid-cluster-0-0-1

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.0 --hybrid_cluster_mean_kl_weight 1.0 --hybrid_cluster_uncertainty_weight 0.0 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-b-hybrid-cluster-0-1-0

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 1.0 --hybrid_cluster_mean_kl_weight 0.0 --hybrid_cluster_uncertainty_weight 0.0 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-b-hybrid-cluster-1-0-0

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.2 --hybrid_cluster_mean_kl_weight 0.6 --hybrid_cluster_uncertainty_weight 0.2 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-b-hybrid-cluster-02-06-02

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.2 --hybrid_cluster_mean_kl_weight 0.2 --hybrid_cluster_uncertainty_weight 0.6 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-b-hybrid-cluster-02-02-06

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.33 --hybrid_cluster_mean_kl_weight 0.33 --hybrid_cluster_uncertainty_weight 0.33 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-b-hybrid-cluster-033-033-033

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.6 --hybrid_cluster_mean_kl_weight 0.2 --hybrid_cluster_uncertainty_weight 0.2 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-b-hybrid-cluster-06-02-02
```

Order C:

```bash
bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.0 --hybrid_cluster_mean_kl_weight 0.0 --hybrid_cluster_uncertainty_weight 1.0 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-c-hybrid-cluster-0-0-1

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.0 --hybrid_cluster_mean_kl_weight 1.0 --hybrid_cluster_uncertainty_weight 0.0 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-c-hybrid-cluster-0-1-0

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 1.0 --hybrid_cluster_mean_kl_weight 0.0 --hybrid_cluster_uncertainty_weight 0.0 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-c-hybrid-cluster-1-0-0

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.2 --hybrid_cluster_mean_kl_weight 0.6 --hybrid_cluster_uncertainty_weight 0.2 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-c-hybrid-cluster-02-06-02

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.2 --hybrid_cluster_mean_kl_weight 0.2 --hybrid_cluster_uncertainty_weight 0.6 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-c-hybrid-cluster-02-02-06

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.33 --hybrid_cluster_mean_kl_weight 0.33 --hybrid_cluster_uncertainty_weight 0.33 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-c-hybrid-cluster-033-033-033

bash scripts/run_paper_ssr_5task_llama2_7b_chat.sh --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.6 --hybrid_cluster_mean_kl_weight 0.2 --hybrid_cluster_uncertainty_weight 0.2 --output_root saves/paper-ssr-5task-llama2-7b-chat-order-c-hybrid-cluster-06-02-02
```

### `llama2-7b` pending runs

Order B:

```bash
bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.0 --hybrid_cluster_mean_kl_weight 0.0 --hybrid_cluster_uncertainty_weight 1.0 --output_root saves/paper-ssr-5task-llama2-7b-order-b-hybrid-cluster-0-0-1

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.0 --hybrid_cluster_mean_kl_weight 1.0 --hybrid_cluster_uncertainty_weight 0.0 --output_root saves/paper-ssr-5task-llama2-7b-order-b-hybrid-cluster-0-1-0

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 1.0 --hybrid_cluster_mean_kl_weight 0.0 --hybrid_cluster_uncertainty_weight 0.0 --output_root saves/paper-ssr-5task-llama2-7b-order-b-hybrid-cluster-1-0-0

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.2 --hybrid_cluster_mean_kl_weight 0.6 --hybrid_cluster_uncertainty_weight 0.2 --output_root saves/paper-ssr-5task-llama2-7b-order-b-hybrid-cluster-02-06-02

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.2 --hybrid_cluster_mean_kl_weight 0.2 --hybrid_cluster_uncertainty_weight 0.6 --output_root saves/paper-ssr-5task-llama2-7b-order-b-hybrid-cluster-02-02-06

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.33 --hybrid_cluster_mean_kl_weight 0.33 --hybrid_cluster_uncertainty_weight 0.33 --output_root saves/paper-ssr-5task-llama2-7b-order-b-hybrid-cluster-033-033-033

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks trans sa qa sum qg --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.6 --hybrid_cluster_mean_kl_weight 0.2 --hybrid_cluster_uncertainty_weight 0.2 --output_root saves/paper-ssr-5task-llama2-7b-order-b-hybrid-cluster-06-02-02
```

Order C:

```bash
bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.0 --hybrid_cluster_mean_kl_weight 0.0 --hybrid_cluster_uncertainty_weight 1.0 --output_root saves/paper-ssr-5task-llama2-7b-order-c-hybrid-cluster-0-0-1

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.0 --hybrid_cluster_mean_kl_weight 1.0 --hybrid_cluster_uncertainty_weight 0.0 --output_root saves/paper-ssr-5task-llama2-7b-order-c-hybrid-cluster-0-1-0

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 1.0 --hybrid_cluster_mean_kl_weight 0.0 --hybrid_cluster_uncertainty_weight 0.0 --output_root saves/paper-ssr-5task-llama2-7b-order-c-hybrid-cluster-1-0-0

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.2 --hybrid_cluster_mean_kl_weight 0.6 --hybrid_cluster_uncertainty_weight 0.2 --output_root saves/paper-ssr-5task-llama2-7b-order-c-hybrid-cluster-02-06-02

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.2 --hybrid_cluster_mean_kl_weight 0.2 --hybrid_cluster_uncertainty_weight 0.6 --output_root saves/paper-ssr-5task-llama2-7b-order-c-hybrid-cluster-02-02-06

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.33 --hybrid_cluster_mean_kl_weight 0.33 --hybrid_cluster_uncertainty_weight 0.33 --output_root saves/paper-ssr-5task-llama2-7b-order-c-hybrid-cluster-033-033-033

bash scripts/run_paper_ssr_5task.sh --model_name_or_path meta-llama/Llama-2-7b-hf --model_family llama2-7b --cuda 0 --tasks sum qg trans qa sa --selector hybrid_cluster --candidate_source refined --sample_memory 200 --hybrid_cluster_diversity_weight 0.6 --hybrid_cluster_mean_kl_weight 0.2 --hybrid_cluster_uncertainty_weight 0.2 --output_root saves/paper-ssr-5task-llama2-7b-order-c-hybrid-cluster-06-02-02
```

## Notes

- Every run in this plan uses the `hybrid_cluster` selector.
- The only new weight relative to the already-finished `llama2-7b-chat` order-A set is `W7 = [0.6, 0.2, 0.2]`.
- This document reflects the scope exactly as requested: completed work is only the 6 `llama2-7b-chat` order-A runs, and the pending work is the order-B/order-C expansion for all listed model families.
- The maintained runner currently supports `llama2-7b-chat` and `llama2-7b`, but not `qwen3` or `llama3.3-7b-chat`.
