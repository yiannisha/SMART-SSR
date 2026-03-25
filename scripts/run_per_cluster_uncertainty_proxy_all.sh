#!/bin/bash

cd /workspace/SMART-SSR
source .venv/bin/activate
source keys.sh

for task in qg sa sum trans; do
  python scripts/run_selection_proxy_iteration.py \
    --base_run saves/paper-ssr-5task-llama2-7b-chat \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --template llama2 \
    --iteration_task "$task" \
    --selector uncertainty_band \
    --uncertainty_selection_mode per_cluster \
    --candidate_source refined \
    --h_min 0.25 \
    --h_max 0.75 \
    --output_root "saves/selection-proxy-llama2-7b-chat-uncertainty-per-cluster/${task}" \
    --cuda 0
done

