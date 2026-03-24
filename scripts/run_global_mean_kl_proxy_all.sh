#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-saves/paper-ssr-5task-llama2-7b-chat}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-2-7b-chat-hf}"
TEMPLATE="${TEMPLATE:-llama2}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-refined}"
SAMPLE_MEMORY="${SAMPLE_MEMORY:-200}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-}"

TASKS=(qg sa sum trans)
EXTRA_ARGS=("$@")

for task in "${TASKS[@]}"; do
  echo "=== Running global mean_kl proxy iteration for ${task} ==="

  cmd=(
    "${PYTHON_BIN}"
    "scripts/run_selection_proxy_iteration.py"
    "--base_run" "${BASE_RUN}"
    "--model_name_or_path" "${MODEL_NAME_OR_PATH}"
    "--template" "${TEMPLATE}"
    "--iteration_task" "${task}"
    "--selector" "mean_kl"
    "--mean_kl_selection_mode" "global"
    "--candidate_source" "${CANDIDATE_SOURCE}"
    "--sample_memory" "${SAMPLE_MEMORY}"
    "--cuda" "${CUDA_DEVICE}"
  )

  if [[ -n "${OUTPUT_ROOT_BASE}" ]]; then
    cmd+=("--output_root" "${OUTPUT_ROOT_BASE}/${task}")
  fi

  if ((${#EXTRA_ARGS[@]} > 0)); then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  printf 'Running: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
done
