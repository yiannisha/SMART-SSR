#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT_DIR"

exec "$REPO_ROOT_DIR/scripts/run_paper_ssr_5task.sh" \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --model_family llama2-7b-chat \
  --output_root saves/paper-ssr-5task-llama2-7b-chat \
  "$@"
