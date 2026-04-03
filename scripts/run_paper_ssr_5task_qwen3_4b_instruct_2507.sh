#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT_DIR"

exec "$REPO_ROOT_DIR/scripts/run_paper_ssr_5task.sh" \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --model_family qwen3-4b-instruct-2507 \
  --output_root saves/paper-ssr-5task-qwen3-4b-instruct-2507 \
  "$@"
