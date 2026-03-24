#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT_DIR"

if [[ -f "$REPO_ROOT_DIR/keys.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT_DIR/keys.sh"
fi

if [[ -n "${HUGGING_FACE_ACCESS_TOKEN:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_ACCESS_TOKEN}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HUGGING_FACE_ACCESS_TOKEN}"
fi

"$REPO_ROOT_DIR/scripts/bootstrap_env.sh"

# shellcheck disable=SC1091
source "$REPO_ROOT_DIR/.venv/bin/activate"

python "$REPO_ROOT_DIR/scripts/prepare_experiment_data.py"

python "$REPO_ROOT_DIR/scripts/run_basic_ssr_5task.py" "$@"
