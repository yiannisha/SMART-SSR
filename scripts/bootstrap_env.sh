#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
STAMP_FILE="${VENV_DIR}/.smart_ssr_bootstrap_v2"

TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  python -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ -f "${STAMP_FILE}" ]] && TORCH_VERSION_EXPECTED="${TORCH_VERSION}" python - <<'PY'
import os
import sys

try:
    import torch
    import transformers
    import peft
    import trl
    import datasets
    import sklearn
    import sentence_transformers
    import huggingface_hub
except Exception:
    sys.exit(1)

base_version = torch.__version__.split("+", 1)[0]
sys.exit(0 if base_version == os.environ["TORCH_VERSION_EXPECTED"] else 1)
PY
then
  exit 0
fi

python -m pip install -U pip "setuptools<82" wheel

if ! TORCH_VERSION_EXPECTED="${TORCH_VERSION}" python - <<'PY'
import sys
import os

try:
    import torch
except Exception:
    sys.exit(1)

base_version = torch.__version__.split("+", 1)[0]
sys.exit(0 if base_version == os.environ["TORCH_VERSION_EXPECTED"] else 1)
PY
then
  python -m pip install --index-url "${PYTORCH_INDEX_URL}" --force-reinstall "torch==${TORCH_VERSION}"
fi

python -m pip install -r "${REPO_ROOT_DIR}/requirements.txt" scikit-learn sentence-transformers huggingface_hub
python -m pip install -e "${REPO_ROOT_DIR}"
touch "${STAMP_FILE}"
