#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <editable-spec> [additional uv pip install args...]" >&2
  echo "Example: $0 '.[yolo]' --group test" >&2
  exit 1
fi

editable_spec="$1"
shift

python -m pip install --upgrade pip setuptools wheel uv
uv venv
uv pip install \
  --python .venv/bin/python \
  --no-sources \
  --torch-backend "${UV_TORCH_BACKEND:-cpu}" \
  --index-strategy "${UV_INDEX_STRATEGY:-unsafe-best-match}" \
  -e "${editable_spec}" \
  "$@"
