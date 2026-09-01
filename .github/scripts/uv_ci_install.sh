#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cpu|cu130> [additional uv sync args...]" >&2
  echo "Example: $0 cpu --extra yolo --group test" >&2
  exit 1
fi

torch_profile="$1"
shift
case "$torch_profile" in
  cpu | cu130) ;;
  *)
    echo "PyTorch profile must be 'cpu' or 'cu130', got '$torch_profile'." >&2
    exit 2
    ;;
esac

uv_version="${UV_VERSION:-0.12.4}"
python -m pip install --upgrade pip setuptools wheel "uv==${uv_version}"
uv sync \
  --locked \
  --no-default-groups \
  --extra "$torch_profile" \
  "$@"
