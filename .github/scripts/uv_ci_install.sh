#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <editable-spec> [additional uv pip install args...]" >&2
  echo "Example: $0 '.[yolo]' --group test" >&2
  exit 1
fi

editable_spec="$1"
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly uv_version_file="$script_dir/../uv-version"
readonly uv_version="$(<"$uv_version_file")"

if [[ ! "$uv_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid uv version pin in $uv_version_file: $uv_version" >&2
  exit 2
fi

read_uv_version() {
  local version_output
  if ! version_output="$(uv --version 2>/dev/null)"; then
    return 1
  fi
  version_output="${version_output#uv }"
  printf '%s\n' "${version_output%% *}"
}

installed_uv_version=""
if command -v uv >/dev/null 2>&1; then
  installed_uv_version="$(read_uv_version || true)"
fi
if [[ "$installed_uv_version" != "$uv_version" ]]; then
  python -m pip install --disable-pip-version-check "uv==$uv_version"
  hash -r
fi

if ! installed_uv_version="$(read_uv_version)"; then
  echo "Unable to execute the pinned uv installation" >&2
  exit 2
fi
if [[ "$installed_uv_version" != "$uv_version" ]]; then
  echo "Expected uv $uv_version, found uv $installed_uv_version" >&2
  exit 2
fi

# Fail before installing anything when project metadata and the committed lockfile
# disagree. The actual environment remains CPU-specific via uv's pip interface.
uv lock --check

constraints_file="$(mktemp "${TMPDIR:-/tmp}/boxmot-lock-constraints.XXXXXX")"
cleanup() {
  rm -f "$constraints_file"
}
trap cleanup EXIT

# Constrain every package to uv.lock without installing unrelated extras. Using
# constraints (instead of requirements) lets --torch-backend select CPU wheels
# without pulling the CUDA packages represented by the platform-neutral lock.
uv export \
  --quiet \
  --locked \
  --all-extras \
  --all-groups \
  --no-emit-project \
  --no-hashes \
  --no-annotate \
  --no-header \
  --output-file "$constraints_file"

uv venv --python python
uv pip install \
  --python .venv/bin/python \
  --no-sources \
  --torch-backend "${UV_TORCH_BACKEND:-cpu}" \
  --constraints "$constraints_file" \
  --build-constraints "$constraints_file" \
  -e "${editable_spec}" \
  "$@"
