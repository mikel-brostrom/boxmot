#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

readonly yolo26n_url="https://github.com/mikel-brostrom/boxmot/releases/download/v22.0.0/yolo26n.pt"
readonly yolo26n_sha256="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"

detector_path="${BOXMOT_CI_DETECTOR:-$repo_root/models/yolo26n.pt}"
if [[ "$detector_path" != /* ]]; then
  detector_path="$repo_root/$detector_path"
fi

bash "$script_dir/fetch_ci_asset.sh" \
  "$yolo26n_url" \
  "$yolo26n_sha256" \
  "$detector_path"

# Make the canonical absolute path available to subsequent GitHub Actions steps.
if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'BOXMOT_CI_DETECTOR=%s\n' "$detector_path" >> "$GITHUB_ENV"
fi

printf '%s\n' "$detector_path"
