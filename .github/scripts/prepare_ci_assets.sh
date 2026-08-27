#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

readonly boxmot_release_base_url="https://github.com/mikel-brostrom/boxmot/releases/download/v22.0.0"
readonly ultralytics_release_base_url="https://github.com/ultralytics/assets/releases/download/v8.4.0"
readonly detector_sha256="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
readonly obb_sha256="b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32"
readonly pose_sha256="eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
readonly seg_sha256="361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"

asset_selection="${*:-detector}"
prepared_assets=""
selected_assets=()
read -r -a selected_assets <<< "$asset_selection"

prepare_asset() {
  local asset_kind="$1"
  local destination
  local env_name
  local filename
  local expected_sha256
  local release_base_url="$boxmot_release_base_url"

  case "$asset_kind" in
    detector)
      filename="yolo26n.pt"
      expected_sha256="$detector_sha256"
      destination="${BOXMOT_CI_DETECTOR:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_DETECTOR"
      ;;
    obb)
      filename="yolo11n-obb.pt"
      expected_sha256="$obb_sha256"
      destination="${BOXMOT_CI_OBB_DETECTOR:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_OBB_DETECTOR"
      ;;
    pose)
      filename="yolo26n-pose.pt"
      expected_sha256="$pose_sha256"
      destination="${BOXMOT_CI_POSE_DETECTOR:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_POSE_DETECTOR"
      release_base_url="$ultralytics_release_base_url"
      ;;
    seg)
      filename="yolo26n-seg.pt"
      expected_sha256="$seg_sha256"
      destination="${BOXMOT_CI_SEG_DETECTOR:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_SEG_DETECTOR"
      ;;
    *)
      echo "Unknown CI asset kind: $asset_kind (expected detector, obb, pose, or seg)" >&2
      return 2
      ;;
  esac

  if [[ "$destination" != /* ]]; then
    destination="$repo_root/$destination"
  fi

  bash "$script_dir/fetch_ci_asset.sh" \
    "$release_base_url/$filename" \
    "$expected_sha256" \
    "$destination"

  # Make the canonical absolute path available to subsequent GitHub Actions steps.
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s=%s\n' "$env_name" "$destination" >> "$GITHUB_ENV"
  fi

  printf '%s=%s\n' "$env_name" "$destination"
}

for asset_kind in "${selected_assets[@]}"; do
  case " $prepared_assets " in
    *" $asset_kind "*) continue ;;
  esac
  prepare_asset "$asset_kind"
  prepared_assets="$prepared_assets $asset_kind"
done

if [[ -z "$prepared_assets" ]]; then
  echo "At least one CI asset kind must be selected" >&2
  exit 2
fi
