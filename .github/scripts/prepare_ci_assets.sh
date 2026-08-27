#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

readonly boxmot_release_base_url="https://github.com/mikel-brostrom/boxmot/releases/download/v22.0.0"
readonly ultralytics_release_base_url="https://github.com/ultralytics/assets/releases/download/v8.4.0"
readonly osnet_url="https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"
readonly lmbn_url="https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_duke.pt"
readonly detector_sha256="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
readonly obb_sha256="b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32"
readonly pose_sha256="eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
readonly seg_sha256="361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"
readonly osnet_sha256="6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99"
readonly lmbn_sha256="bbc2080e15b1b819a3c3cf6d007963e6ab33ac87eb2781e34196164de10f081f"

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
  local url

  case "$asset_kind" in
    detector)
      filename="yolo26n.pt"
      expected_sha256="$detector_sha256"
      destination="${BOXMOT_CI_DETECTOR:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_DETECTOR"
      url="$boxmot_release_base_url/$filename"
      ;;
    obb)
      filename="yolo11n-obb.pt"
      expected_sha256="$obb_sha256"
      destination="${BOXMOT_CI_OBB_DETECTOR:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_OBB_DETECTOR"
      url="$boxmot_release_base_url/$filename"
      ;;
    pose)
      filename="yolo26n-pose.pt"
      expected_sha256="$pose_sha256"
      destination="${BOXMOT_CI_POSE_DETECTOR:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_POSE_DETECTOR"
      url="$ultralytics_release_base_url/$filename"
      ;;
    seg)
      filename="yolo26n-seg.pt"
      expected_sha256="$seg_sha256"
      destination="${BOXMOT_CI_SEG_DETECTOR:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_SEG_DETECTOR"
      url="$boxmot_release_base_url/$filename"
      ;;
    reid-osnet)
      filename="osnet_x0_25_msmt17.pt"
      expected_sha256="$osnet_sha256"
      destination="${BOXMOT_CI_REID_OSNET:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_REID_OSNET"
      url="$osnet_url"
      ;;
    reid-lmbn)
      filename="lmbn_n_duke.pt"
      expected_sha256="$lmbn_sha256"
      destination="${BOXMOT_CI_REID_LMBN:-$repo_root/models/$filename}"
      env_name="BOXMOT_CI_REID_LMBN"
      url="$lmbn_url"
      ;;
    *)
      echo "Unknown CI asset kind: $asset_kind (expected detector, obb, pose, seg, reid-osnet, or reid-lmbn)" >&2
      return 2
      ;;
  esac

  if [[ "$destination" != /* ]]; then
    destination="$repo_root/$destination"
  fi

  bash "$script_dir/fetch_ci_asset.sh" \
    "$url" \
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
