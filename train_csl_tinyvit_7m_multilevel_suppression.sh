#!/usr/bin/env bash
set -euo pipefail

MARKET1501_DIR="${MARKET1501_DIR:-Market-1501-v15.09.15}"
PAV_METADATA_DIR="${PAV_METADATA_DIR:-Market-1501-pav-metadata-clean}"
MULTILEVEL_SUPPRESSION_PROJECT="${MULTILEVEL_SUPPRESSION_PROJECT:-runs/csl_tinyvit_7m_multilevel_suppression}"
MULTILEVEL_SUPPRESSION_NAME="${MULTILEVEL_SUPPRESSION_NAME:-class_cam_q15_v2_seed0}"
MULTILEVEL_SUPPRESSION_DEVICE="${MULTILEVEL_SUPPRESSION_DEVICE:-0}"
MULTILEVEL_SUPPRESSION_NUM_WORKERS="${MULTILEVEL_SUPPRESSION_NUM_WORKERS:-4}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 2
}

[[ -d "$MARKET1501_DIR" ]] || fail "Market-1501 directory does not exist: $MARKET1501_DIR"
for split in bounding_box_train query bounding_box_test; do
  split_dir="$MARKET1501_DIR/$split"
  [[ -d "$split_dir" ]] || fail "$split split does not exist: $split_dir"
  if ! find "$split_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print -quit | grep -q .; then
    fail "$split split contains no valid JPEG images"
  fi
done

metadata_file="$PAV_METADATA_DIR/metadata.json"
[[ -f "$metadata_file" ]] || fail "PAV metadata file does not exist: $metadata_file"

if ! python3 - "$MARKET1501_DIR" "$PAV_METADATA_DIR" <<'PY'
import json
import pathlib
import sys

market_root = pathlib.Path(sys.argv[1])
metadata_root = pathlib.Path(sys.argv[2])
try:
    payload = json.loads((metadata_root / "metadata.json").read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    print(f"Error: invalid PAV metadata: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

images = payload.get("images")
if not isinstance(images, dict):
    print("Error: PAV metadata 'images' must be an object", file=sys.stderr)
    raise SystemExit(1)

training_images = {
    path.relative_to(market_root).as_posix()
    for path in (market_root / "bounding_box_train").rglob("*")
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
}
matching = training_images.intersection(images)
if not matching:
    print("Error: PAV metadata does not match any Market-1501 training image", file=sys.stderr)
    raise SystemExit(1)

for image_key in matching:
    record = images[image_key]
    if not isinstance(record, dict):
        print(f"Error: invalid PAV metadata record for {image_key}", file=sys.stderr)
        raise SystemExit(1)
    mask_ref = record.get("person_mask")
    if mask_ref and not (metadata_root / mask_ref).is_file():
        print(f"Error: PAV person mask does not exist: {mask_ref}", file=sys.stderr)
        raise SystemExit(1)
PY
then
  exit 2
fi

command=(
  uv run --no-sync python -m boxmot.engine.cli train-reid --recipe csl_tinyvit_7m_multilevel_suppression
  --data-dir "$MARKET1501_DIR"
  --anatomical-metadata-dir "$PAV_METADATA_DIR"
  --project "$MULTILEVEL_SUPPRESSION_PROJECT"
  --name "$MULTILEVEL_SUPPRESSION_NAME"
  --device "$MULTILEVEL_SUPPRESSION_DEVICE"
  --num-workers "$MULTILEVEL_SUPPRESSION_NUM_WORKERS"
)

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  printf 'Validated multilevel-suppression inputs. Resolved command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
