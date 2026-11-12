#!/usr/bin/env bash
# Launch the CPU webcam tracker, vLLM, and PostgreSQL from a repository checkout.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./docker/webcam.sh

Build and run webcam tracking with vLLM attributes, PostgreSQL, and a preview window.
Run as your desktop user; Docker uses sudo automatically when needed.

Environment overrides:
  VIDEO_DEVICE     Webcam device (default: /dev/video0)
  HF_CACHE_DIR     Hugging Face cache (default: $HOME/.cache/huggingface)
  BOXMOT_DETECTOR  Detector profile (default: yolo26n)
  BOXMOT_TRACKER   Tracker (default: bytetrack)
  BOXMOT_RUN_NAME  Output directory under runs/ (default: compose)
  BOXMOT_POSTGRES_PORT  PostgreSQL loopback host port (default: 5432)

Other Compose environment settings are also supported. See
docs/integrations/docker.md for details. Press Ctrl+C to stop.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# )); then
  usage >&2
  exit 2
fi
if [[ -n ${SUDO_USER:-} && $EUID == 0 ]]; then
  echo "Run ./docker/webcam.sh without sudo so directories belong to your desktop user." >&2
  exit 1
fi
if [[ -z ${DISPLAY:-} ]]; then
  echo "DISPLAY is not set. Run ./docker/webcam.sh from your Linux desktop terminal." >&2
  exit 1
fi
if ! command -v docker >/dev/null; then
  echo "Docker is required. Install Docker and its Compose plugin first." >&2
  exit 1
fi
docker compose version >/dev/null

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"

BOXMOT_UID=${BOXMOT_UID:-$(id -u)}
BOXMOT_GID=${BOXMOT_GID:-$(id -g)}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
VIDEO_DEVICE=${VIDEO_DEVICE:-/dev/video0}
if [[ ! -c $VIDEO_DEVICE ]]; then
  echo "Webcam device not found: $VIDEO_DEVICE. Set VIDEO_DEVICE to your camera's device path." >&2
  exit 1
fi
VIDEO_GID=${VIDEO_GID:-$(stat -c '%g' -- "$VIDEO_DEVICE")}
mkdir -p -- data models runs "$HF_CACHE_DIR"
# Compose resolves relative bind paths beside its YAML files.
HF_CACHE_DIR=$(cd -- "$HF_CACHE_DIR" && pwd)

# Pass only Compose settings through sudo, retaining the desktop user's values.
compose_env=()
for name in BOXMOT_UID BOXMOT_GID HF_CACHE_DIR DISPLAY VIDEO_DEVICE VIDEO_GID \
  BOXMOT_INPUT_DIR BOXMOT_SOURCE BOXMOT_DETECTOR BOXMOT_TRACKER BOXMOT_REID \
  BOXMOT_RUN_NAME VLLM_CPU_IMAGE VLLM_PORT OLLAMA_PORT \
  BOXMOT_POSTGRES_USER BOXMOT_POSTGRES_PASSWORD BOXMOT_POSTGRES_DB BOXMOT_POSTGRES_PORT \
  VLLM_CPU_OMP_THREADS_BIND VLLM_CPU_NUM_OF_RESERVED_CPU; do
  if [[ -v $name ]]; then
    compose_env+=("$name=${!name}")
  fi
done

docker_command=(env "${compose_env[@]}" docker)
if ! docker info >/dev/null 2>&1; then
  echo "Retrying Docker with sudo; you may be asked for your password." >&2
  docker_command=(sudo env "${compose_env[@]}" docker)
fi
exec "${docker_command[@]}" compose \
  -f docker/compose.attributes.yaml -f docker/compose.webcam.yaml \
  up --build --abort-on-container-exit --exit-code-from boxmot boxmot
