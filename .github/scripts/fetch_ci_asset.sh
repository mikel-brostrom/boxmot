#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <url> <sha256> <destination>" >&2
  exit 2
fi

url="$1"
expected_sha256="$2"
destination="$3"

if ! [[ "$expected_sha256" =~ ^[[:xdigit:]]{64}$ ]]; then
  echo "Expected SHA-256 must contain exactly 64 hexadecimal characters" >&2
  exit 2
fi
expected_sha256="$(printf '%s' "$expected_sha256" | tr '[:upper:]' '[:lower:]')"

sha256_of() {
  local path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo "Neither sha256sum nor shasum is available" >&2
    return 127
  fi
}

has_expected_checksum() {
  local path="$1"
  local actual_sha256

  [[ -f "$path" ]] || return 1
  actual_sha256="$(sha256_of "$path")"
  [[ "$actual_sha256" == "$expected_sha256" ]]
}

if has_expected_checksum "$destination"; then
  echo "Using verified CI asset: $destination"
  exit 0
fi

mkdir -p "$(dirname "$destination")"
temporary="$(mktemp "${destination}.part.XXXXXX")"
cleanup() {
  rm -f "$temporary"
}
trap cleanup EXIT

curl \
  --fail \
  --location \
  --retry 5 \
  --retry-all-errors \
  --show-error \
  --silent \
  --output "$temporary" \
  "$url"

actual_sha256="$(sha256_of "$temporary")"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "Checksum mismatch for $url" >&2
  echo "Expected: $expected_sha256" >&2
  echo "Actual:   $actual_sha256" >&2
  exit 1
fi

chmod 0644 "$temporary"
mv -f "$temporary" "$destination"
trap - EXIT
echo "Downloaded and verified CI asset: $destination"
