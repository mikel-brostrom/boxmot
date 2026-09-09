# Docker images

BoxMOT uses one shared multi-stage Dockerfile for four independently built
images:

| Target | BoxMOT v24 tag | Rolling tag | Contents |
| --- | --- | --- | --- |
| `cli-gpu` | `boxmot/boxmot:24.0.0` | `boxmot/boxmot:latest` | Full detector, ReID, CLI, and evaluation stack with CUDA 13.0 PyTorch |
| `cli-cpu` | `boxmot/boxmot:24.0.0-cpu` | `boxmot/boxmot:latest-cpu` | The same full stack with CPU-only PyTorch |
| `service-cpu` | `boxmot/boxmot-service:24.0.0` | `boxmot/boxmot-service:latest` | Non-root CPU geometry-only detection-to-track HTTP service |
| `service-gpu` | `boxmot/boxmot-service:24.0.0-gpu` | `boxmot/boxmot-service:latest-gpu` | Non-root CUDA/ReID detection-to-track HTTP service |

The CPU and CUDA selections come from mutually exclusive, lockfile-backed `cpu`
and `cu130` extras in the root project. Docker, local development, and CI all
consume the same `pyproject.toml` and `uv.lock`. The CPU service combines its
minimal service runtime with the CPU Torch profile, so canonical structures are
available without CUDA and uses headless OpenCV. The GPU service selects CUDA
13.0 Torch plus ReID and HTTP dependencies, without detector or evaluation
extras. A final `default` stage aliases `cli-gpu`, so no-target builds still
produce the full CUDA image.

The GPU target intentionally starts from the same Python slim base as the CPU
target. Its locked `+cu130` PyTorch wheels provide the matching user-space CUDA
libraries, avoiding a second CUDA runtime from a PyTorch base image. The host
still needs a compatible NVIDIA driver and NVIDIA Container Toolkit.

## Build

Run builds from the repository root:

```bash
docker build --target cli-gpu -f docker/Dockerfile -t boxmot/boxmot:local .
docker build --target cli-cpu -f docker/Dockerfile -t boxmot/boxmot:local-cpu .
docker build --target service-cpu -f docker/Dockerfile -t boxmot/boxmot-service:local .
docker build --target service-gpu -f docker/Dockerfile -t boxmot/boxmot-service:local-gpu .
```

The default build is equivalent to `--target cli-gpu`:

```bash
docker build -f docker/Dockerfile -t boxmot/boxmot:local .
```

When dependencies change, regenerate the single root lock. The root
`pyproject.toml` declares and enforces the supported uv version; keep the
Dockerfile's bootstrap uv version aligned with it:

```bash
uv lock
```

## Build cache

System packages, native compilation, Python dependencies, and BoxMOT installation
have separate build layers. Changes to Python code, README, or the release
version leave the system and native layers reusable. Native source changes
rebuild the shared native libraries and the CLI wheel.

The dependency preparation stage copies the canonical `pyproject.toml` and
`uv.lock`, verifies their BoxMOT versions agree, and normalizes only the local
project version in those temporary copies. CPU and CUDA dependency stages
consume those copies with `uv sync --locked`; all dependency constraints,
markers, sources, and artifact hashes are preserved. Version-only releases
therefore reuse the same dependency inputs. Actual dependency changes still
invalidate the corresponding layers.

The CLI wheel is built separately from the original release metadata. Each CLI
runtime first copies its dependency environment, then installs only that wheel
with `--no-deps`. Updating BoxMOT changes the application layer without copying
the complete CUDA environment again. Service images likewise copy dependencies
and application source separately. Published package versions are unchanged by
dependency preparation.

GitHub Actions exports intermediate layers with `mode=max` and a separate cache
scope per image target. Reuse still depends on an accessible, retained cache;
the first build needs to populate it. The uv cache mounts speed up repeated
commands within a builder but are not exported as download caches between jobs.

## Publish

GitHub Actions builds, smoke-tests, and pushes `cli-gpu`, `cli-cpu`, and
`service-cpu` when a GitHub release is published. The `service-gpu` target is
disabled by default. Enable it by setting the repository Actions variable
`BOXMOT_GPU_SERVICE_CI=true` after registering a `gpu-latest` Linux NVIDIA runner
with Docker and the NVIDIA Container Toolkit. This also enables its release
gate and manual-workflow checks.

The same workflow can be dispatched manually with an
exact commit SHA and `v<version>` release tag; manual runs validate only unless
`push_images` is explicitly enabled. Pull requests and branch pushes do not
build or publish images. Each target is pushed only after its smoke test passes,
and `<version>` must exactly match `[project].version` in `pyproject.toml`.

Each image also receives an immutable `sha-<commit>` tag with the same CPU or
GPU suffix shown above.

## Run

Run the CUDA CLI image with the NVIDIA Container Toolkit:

```bash
docker run --rm -it --ipc=host --gpus all boxmot/boxmot:local
```

Run the CPU CLI image:

```bash
docker run --rm -it --ipc=host boxmot/boxmot:local-cpu
```

Both CLI images include the native ReID library and the live C API backends for
BotSort, ByteTrack, OcSort, OccluBoost, and SFSORT. Selecting
`--tracker-backend cpp` does not require a compiler or CMake inside the runtime
container. The service images use the Python tracker backends.

### Persist datasets, builds, and models

Mount raw datasets, immutable builds, and downloaded model artifacts outside
the container. Dataset configs resolve beneath `/opt/boxmot/datasets/mot` by
default, build IDs resolve beneath `BOXMOT_BUILDS_DIR`, and built-in model paths
resolve beneath the CLI image's `/opt/boxmot` working directory:

```bash
mkdir -p "$PWD/datasets/mot" "$PWD/runs/materializations" "$PWD/models"

docker run --rm --gpus all --ipc=host \
  -v "$PWD/datasets/mot:/opt/boxmot/datasets/mot" \
  -v "$PWD/runs/materializations:/materializations" \
  -v "$PWD/models:/opt/boxmot/models" \
  -e BOXMOT_BUILDS_DIR=/materializations \
  boxmot/boxmot:24.0.0 \
  boxmot materialize \
    --experiment mot17/ablation-yolox-lmbn.yaml \
    --device 0
```

Use the build ID printed by `materialize` for evaluation. Reuse the same mounts
so evaluation can verify the source catalog and component artifacts:

```bash
BUILD_ID=replace-with-the-64-character-build-id

docker run --rm --gpus all --ipc=host \
  -v "$PWD/datasets/mot:/opt/boxmot/datasets/mot:ro" \
  -v "$PWD/runs/materializations:/materializations:ro" \
  -v "$PWD/models:/opt/boxmot/models:ro" \
  -e BOXMOT_BUILDS_DIR=/materializations \
  boxmot/boxmot:24.0.0 \
  boxmot eval \
    --experiment mot17/ablation-yolox-lmbn.yaml \
    --build "$BUILD_ID" \
    --tracker occluboost
```

For CPU-only execution, use `boxmot/boxmot:24.0.0-cpu`, omit `--gpus all`,
and pass `--device cpu` to `materialize`.

Run the CPU geometry-only detection-to-track service:

```bash
docker run --rm -p 8000:8000 boxmot/boxmot-service:local
```

Select the geometric association function process-wide with
`BOXMOT_SERVICE_ASSO_FUNC`:

```bash
docker run --rm -p 8000:8000 \
  -e BOXMOT_SERVICE_TRACKER=bytetrack \
  -e BOXMOT_SERVICE_ASSO_FUNC=centroid \
  boxmot/boxmot-service:local
```

Available choices for AABB and OBB sessions are `iou`, `giou`, `diou`, `ciou`,
`hmiou`, and `centroid`. The CPU service uses each request's declared width and
height to initialize centroid normalization, so it still does not need image
pixels.

For OBB sessions, `iou` uses oriented-rectangle overlap, `giou` uses the joint
convex hull, and `diou`/`ciou` use the rotation-invariant minimum-area joint
oriented enclosure for center-distance normalization. OBB `ciou` is a custom
experimental long/short-side aspect adaptation. OBB `hmiou` is an experimental
product of oriented IoU and global-y projection IoU; use it only when image
vertical is a meaningful height or depth cue. See the
[association function guide](../docs/config/trackers.md#association-function)
for all OBB definitions and score normalization.

From another terminal, verify that it is ready:

```bash
curl --fail http://127.0.0.1:8000/healthz
```

It supports ByteTrack, OcSort, and SFSORT and does not require source image
pixels. Send one request per frame, including `width` and `height` when no image
is supplied.

In-process BoxMOT trackers infer AABB or OBB mode automatically from each
non-empty detection row's column count. The HTTP API still declares `box_type`
because it fixes one session schema before detections reach the tracker, keeps
empty frames unambiguous, and determines the response's `track_columns` layout.
The server checks non-empty row widths against that declaration. AABB is the
HTTP default; OBB sessions must set `"box_type": "obb"`.

AABB detection rows use `(x1, y1, x2, y2, confidence, class_id)`:

```bash
curl --fail --request POST \
  --url http://127.0.0.1:8000/v1/streams/camera-01/sessions/aabb-demo/frames \
  --header 'content-type: application/json' \
  --data '{
    "frame_id": 0,
    "width": 640,
    "height": 480,
    "box_type": "aabb",
    "detections": [[10, 20, 60, 120, 0.95, 0]]
  }'
```

OBB detection rows use
`(center_x, center_y, width, height, angle_radians, confidence, class_id)`:

```bash
curl --fail --request POST \
  --url http://127.0.0.1:8000/v1/streams/camera-01/sessions/obb-demo/frames \
  --header 'content-type: application/json' \
  --data '{
    "frame_id": 0,
    "width": 640,
    "height": 480,
    "box_type": "obb",
    "detections": [[35, 70, 50, 100, 0.1, 0.95, 0]]
  }'
```

Use a separate session when changing `box_type`. Continue each session with
contiguous `frame_id` values (`1`, `2`, ...) and send `"detections": []` when
a frame has no detections. The response's `track_columns` field defines the
column order of each returned track.

Prediction uses fixed steps by default, preserving established tracker tuning.
Optional `timestamp_s` values remain metadata. To try experimental prediction
in elapsed seconds, start the container with `-e BOXMOT_VARIABLE_DT=true` and
include finite, strictly increasing capture timestamps on frame 0 and every
subsequent request. Timestamps `12.0` and `12.04` on frames 0 and 1 then give
the tracker a prediction interval of 0.04 seconds. Exact retries retain their
original timestamps. Motion priors use a fixed `1/30`-second reference for unit
conversion; actual prediction intervals come from the timestamps. This mode
needs separate motion-noise calibration and does not support SFSORT; track
expiration still counts updates. See
[capture timestamps](../docs/guides/deployment.md#capture-timestamps) for session
validation and tuning limitations.

Run the CUDA/ReID service with an NVIDIA GPU and a mounted checkpoint:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models/osnet_x0_25_msmt17.pt:/models/osnet_x0_25_msmt17.pt:ro" \
  -e BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt \
  boxmot/boxmot-service:local-gpu
```

The GPU service defaults to BotSort and also supports StrongSort, DeepOcSort,
HybridSort, BoostTrack, and OccluBoost. Its request must contain a raw
base64-encoded JPEG or PNG in `image_base64` for every frame, even when
`detections` is empty. Base64 increases the compressed payload by roughly 33%,
so prefer compressed JPEG for high-volume streams and enforce request-size
limits at ingress.

For example, send `frame.jpg` from Python. The service infers its dimensions:

```python
import base64
from pathlib import Path

import requests

payload = {
    "frame_id": 0,
    "frame_rate": 30,
    "box_type": "aabb",
    "detections": [[10, 20, 60, 120, 0.95, 0]],
    "image_base64": base64.b64encode(Path("frame.jpg").read_bytes()).decode("ascii"),
}
response = requests.post(
    "http://127.0.0.1:8000/v1/streams/camera-01/sessions/gpu-demo/frames",
    json=payload,
    timeout=30,
)
response.raise_for_status()
print(response.json())
```

Omit `width` and `height` when supplying an image. If provided, they must match
the encoded image. Frame dimensions must remain fixed within a session. Send
only the raw base64 text, without a `data:image/...;base64,` prefix.

Neither service runs detector inference. Keep one service process per
container; the GPU process loads and warms one ReID model shared by its tracker
sessions and defaults to one concurrent tracker update. Scale with multiple
containers and route every stream/session consistently to the same instance.
See the [deployment guide](../docs/guides/deployment.md) for the request schema,
state model, and scaling constraints.

A GitHub Actions runner is intentionally not included. A self-hosted runner is
CI infrastructure with a different security and lifecycle model, not a BoxMOT
runtime image.
