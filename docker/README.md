# Docker images

BoxMOT uses one shared multi-stage Dockerfile for four independently built
images:

| Target | Suggested tag | Contents |
| --- | --- | --- |
| `cli-gpu` | `boxmot/boxmot:latest` | Full detector, ReID, CLI, and evaluation stack with CUDA 13.0 PyTorch |
| `cli-cpu` | `boxmot/boxmot:latest-cpu` | The same full stack with CPU-only PyTorch |
| `service-cpu` | `boxmot/boxmot-service:latest` | Non-root CPU geometry-only detection-to-track HTTP service |
| `service-gpu` | `boxmot/boxmot-service:latest-gpu` | Non-root CUDA/ReID detection-to-track HTTP service |

The CPU and CUDA selections come from mutually exclusive, lockfile-backed `cpu`
and `cu130` extras in the root project. Docker, local development, and CI all
consume the same `pyproject.toml` and `uv.lock`. The CPU service installs only
the minimal `service-runtime` group and runs BoxMOT directly from source, so it
contains neither PyTorch nor CUDA and uses headless OpenCV. The GPU service
selects the root CUDA/ReID and HTTP extras, without detector or evaluation
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

## Publish

GitHub Actions builds, smoke-tests, and pushes all four targets only when a
GitHub release is published. Pull requests, branch pushes, and manual workflow
runs do not build or publish these images. Each target is pushed only after its
smoke test passes. The release tag may optionally start with `v`, but its
remaining value must match `[project].version` in `pyproject.toml`.

## Run

Run the CUDA CLI image with the NVIDIA Container Toolkit:

```bash
docker run --rm -it --ipc=host --gpus all boxmot/boxmot:local
```

Run the CPU CLI image:

```bash
docker run --rm -it --ipc=host boxmot/boxmot:local-cpu
```

Run the CPU geometry-only detection-to-track service:

```bash
docker run --rm -p 8000:8000 boxmot/boxmot-service:local
```

From another terminal, verify that it is ready:

```bash
curl --fail http://127.0.0.1:8000/healthz
```

It supports ByteTrack, OCSort, and SFSORT and does not need image pixels. Send
one request per frame. AABB detection rows use
`(x1, y1, x2, y2, confidence, class_id)`:

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

Run the CUDA/ReID service with an NVIDIA GPU and a mounted checkpoint:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models/osnet_x0_25_msmt17.pt:/models/osnet_x0_25_msmt17.pt:ro" \
  -e BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt \
  boxmot/boxmot-service:local-gpu
```

The GPU service defaults to BotSORT and also supports StrongSORT, DeepOCSORT,
HybridSORT, BoostTrack, and OccluBoost. Its request must contain a raw
base64-encoded JPEG or PNG in `image_base64` for every frame, even when
`detections` is empty. Base64 increases the compressed payload by roughly 33%,
so prefer compressed JPEG for high-volume streams and enforce request-size
limits at ingress.

For example, if `frame.jpg` is exactly 640 by 480 pixels, stream its base64
bytes directly into an AABB request without storing the encoded image in a
shell variable:

```bash
{
  printf '%s' \
    '{"frame_id":0,"width":640,"height":480,"frame_rate":30,' \
    '"box_type":"aabb","detections":[[10,20,60,120,0.95,0]],' \
    '"image_base64":"'
  base64 < frame.jpg | tr -d '\r\n'
  printf '%s' '"}'
} | curl --fail --request POST \
  --url http://127.0.0.1:8000/v1/streams/camera-01/sessions/gpu-demo/frames \
  --header 'content-type: application/json' \
  --data-binary @-
```

The declared `width` and `height` must exactly match the encoded image. Send
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
