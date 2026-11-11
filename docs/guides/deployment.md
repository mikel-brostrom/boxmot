# Tracker service deployment

The tracker service has two deployment profiles. Both receive detections from
an external detector and associate them into persistent tracks; neither image
runs detector inference. Each stream/session URL owns one tracker instance.

| Profile | Build target | Image | Frame pixels |
| --- | --- | --- | --- |
| CPU geometry | `service-cpu` | `boxmot/boxmot-service:latest` | Not required |
| CUDA/ReID | `service-gpu` | Local build, shown below | Required on every frame |

The CPU service uses CPU-only Torch for canonical structures and contains no
CUDA runtime. The GPU service uses CUDA 13.0 Torch and a shared ReID encoder.
Neither service runs detector inference. The CPU profile supports ByteTrack,
OcSort, and SFSORT; the GPU profile supports StrongSort, BotSort (default),
DeepOcSort, HybridSort, BoostTrack, and OccluBoost.

## Build and run

Run the published CPU image:

```bash
docker pull boxmot/boxmot-service:latest
docker run --rm -p 8000:8000 boxmot/boxmot-service:latest
```

For GPU tracking, build the image from the repository root. Run it with the
NVIDIA Container Toolkit and a ReID checkpoint mounted at
`/models/osnet_x0_25_msmt17.pt`:

```bash
docker build --target service-gpu -f docker/Dockerfile -t boxmot/boxmot-service:local-gpu .
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models/osnet_x0_25_msmt17.pt:/models/osnet_x0_25_msmt17.pt:ro" \
  -e BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt \
  boxmot/boxmot-service:local-gpu
```

To build the CPU service locally instead:

```bash
docker build --target service-cpu -f docker/Dockerfile -t boxmot/boxmot-service:local .
docker run --rm -p 8000:8000 boxmot/boxmot-service:local
```

Check `http://localhost:8000/healthz` for liveness,
`http://localhost:8000/readyz` for capacity, and
`http://localhost:8000/docs` for the generated OpenAPI interface.

`/readyz` reports `active_streams` and `max_streams`; it still returns HTTP 200
when all slots are occupied. New sessions receive HTTP 503 until capacity is
available. Existing sessions can continue sending frames.

### Run without Docker

The installed package also provides a single-worker service entry point:

```bash
boxmot install --extra service
boxmot-service
```

It defaults to the CPU profile and listens on `0.0.0.0:8000`. Set the same
environment variables listed below to configure it. For the GPU profile, select
a CUDA PyTorch environment and supply an existing ReID checkpoint; the service
does not download missing weights at startup.

## Send detections

Send exactly one request for each frame, including frames with no detections.
The HTTP `box_type` fixes the session schema, the canonical `TrackerSpec`
geometry, and the response layout before the first update. It also makes empty
frames unambiguous. Non-empty row widths are validated against it. AABB is the
HTTP default; OBB sessions must set `"box_type": "obb"`.

The default AABB row is `(x1, y1, x2, y2, confidence, class_id)`:

```bash
curl --request POST \
  --url http://localhost:8000/v1/streams/camera-01/sessions/run-01/frames \
  --header 'content-type: application/json' \
  --data '{
    "frame_id": 0,
    "width": 1920,
    "height": 1080,
    "frame_rate": 30,
    "box_type": "aabb",
    "detections": [
      [620.0, 210.0, 790.0, 690.0, 0.94, 0],
      [910.0, 240.0, 1050.0, 680.0, 0.88, 0]
    ]
  }'
```

For an empty frame, send `"detections": []` with the next `frame_id`. For OBB,
set `"box_type": "obb"` and use
`(cx, cy, width, height, angle_radians, confidence, class_id)` rows.

Coordinates are pixel values, with positive box width/height; scores must be
between 0 and 1 and class IDs must be nonnegative integers. Unknown request
fields are rejected. Stream and session IDs must start with a letter or digit,
contain only letters, digits, `_`, `.`, and `-`, and be at most 128 characters.

The CPU profile does not require source image pixels; requests without an image
must include both `width` and `height`. When `image_base64` is supplied, the
service infers dimensions from the decoded image. The GPU profile requires
`image_base64` to contain the raw base64 text of a valid JPEG or PNG for every
request, including frames whose `detections` array is empty:

```json
{
  "frame_id": 0,
  "frame_rate": 30,
  "box_type": "aabb",
  "detections": [[620.0, 210.0, 790.0, 690.0, 0.94, 0]],
  "image_base64": "<raw base64-encoded JPEG or PNG>"
}
```

Do not include a `data:image/...;base64,` prefix. Omit `width` and `height` when
supplying an image; if provided, they must match the decoded image. ReID and
camera-motion compensation need the real frame even when the detector found
nothing, so an empty detection frame cannot omit `image_base64`.

For example, a Python client can attach a compressed frame without specifying
its dimensions:

```python
import base64
from pathlib import Path

import requests

payload = {
    "frame_id": 0,
    "frame_rate": 30,
    "box_type": "aabb",
    "detections": [[620.0, 210.0, 790.0, 690.0, 0.94, 0]],
    "image_base64": base64.b64encode(Path("frame.jpg").read_bytes()).decode("ascii"),
}
requests.post(
    "http://localhost:8000/v1/streams/camera-01/sessions/run-01/frames",
    json=payload,
    timeout=30,
).raise_for_status()
```

Base64 adds roughly 33% to the compressed image size and JSON adds a small
additional cost. Prefer compressed JPEG where its quality is sufficient, and
configure ingress request-body limits and timeouts for the resulting payloads.
The service also validates compressed-byte and decoded-pixel limits described below.

The response includes a `track_columns` array that describes each dense track
row. AABB output is
`(x1, y1, x2, y2, id, confidence, class_id, detection_index)`; OBB adds the
angle before `id`. Use `detection_index` to relate a returned track to the
corresponding input row. Track and detection counts are not guaranteed to be
equal.

Frame dimensions, whether inferred or explicit, frame rate, and box type are
fixed after the first request for a session. Frames must then be contiguous,
with at most one request in flight for each session. A gap or conflicting retry
returns HTTP 409 without advancing the tracker. Repeating the most recent frame
with the exact same body safely replays its cached response. A new session must
start with frame 0; this also prevents an expired or misrouted session from
silently restarting midway through a sequence.

The response includes `next_frame_id`, which is the next expected frame, and
`replayed`, which is true when the most recent successful result is returned
without updating the tracker again. Track IDs are session-local; identify a
track by its stream, session, and track ID together.

### Handle failures

| HTTP status | Meaning | Client action |
| --- | --- | --- |
| `422` | Invalid request, detections, image, or first-frame timing | Correct the payload before retrying; tracking has not advanced. |
| `409` | Frame order, retry content, timing, or session dimensions conflict | Send the expected frame with unchanged session settings, or start a new session. |
| `503` | New-session capacity is exhausted | Honor `Retry-After: 1`, or delete an unused session. |
| `500` | Tracker creation or update failed | Inspect the server error. A failed update discards the session; start from frame 0 under a new session ID. |

After a network timeout, retry the same most recent request before advancing.
Only the last successful frame is cached. Keep encoded image bytes unchanged
on retries, because they participate in request identity.

Delete a session to release its tracker immediately:

```bash
curl --request DELETE \
  http://localhost:8000/v1/streams/camera-01/sessions/run-01
```

## Capture timestamps

The service defaults to fixed-step prediction (`BOXMOT_VARIABLE_DT=false`),
matching established tracker tuning. The optional `timestamp_s` request field
is metadata by default and does not change motion prediction.

To try experimental prediction in elapsed seconds, start the service with
`BOXMOT_VARIABLE_DT=true`. This mode needs separate motion-noise calibration;
existing benchmark accuracy is not guaranteed. Supply a finite capture or
media timestamp in seconds on every frame, including frames without
detections. For example, the CPU service accepts this first request:

```json
{
  "frame_id": 0,
  "timestamp_s": 12.0,
  "width": 1920,
  "height": 1080,
  "detections": [[620.0, 210.0, 790.0, 690.0, 0.94, 0]]
}
```

The first timestamp establishes the session clock. A following request with
`"frame_id": 1` and `"timestamp_s": 12.04` advances Kalman prediction by 0.04
seconds. The service passes the capture timestamp to the tracker, which
computes the interval internally. GPU requests use the same field alongside
their required image.
Use capture or media timestamps rather than processing or network arrival
times, and keep frame IDs contiguous even when capture intervals vary.

Timing mode is process configuration; request timestamps and `frame_rate` do
not enable it. Request `frame_rate` is stream metadata; it does not scale
ByteTrack or BotSort's `max_lost_frames` lifetime, which counts tracker updates.
When enabled, a missing timestamp on a new session's first
frame returns HTTP 422 before creating a tracker. Missing or non-increasing
timestamps on subsequent frames return HTTP 409 without advancing the tracker.
When disabled, optional timestamps may appear, disappear, or change without
affecting prediction. In either mode, changing a timestamp on a retry returns
HTTP 409; an exact retry of the most recent frame replays its cached response.

Service trackers except SFSORT support this option. SFSORT uses fixed steps
and rejects `BOXMOT_VARIABLE_DT=true`. When enabled, motion priors are converted to
seconds using a fixed `1/30`-second reference interval, then integrated over
the actual capture interval. The reference does not replace capture timestamps;
review [elapsed-time prediction](../python/index.md#elapsed-time) when tuning
noise. Track expiration and confirmation settings remain counts of updates.

## Configure the process

The CPU image defaults to ByteTrack; the GPU image defaults to BotSort. Their
process-level settings are:

| Variable | CPU default | GPU default | Purpose |
| --- | --- | --- | --- |
| `BOXMOT_SERVICE_PROFILE` | `cpu` | `gpu` | Selects the tracker allowlist and whether images/ReID are required. Use the profile built into the image. |
| `BOXMOT_SERVICE_TRACKER` | `bytetrack` | `botsort` | CPU: `bytetrack`, `ocsort`, or `sfsort`. GPU: `strongsort`, `botsort`, `deepocsort`, `hybridsort`, `boosttrack`, or `occluboost`. |
| `BOXMOT_SERVICE_ASSO_FUNC` | `iou` | `iou` | Geometry used for AABB or OBB detection-track matching: `iou`, `giou`, `diou`, `ciou`, `hmiou`, or `centroid`. |
| `BOXMOT_VARIABLE_DT` | `false` | `false` | Opt into experimental elapsed-seconds motion; requires `timestamp_s` on every frame. Unsupported by SFSORT. |
| `BOXMOT_SERVICE_DEVICE` | `cpu` | `0` | ReID device passed to the GPU backend. |
| `BOXMOT_SERVICE_HALF` | `false` | `true` | Enables FP16 ReID inference; relevant to the GPU profile. |
| `BOXMOT_SERVICE_REID_WEIGHTS` | Not used | `/models/osnet_x0_25_msmt17.pt` | Mounted ReID checkpoint path. |
| `BOXMOT_SERVICE_PORT` | `8000` | `8000` | Internal HTTP port. |
| `BOXMOT_SERVICE_MAX_STREAMS` | `256` | `256` | Maximum resident tracker sessions. |
| `BOXMOT_SERVICE_STREAM_TTL_SECONDS` | `900` | `900` | Idle time before an unlocked session can be evicted. |
| `BOXMOT_SERVICE_MAX_DETECTIONS` | `1000` | `1000` | Per-frame detection limit; configurable up to 2000. |
| `BOXMOT_SERVICE_MAX_CLASSES` | `32` | `32` | Maximum distinct class IDs retained by one session. |
| `BOXMOT_SERVICE_MAX_CONCURRENT_UPDATES` | `4` | `1` | Process-wide tracker updates. Keep the GPU default when sharing one ReID model. |
| `BOXMOT_SERVICE_MAX_IMAGE_BYTES` | `20000000` | `20000000` | Maximum compressed image size after base64 decoding. |
| `BOXMOT_SERVICE_MAX_FRAME_PIXELS` | `33177600` | `33177600` | Maximum decoded pixel count when an image is supplied. |

For example:

```bash
docker run --rm \
  -p 8080:8080 \
  -e BOXMOT_SERVICE_PORT=8080 \
  -e BOXMOT_SERVICE_TRACKER=ocsort \
  -e BOXMOT_SERVICE_ASSO_FUNC=centroid \
  -e BOXMOT_SERVICE_MAX_STREAMS=512 \
  boxmot/boxmot-service:latest
```

Centroid normalization uses the session's fixed `width` and `height`. The CPU
profile does not require source image pixels when centroid is selected.

For OBB sessions, `iou` uses oriented-rectangle overlap, `giou` uses the joint
convex hull, and `diou`/`ciou` use the rotation-invariant minimum-area joint
oriented enclosure for center-distance normalization. OBB `ciou` is a custom
experimental long/short-side aspect adaptation. OBB `hmiou` is an experimental
product of oriented IoU and global-y projection IoU and is intended only for
scenes where image vertical is a meaningful height or depth cue. The
[association function guide](../config/trackers.md#association-function)
defines every OBB mode and its score normalization.

## Scale replicas

Run one Uvicorn worker per container. Tracker objects are mutable and live in
that process, so all requests for the same stream/session must reach the same
replica. Configure the load balancer for sticky routing or consistent hashing
on the stream/session key, then scale by adding containers.

The GPU process loads one ReID model, then shares it across all of
that process's tracker sessions. Its default concurrency is therefore one.
Keep one worker process per GPU container unless model memory, inference
synchronization, and capacity have been explicitly designed for a different
layout. Add GPU containers to scale and retain sticky stream/session routing.

A container restart discards its sessions. Start a new session ID and replay
the sequence from its first required frame if state must be reconstructed.
Persistent shared storage alone cannot move a live tracker object between
workers.

Both service images include Torch because HTTP rows are decoded into canonical
structures before tracking. The CPU image uses a CPU-only build and omits CUDA
and perception model runtimes. The GPU image instead uses the locked CUDA 13.0
build. Its process owns one shared appearance encoder and enriches
requests before advancing per-stream trackers; injected segmentors follow the
same engine-owned pattern. For detector and batch workflows, use the
[CLI Docker images](../integrations/docker.md). The service images use Python
tracker backends. Place authentication,
TLS, request-size limits, and rate limiting at the ingress or API gateway.

## Run persistent CLI jobs

Use the CLI image for materialization and evaluation. Keep datasets, builds,
and models on host volumes so they persist between runs; the
[Docker batch-job guide](../integrations/docker.md#reproducible-batch-jobs)
provides the mount paths and commands.
