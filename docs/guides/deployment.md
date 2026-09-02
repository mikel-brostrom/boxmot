# Tracker service deployment

The tracker service has two deployment profiles. Both receive detections from
an external detector and associate them into persistent tracks; neither image
runs detector inference. Each stream/session URL owns one tracker instance.

| Profile | Image | Trackers | Frame pixels |
| --- | --- | --- | --- |
| CPU geometry | `boxmot/boxmot-service:latest` | ByteTrack, OCSort, SFSORT | Not required |
| CUDA/ReID | `boxmot/boxmot-service:latest-gpu` | StrongSORT, BotSORT (default), DeepOCSORT, HybridSORT, BoostTrack, OccluBoost | Required on every frame |

## Build and run

Run the published CPU image:

```bash
docker pull boxmot/boxmot-service:latest
docker run --rm -p 8000:8000 boxmot/boxmot-service:latest
```

Run the GPU image with the NVIDIA Container Toolkit and mount the ReID
checkpoint at `/models/osnet_x0_25_msmt17.pt`:

```bash
docker pull boxmot/boxmot-service:latest-gpu
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models/osnet_x0_25_msmt17.pt:/models/osnet_x0_25_msmt17.pt:ro" \
  -e BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt \
  boxmot/boxmot-service:latest-gpu
```

Build either production target from the repository root:

```bash
docker build \
  --target service-cpu \
  -f docker/Dockerfile \
  -t boxmot/boxmot-service:local \
  .

docker build \
  --target service-gpu \
  -f docker/Dockerfile \
  -t boxmot/boxmot-service:local-gpu \
  .

docker run --rm -p 8000:8000 boxmot/boxmot-service:local
```

Check `http://localhost:8000/healthz` for liveness,
`http://localhost:8000/readyz` for capacity, and
`http://localhost:8000/docs` for the generated OpenAPI interface.

## Send detections

Send exactly one request for each frame, including frames with no detections.
In-process BoxMOT trackers infer AABB or OBB mode automatically from each
non-empty detection row's column count. The HTTP contract still declares
`box_type` because it fixes the session schema before tracker input, makes empty
frames unambiguous, and determines the response column layout. Non-empty row
widths are validated against it. AABB is the HTTP default; OBB sessions must set
`"box_type": "obb"`.

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

The CPU profile does not require pixels. The GPU profile requires
`image_base64` to contain the raw base64 text of a valid JPEG or PNG for every
request, including frames whose `detections` array is empty:

```json
{
  "frame_id": 0,
  "width": 1920,
  "height": 1080,
  "frame_rate": 30,
  "box_type": "aabb",
  "detections": [[620.0, 210.0, 790.0, 690.0, 0.94, 0]],
  "image_base64": "<raw base64-encoded JPEG or PNG>"
}
```

Do not include a `data:image/...;base64,` prefix. The decoded image dimensions
must exactly match `width` and `height`. ReID and camera-motion compensation
need the real frame even when the detector found nothing, so an empty detection
frame cannot omit `image_base64`.

For example, a Python client can attach a compressed frame to the same metadata
used by the CPU endpoint:

```python
import base64
from pathlib import Path

import requests

payload = {
    "frame_id": 0,
    "width": 1920,
    "height": 1080,
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

Width, height, frame rate, and box type are fixed after the first request for a
session. Frames must then be contiguous, with at most one request in flight for
each session. A gap or conflicting retry returns HTTP 409 without advancing the
tracker. Repeating the most recent frame with the exact same body safely
replays its cached response. A new session must start with frame 0; this also
prevents an expired or misrouted session from silently restarting midway
through a sequence.

Delete a session to release its tracker immediately:

```bash
curl --request DELETE \
  http://localhost:8000/v1/streams/camera-01/sessions/run-01
```

## Configure the process

The CPU image defaults to ByteTrack; the GPU image defaults to BotSORT. Their
process-level settings are:

| Variable | CPU default | GPU default | Purpose |
| --- | --- | --- | --- |
| `BOXMOT_SERVICE_PROFILE` | `cpu` | `gpu` | Selects the tracker allowlist and whether images/ReID are required. Use the profile built into the image. |
| `BOXMOT_SERVICE_TRACKER` | `bytetrack` | `botsort` | CPU: `bytetrack`, `ocsort`, or `sfsort`. GPU: `strongsort`, `botsort`, `deepocsort`, `hybridsort`, `boosttrack`, or `occluboost`. |
| `BOXMOT_SERVICE_ASSO_FUNC` | `iou` | `iou` | Geometry used for AABB or OBB detection-track matching: `iou`, `giou`, `diou`, `ciou`, `hmiou`, or `centroid`. |
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
profile therefore remains pixel-free when centroid is selected.

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

The GPU process loads and warms one ReID model, then shares it across all of
that process's tracker sessions. Its default concurrency is therefore one.
Keep one worker process per GPU container unless model memory, inference
synchronization, and capacity have been explicitly designed for a different
layout. Add GPU containers to scale and retain sticky stream/session routing.

A container restart discards its sessions. Start a new session ID and replay
the sequence from its first required frame if state must be reconstructed.
Persistent shared storage alone cannot move a live tracker object between
workers.

The CPU service omits detector and evaluation extras and contains neither
PyTorch nor CUDA. The GPU service adds CUDA/ReID but still does not run a
detector. Use `boxmot/boxmot:latest` for full GPU CLI/detector workflows or
`boxmot/boxmot:latest-cpu` for their CPU counterpart. Place authentication,
TLS, request-size limits, and rate limiting at the ingress or API gateway.
