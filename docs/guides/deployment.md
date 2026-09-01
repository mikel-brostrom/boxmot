# Tracker service deployment

Use the service image when detection runs elsewhere and BoxMOT only needs to
associate detections into persistent tracks. Each stream/session URL owns one
tracker instance.

## Build and run

Build the production target from the repository root:

```bash
docker build \
  --target service \
  -f docker/Dockerfile \
  -t boxmot/boxmot-service:local \
  .

docker run --rm -p 8000:8000 boxmot/boxmot-service:local
```

Check `http://localhost:8000/healthz` for liveness,
`http://localhost:8000/readyz` for capacity, and
`http://localhost:8000/docs` for the generated OpenAPI interface.

## Send detections

Send exactly one request for each frame, including frames with no detections.
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

The service defaults to ByteTrack. Its process-level settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BOXMOT_SERVICE_TRACKER` | `bytetrack` | Geometry-only tracker: `bytetrack` or `ocsort`. |
| `BOXMOT_SERVICE_PORT` | `8000` | Internal HTTP port. |
| `BOXMOT_SERVICE_MAX_STREAMS` | `256` | Maximum resident tracker sessions. |
| `BOXMOT_SERVICE_STREAM_TTL_SECONDS` | `900` | Idle time before an unlocked session can be evicted. |
| `BOXMOT_SERVICE_MAX_DETECTIONS` | `1000` | Per-frame detection limit; configurable up to 2000. |
| `BOXMOT_SERVICE_MAX_CLASSES` | `32` | Maximum distinct class IDs retained by one session. |
| `BOXMOT_SERVICE_MAX_CONCURRENT_UPDATES` | `4` | Process-wide tracker updates allowed at once. |

For example:

```bash
docker run --rm \
  -p 8080:8080 \
  -e BOXMOT_SERVICE_PORT=8080 \
  -e BOXMOT_SERVICE_TRACKER=ocsort \
  -e BOXMOT_SERVICE_MAX_STREAMS=512 \
  boxmot/boxmot-service:local
```

## Scale replicas

Run one Uvicorn worker per container. Tracker objects are mutable and live in
that process, so all requests for the same stream/session must reach the same
replica. Configure the load balancer for sticky routing or consistent hashing
on the stream/session key, then scale by adding containers.

A container restart discards its sessions. Start a new session ID and replay
the sequence from its first required frame if state must be reconstructed.
Persistent shared storage alone cannot move a live tracker object between
workers.

The service does not receive image pixels. It therefore limits the tracker
choice to ByteTrack and OCSort and does not provide ReID, camera-motion
compensation, or detector inference. Use the CLI image or Python API when those
features are needed. The image omits detector and evaluation extras but still
contains BoxMOT's shared core ML dependencies. Place authentication, TLS,
request-size limits, and rate limiting at the ingress or API gateway.
