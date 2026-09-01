# Docker images

The shared Dockerfile has two named targets:

- `cli`: detector, CLI, evaluation, and interactive workflows with the `yolo`
  and `trackeval` extras.
- `service`: a non-root HTTP tracker that accepts detections and keeps isolated
  state for each stream/session.

The service target omits detector and evaluation extras. It still contains the
current BoxMOT core dependency set, including its shared ML runtime.

Build either image from the repository root:

```bash
docker build --target cli -f docker/Dockerfile -t boxmot/boxmot:local .
docker build --target service -f docker/Dockerfile -t boxmot/boxmot-service:local .
```

Run the detection-to-track service:

```bash
docker run --rm -p 8000:8000 boxmot/boxmot-service:local
```

See the [deployment guide](../docs/guides/deployment.md) for the request schema,
state model, and scaling constraints.
