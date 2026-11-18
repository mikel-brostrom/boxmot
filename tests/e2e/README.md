# Real Compose smoke and recovery test

Run from the repository root with Docker daemon access and host `ffmpeg`:

```bash
uv run --no-sync python -m tests.e2e.compose_attributes --profile cpu
```

The test builds the current BoxMOT CPU image and starts PostgreSQL and vLLM.
It uses actual YOLO detection and attribute inference on a small crop from the
bundled MOT17 fixture. A successful tracking exit alone is insufficient: the
test requires completed attributes in both exported JSONL and PostgreSQL.

It then serves a repeating JPEG as a real HTTP camera stream and exercises:

- continuous capture and attribute export;
- competing camera ownership, requiring an ownership rejection and unchanged lease/session identities;
- camera disconnect/reconnect with fresh session IDs;
- SIGTERM during an outstanding inference request, requiring an explicit cancellation for the observed batch and a clean exit;
- database loss, worker termination, preserved event IDs/payloads, and a newly completed result after recovery;
- correspondence between synchronized JSONL payloads and database acknowledgments, including consistency of any duplicate event IDs.

The shutdown check briefly pauses this test's vLLM container and starts a fresh
worker session, so inference cannot finish before SIGTERM reaches the worker.
The server is unpaused in cleanup. No model responses are fabricated.

Each invocation uses a unique Compose project and database volume. Existing
application databases and running Compose projects are not selected. Model
caches are reused. Exact commands, logs, JSONL, and `result.json` are retained
under the printed `runs/boxmot-e2e-.../` directory. The test stops its containers
on completion or failure and preserves its database volume for inspection.

For the GPU Compose variant, use `--profile gpu`. Docker must support NVIDIA
GPU allocation, and the GPU must support the selected model/image. The existing
Compose defaults share GPU 0; set `BOXMOT_GPU_ID=0 VLLM_GPU_ID=1` to test distinct
devices on a two-GPU host. This does not test Kubernetes GPU scheduling.

To reuse an already verified BoxMOT image, pass `--image IMAGE_REFERENCE`. This
skips the build; the image must contain the code you intend to test. The default
always builds the current checkout.

`--prepare-only` generates the fixture and checks Compose configuration without
starting containers. It does **not** establish that model inference works.

This bounded test does not replace a 24-hour deployment soak, physical-camera
testing, Kubernetes/PVC validation, or the dedicated retention and replay tests.
It does not inject output failures, a crash between synchronization and acknowledgment,
or a blocked capture backend. Queue capacity, rotation, retention, stale inference
responses after takeover, and bounded memory need their dedicated tests.
Cold image builds, downloads, and CPU inference may take several minutes. Each
stage has a timeout and retains its log on failure.
