# Performance benchmarks

This directory contains opt-in benchmark CLIs. They report measurements instead
of enforcing machine-specific timing thresholds during the normal pytest suite.
Functional behavior remains covered under `tests/unit`.

The benchmarks are grouped by domain:

- `trackers/motion/benchmark_cmc.py` measures camera-motion compensation on the bundled
  MOT17 mini frames.
- `trackers/motion/benchmark_kalman.py` compares scalar and batched prediction and
  correction for every shared Kalman model and EagerMOT's 3D model. It verifies
  numerical parity, limits BLAS to one thread, and includes object bookkeeping
  for stateful filters. Pass `--baseline-root` with a prior checkout to also
  measure improvements over existing XYAH/XYWH batch prediction.
- `trackers/benchmark_kitti_kalman.py` compares EagerMOT KITTI evaluation against
  a saved package snapshot. It holds sequence workers and numerical threads
  constant, records replay/evaluation timings separately, and checks every
  tracking file and metric across warmups and measured runs.
- `trackers/benchmark_fps.py` measures tracker-update throughput with synthetic
  detections. ReID trackers use precomputed embeddings by default so the timing
  isolates tracking; pass `--reid-mode live` to include ReID inference.
- `reid/benchmark_inference.py` compares ReID runtime latency in isolated worker
  processes and can export missing ONNX/Core ML artifacts.
- `benchmark_eval.py` times fresh evaluation CLI invocations on an existing
  materialized build, records setup/replay/metrics separately, and verifies
  tracking-file hashes and metrics across repetitions.
- `benchmark_cli_startup.py` measures launch-to-first-visible-output latency
  through a real terminal on macOS/Linux, without importing workflow modules
  ahead of the CLI. It also records the first Setup and workflow titles.

Run each benchmark as a module from the repository root:

```bash
uv run --no-sync python -m tests.performance.trackers.motion.benchmark_cmc
uv run --no-sync python -m tests.performance.trackers.motion.benchmark_kalman --json /tmp/kalman.json
uv run --no-sync python -m tests.performance.trackers.benchmark_fps
uv run --no-sync python -m tests.performance.reid.benchmark_inference --weights models/osnet_x0_25_msmt17.pt
```

Use `--help` for benchmark-specific controls and JSON/CSV output options. Warmup
work and synthetic-input generation are kept outside measured intervals. Compare
results only on the same machine and software stack.

For cached MOT17 evaluation, keep the tracker, build, and worker count fixed
while comparing code changes:

```bash
uv run --no-sync python -m tests.performance.benchmark_eval \
  --build runs/materializations/BUILD_ID --tracker sfsort --workers 7 \
  --repeat 3 --label baseline
uv run --no-sync python -m tests.performance.benchmark_eval \
  --build runs/materializations/BUILD_ID --tracker sfsort --workers 7 \
  --repeat 3 --label optimized \
  --compare-to runs/replay-benchmark/baseline-1.json
```

Each run writes a log, stage timings, runtime metadata, metrics, and SHA-256
checksums under `runs/replay-benchmark`. `--reuse-build` also times the MOT17
ablation component-selection workflow (`yolox-x-mot17`, `lmbn-n-duke`, MPS),
and checks that it selects the supplied build. `--profile --repeat 1` writes
one cProfile file per sequence; inspect these separately from unprofiled
timings. Worker profiles cover the replay thread; prefetched input decoding
runs on a separate thread.

Use repeated medians with warm filesystem caches, and run benchmarks one at
a time. Output is redirected to logs, so interactive terminal rendering costs
can differ. Invocation time includes process startup/shutdown and the small
checksum/JSON-reporting overhead. Profiled timings include instrumentation
overhead and should not be used to claim speedups.

For startup and interactive UI measurements, run the command through a PTY:

```bash
uv run --no-sync python -m tests.performance.benchmark_cli_startup \
  --repeat 3 --label startup -- \
  eval --dataset mot17 --split ablation --detector yolox-x-mot17 \
  --reid lmbn-n-duke --tracker sfsort --device mps
```

This benchmark writes terminal transcripts and timestamp JSON files under
`runs/replay-benchmark/startup`. Its invocation timer includes interpreter
startup, runtime imports, interactive rendering, and shutdown. Workflow-title
timestamps indicate the first displayed panel, including transient Setup
panels; use `benchmark_eval.py` for execution-stage timings and output equality.
