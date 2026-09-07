# Performance benchmarks

This directory contains opt-in benchmark CLIs. They report measurements instead
of enforcing machine-specific timing thresholds during the normal pytest suite.
Functional behavior remains covered under `tests/unit`.

The benchmarks are grouped by domain:

- `motion/benchmark_cmc.py` measures camera-motion compensation on the bundled
  MOT17 mini frames.
- `trackers/benchmark_fps.py` measures tracker-update throughput with synthetic
  detections. ReID trackers use precomputed embeddings by default so the timing
  isolates tracking; pass `--reid-mode live` to include ReID inference.
- `reid/benchmark_inference.py` compares ReID runtime latency in isolated worker
  processes and can export missing ONNX/Core ML artifacts.
- `benchmark_eval.py` times fresh evaluation CLI invocations on an existing
  materialized build, records setup/replay/metrics separately, and verifies
  tracking-file hashes and metrics across repetitions.

Run each benchmark as a module from the repository root:

```bash
uv run --no-sync python -m tests.performance.motion.benchmark_cmc
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
