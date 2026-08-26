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

Run each benchmark as a module from the repository root:

```bash
uv run python -m tests.performance.motion.benchmark_cmc
uv run python -m tests.performance.trackers.benchmark_fps
uv run python -m tests.performance.reid.benchmark_inference --weights models/osnet_x0_25_msmt17.pt
```

Use `--help` for benchmark-specific controls and JSON/CSV output options. Warmup
work and synthetic-input generation are kept outside measured intervals. Compare
results only on the same machine and software stack.
