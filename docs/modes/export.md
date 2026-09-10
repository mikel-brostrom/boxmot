# Export

Use `export` to convert ReID models to TorchScript, ONNX, OpenVINO, TensorRT,
native Core ML, or TFLite.

Install format-specific dependencies explicitly before exporting, for example
with `boxmot install --extra onnx`. Exporters validate their requirements and
report missing packages. See [Install dependencies](install.md) for other
extras and TensorRT setup.

TensorRT and OpenVINO use ONNX as an intermediate. If you request only `engine` or `openvino`, BoxMOT creates or reuses a fresh `.onnx` file next to the source weights before building the requested format.

Core ML export is native and does not pass through ONNX Runtime. It produces an
FP16 MLProgram bundle with static batch buckets (1, 8, 16, and 32 by default).
The runtime pads or chunks arbitrary detection counts and lazily keeps one
compiled package resident. Conversion workers have configurable time and RAM
limits to prevent runaway Apple graph compilation.

## Examples

`--device` follows the shared [device selection rules](track.md#device-selection).
Choose a device supported by the requested export format; TensorRT requires
CUDA, while the Core ML example below performs conversion on CPU.

!!! example

    === "CLI"

        ```bash
        boxmot export --weights osnet_x0_25_msmt17.pt --include onnx
        ```

        Export a transformer ReID model for Apple GPU/CPU inference:

        ```bash
        boxmot export \
          --weights runs/reid_train/exp/best.pt \
          --include coreml \
          --device cpu \
          --coreml-batch-buckets 1,8,16,32 \
          --coreml-minimum-deployment-target macOS15 \
          --coreml-compute-units CPUAndGPU \
          --coreml-timeout 600 \
          --coreml-max-memory-gb 16
        ```

        The output is `best_coreml_model/`. Pass that directory directly as
        ReID weights. `BOXMOT_COREML_MAX_LOADED_BUCKETS=1` is the safe default;
        increasing it trades RAM for fewer bucket recompilations.

        Export multiple formats:

        ```bash
        boxmot export \
          --weights osnet_x0_25_msmt17.pt \
          --include onnx \
          --include engine \
          --dynamic \
          --batch-size 16 \
          --device 0
        ```

        Export calibrated TFLite int8 using representative ReID crops:

        ```bash
        boxmot export \
          --weights runs/reid_train/exp/best.pt \
          --include tflite \
          --tflite-quantize static \
          --tflite-calibration-data datasets/reid/Market-1501-v15.09.15/bounding_box_train \
          --tflite-calibration-samples 512 \
          --tflite-calibration-seed 0 \
          --tflite-calibration-update minmax \
          --tflite-static-activation-bits 16
        ```

        Static TFLite uses int8 weights. The default `--tflite-static-activation-bits 16`
        preserves ReID embedding parity better but can be slower on CPU; use `8` only
        for strict int8 activation ablations.

    === "Python"

        Reusable exporter implementations live in `boxmot.reid.exporters.backends`.
        Shared configuration, registry, and model preparation live in `boxmot.reid.exporters`.
        CLI parsing, output naming, and workflow orchestration remain engine
        concerns and are intentionally absent from the package root.

## Typical use cases

- deploy a ReID backbone outside BoxMOT
- prepare ReID models for inference benchmarks
- build an optimized runtime for a tracker that uses appearance features

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.reid.export
    :command: export
    :depth: 0
    :style: table
    :prog_name: boxmot export
