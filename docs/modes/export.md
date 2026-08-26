# Export

Use `export` to convert ReID models to TorchScript, ONNX, OpenVINO, TensorRT,
native Core ML, or TFLite.

Format-specific Python packages are installed on first use when possible. TensorRT export also attempts to install `nvidia-tensorrt`, but the resulting wheel still needs a compatible CUDA/NVIDIA runtime.

TensorRT and OpenVINO use ONNX as an intermediate. If you request only `engine` or `openvino`, BoxMOT creates or reuses a fresh `.onnx` file next to the source weights before building the requested format.

Core ML export is native and does not pass through ONNX Runtime. It produces an
FP16 MLProgram bundle with static batch buckets (1, 8, 16, and 32 by default).
The runtime pads or chunks arbitrary detection counts and lazily keeps one
compiled package resident. Conversion workers have configurable time and RAM
limits to prevent runaway Apple graph compilation.

## Examples

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
          --tflite-calibration-data Market-1501-v15.09.15/bounding_box_train \
          --tflite-calibration-samples 512 \
          --tflite-calibration-seed 0 \
          --tflite-calibration-update minmax \
          --tflite-static-activation-bits 16
        ```

        Static TFLite uses int8 weights. The default `--tflite-static-activation-bits 16`
        preserves ReID embedding parity better but can be slower on CPU; use `8` only
        for strict int8 activation ablations.

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(reid="osnet_x0_25_msmt17")
        exported = boxmot.export(
            include=("onnx", "engine"),
            dynamic=True,
            batch_size=16,
            device="0",
        )
        print(exported.files)

        reid = BoxMOT(reid="models/lmbn_n_duke.pt")
        exported = reid.export(format="onnx")
        embeddings = exported.embed(source="path/to/image.jpg")

        apple_reid = BoxMOT(reid="runs/reid_train/exp/best.pt")
        apple_export = apple_reid.export(
            format="coreml",
            coreml_batch_buckets=(1, 8, 16, 32),
        )
        print(apple_export.files["coreml"])
        ```

## Typical use cases

- deploy a ReID backbone outside BoxMOT
- prepare ReID models for inference benchmarks
- build an optimized runtime for a tracker that uses appearance features

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: export
    :style: table
    :prog_name: boxmot export
