# Export

Use `export` to convert ReID models to deployment formats such as ONNX and TensorRT.

Format-specific Python packages are installed on first use when possible. TensorRT export also attempts to install `nvidia-tensorrt`, but the resulting wheel still needs a compatible CUDA/NVIDIA runtime.

TensorRT and OpenVINO use ONNX as an intermediate. If you request only `engine` or `openvino`, BoxMOT creates or reuses a fresh `.onnx` file next to the source weights before building the requested format.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot export --weights osnet_x0_25_msmt17.pt --include onnx
        ```

        Export multiple formats:

        ```bash
        boxmot export \
          --weights osnet_x0_25_msmt17.pt \
          --include engine \
          --dynamic
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
        exported = boxmot.export(include=("onnx", "engine"), dynamic=True)
        print(exported.files)

        reid = BoxMOT(reid="models/lmbn_n_duke.pt")
        reid = reid.export(format="onnx", half=True)
        embeddings = reid.embed(source="path/to/image.jpg")
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
