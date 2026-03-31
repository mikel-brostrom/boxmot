---
description: Export BoxMOT ReID models to TorchScript, ONNX, OpenVINO, TensorRT, and TFLite.
---

# Export Mode

`export` converts BoxMOT ReID backbones into deployment formats. This mode is specifically for ReID models and does not export detector weights.

!!! example "Export from CLI or Python"

    === "CLI"

        ```bash
        boxmot export --weights osnet_x0_25_msmt17.pt --include onnx
        ```

    === "Python"

        ```python
        from boxmot import boxmot

        model = boxmot(reid="osnet_x0_25_msmt17")
        results = model.export(include=("onnx",), device="cpu")
        print(results.onnx)
        ```

## Supported Export Targets

BoxMOT supports these `--include` values:

- `torchscript`
- `onnx`
- `openvino`
- `engine`
- `tflite`

You can request more than one format in the same command.

## Common Examples

```bash
# TorchScript
boxmot export --weights osnet_x0_25_msmt17.pt --include torchscript

# ONNX
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --device cpu

# OpenVINO
boxmot export --weights osnet_x0_25_msmt17.pt --include openvino --device cpu

# TensorRT
boxmot export --weights osnet_x0_25_msmt17.pt --include engine --device 0 --dynamic --half
```

## Export Notes

- BoxMOT resolves the runtime device from `--device`.
- `--half` is only valid on GPU exports.
- `--optimize` applies to CPU TorchScript export.
- TensorRT exports use `--workspace` to control workspace size.
- Exported files are written next to the resolved weights file under the BoxMOT weights directory.

During export, BoxMOT prints the resolved input and output tensor shapes so you can verify the model contract before deployment.

!!! note

    `boxmot(...).export(...)` is the public Python API. Internally it dispatches to the same export engine used by the CLI.

## CLI Reference

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: export
    :style: table
    :prog_name: boxmot export
