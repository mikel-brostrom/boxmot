# Export

Use `export` to convert ReID models to deployment formats such as ONNX and TensorRT.

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
          --include onnx \
          --include engine \
          --dynamic
        ```

    === "Python"

        ```python
        from boxmot.api import Boxmot

        boxmot = Boxmot(reid="osnet_x0_25_msmt17")
        exported = boxmot.export(include=("onnx", "engine"), dynamic=True)
        print(exported.files)
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
