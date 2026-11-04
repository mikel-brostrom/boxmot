# Export

Use `export` to convert ReID models to TorchScript, ONNX, OpenVINO, TensorRT,
native Core ML, or TFLite.

This command exports appearance encoders. The separate registered
`export-edgetam` command is unfinished: its implementation module is absent
from this checkout, so it cannot produce an EdgeTAM bundle. Use the supported
PyTorch workflows on the [instance masks page](../tasks/masks.md).

Install format-specific dependencies explicitly before exporting, for example
with `boxmot install --extra onnx`. Exporters validate their requirements and
report missing packages. See [Install dependencies](install.md) for other
extras and TensorRT setup.

TensorRT and OpenVINO use ONNX as an intermediate. If you request only `engine`
or `openvino`, BoxMOT exports `.onnx` next to the source weights first. The
downstream exporter reuses it when its content and export contract match.

Core ML export is native and does not pass through ONNX Runtime. It produces an
FP16 MLProgram bundle with static batch buckets (1, 8, 16, and 32 by default).
The runtime pads or chunks arbitrary detection counts and lazily keeps one
compiled package resident. Conversion workers have configurable time and RAM
limits to prevent runaway Apple graph compilation.

## Examples

`--device` follows the shared [device selection rules](track.md#device-selection).
Choose a device supported by the requested export format; TensorRT requires
CUDA, while the Core ML example below performs conversion on CPU.

Both tabs run the same export and parity-check workflow. The named OSNet
checkpoint can be downloaded on first use; replace `runs/reid_train/exp/best.pt`
with your own training output for the other examples. Python options use
underscores, and repeated `--include` values become a tuple.

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

        ```python
        from boxmot.engine.commands.reid.export import main as export_reid

        if __name__ == "__main__":
            result = export_reid(
                weights="osnet_x0_25_msmt17.pt",
                include=("onnx",),
            )
            print(result.files)
        ```

        Export a transformer ReID model for Apple GPU/CPU inference:

        ```python
        from boxmot.engine.commands.reid.export import main as export_reid

        if __name__ == "__main__":
            result = export_reid(
                weights="runs/reid_train/exp/best.pt",
                include=("coreml",),
                device="cpu",
                coreml_batch_buckets=(1, 8, 16, 32),
                coreml_minimum_deployment_target="macOS15",
                coreml_compute_units="CPUAndGPU",
                coreml_timeout=600,
                coreml_max_memory_gb=16,
            )
            print(result.files)
        ```

        The output is `best_coreml_model/`. Pass that directory directly as
        ReID weights. `BOXMOT_COREML_MAX_LOADED_BUCKETS=1` is the safe default;
        increasing it trades RAM for fewer bucket recompilations.

        Export multiple formats:

        ```python
        from boxmot.engine.commands.reid.export import main as export_reid

        if __name__ == "__main__":
            result = export_reid(
                weights="osnet_x0_25_msmt17.pt",
                include=("onnx", "engine"),
                dynamic=True,
                batch_size=16,
                device="cuda:0",
            )
            print(result.files)
        ```

        Export calibrated TFLite int8 using representative ReID crops:

        ```python
        from boxmot.engine.commands.reid.export import main as export_reid

        if __name__ == "__main__":
            result = export_reid(
                weights="runs/reid_train/exp/best.pt",
                include=("tflite",),
                tflite_quantize="static",
                tflite_calibration_data=(
                    "datasets/reid/Market-1501-v15.09.15/bounding_box_train"
                ),
                tflite_calibration_samples=512,
                tflite_calibration_seed=0,
                tflite_calibration_update="minmax",
                tflite_static_activation_bits=16,
            )
            print(result.files)
        ```

        Static TFLite uses int8 weights. The default
        `tflite_static_activation_bits=16` preserves ReID embedding parity
        better but can be slower on CPU; use `8` only for strict int8
        activation ablations.

## Typical use cases

- deploy a ReID backbone outside BoxMOT
- prepare ReID models for inference benchmarks
- build an optimized runtime for a tracker that uses appearance features

## Shapes and artifacts

Exports use a fixed RGB crop in height-width order, taken from checkpoint model
metadata or the registered backbone default. Vehicle checkpoints identified by
`veri` or `vehicleid` use `256x256`. The current implementation replaces the
parsed `--imgsz` value with this resolved size; that flag does not override the
export crop.

`--dynamic` makes the batch dimension variable for ONNX and OpenVINO while
keeping the crop dimensions fixed. For TensorRT, `--batch-size` sets the maximum
dynamic batch; use a value larger than one. TFLite attempts a dynamic batch and
falls back to static when conversion cannot support it. Core ML uses its
explicit bucket list independently of `--dynamic`.

Files are written beside the source checkpoint:

| Format | Output for `best.pt` |
| --- | --- |
| TorchScript | `best.torchscript` |
| ONNX | `best.onnx` |
| OpenVINO | `best_openvino_model/` with XML and BIN files |
| TensorRT | `best.engine` |
| Core ML | `best_coreml_model/` with its manifest and batch packages |
| TFLite | `best.tflite`; quantized exports also retain `best.float.tflite` |

Keep generated metadata sidecars with the exported artifacts. For example,
use `boxmot track --source video.mp4 --detector yolo26n --tracker botsort
--reid runs/reid_train/exp/best.onnx` after exporting that checkpoint.

## Validate the result

BoxMOT compares supported exports with the PyTorch model on deterministic
synthetic input and reports numerical error and embedding cosine similarity.
This check is informational: a failed check does not undo the exported file or
fail the command. TensorRT parity is skipped by this workflow. Validate
retrieval quality and end-to-end tracking on your target runtime and device,
especially after quantization; a conversion check is not a dataset evaluation.

TFLite calibration uses ReID crops, with preprocessing selected independently
by `--tflite-calibration-preprocess`. Match it to training/inference (`resize`
or `resize_pad`) and use representative images from your training data.

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.reid.export
    :command: export
    :depth: 0
    :style: table
    :prog_name: boxmot export
