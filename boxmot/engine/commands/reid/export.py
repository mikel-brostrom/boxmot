"""CLI and presentation adapter for reusable ReID export workflows."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import click

from boxmot.engine.commands._options import _click_imgsz_default, _parse_imgsz, _parse_int_tuple
from boxmot.engine.commands._support import _run_engine_workflow
from boxmot.reid.exporters.config import build_export_namespace, load_export_defaults

EXPORT_DEFAULTS = load_export_defaults()


def _parse_coreml_buckets(ctx: click.Context, param: click.Parameter, value: Any) -> tuple[int, ...]:
    """Parse safe, positive CoreML batch buckets capped at 32."""

    parts = _parse_int_tuple(ctx, param, value)
    if not parts:
        raise click.BadParameter("must contain at least one batch size")
    if any(part < 1 for part in parts):
        raise click.BadParameter("batch buckets must be positive integers")
    if max(parts) > 32:
        raise click.BadParameter("batch buckets are capped at 32; larger inputs are chunked")
    return tuple(sorted(set(parts)))


def _parse_tflite_static_activation_bits(
    _ctx: click.Context,
    _param: click.Parameter,
    value: Any,
) -> int:
    """Validate supported TFLite static activation precision."""

    bits = int(value)
    if bits not in {8, 16}:
        raise click.BadParameter("must be 8 or 16")
    return bits


def export_options(func: Callable) -> Callable:
    """Attach ReID export options sourced from domain-owned defaults."""

    options = [
        click.option(
            "--batch-size",
            type=int,
            default=EXPORT_DEFAULTS.batch_size,
            help="Batch size for export",
        ),
        click.option(
            "--imgsz",
            "--img",
            "--img-size",
            callback=_parse_imgsz,
            type=str,
            default=_click_imgsz_default(EXPORT_DEFAULTS.imgsz),
            help="Image size as H,W (e.g. 256,128)",
        ),
        click.option(
            "--device",
            default=EXPORT_DEFAULTS.device,
            help="CUDA device (e.g., '0', '0,1,2,3', or 'cpu')",
        ),
        click.option(
            "--optimize",
            is_flag=True,
            default=EXPORT_DEFAULTS.optimize,
            help="Optimize TorchScript for mobile (CPU export only)",
        ),
        click.option(
            "--dynamic",
            is_flag=True,
            default=EXPORT_DEFAULTS.dynamic,
            help="Enable dynamic axes for ONNX/TensorRT export",
        ),
        click.option(
            "--simplify",
            is_flag=True,
            default=EXPORT_DEFAULTS.simplify,
            help="Simplify ONNX model",
        ),
        click.option("--opset", type=int, default=EXPORT_DEFAULTS.opset, help="ONNX opset version"),
        click.option(
            "--workspace",
            type=int,
            default=EXPORT_DEFAULTS.workspace,
            help="TensorRT workspace size (GB)",
        ),
        click.option("--verbose", is_flag=True, help="Enable verbose logging for TensorRT"),
        click.option(
            "--weights",
            type=Path,
            default=EXPORT_DEFAULTS.weights,
            help="Path to the model weights (.pt file)",
        ),
        click.option(
            "--half",
            is_flag=True,
            default=EXPORT_DEFAULTS.half,
            help="Enable FP16 half-precision export (GPU only)",
        ),
        click.option(
            "--coreml-batch-buckets",
            type=str,
            callback=_parse_coreml_buckets,
            default=",".join(str(value) for value in EXPORT_DEFAULTS.coreml_batch_buckets),
            show_default=True,
            help="Static MLProgram batch buckets; values above 32 are rejected",
        ),
        click.option(
            "--coreml-minimum-deployment-target",
            type=click.Choice(("macOS12", "macOS13", "macOS14", "macOS15", "macOS26")),
            default=EXPORT_DEFAULTS.coreml_minimum_deployment_target,
            show_default=True,
            help="Minimum macOS target; macOS15 enables native SDPA",
        ),
        click.option(
            "--coreml-compute-units",
            type=click.Choice(("ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly")),
            default=EXPORT_DEFAULTS.coreml_compute_units,
            show_default=True,
            help="CoreML compute units used when compiling MLPrograms",
        ),
        click.option(
            "--coreml-timeout",
            type=click.FloatRange(min=1.0),
            default=EXPORT_DEFAULTS.coreml_timeout,
            show_default=True,
            help="Per-bucket CoreML conversion timeout in seconds",
        ),
        click.option(
            "--coreml-max-memory-gb",
            type=click.FloatRange(min=1.0),
            default=EXPORT_DEFAULTS.coreml_max_memory_gb,
            show_default=True,
            help="Per-bucket CoreML conversion process memory limit",
        ),
        click.option(
            "--tflite-quantize",
            type=click.Choice(("none", "weight", "dynamic", "static"), case_sensitive=False),
            default=EXPORT_DEFAULTS.tflite_quantize,
            show_default=True,
            help=(
                "Post-quantize TFLite export: weight=int8 weights with float compute, "
                "dynamic=int8 dynamic range, static=int8 weights with calibrated activations"
            ),
        ),
        click.option(
            "--tflite-calibration-data",
            type=Path,
            default=EXPORT_DEFAULTS.tflite_calibration_data,
            help="Image, image-list .txt, or directory of ReID crops for TFLite static calibration",
        ),
        click.option(
            "--tflite-calibration-samples",
            type=int,
            default=EXPORT_DEFAULTS.tflite_calibration_samples,
            show_default=True,
            help="Maximum number of calibration images for TFLite static export",
        ),
        click.option(
            "--tflite-calibration-preprocess",
            type=click.Choice(("resize", "resize_pad"), case_sensitive=False),
            default=EXPORT_DEFAULTS.tflite_calibration_preprocess,
            show_default=True,
            help="Crop preprocessing for TFLite static calibration images",
        ),
        click.option(
            "--tflite-calibration-seed",
            type=int,
            default=EXPORT_DEFAULTS.tflite_calibration_seed,
            show_default=True,
            help="Seed for nested directory sampling in TFLite static calibration",
        ),
        click.option(
            "--tflite-calibration-update",
            type=click.Choice(("minmax", "moving_average"), case_sensitive=False),
            default=EXPORT_DEFAULTS.tflite_calibration_update,
            show_default=True,
            help="Activation range update rule for TFLite static calibration",
        ),
        click.option(
            "--tflite-static-activation-bits",
            type=int,
            callback=_parse_tflite_static_activation_bits,
            default=EXPORT_DEFAULTS.tflite_static_activation_bits,
            show_default=True,
            help="Activation precision for TFLite static quantization; weights remain int8",
        ),
        click.option(
            "--include",
            multiple=True,
            default=EXPORT_DEFAULTS.include,
            help="Export formats to include. Options: torchscript, onnx, openvino, engine, coreml, tflite",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def main(args: Any):
    """Render export progress while delegating model work to ``boxmot.reid``."""

    from boxmot.engine.ui.reporters.export import ExportWorkflowReporter
    from boxmot.reid.exporters import workflow

    pipeline = ExportWorkflowReporter(args).pipeline()
    start_time = time.time()
    with pipeline:
        pipeline.update("Loading ReID model...")
        model, dummy_input = workflow.prepare_export(args)

        output = model(dummy_input)
        output_tensor = output[0] if isinstance(output, tuple) else output
        output_shape = tuple(output_tensor.shape)
        checkpoint_size_mb = Path(args.weights).stat().st_size / 1e6
        pipeline.update(
            f"Input shape:  {tuple(dummy_input.shape)}\n"
            f"Output shape: {output_shape} ({checkpoint_size_mb:.1f} MB)"
        )
        pipeline.advance("Exporting model...")

        formats = list(getattr(args, "include", ()) or ())
        pipeline.update(
            f"Exporting to {len(formats)} format(s): {', '.join(formats) if formats else 'none'}"
        )
        exported_files = workflow.execute_export(args, model, dummy_input)
        parity_report: dict[str, dict[str, Any]] = {}
        if exported_files:
            with workflow.suppress_export_noise(not args.verbose):
                parity_report = workflow.verify_export_parity(args, model, dummy_input, exported_files)
        result = workflow.ReIDExportResult(
            weights=args.weights,
            files=exported_files,
            parity=parity_report,
            half=bool(getattr(args, "half", False)),
        )

        elapsed_time = time.time() - start_time
        if result.files:
            lines = [f"Time: {elapsed_time:.1f}s", f"Saved to: {args.weights.parent.resolve()}", "", "Files:"]
            for fmt, output_path in result.files.items():
                stats = parity_report.get(fmt)
                if stats is None:
                    suffix = " — parity: skipped"
                elif "error" in stats:
                    suffix = f" — parity: error ({stats['error']})"
                else:
                    parity_status = "OK" if stats.get("parity_ok") else "MISMATCH"
                    embedding_status = "OK" if stats.get("embedding_ok") else "MISMATCH"
                    suffix = (
                        f" — parity {parity_status} (maxΔ={stats['max_abs']:.1e}), "
                        f"embedding {embedding_status} (cos={stats['cosine']:.4f})"
                    )
                lines.append(f"  • {fmt}: {Path(output_path).name}{suffix}")
            lines.extend(
                (
                    "",
                    f"Overall parity: {'acceptable' if result.parity_ok else 'out of tolerance'}",
                    "",
                    "Visualize: https://netron.app",
                )
            )
            pipeline.update("\n".join(lines))
        else:
            pipeline.update(f"Export complete in {elapsed_time:.1f}s")
        pipeline.finish()
        return result


@click.command(name="export", help="Export ReID models")
@export_options
def export(**kwargs: Any) -> Any:
    """Export ReID weights and configurations for deployment."""

    args = build_export_namespace(kwargs)
    return _run_engine_workflow(__name__, args)


__all__ = ("export", "export_options", "main")
