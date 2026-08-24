"""Native CoreML MLProgram export for Apple deployment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from boxmot.reid.core.artifacts import source_artifact_metadata, write_artifact_metadata
from boxmot.reid.exporters.base_exporter import BaseExporter, as_inference_export_model
from boxmot.reid.exporters.process import run_limited
from boxmot.utils import logger as LOGGER

COREML_RESULT_PREFIX = "BOXMOT_COREML_EXPORT_RESULT="
COREML_DEFAULT_BUCKETS = (1, 8, 16, 32)
COREML_MAX_SAFE_BUCKET = 32
COREML_TARGETS = ("macOS12", "macOS13", "macOS14", "macOS15", "macOS26")
COREML_COMPUTE_UNITS = ("ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly")


def parse_coreml_buckets(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Normalize unique positive CoreML batch buckets."""
    parts = value if isinstance(value, (tuple, list)) else value.replace(";", ",").split(",")
    try:
        buckets = tuple(sorted({int(part) for part in parts if str(part).strip()}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CoreML batch buckets: {value!r}") from exc
    if not buckets or buckets[0] < 1:
        raise ValueError("CoreML batch buckets must contain positive integers")
    if buckets[-1] > COREML_MAX_SAFE_BUCKET:
        raise ValueError(
            f"CoreML batch buckets are capped at {COREML_MAX_SAFE_BUCKET}; larger batches are chunked by the runtime"
        )
    return buckets


def _coreml_output_dir(source: str | Path) -> Path:
    path = Path(source)
    return path.parent / f"{path.stem}_coreml_model"


def _extract_worker_result(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(COREML_RESULT_PREFIX):
            return json.loads(line.removeprefix(COREML_RESULT_PREFIX))
    detail = process.stdout.strip().splitlines()
    suffix = detail[-1] if detail else "worker produced no diagnostics"
    raise RuntimeError(f"CoreML conversion worker exited {process.returncode}: {suffix}")


class _CoreMLFrozenBatchNorm(nn.Module):
    """Single-output affine equivalent of an evaluation BatchNorm module.

    Torch 2.11 exports evaluation BatchNorm as an ATen operation with three
    outputs. CoreMLTools 9 only binds the normalized tensor and then fails
    while lowering the two unused auxiliary outputs. Materializing the frozen
    evaluation transform as multiply/add avoids that converter limitation.
    """

    def __init__(self, batch_norm: nn.modules.batchnorm._BatchNorm) -> None:
        super().__init__()
        if batch_norm.training:
            raise RuntimeError("CoreML BatchNorm freezing requires eval mode")
        if batch_norm.running_mean is None or batch_norm.running_var is None:
            raise RuntimeError("CoreML export requires BatchNorm modules with tracked running statistics")

        running_mean = batch_norm.running_mean.detach()
        running_var = batch_norm.running_var.detach()
        weight = (
            batch_norm.weight.detach()
            if batch_norm.weight is not None
            else torch.ones_like(running_mean)
        )
        bias = (
            batch_norm.bias.detach()
            if batch_norm.bias is not None
            else torch.zeros_like(running_mean)
        )
        scale = weight * torch.rsqrt(running_var + batch_norm.eps)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias - running_mean * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_shape = (1, -1, *((1,) * (x.ndim - 2)))
        return x * self.scale.reshape(channel_shape) + self.bias.reshape(channel_shape)


class _CoreMLGlobalMaxPool2d(nn.Module):
    """Index-free equivalent of AdaptiveMaxPool2d((1, 1))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=(-2, -1), keepdim=True)


def prepare_coreml_export_model(model: nn.Module) -> nn.Module:
    """Rewrite frozen modules that expose unsupported auxiliary ATen outputs."""
    if model.training:
        raise RuntimeError("CoreML deployment rewrites require eval mode")

    def rewrite(parent: nn.Module) -> tuple[int, int]:
        batch_norm_count = 0
        max_pool_count = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                setattr(parent, name, _CoreMLFrozenBatchNorm(child))
                batch_norm_count += 1
                continue
            if isinstance(child, nn.AdaptiveMaxPool2d) and child.output_size in {1, (1, 1)}:
                setattr(parent, name, _CoreMLGlobalMaxPool2d())
                max_pool_count += 1
                continue
            child_batch_norms, child_max_pools = rewrite(child)
            batch_norm_count += child_batch_norms
            max_pool_count += child_max_pools
        return batch_norm_count, max_pool_count

    batch_norm_count, max_pool_count = rewrite(model)
    if batch_norm_count or max_pool_count:
        LOGGER.debug(
            "Prepared CoreML graph with "
            f"{batch_norm_count} frozen BatchNorm and {max_pool_count} index-free global max-pool rewrites"
        )
    return model


class CoreMLExporter(BaseExporter):
    """Export FP16 MLProgram packages for a bounded set of static batch buckets."""

    group = "coreml"

    def __init__(
        self,
        model,
        im,
        file,
        *,
        batch_buckets: tuple[int, ...] = COREML_DEFAULT_BUCKETS,
        minimum_deployment_target: str = "macOS15",
        compute_units: str = "CPUAndGPU",
        timeout_s: float = 600.0,
        max_memory_gb: float = 16.0,
        verbose: bool = True,
    ) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("CoreML export is only supported on macOS")
        super().__init__(
            model,
            im,
            file,
            optimize=True,
            dynamic=False,
            half=True,
            simplify=False,
            verbose=verbose,
        )
        self.batch_buckets = parse_coreml_buckets(batch_buckets)
        if minimum_deployment_target not in COREML_TARGETS:
            raise ValueError(
                f"Unsupported CoreML deployment target {minimum_deployment_target!r}; choose from {COREML_TARGETS}"
            )
        if compute_units not in COREML_COMPUTE_UNITS:
            raise ValueError(f"Unsupported CoreML compute units {compute_units!r}; choose from {COREML_COMPUTE_UNITS}")
        if timeout_s <= 0:
            raise ValueError("CoreML conversion timeout must be positive")
        if max_memory_gb <= 0:
            raise ValueError("CoreML conversion memory limit must be positive")
        self.minimum_deployment_target = minimum_deployment_target
        self.compute_units = compute_units
        self.timeout_s = float(timeout_s)
        self.max_memory_gb = float(max_memory_gb)

    def export(self) -> Path:
        output_dir = _coreml_output_dir(self.file)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.",
                dir=output_dir.parent,
            )
        )
        staging_bundle = staging_root / output_dir.name
        staging_bundle.mkdir()
        export_model = prepare_coreml_export_model(as_inference_export_model(self.model))
        source_metadata = source_artifact_metadata(self.file)
        bucket_metadata: dict[str, dict[str, Any]] = {}

        try:
            with tempfile.TemporaryDirectory(prefix="boxmot-coreml-program-") as program_dir:
                for batch_size in self.batch_buckets:
                    if self.verbose:
                        LOGGER.info(
                            f"Exporting CoreML MLProgram batch={batch_size} "
                            f"target={self.minimum_deployment_target} FP16..."
                        )
                    sample = torch.zeros(
                        (batch_size, *self.im.shape[1:]),
                        dtype=self.im.dtype,
                        device=self.im.device,
                    )
                    exported_program = torch.export.export(
                        export_model,
                        (sample,),
                        strict=False,
                    ).run_decompositions({})
                    program_path = Path(program_dir) / f"batch{batch_size}.pt2"
                    torch.export.save(exported_program, program_path)

                    package_name = f"batch{batch_size}.mlpackage"
                    package_path = staging_bundle / package_name
                    process = run_limited(
                        [
                            sys.executable,
                            "-m",
                            "boxmot.reid.exporters.coreml_exporter",
                            "--worker",
                            "--program",
                            str(program_path),
                            "--output",
                            str(package_path),
                            "--target",
                            self.minimum_deployment_target,
                            "--compute-units",
                            self.compute_units,
                        ],
                        timeout_s=self.timeout_s,
                        max_memory_gb=self.max_memory_gb,
                    )
                    result = _extract_worker_result(process)
                    if process.returncode != 0 or result.get("status") != "ok":
                        raise RuntimeError(result.get("error", f"CoreML worker exited {process.returncode}"))
                    bucket_metadata[str(batch_size)] = {
                        "package": package_name,
                        "input_name": result["input_name"],
                        "output_name": result["output_name"],
                        "output_shape": result["output_shape"],
                        "operation_count": result["operation_count"],
                        "scaled_dot_product_attention_count": result["scaled_dot_product_attention_count"],
                    }

            input_names = {entry["input_name"] for entry in bucket_metadata.values()}
            output_names = {entry["output_name"] for entry in bucket_metadata.values()}
            if len(input_names) != 1 or len(output_names) != 1:
                raise RuntimeError("CoreML bucket input/output names are inconsistent")
            output_widths = set()
            operation_counts = set()
            for batch_size in self.batch_buckets:
                entry = bucket_metadata[str(batch_size)]
                output_shape = entry["output_shape"]
                if not output_shape or int(output_shape[0]) != batch_size:
                    raise RuntimeError(f"CoreML batch {batch_size} has invalid output shape {output_shape}")
                output_widths.add(tuple(output_shape[1:]))
                operation_counts.add(int(entry["operation_count"]))
            if len(output_widths) != 1 or len(operation_counts) != 1:
                raise RuntimeError("CoreML bucket output/graph metadata is inconsistent")

            target_supports_native_sdpa = self.minimum_deployment_target in {"macOS15", "macOS26"}
            is_csl_tinyvit = str(source_metadata.get("model_name") or "").startswith("csl_tinyvit")
            if target_supports_native_sdpa and is_csl_tinyvit:
                missing_sdpa = [
                    batch_size
                    for batch_size in self.batch_buckets
                    if not bucket_metadata[str(batch_size)]["scaled_dot_product_attention_count"]
                ]
                if missing_sdpa:
                    raise RuntimeError(
                        "CoreML conversion decomposed CSL attention instead of "
                        f"retaining native SDPA for batch buckets {missing_sdpa}"
                    )

            write_artifact_metadata(
                staging_bundle,
                {
                    **source_metadata,
                    "format": "coreml_mlprogram",
                    "precision": "float16",
                    "io_dtype": "float32",
                    "minimum_deployment_target": self.minimum_deployment_target,
                    "compute_units": self.compute_units,
                    "input_name": next(iter(input_names)),
                    "output_name": next(iter(output_names)),
                    "input_shape": list(self.im.shape[1:]),
                    "batch_buckets": list(self.batch_buckets),
                    "buckets": bucket_metadata,
                },
            )
            self._replace_bundle(staging_bundle, output_dir)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

        return output_dir

    @staticmethod
    def _replace_bundle(staging_bundle: Path, output_dir: Path) -> None:
        backup = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.backup")
        if backup.exists():
            shutil.rmtree(backup)
        if output_dir.exists():
            output_dir.replace(backup)
        try:
            staging_bundle.replace(output_dir)
        except BaseException:
            if backup.exists() and not output_dir.exists():
                backup.replace(output_dir)
            raise
        finally:
            shutil.rmtree(backup, ignore_errors=True)


def _coreml_operations(model) -> list[str]:
    operations: list[str] = []
    program = model.get_spec().mlProgram
    for function in program.functions.values():
        for block in function.block_specializations.values():
            operations.extend(operation.type for operation in block.operations)
    return operations


def _worker_main(args: argparse.Namespace) -> int:
    try:
        import coremltools as ct

        exported_program = torch.export.load(args.program)
        target = getattr(ct.target, args.target)
        compute_units = {
            "ALL": ct.ComputeUnit.ALL,
            "CPUAndGPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPUAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
            "CPUOnly": ct.ComputeUnit.CPU_ONLY,
        }[args.compute_units]
        converted = ct.convert(
            exported_program,
            convert_to="mlprogram",
            minimum_deployment_target=target,
            compute_precision=ct.precision.FLOAT16,
            compute_units=compute_units,
            skip_model_load=True,
            package_dir=str(args.output),
        )
        spec = converted.get_spec()
        operations = _coreml_operations(converted)
        output_shape = list(spec.description.output[0].type.multiArrayType.shape)
        result = {
            "status": "ok",
            "input_name": spec.description.input[0].name,
            "output_name": spec.description.output[0].name,
            "output_shape": output_shape,
            "operation_count": len(operations),
            "scaled_dot_product_attention_count": operations.count("scaled_dot_product_attention"),
        }
    except Exception as exc:
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(COREML_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ok" else 1


def _build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=COREML_TARGETS, required=True)
    parser.add_argument("--compute-units", choices=COREML_COMPUTE_UNITS, required=True)
    return parser


if __name__ == "__main__":
    worker_args = _build_worker_parser().parse_args()
    raise SystemExit(_worker_main(worker_args))
