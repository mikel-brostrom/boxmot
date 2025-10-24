from __future__ import annotations

import json
from pathlib import Path

from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.appearance.exporters.onnx_exporter import ONNXExporter
from boxmot.utils import logger as LOGGER


def _is_jetson() -> bool:
    """Best-effort Jetson detection."""
    try:
        return Path("/etc/nv_tegra_release").exists()
    except Exception:
        return False


class EngineExporter(BaseExporter):
    """
    TensorRT engine exporter aligned with the provided onnx2engine pattern,
    adapted for person re-identification models.

    Notes
    -----
    - INT8 is intentionally omitted per request.
    - Writes optional metadata header (length + JSON) before engine bytes, matching your example.
      If you prefer a pure-TRT engine without a custom header, comment out the metadata block.
    """

    required_packages = ("nvidia-tensorrt",)
    cmds = "--extra-index-url https://pypi.ngc.nvidia.com"

    # Optional knobs the runner/CLI may attach to this instance (safe fallbacks here):
    dla: int | None = None          # DLA core index for Jetson devices, else None
    verbose: bool = False           # TensorRT verbose logging
    metadata: dict | None = None    # custom metadata to prepend to the engine file

    def export(self):
        # --- Preconditions ---------------------------------------------------
        assert (
            self.im.device.type != "cpu"
        ), "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"

        try:
            import tensorrt as trt  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TensorRT not found. Install `nvidia-tensorrt` (often via pip + NGC index) and ensure CUDA-compatible drivers."
            ) from e

        # --- Export ONNX first ----------------------------------------------
        onnx_file = self.export_onnx()
        assert Path(onnx_file).exists(), f"Failed to export ONNX file: {onnx_file}"

        # --- Build TRT engine ------------------------------------------------
        return self._onnx2engine(
            onnx_file=str(onnx_file),
            engine_file=str(self.file.with_suffix(".engine")),
            workspace=getattr(self, "workspace", None),     # GB (float or int); handled below
            half=bool(self.half),
            dynamic=bool(self.dynamic),
            shape=self._infer_shape_for_reid(),
            dla=getattr(self, "dla", None),
            metadata=getattr(self, "metadata", None),
            verbose=bool(getattr(self, "verbose", False)),
            prefix="TRT:",
        )

    # ------------------------------------------------------------------------
    # Helper: keep API close to your function (INT8 removed)
    # ------------------------------------------------------------------------
    def _onnx2engine(
        self,
        onnx_file: str,
        engine_file: str | None = None,
        workspace: int | float | None = None,
        half: bool = False,
        dynamic: bool = False,
        shape: tuple[int, int, int, int] = (1, 3, 256, 128),  # (B,C,H,W) – ReID-friendly default
        dla: int | None = None,
        metadata: dict | None = None,
        verbose: bool = False,
        prefix: str = "",
    ) -> Path:
        import tensorrt as trt  # noqa

        engine_path = Path(engine_file or Path(onnx_file).with_suffix(".engine"))

        LOGGER.info(f"\nStarting export with TensorRT {trt.__version__}...")
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        # Builder + config
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

        # Workspace (GB -> bytes); safely handle None
        ws_gb = float(workspace) if workspace is not None else 0.0
        workspace_bytes = int(ws_gb * (1 << 30))

        is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10
        if is_trt10 and workspace_bytes > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        elif workspace_bytes > 0:  # TRT 7/8 path
            config.max_workspace_size = workspace_bytes

        # Explicit batch
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)

        # Capabilities
        half = builder.platform_has_fast_fp16 and half

        # Optional DLA (Jetson only)
        if dla is not None:
            if not _is_jetson():
                raise ValueError("DLA is only available on NVIDIA Jetson devices.")
            LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
            if not half:
                raise ValueError("DLA requires 'half=True' (FP16). Enable FP16 and try again.")
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(dla)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # Parse ONNX
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_file):
            raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

        # Describe network IO
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        LOGGER.info("Network Description:")
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape {inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape {out.shape} {out.dtype}')

        # Dynamic shapes (keep simple & safe ordering min <= opt <= max)
        if dynamic:
            if self.im.shape[0] <= 1:
                LOGGER.warning("WARNING: --dynamic model usually benefits from a larger --batch-size")
            b, c, h, w = shape
            profile = builder.create_optimization_profile()

            # Conservative min/opt/max suitable for ReID pipelines
            min_shape = (1, c, max(32, h // 4), max(32, w // 4))
            opt_shape = (max(1, b // 2), c, h, w)
            max_shape = (max(b, 4), c, max(h, h * 2), max(w, w * 2))

            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)

        # Precision flags
        LOGGER.info(f"{prefix} building {'FP16' if half else 'FP32'} engine as {engine_path}")
        if half:
            config.set_flag(trt.BuilderFlag.FP16)

        # Build & write
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(engine_path, "wb") as t:
            # Optional metadata block (length + JSON) BEFORE engine bytes to mirror your example
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            # Engine
            t.write(engine if is_trt10 else engine.serialize())

        return engine_path

    # ------------------------------------------------------------------------
    # Utility: infer reasonable (B,C,H,W) from self.im for ReID
    # ------------------------------------------------------------------------
    def _infer_shape_for_reid(self) -> tuple[int, int, int, int]:
        """
        Returns a (B,C,H,W) tuple from the current input tensor, with a ReID-friendly fallback.
        Common ReID sizes: (1,3,256,128) or (1,3,384,128). We respect the actual tensor when available.
        """
        try:
            b, c, h, w = map(int, self.im.shape)  # torch Tensor-like
            return (b or 1, c or 3, h or 256, w or 128)
        except Exception:
            return (1, 3, 256, 128)

    # --- Keep your ONNX path generation encapsulated ------------------------
    def export_onnx(self) -> Path:
        onnx_exporter = ONNXExporter(
            self.model,
            self.im,
            self.file,
            self.optimize,
            self.dynamic,
            self.half,
            self.simplify,
        )
        return onnx_exporter.export()
