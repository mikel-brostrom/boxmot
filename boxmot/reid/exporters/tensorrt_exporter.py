from boxmot.reid.backends.dependencies import ensure_reid_backend_requirements
from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.reid.exporters.onnx_exporter import ensure_onnx_export
from boxmot.utils import logger as LOGGER


class EngineExporter(BaseExporter):
    def __init__(
        self,
        model,
        im,
        file,
        opset: int | None = None,
        dynamic: bool = False,
        half: bool = True,
        simplify: bool = True,
        verbose: bool = True,
        workspace: int = 4,
    ):
        super().__init__(
            model=model,
            im=im,
            file=file,
            optimize=False,
            dynamic=dynamic,
            half=half,
            simplify=simplify,
            verbose=verbose,
        )
        self.opset = opset
        self.workspace = workspace

    def export(self):
        assert (
            self.im.device.type != "cpu"
        ), "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
        ensure_reid_backend_requirements(self.checker, "tensorrt")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "TensorRT auto-install completed, but the 'tensorrt' module still "
                "could not be imported. Check CUDA, Python, and NVIDIA package compatibility."
            ) from exc

        onnx_file = self.export_onnx()
        LOGGER.info(f"\nStarting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        assert onnx_file.exists(), f"Failed to export ONNX file: {onnx_file}"
        f = self.file.with_suffix(".engine")
        logger = trt.Logger(trt.Logger.INFO)
        if self.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace = int(self.workspace * (1 << 30))
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_file)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            message = f"Failed to load ONNX file: {onnx_file}"
            if errors:
                message = f"{message}\n{errors}"
            raise RuntimeError(message)

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        LOGGER.info("Network Description:")
        for inp in inputs:
            LOGGER.info(f'\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        if self.dynamic:
            if self.im.shape[0] <= 1:
                LOGGER.warning("WARNING: --dynamic model requires maximum --batch-size argument")
            profile = builder.create_optimization_profile()
            for inp in inputs:
                if self.half:
                    inp.dtype = trt.float16
                profile.set_shape(
                    inp.name,
                    (1, *self.im.shape[1:]),
                    (max(1, self.im.shape[0] // 2), *self.im.shape[1:]),
                    self.im.shape,
                )
            config.add_optimization_profile(profile)

        LOGGER.info(
            f"Building FP{16 if builder.platform_has_fast_fp16 and self.half else 32} engine in {f}"
        )
        if builder.platform_has_fast_fp16 and self.half:
            config.set_flag(trt.BuilderFlag.FP16)
            config.default_device_type = trt.DeviceType.GPU

        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            t.write(engine if is_trt10 else engine.serialize())

        return f

    def export_onnx(self):
        return ensure_onnx_export(
            model=self.model,
            im=self.im,
            file=self.file,
            opset=self.opset,
            dynamic=self.dynamic,
            half=self.half,
            simplify=self.simplify,
            verbose=self.verbose,
        )
