import platform
import torch
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.appearance.exporters.onnx_exporter import ONNXExporter
from boxmot.utils import logger as LOGGER


class EngineExporter(BaseExporter):
    required_packages = ("nvidia-tensorrt",)
    cmds = '--extra-index-url https://pypi.ngc.nvidia.com'
    
    def export(self):

        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
        try:
            import tensorrt as trt
        except ImportError:
            import tensorrt as trt

        onnx_file = self.export_onnx()
        LOGGER.info(f"\nStarting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        assert onnx_file.exists(), f"Failed to export ONNX file: {onnx_file}"
        f = self.file.with_suffix(".engine")
        logger = trt.Logger(trt.Logger.INFO)
        if True:
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
            raise RuntimeError(f"Failed to load ONNX file: {onnx_file}")

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

        LOGGER.info(f"Building FP{16 if builder.platform_has_fast_fp16 and self.half else 32} engine in {f}")
        if builder.platform_has_fast_fp16 and self.half:
            config.set_flag(trt.BuilderFlag.FP16)
            config.default_device_type = trt.DeviceType.GPU

        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            t.write(engine if is_trt10 else engine.serialize())

        return f


    def export_onnx(self):
        onnx_exporter = ONNXExporter(self.model, self.im, self.file, self.optimize, self.dynamic, self.half, self.simplify)
        return onnx_exporter.export()
