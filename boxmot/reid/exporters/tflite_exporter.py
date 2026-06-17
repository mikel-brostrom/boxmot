import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class TFLiteExporter(BaseExporter):
    group = "tflite"
    _IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    _QUANTIZATION_MODES = {"none", "weight", "dynamic", "static"}
    _CALIBRATION_UPDATE_MODES = {"minmax", "moving_average"}
    _STATIC_ACTIVATION_BITS = {8, 16}
    _STATIC_QUANT_OP_NAMES = (
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "BATCH_MATMUL",
    )

    def __init__(
        self,
        model,
        im,
        file,
        opset=None,
        dynamic=False,
        half=False,
        simplify=False,
        quantize="none",
        calibration_data=None,
        calibration_samples=256,
        calibration_preprocess="resize",
        calibration_seed=0,
        calibration_update="minmax",
        static_activation_bits=16,
    ):
        super().__init__(model, im, file, optimize=False, dynamic=dynamic, half=half, simplify=simplify)
        self.opset = opset
        self.quantize = str(quantize or "none").lower()
        if self.quantize not in self._QUANTIZATION_MODES:
            raise ValueError(
                f"Unsupported TFLite quantization mode '{quantize}'. "
                f"Expected one of {sorted(self._QUANTIZATION_MODES)}."
            )
        self.calibration_data = Path(calibration_data) if calibration_data else None
        self.calibration_samples = int(calibration_samples or 0)
        self.calibration_preprocess = calibration_preprocess or "resize"
        self.calibration_seed = int(calibration_seed)
        self.calibration_update = str(calibration_update or "minmax").lower()
        self.static_activation_bits = int(static_activation_bits)
        if self.calibration_update not in self._CALIBRATION_UPDATE_MODES:
            raise ValueError(
                f"Unsupported TFLite calibration update mode '{calibration_update}'. "
                f"Expected one of {sorted(self._CALIBRATION_UPDATE_MODES)}."
            )
        if self.static_activation_bits not in self._STATIC_ACTIVATION_BITS:
            raise ValueError(
                f"Unsupported TFLite static activation bits '{static_activation_bits}'. "
                f"Expected one of {sorted(self._STATIC_ACTIVATION_BITS)}."
            )
        if self.calibration_data is not None and self.quantize != "static":
            raise ValueError("--tflite-calibration-data is only used with --tflite-quantize static")
        if self.quantize == "static":
            if self.calibration_data is None:
                raise ValueError("--tflite-quantize static requires --tflite-calibration-data")
            if self.calibration_samples <= 0:
                raise ValueError("--tflite-calibration-samples must be positive for static TFLite quantization")

    def export(self) -> str:
        import litert_torch

        tflite_path = self.file.with_suffix(".tflite")
        export_path = tflite_path if self.quantize == "none" else self.file.with_suffix(".float.tflite")
        version = getattr(litert_torch, "__version__", "unknown")
        LOGGER.info(f"Exporting TFLite with litert-torch {version}...")

        sample_inputs = self._sample_inputs(self.im)
        model = self._prepare_model_for_litert(self.model.eval(), sample_inputs)

        # Attempt dynamic batch export; fall back to static if unsupported.
        edge_model = None
        if self.dynamic and sample_inputs:
            batch_dim = torch.export.Dim("batch", min=1, max=128)
            dynamic_shapes = tuple(
                {0: batch_dim} if isinstance(t, torch.Tensor) and t.dim() >= 1 else {}
                for t in sample_inputs
            )
            try:
                edge_model = litert_torch.convert(
                    model,
                    sample_inputs,
                    dynamic_shapes=dynamic_shapes,
                )
            except Exception as exc:
                if self.verbose:
                    LOGGER.warning(f"Dynamic batch export unsupported ({exc}); falling back to static batch.")
                else:
                    LOGGER.info("Dynamic batch export unsupported; falling back to static batch.")

        if edge_model is None:
            edge_model = litert_torch.convert(model, sample_inputs)

        edge_model.export(str(export_path))

        if not export_path.is_file():
            raise RuntimeError(f"litert-torch completed without producing {export_path}")

        if self.quantize != "none":
            tflite_path = self._quantize_tflite(export_path, tflite_path)
            export_path.unlink(missing_ok=True)

        return str(tflite_path)

    def _quantize_tflite(self, float_path, output_path) -> Any:
        from ai_edge_quantizer import qtyping, quantizer

        LOGGER.info(f"Applying TFLite {self.quantize} int8 quantization...")
        tflite_quantizer = quantizer.Quantizer(float_path)
        calibration_result = None
        if self.quantize == "weight":
            tflite_quantizer.add_weight_only_config(
                regex=".*",
                operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                num_bits=8,
            )
        elif self.quantize == "dynamic":
            tflite_quantizer.add_dynamic_config(
                regex=".*",
                operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                num_bits=8,
            )
        elif self.quantize == "static":
            for op_name in self._STATIC_QUANT_OP_NAMES:
                tflite_quantizer.add_static_config(
                    regex=".*",
                    operation_name=getattr(qtyping.TFLOperationName, op_name),
                    activation_num_bits=self.static_activation_bits,
                    weight_num_bits=8,
                )
            calibration_data, sample_count = self._build_static_calibration_data(float_path)
            LOGGER.info(
                f"Calibrating TFLite static quantization "
                f"(int8 weights, int{self.static_activation_bits} activations) with "
                f"{sample_count} image(s) from {self.calibration_data}..."
            )
            calibration_result = self._calibrate_static_quantizer(
                tflite_quantizer,
                calibration_data,
                float_path,
            )
            if not calibration_result:
                raise RuntimeError("TFLite static quantization calibration produced no ranges.")
        quantized = (
            tflite_quantizer.quantize(calibration_result)
            if calibration_result is not None
            else tflite_quantizer.quantize()
        )
        quantized.export_model(output_path, overwrite=True)
        if not output_path.is_file():
            raise RuntimeError(f"TFLite quantization completed without producing {output_path}")
        return output_path

    def _calibrate_static_quantizer(self, tflite_quantizer, calibration_data, float_path):
        if self.calibration_update == "moving_average":
            return tflite_quantizer.calibrate(calibration_data)

        from ai_edge_quantizer import calibrator, recipe_manager
        from ai_edge_quantizer.utils import qsv_utils

        manager = recipe_manager.RecipeManager()
        manager.load_quantization_recipe(tflite_quantizer.get_quantization_recipe())
        calib = calibrator.Calibrator(
            str(float_path),
            qsv_update_func=qsv_utils.min_max_update,
        )
        calib.calibrate(calibration_data, manager)
        return calib.get_model_qsvs()

    def _build_static_calibration_data(self, float_path) -> tuple[dict[str, list[dict[str, Any]]], int]:
        import numpy as np
        from PIL import Image

        from boxmot.reid.datasets.transforms import build_test_transforms

        signature_key, input_name, input_shape, input_dtype = self._tflite_signature_input(float_path)
        batch_size, layout, image_size = self._calibration_input_spec(input_shape)
        paths = self._calibration_image_paths()
        transform = build_test_transforms(image_size, preprocess=self.calibration_preprocess)

        batches = []
        current = []
        for path in paths:
            try:
                image = Image.open(path).convert("RGB")
            except Exception as exc:
                raise ValueError(f"Unable to read TFLite calibration image: {path}") from exc
            tensor = transform(image).numpy()
            if layout == "nhwc":
                tensor = tensor.transpose(1, 2, 0)
            current.append(tensor.astype(input_dtype, copy=False))
            if len(current) == batch_size:
                batches.append({input_name: np.stack(current, axis=0)})
                current = []

        if current:
            while len(current) < batch_size:
                current.append(current[-1].copy())
            batches.append({input_name: np.stack(current, axis=0)})

        return {signature_key: batches}, len(paths)

    def _calibration_image_paths(self) -> list[Path]:
        if self.calibration_data is None:
            raise ValueError("Missing TFLite calibration data path.")

        path = self.calibration_data
        if not path.exists():
            raise FileNotFoundError(f"TFLite calibration data path does not exist: {path}")

        nested_sample = False
        if path.is_dir():
            images = sorted(p for p in path.rglob("*") if p.suffix.lower() in self._IMAGE_SUFFIXES)
            nested_sample = True
        elif path.suffix.lower() == ".txt":
            images = []
            for line in path.read_text().splitlines():
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                images.append(candidate)
        elif path.suffix.lower() in self._IMAGE_SUFFIXES:
            images = [path]
        else:
            raise ValueError(
                "TFLite calibration data must be an image, a text file of image paths, "
                f"or a directory containing images. Got: {path}"
            )

        images = [p for p in images if p.suffix.lower() in self._IMAGE_SUFFIXES]
        if not images:
            raise ValueError(f"No calibration images found under {path}")

        if len(images) > self.calibration_samples:
            if nested_sample:
                import numpy as np

                rng = np.random.default_rng(self.calibration_seed)
                indices = rng.permutation(len(images))[:self.calibration_samples]
                images = [images[index] for index in indices]
            else:
                images = images[:self.calibration_samples]
        return images

    @staticmethod
    def _tflite_signature_input(float_path) -> tuple[str, str, tuple[int, ...], Any]:
        from importlib import import_module

        litert = import_module("ai_edge_litert.interpreter")
        interpreter = litert.Interpreter(model_path=str(float_path))
        interpreter.allocate_tensors()
        signatures = interpreter.get_signature_list()
        if not signatures:
            raise RuntimeError("TFLite static calibration requires a model with a LiteRT signature.")
        signature_key = next(iter(signatures))
        input_names = signatures[signature_key].get("inputs") or []
        if len(input_names) != 1:
            raise RuntimeError(
                f"TFLite static calibration expects one model input, got {input_names}"
            )
        input_name = input_names[0]
        input_detail = interpreter.get_signature_runner(signature_key).get_input_details()[input_name]
        input_shape = tuple(int(dim) for dim in input_detail["shape"])
        return signature_key, input_name, input_shape, input_detail.get("dtype")

    @staticmethod
    def _calibration_input_spec(input_shape: tuple[int, ...]) -> tuple[int, str, tuple[int, int]]:
        if len(input_shape) != 4:
            raise RuntimeError(f"TFLite static calibration expects a 4D image input, got {input_shape}")

        batch_size = max(int(input_shape[0]), 1)
        if input_shape[-1] == 3 and input_shape[1] != 3:
            return batch_size, "nhwc", (int(input_shape[1]), int(input_shape[2]))
        if input_shape[1] == 3:
            return batch_size, "nchw", (int(input_shape[2]), int(input_shape[3]))
        raise RuntimeError(f"Unable to infer TFLite image input layout from shape {input_shape}")

    def _prepare_model_for_litert(self, model: nn.Module, sample_inputs: tuple[Any, ...]) -> nn.Module:
        return self._replace_static_adaptive_max_pool2d(model, sample_inputs)

    @staticmethod
    def _sample_inputs(im: Any) -> tuple[Any, ...]:
        if isinstance(im, tuple):
            return im
        if isinstance(im, list):
            return tuple(im)
        return (im,)

    def _replace_static_adaptive_max_pool2d(self, model: nn.Module, sample_inputs: tuple[Any, ...]) -> nn.Module:
        adaptive_pools = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.AdaptiveMaxPool2d) and self._pool_output_size(module) == (1, 1)
        }
        if not adaptive_pools:
            return model

        input_shapes: dict[str, list[tuple[int, int]]] = {name: [] for name in adaptive_pools}
        handles = []
        for name, module in adaptive_pools.items():
            handles.append(
                module.register_forward_hook(
                    lambda _module, args, _output, pool_name=name: input_shapes[pool_name].append(
                        tuple(int(dim) for dim in args[0].shape[-2:])
                    )
                )
            )

        try:
            with torch.inference_mode():
                model(*sample_inputs)
        except Exception as exc:
            if self.verbose:
                LOGGER.warning(f"Unable to inspect adaptive max-pool shapes for LiteRT export: {exc}")
            return model
        finally:
            for handle in handles:
                handle.remove()

        replacements = {}
        for name, shapes in input_shapes.items():
            unique_shapes = set(shapes)
            if len(unique_shapes) == 1:
                replacements[name] = unique_shapes.pop()
            elif self.verbose:
                LOGGER.warning(
                    f"Keeping {name} as AdaptiveMaxPool2d because it saw multiple input shapes: {sorted(unique_shapes)}"
                )

        if not replacements:
            return model

        export_model = copy.deepcopy(model).eval()
        for name, kernel_size in replacements.items():
            self._set_submodule(
                export_model,
                name,
                nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size),
            )

        if self.verbose:
            LOGGER.info(
                f"Replaced {len(replacements)} AdaptiveMaxPool2d layer(s) with static MaxPool2d for LiteRT export."
            )
        return export_model

    @staticmethod
    def _pool_output_size(module: nn.AdaptiveMaxPool2d) -> tuple[int, int]:
        output_size = module.output_size
        if isinstance(output_size, int):
            return (output_size, output_size)
        return tuple(output_size)

    @staticmethod
    def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, module)
