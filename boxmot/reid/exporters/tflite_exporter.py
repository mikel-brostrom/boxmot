import copy
import sys
from typing import Any

import torch
from torch import nn

from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class TFLiteExporter(BaseExporter):
    group = "tflite"

    def __init__(self, model, im, file, opset=None, dynamic=False, half=False, simplify=False):
        super().__init__(model, im, file, optimize=False, dynamic=dynamic, half=half, simplify=simplify)
        self.opset = opset

    def export(self) -> str:
        if sys.version_info < (3, 10):
            raise RuntimeError("TFLite export with litert-torch requires Python 3.10 or newer.")

        import litert_torch

        tflite_path = self.file.with_suffix(".tflite")
        version = getattr(litert_torch, "__version__", "unknown")
        LOGGER.info(f"Exporting TFLite with litert-torch {version}...")

        sample_inputs = self._sample_inputs(self.im)
        model = self._prepare_model_for_litert(self.model.eval(), sample_inputs)

        # Attempt dynamic batch export; fall back to static if unsupported.
        edge_model = None
        if self.dynamic and sample_inputs:
            batch_dim = torch.export.Dim("batch", min=1, max=128)
            dynamic_shapes = [
                {0: batch_dim} if isinstance(t, torch.Tensor) and t.dim() >= 1 else {}
                for t in sample_inputs
            ]
            try:
                edge_model = litert_torch.convert(
                    model,
                    sample_inputs,
                    dynamic_shapes=dynamic_shapes,
                )
            except Exception:
                LOGGER.info("Dynamic batch export unsupported; falling back to static batch.")

        if edge_model is None:
            edge_model = litert_torch.convert(model, sample_inputs)

        edge_model.export(str(tflite_path))

        if not tflite_path.is_file():
            raise RuntimeError(f"litert-torch completed without producing {tflite_path}")

        return str(tflite_path)

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
