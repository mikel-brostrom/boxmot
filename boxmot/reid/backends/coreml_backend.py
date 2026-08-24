"""Native Core ML runtime for statically bucketed ReID MLPrograms."""

from __future__ import annotations

import gc
import os
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.reid.backends.dependencies import ensure_reid_backend_requirements
from boxmot.reid.core.artifacts import read_artifact_metadata
from boxmot.utils import logger as LOGGER


class CoreMLBackend(BaseModelBackend):
    """Run native MLProgram packages with bounded static-batch routing.

    Only one compiled bucket is retained by default. This avoids the large
    resident-memory spikes caused by compiling several copies of a transformer
    graph at once. Larger input batches are chunked, and incomplete chunks are
    padded to the smallest available bucket that fits.
    """

    _COMPUTE_UNITS = {
        "ALL": "ALL",
        "CPUAndGPU": "CPU_AND_GPU",
        "CPUAndNeuralEngine": "CPU_AND_NE",
        "CPUOnly": "CPU_ONLY",
    }
    build_source_model = False

    def __init__(self, weights, device, half, preprocess=None):
        self._requested_device = device
        # Core ML accepts numpy inputs. Keep crop construction and normalization
        # on CPU rather than copying CPU -> MPS -> CPU before every prediction.
        super().__init__(weights, torch.device("cpu"), False, preprocess=preprocess)
        self.nhwc = False
        self.half = False
        self.input_shape = self._crop_shape[1:]

    @staticmethod
    def _positive_env_int(name: str, default: int) -> int:
        value = os.environ.get(name, "").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            return default
        return max(parsed, 1)

    def load_model(self, w) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("Native Core ML inference is only supported on macOS")
        ensure_reid_backend_requirements(self.checker, "coreml")

        self._bundle = Path(w)
        manifest = read_artifact_metadata(self._bundle)
        if manifest.get("format") != "coreml_mlprogram":
            raise RuntimeError(f"{self._bundle} is missing a valid Core ML MLProgram manifest")

        bucket_entries = manifest.get("buckets")
        if not isinstance(bucket_entries, dict) or not bucket_entries:
            raise RuntimeError("Core ML manifest does not contain any batch buckets")

        self._bucket_entries: dict[int, dict[str, Any]] = {}
        for raw_batch, raw_entry in bucket_entries.items():
            try:
                batch_size = int(raw_batch)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid Core ML batch bucket {raw_batch!r}") from exc
            if batch_size < 1 or not isinstance(raw_entry, dict):
                raise RuntimeError(f"Invalid Core ML batch metadata for {raw_batch!r}")
            package = self._bundle / str(raw_entry.get("package", ""))
            if not package.is_dir():
                raise FileNotFoundError(f"Missing Core ML package for batch {batch_size}: {package}")
            self._bucket_entries[batch_size] = dict(raw_entry)

        self._buckets = tuple(sorted(self._bucket_entries))
        self._input_name = str(manifest.get("input_name") or "")
        self._output_name = str(manifest.get("output_name") or "")
        if not self._input_name or not self._output_name:
            raise RuntimeError("Core ML manifest is missing input/output names")

        shape = manifest.get("input_shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise RuntimeError("Core ML manifest input_shape must be [C, H, W]")
        self._crop_shape = tuple(int(dim) for dim in shape)
        self._output_width = self._infer_output_width()
        self._pad_buffers = {
            batch_size: np.zeros((batch_size, *self._crop_shape), dtype=np.float32) for batch_size in self._buckets
        }
        self._models: OrderedDict[int, Any] = OrderedDict()
        self._max_loaded_buckets = self._positive_env_int("BOXMOT_COREML_MAX_LOADED_BUCKETS", 1)
        self._compute_units_name = os.environ.get(
            "BOXMOT_COREML_COMPUTE_UNITS",
            str(manifest.get("compute_units") or "CPUAndGPU"),
        )
        if self._compute_units_name not in self._COMPUTE_UNITS:
            raise ValueError(
                f"Invalid BOXMOT_COREML_COMPUTE_UNITS={self._compute_units_name!r}; "
                f"choose from {tuple(self._COMPUTE_UNITS)}"
            )

        LOGGER.info(
            f"Core ML ReID buckets={list(self._buckets)} "
            f"compute_units={self._compute_units_name} lazy_cache={self._max_loaded_buckets}"
        )

    def _infer_output_width(self) -> int:
        for batch_size in self._buckets:
            shape = self._bucket_entries[batch_size].get("output_shape")
            if isinstance(shape, list) and len(shape) >= 2:
                try:
                    return int(np.prod(shape[1:]))
                except (TypeError, ValueError):
                    continue
        return 0

    def _load_bucket(self, batch_size: int):
        cached = self._models.pop(batch_size, None)
        if cached is not None:
            self._models[batch_size] = cached
            return cached

        import coremltools as ct

        compute_units = getattr(ct.ComputeUnit, self._COMPUTE_UNITS[self._compute_units_name])
        package = self._bundle / self._bucket_entries[batch_size]["package"]
        model = ct.models.MLModel(str(package), compute_units=compute_units)
        self._models[batch_size] = model

        while len(self._models) > self._max_loaded_buckets:
            _, evicted = self._models.popitem(last=False)
            del evicted
            gc.collect()
        return model

    def _choose_bucket(self, total_batch: int) -> int:
        """Choose one package for an entire call to avoid compile-cache thrash."""
        largest = self._buckets[-1]
        if total_batch > largest:
            return largest
        return next(size for size in self._buckets if size >= total_batch)

    def forward(self, im_batch) -> np.ndarray:
        array = self.to_numpy(im_batch).astype(np.float32, copy=False)
        if array.ndim != 4 or tuple(array.shape[1:]) != self._crop_shape:
            raise ValueError(f"Core ML expected NCHW input (*, {self._crop_shape}), got {tuple(array.shape)}")
        if array.shape[0] == 0:
            return np.empty((0, self._output_width), dtype=np.float32)

        outputs: list[np.ndarray] = []
        offset = 0
        bucket_size = self._choose_bucket(array.shape[0])
        model = self._load_bucket(bucket_size)
        while offset < array.shape[0]:
            valid = min(bucket_size, array.shape[0] - offset)
            if valid == bucket_size:
                chunk = np.ascontiguousarray(array[offset : offset + valid])
            else:
                chunk = self._pad_buffers[bucket_size]
                chunk.fill(0)
                chunk[:valid] = array[offset : offset + valid]

            prediction = model.predict({self._input_name: chunk})
            output = np.asarray(prediction[self._output_name], dtype=np.float32)
            outputs.append(output[:valid])
            offset += valid

        return outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)
