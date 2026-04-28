"""Shared helpers for native (C++) tracker backends.

Centralizes functionality that was previously duplicated across each
``boxmot/native/<tracker>_cpp.py`` module:

* ReID model resolution and ONNX auto-export (used by BoTSORT and OccluBoost).
* ``dets_n_embs`` cache root construction (used by every native replay backend).
* Progress / stderr / summary parsing helpers shared by every native runner.
"""

from __future__ import annotations

import inspect
import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from boxmot.utils.misc import resolve_model_path

PROGRESS_PREFIX = "BOXMOT_PROGRESS\t"

# Module-wide lock used to serialize potentially expensive ONNX exports.
EXPORT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# dets_n_embs cache layout
# ---------------------------------------------------------------------------

def dets_n_embs_root(project_root: str | Path, dataset_name: str | None = None) -> Path:
    """Return the canonical ``dets_n_embs`` cache root for a project.

    Mirrors the layout used by :mod:`boxmot.engine.cache` and the native
    replay binaries: ``<project_root>/dets_n_embs[/<dataset>]``.
    """
    root = Path(project_root) / "dets_n_embs"
    if dataset_name:
        root = root / dataset_name
    return root


def cached_embedding_path(
    project_root: str | Path,
    detector_name: str,
    reid_name: str,
    sequence_name: str,
    *,
    dataset_name: str | None = None,
    preprocess_name: str | None = None,
) -> Path:
    """Return the expected path of a cached embedding ``.npy`` for a sequence."""
    detector_key = _stem_key(detector_name)
    reid_key = _stem_key(reid_name)
    preprocess_key = str(preprocess_name or "resize")
    return (
        dets_n_embs_root(project_root, dataset_name)
        / detector_key
        / "embs"
        / reid_key
        / preprocess_key
        / f"{sequence_name}.npy"
    )


def _stem_key(name: str | Path) -> str:
    path = Path(name)
    return path.stem if path.suffix else str(name)


# ---------------------------------------------------------------------------
# ReID model resolution + ONNX export
# ---------------------------------------------------------------------------

def native_onnx_cache_path(weights: Path) -> Path:
    """Path of the OpenCV-compatible ONNX cache produced from a ``.pt`` file."""
    return weights.with_name(f"{weights.stem}_opencv.onnx")


def resolve_reid_model_ref(reid_weights: str | Path | None) -> Path | None:
    """Resolve a user-provided ReID weight reference to a concrete file path.

    Mirrors the lookup precedence used by the native trackers: prefer an
    OpenCV-compatible ``*_opencv.onnx`` cache when one is available, otherwise
    fall back to the original ONNX or PyTorch weights.
    """
    if reid_weights is None:
        return None

    path = Path(reid_weights)
    if path.suffix.lower() == ".onnx" and not path.stem.endswith("_opencv"):
        resolved_onnx = resolve_model_path(path)
        opencv_candidate = resolved_onnx.with_name(f"{resolved_onnx.stem}_opencv.onnx")
        if opencv_candidate.exists():
            return opencv_candidate
        return resolved_onnx

    if path.suffix.lower() == ".pt":
        resolved_pt = resolve_model_path(path)
        opencv_candidate = native_onnx_cache_path(resolved_pt)
        if opencv_candidate.exists():
            return opencv_candidate
        return resolved_pt

    if not path.suffix:
        pt_candidate = resolve_model_path(path.with_suffix(".pt"))
        opencv_candidate = native_onnx_cache_path(pt_candidate)
        if opencv_candidate.exists():
            return opencv_candidate
        onnx_candidate = resolve_model_path(path.with_suffix(".onnx"))
        if onnx_candidate.exists():
            if not onnx_candidate.stem.endswith("_opencv"):
                sibling_opencv = onnx_candidate.with_name(f"{onnx_candidate.stem}_opencv.onnx")
                if sibling_opencv.exists():
                    return sibling_opencv
            return onnx_candidate
        return pt_candidate
    return resolve_model_path(path)


def infer_onnx_output_names(model, dummy_input) -> list[str]:
    import torch

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    if isinstance(output, (tuple, list)):
        return [f"output{index}" for index in range(len(output))]
    return ["output0"]


def export_reid_to_onnx(weights: Path, *, display_name: str = "ReID") -> Path:
    """Export ``.pt`` ReID weights to an OpenCV-compatible ONNX file."""
    from boxmot.engine.export import setup_model
    import torch

    args = SimpleNamespace(
        weights=weights,
        device="cpu",
        half=False,
        optimize=False,
        batch_size=1,
        imgsz=None,
    )
    model, dummy_input = setup_model(args)

    output_names = infer_onnx_output_names(model, dummy_input)
    onnx_path = native_onnx_cache_path(weights)
    export_kwargs = {
        "opset_version": 17,
        "input_names": ["images"],
        "output_names": output_names,
        "dynamic_axes": {
            "images": {0: "batch"},
            **{name: {0: "batch"} for name in output_names},
        },
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        **export_kwargs,
    )
    if not onnx_path.exists():
        raise RuntimeError(
            f"Failed to export native {display_name} ReID model to ONNX: {weights}"
        )
    return onnx_path


def ensure_native_reid_model_path(
    reid_weights: str | Path | None,
    *,
    display_name: str = "ReID",
    exporter: Callable[[Path], Path] | None = None,
    resolver: Callable[[str | Path | None], Path | None] | None = None,
) -> Path | None:
    """Resolve ReID weights to a native-ready file, exporting ONNX if needed.

    ``exporter`` and ``resolver`` are injection points so per-tracker modules
    keep monkeypatchable thin wrappers.
    """
    resolve = resolver or resolve_reid_model_ref
    resolved = resolve(reid_weights)
    if resolved is None:
        return None

    suffix = resolved.suffix.lower()
    if suffix == ".onnx":
        return resolved
    if suffix != ".pt":
        raise RuntimeError(
            f"Native {display_name} ReID supports ONNX directly and can auto-export "
            f"PyTorch '.pt' weights only: {resolved}"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"Native {display_name} ReID weights not found: {resolved}")

    onnx_path = native_onnx_cache_path(resolved)
    if onnx_path.exists() and onnx_path.stat().st_mtime >= resolved.stat().st_mtime:
        return onnx_path

    export = exporter or (lambda weights: export_reid_to_onnx(weights, display_name=display_name))
    with EXPORT_LOCK:
        if onnx_path.exists() and onnx_path.stat().st_mtime >= resolved.stat().st_mtime:
            return onnx_path
        return export(resolved)


# ---------------------------------------------------------------------------
# Stdout / stderr parsing helpers shared by every native runner
# ---------------------------------------------------------------------------

def parse_progress_line(line: str) -> tuple[str, int, int] | None:
    text = str(line).strip()
    if not text.startswith(PROGRESS_PREFIX):
        return None
    parts = text.split("\t")
    if len(parts) != 4:
        return None
    _, seq_name, current, total = parts
    try:
        return seq_name, int(current), int(total)
    except ValueError:
        return None


def drain_native_stderr(stderr_stream, progress_queue, stderr_lines: list[str]) -> None:
    if stderr_stream is None:
        return
    for raw_line in stderr_stream:
        progress = parse_progress_line(raw_line)
        if progress is not None:
            if progress_queue is not None:
                try:
                    progress_queue.put_nowait(progress)
                except (OSError, queue.Full):
                    pass
            continue
        line = str(raw_line).strip()
        if line:
            stderr_lines.append(line)


def parse_summary(stdout: str, *, display_name: str = "native tracker") -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise RuntimeError(f"Native {display_name} runner produced no stdout.")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        f"Failed to parse native {display_name} summary JSON from stdout:\n{text}"
    )
