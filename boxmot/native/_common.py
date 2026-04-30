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
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from boxmot.utils.misc import resolve_model_path

PROGRESS_PREFIX = "BOXMOT_PROGRESS\t"

# Module-wide lock used to serialize potentially expensive ONNX exports.
EXPORT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Native source/build/install layout
# ---------------------------------------------------------------------------

def package_native_root() -> Path:
    """Return the ``boxmot/native`` directory inside the installed package."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Return the repository root.

    Only valid for editable / source checkouts. Wheels installed via pip will
    typically not have a meaningful repo root above the package, so callers
    must treat the returned path as best-effort.
    """
    return Path(__file__).resolve().parents[2]


def tracker_source_dir(name: str) -> Path:
    """Directory containing the C++ sources for a given native tracker.

    After the relocation of ``native/trackers`` into the package, this lives
    at ``boxmot/native/trackers/<name>``. The path is the same whether the
    package is imported from a source checkout or an installed wheel.
    """
    return package_native_root() / "trackers" / str(name)


def tracker_build_dir(name: str) -> Path:
    """Out-of-tree CMake build directory used by editable / dev installs.

    Located at ``<repo>/build/native/<name>``. Wheels never write here.
    """
    return repo_root() / "build" / "native" / str(name)


def installed_library_candidates(name: str, lib_filename: str) -> list[Path]:
    """Where scikit-build-core places the shared library inside the wheel.

    The build configuration installs the shared library beside the C++ source
    directory so it ships with the package and is loadable without re-running
    CMake at runtime.
    """
    src = tracker_source_dir(name)
    return [src / lib_filename, src / "lib" / lib_filename]


def installed_executable_candidates(name: str, exe_filename: str) -> list[Path]:
    """Where the native replay executable is shipped inside the wheel."""
    src = tracker_source_dir(name)
    return [src / exe_filename, src / "bin" / exe_filename]


def build_library_candidates(name: str, lib_filename: str) -> list[Path]:
    """Editable-install fallback locations for the shared library."""
    bd = tracker_build_dir(name)
    return [bd / lib_filename, bd / "Release" / lib_filename, bd / "Debug" / lib_filename]


def build_executable_candidates(name: str, exe_filename: str) -> list[Path]:
    """Editable-install fallback locations for the replay executable."""
    bd = tracker_build_dir(name)
    return [bd / exe_filename, bd / "Release" / exe_filename, bd / "Debug" / exe_filename]


# ---------------------------------------------------------------------------
# Platform-aware filename + candidate helpers (per-tracker convenience)
# ---------------------------------------------------------------------------

def executable_filename(tracker_name: str) -> str:
    """Return the replay executable filename for a tracker on the current OS.

    Convention: ``<tracker>_replay`` (with ``.exe`` on Windows).
    """
    return f"{tracker_name}_replay.exe" if os.name == "nt" else f"{tracker_name}_replay"


def library_filename(tracker_name: str) -> str:
    """Return the C-API shared library filename for a tracker on the current OS.

    Convention: ``<tracker>_capi`` with the platform's shared-library suffix.
    """
    if os.name == "nt":
        return f"{tracker_name}_capi.dll"
    if sys.platform == "darwin":
        return f"{tracker_name}_capi.dylib"
    return f"{tracker_name}_capi.so"


def candidate_executables(tracker_name: str) -> list[Path]:
    """Installed-then-built search paths for the replay executable."""
    name = executable_filename(tracker_name)
    return (
        installed_executable_candidates(tracker_name, name)
        + build_executable_candidates(tracker_name, name)
    )


def candidate_libraries(tracker_name: str) -> list[Path]:
    """Installed-then-built search paths for the C-API shared library."""
    name = library_filename(tracker_name)
    return (
        installed_library_candidates(tracker_name, name)
        + build_library_candidates(tracker_name, name)
    )


def build_native_target(
    *,
    tracker_name: str,
    display_name: str,
    target: str,
    candidates: list[Path],
    force_rebuild: bool,
    not_found_message: str,
    build_lock: threading.Lock,
) -> Path:
    """Configure and build a single CMake target for a native tracker.

    Returns the first existing candidate path after the build (or before, if
    one already exists and ``force_rebuild`` is False). Raises ``RuntimeError``
    on configure/build failure or if the expected artifact is still missing.
    """
    with build_lock:
        if not force_rebuild:
            for candidate in candidates:
                if candidate.exists():
                    return candidate

        source_dir = tracker_source_dir(tracker_name)
        build_dir = tracker_build_dir(tracker_name)
        build_dir.mkdir(parents=True, exist_ok=True)

        configure_cmd = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        configure = subprocess.run(
            configure_cmd, capture_output=True, text=True, check=False
        )
        if configure.returncode != 0:
            raise RuntimeError(
                f"Failed to configure native {display_name}.\n"
                "Requirements: CMake 3.16+, OpenCV 4.x, Eigen3 3.3+.\n"
                f"Command: {' '.join(configure_cmd)}\n"
                f"{configure.stderr.strip()}"
            )

        build_cmd = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            target,
        ]
        build = subprocess.run(build_cmd, capture_output=True, text=True, check=False)
        if build.returncode != 0:
            raise RuntimeError(
                f"Failed to build native {display_name}.\n"
                "Requirements: C++17 compiler, OpenCV 4.x, Eigen3 3.3+.\n"
                f"Command: {' '.join(build_cmd)}\n"
                f"{build.stderr.strip()}"
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise RuntimeError(not_found_message)


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
    """Return the expected path of a cached embedding ``.npy`` for a sequence.

    Prefers the new suffix-included cache directory (e.g. ``lmbn_n_duke.pt``)
    but transparently falls back to the legacy stem-only directory
    (``lmbn_n_duke``) when only the legacy cache exists on disk.
    """
    detector_key = _stem_key(detector_name)
    preprocess_key = str(preprocess_name or "resize")
    embs_root = dets_n_embs_root(project_root, dataset_name) / detector_key / "embs"
    reid_key_new = _name_key(reid_name)
    new_path = embs_root / reid_key_new / preprocess_key / f"{sequence_name}.npy"
    if new_path.exists():
        return new_path
    legacy_key = _stem_key(reid_name)
    if legacy_key and legacy_key != reid_key_new:
        legacy_path = embs_root / legacy_key / preprocess_key / f"{sequence_name}.npy"
        if legacy_path.exists():
            return legacy_path
    return new_path


def _stem_key(name: str | Path) -> str:
    path = Path(name)
    return path.stem if path.suffix else str(name)


def _name_key(name: str | Path) -> str:
    path = Path(name)
    return path.name if path.suffix else str(name)


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
