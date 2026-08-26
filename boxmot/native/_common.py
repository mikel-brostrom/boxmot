"""Shared helpers for native (C++) tracker backends.

Centralizes functionality that was previously duplicated across each
``boxmot/native/trackers/<tracker>.py`` module:

* ReID model resolution and ONNX auto-export (used by BoTSORT and OccluBoost).
* ``dets_n_embs`` cache root construction (used by every native replay backend).
* Progress / stderr / summary parsing helpers shared by every native runner.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from boxmot.utils.misc import resolve_model_path

PROGRESS_PREFIX = "BOXMOT_PROGRESS\t"

# Module-wide lock used to serialize potentially expensive ONNX exports.
EXPORT_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Build status reporting
# ---------------------------------------------------------------------------

_build_status_state = threading.local()


def set_build_status_fn(status_fn: Any) -> None:
    """Register a callback used to report native build progress.

    The callback is typically a ``WorkflowDetailCallback`` whose ``__call__``
    routes a status message into an active Rich workflow panel. When set,
    :func:`run_build_step` streams CMake output into the panel instead of
    printing it to stdout (which would corrupt the Rich Live region).

    Pass ``None`` to clear the registration.
    """
    _build_status_state.status_fn = status_fn


def get_build_status_fn() -> Any:
    """Return the currently registered build status callback, if any."""
    return getattr(_build_status_state, "status_fn", None)


def run_build_step(
    *,
    cmd: list[str],
    label: str,
    status_fn: Any | None = None,
) -> int:
    """Run a CMake build subcommand, routing output through ``status_fn``.

    When ``status_fn`` is callable (or registered via
    :func:`set_build_status_fn`), captures stdout/stderr line-by-line and
    forwards each line to the callback so it appears inside the active Rich
    workflow panel. When no callback is active, falls back to the legacy
    behaviour of streaming output to the terminal.
    """
    if status_fn is None:
        status_fn = get_build_status_fn()

    if not callable(status_fn):
        print(f"[boxmot build] {label}", flush=True)
        result = subprocess.run(cmd, check=False)
        return result.returncode

    status_fn(f"{label}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_line = label
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if not line:
                continue
            last_line = line
            status_fn(f"{label}\n{line}")
    finally:
        process.wait()
    if process.returncode != 0:
        status_fn(f"{label} (failed)\n{last_line}")
    return process.returncode


@contextlib.contextmanager
def _cross_process_build_lock(build_dir: Path):
    """Serialize CMake configure/build across threads *and* subprocesses.

    The native trackers can be invoked concurrently from a thread pool **and**
    from multiple worker subprocesses (e.g. ``--replay-backend process``).
    A simple ``threading.Lock`` only protects threads inside one process, so
    parallel workers race on the same ``build/native/<name>`` directory and
    corrupt CMake's cache. This context manager wraps the build with a POSIX
    ``fcntl.flock`` (or ``msvcrt.locking`` on Windows) on a sentinel file so
    only one process at a time runs ``cmake configure`` / ``cmake --build``.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    lock_path = build_dir.parent / f"{build_dir.name}.lock"

    fh = open(lock_path, "w")
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


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

    Native C++ sources live at ``boxmot/native/cpp/trackers/<name>``.
    The path is the same whether the package is imported from a source
    checkout or an installed wheel.
    """
    return package_native_root() / "cpp" / "trackers" / str(name)


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
    return installed_executable_candidates(tracker_name, name) + build_executable_candidates(tracker_name, name)


def candidate_libraries(tracker_name: str) -> list[Path]:
    """Installed-then-built search paths for the C-API shared library."""
    name = library_filename(tracker_name)
    return installed_library_candidates(tracker_name, name) + build_library_candidates(tracker_name, name)


_NATIVE_BUILD_INPUT_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cmake"})

_NATIVE_DEPENDENCY_CACHE_KEYS = frozenset(
    {
        "CMAKE_CXX_COMPILER",
        "CMAKE_MAKE_PROGRAM",
        "CMAKE_TOOLCHAIN_FILE",
        "Eigen3_DIR",
        "ONNXRUNTIME_INCLUDE_DIR",
        "ONNXRUNTIME_LIB",
        "ONNXRUNTIME_ROOT",
        "OpenCV_DIR",
        "onnxruntime_DIR",
    }
)

_NATIVE_ORT_DISCOVERY_ROOTS = (
    Path("/opt/homebrew/opt/onnxruntime"),
    Path("/opt/homebrew/lib/cmake/onnxruntime"),
    Path("/usr/local/opt/onnxruntime"),
    Path("/usr/local/lib/cmake/onnxruntime"),
    Path("/usr/lib/cmake/onnxruntime"),
    Path("/usr/lib64/cmake/onnxruntime"),
)


def _native_build_input_files(tracker_name: str) -> list[Path]:
    """Return source and CMake inputs that can affect a tracker artifact."""
    native_cpp_root = package_native_root() / "cpp"
    roots = (
        tracker_source_dir(tracker_name),
        tracker_source_dir("base"),
        native_cpp_root / "cmake",
    )
    inputs: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            is_build_input = path.name == "CMakeLists.txt" or path.suffix.lower() in _NATIVE_BUILD_INPUT_SUFFIXES
            if path.is_file() and is_build_input:
                inputs.add(path)
    return sorted(inputs, key=lambda path: str(path))


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _native_build_fingerprint(tracker_name: str) -> str:
    """Hash native sources and CMake files used by a tracker build."""
    digest = hashlib.sha256()
    native_root = package_native_root()
    for path in _native_build_input_files(tracker_name):
        try:
            identity = path.relative_to(native_root).as_posix()
        except ValueError:
            identity = str(path.resolve())
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_cmake_cache(build_dir: Path) -> dict[str, tuple[str, str]]:
    """Parse stable key/type/value entries from a CMake cache."""
    cache_path = build_dir / "CMakeCache.txt"
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    entries: dict[str, tuple[str, str]] = {}
    for line in lines:
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        if ":" not in key_and_type:
            continue
        key, entry_type = key_and_type.rsplit(":", 1)
        entries[key] = (entry_type, value)
    return entries


def _update_path_fingerprint(digest: Any, label: str, path: Path, *, hash_cmake_files: bool = False) -> None:
    """Add path identity and dependency configuration contents to ``digest``."""
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(path).encode("utf-8"))
    digest.update(b"\0")

    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError:
        digest.update(b"missing\0")
        return

    digest.update(str(resolved).encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"{stat.st_mode}:{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    digest.update(b"\0")

    if resolved.is_file() and (hash_cmake_files or resolved.suffix.lower() == ".cmake"):
        digest.update(_sha256_file(resolved).encode("ascii"))
        digest.update(b"\0")
    elif resolved.is_dir() and hash_cmake_files:
        for cmake_path in sorted(resolved.rglob("*.cmake"), key=lambda item: str(item)):
            if not cmake_path.is_file():
                continue
            digest.update(str(cmake_path.relative_to(resolved)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_file(cmake_path).encode("ascii"))
            digest.update(b"\0")


def _native_dependency_probe_paths(cache: dict[str, tuple[str, str]]) -> list[tuple[str, Path, bool]]:
    """Return toolchain and dependency locations that affect CMake discovery."""
    probes: list[tuple[str, Path, bool]] = []
    for key in sorted(_NATIVE_DEPENDENCY_CACHE_KEYS):
        value = cache.get(key, ("", ""))[1]
        if not value or value.endswith("-NOTFOUND"):
            continue
        hash_cmake_files = key.endswith("_DIR") or key == "CMAKE_TOOLCHAIN_FILE"
        probes.append((f"cache:{key}", Path(value), hash_cmake_files))

    for root in _NATIVE_ORT_DISCOVERY_ROOTS:
        probes.append(("ort-discovery", root, True))

    return probes


def _native_build_configuration_fingerprint(build_dir: Path) -> str:
    """Hash the effective CMake, toolchain, and dependency configuration.

    The configured cache captures the effective generator/options and resolved
    OpenCV, Eigen, and ONNX Runtime locations. Filesystem probes also make an
    optional ONNX Runtime install/removal visible before CMake is run. Raw
    compiler/generator environment variables are deliberately excluded: CMake
    ignores those initial-only inputs once a build directory has a cache.
    This fingerprint must be recomputed after configure because the first
    configure creates the cache and compiler-identification files.
    """
    digest = hashlib.sha256()
    digest.update(b"boxmot-native-build-configuration-v1\0")
    digest.update(
        f"{os.name}:{sys.platform}:{platform.machine()}:{platform.system()}:{platform.release()}".encode("utf-8")
    )
    digest.update(b"\0")

    cmake_executable = shutil.which("cmake")
    _update_path_fingerprint(
        digest,
        "cmake-executable",
        Path(cmake_executable) if cmake_executable else Path("cmake-not-found"),
    )

    cache = _read_cmake_cache(build_dir)
    for key, (entry_type, value) in sorted(cache.items()):
        digest.update(f"{key}:{entry_type}={value}".encode("utf-8"))
        digest.update(b"\0")

    compiler_state_files = sorted((build_dir / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    system_state_files = sorted((build_dir / "CMakeFiles").glob("*/CMakeSystem.cmake"))
    for state_path in compiler_state_files + system_state_files:
        _update_path_fingerprint(digest, "cmake-state", state_path, hash_cmake_files=True)

    for label, path, hash_cmake_files in _native_dependency_probe_paths(cache):
        _update_path_fingerprint(digest, label, path, hash_cmake_files=hash_cmake_files)
    return digest.hexdigest()


def _native_build_stamp_path(build_dir: Path, target: str) -> Path:
    return build_dir / f".{target}.source-sha256"


def _read_native_build_stamp(build_dir: Path, target: str) -> dict[str, Any] | None:
    try:
        stamp = json.loads(_native_build_stamp_path(build_dir, target).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


def _native_build_is_current(
    build_dir: Path,
    target: str,
    source_fingerprint: str,
    configuration_fingerprint: str,
    artifact: Path,
) -> bool:
    stamp = _read_native_build_stamp(build_dir, target)
    if stamp is None:
        return False
    try:
        artifact_stat = artifact.stat()
        artifact_sha256 = _sha256_file(artifact)
    except OSError:
        return False
    return (
        stamp.get("artifact") == str(artifact.resolve())
        and stamp.get("artifact_size") == artifact_stat.st_size
        and stamp.get("artifact_sha256") == artifact_sha256
        and stamp.get("source_sha256") == source_fingerprint
        and stamp.get("configuration_sha256") == configuration_fingerprint
    )


def _write_native_build_stamp(
    build_dir: Path,
    target: str,
    source_fingerprint: str,
    configuration_fingerprint: str,
    artifact: Path,
) -> None:
    """Atomically record build inputs and the exact artifact contents."""
    stamp_path = _native_build_stamp_path(build_dir, target)
    pending_path = stamp_path.with_suffix(f"{stamp_path.suffix}.{os.getpid()}.tmp")
    artifact_stat = artifact.stat()
    stamp = {
        "artifact": str(artifact.resolve()),
        "artifact_mtime_ns": artifact_stat.st_mtime_ns,
        "artifact_size": artifact_stat.st_size,
        "artifact_sha256": _sha256_file(artifact),
        "configuration_sha256": configuration_fingerprint,
        "source_sha256": source_fingerprint,
    }
    pending_path.write_text(json.dumps(stamp, sort_keys=True) + "\n", encoding="utf-8")
    pending_path.replace(stamp_path)


def _is_installed_native_artifact(candidate: Path, source_dir: Path) -> bool:
    """Return whether ``candidate`` is a packaged artifact beside its sources."""
    try:
        candidate.resolve().relative_to(source_dir.resolve())
    except (OSError, ValueError):
        return False
    return True


def _is_native_source_checkout() -> bool:
    """Return whether native sources are being loaded from a Git checkout."""
    return (repo_root() / ".git").exists()


def _current_native_candidate(
    candidates: list[Path],
    *,
    source_dir: Path,
    build_dir: Path,
    target: str,
    source_fingerprint: str,
    configuration_fingerprint: str,
    trust_installed: bool,
) -> Path | None:
    for candidate in candidates:
        if not candidate.exists():
            continue
        if _is_installed_native_artifact(candidate, source_dir):
            if trust_installed:
                return candidate
            continue
        if _native_build_is_current(
            build_dir,
            target,
            source_fingerprint,
            configuration_fingerprint,
            candidate,
        ):
            return candidate
    return None


def _native_artifact_state(path: Path) -> tuple[int, int, str] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, _sha256_file(path)
    except OSError:
        return None


def _candidate_configuration_rank(candidate: Path, build_dir: Path) -> tuple[int, str]:
    """Rank a built candidate for the generator's requested Release config."""
    cache = _read_cmake_cache(build_dir)
    multi_config = bool(cache.get("CMAKE_CONFIGURATION_TYPES", ("", ""))[1])
    try:
        relative = candidate.resolve().relative_to(build_dir.resolve())
    except (OSError, ValueError):
        return 3, str(candidate)

    in_release_dir = bool(relative.parts) and relative.parts[0].lower() == "release"
    at_build_root = len(relative.parts) == 1
    if multi_config:
        return (0 if in_release_dir else 1 if at_build_root else 2), str(candidate)
    return (0 if at_build_root else 1 if in_release_dir else 2), str(candidate)


def _select_built_candidate(
    candidates: list[Path],
    *,
    source_dir: Path,
    build_dir: Path,
    before_build: dict[Path, tuple[int, int, str] | None],
) -> Path | None:
    """Select an artifact that was actually produced by the target build."""
    editable = [
        candidate
        for candidate in candidates
        if candidate.exists() and not _is_installed_native_artifact(candidate, source_dir)
    ]
    produced = [candidate for candidate in editable if _native_artifact_state(candidate) != before_build.get(candidate)]
    if not produced:
        return None
    return min(produced, key=lambda candidate: _candidate_configuration_rank(candidate, build_dir))


def _remove_stale_native_candidates(
    candidates: list[Path],
    *,
    source_dir: Path,
    build_dir: Path,
) -> set[Path]:
    """Remove only the requested editable artifacts so CMake must relink them.

    A source/configuration fingerprint mismatch or a content-hash mismatch
    means an existing artifact cannot be trusted. Removing the exact target
    outputs after configure avoids ``--clean-first`` (which also deletes
    sibling replay/CAPI artifacts) and prevents CMake from accepting a
    tampered artifact whose timestamp happens to look current.
    """
    resolved_build_dir = build_dir.resolve()
    removed: set[Path] = set()
    for candidate in candidates:
        if not candidate.exists() or _is_installed_native_artifact(candidate, source_dir):
            continue
        try:
            candidate.resolve().relative_to(resolved_build_dir)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Refusing to remove native artifact outside its build directory: {candidate}") from exc
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"Failed to replace stale native artifact {candidate}: {exc}") from exc
        removed.add(candidate)
    return removed


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

    Packaged artifacts installed beside their sources are trusted as immutable.
    Editable-build artifacts are reused only when their recorded source/CMake
    fingerprint matches the current tree. Raises ``RuntimeError`` on
    configure/build failure or if the expected artifact is still missing.
    """
    with build_lock:
        source_dir = tracker_source_dir(tracker_name)
        build_dir = tracker_build_dir(tracker_name)
        source_fingerprint = _native_build_fingerprint(tracker_name)
        configuration_fingerprint = _native_build_configuration_fingerprint(build_dir)
        trust_installed = not _is_native_source_checkout() and not force_rebuild

        if not force_rebuild:
            current_candidate = _current_native_candidate(
                candidates,
                source_dir=source_dir,
                build_dir=build_dir,
                target=target,
                source_fingerprint=source_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
                trust_installed=trust_installed,
            )
            if current_candidate is not None:
                return current_candidate

        build_dir.mkdir(parents=True, exist_ok=True)

        # Cross-process lock: prevents racing CMake invocations from multiple
        # worker subprocesses (e.g. ``--replay-backend process``) trampling
        # each other's CMake cache in the shared build directory.
        with _cross_process_build_lock(build_dir):
            # The cache or source tree may have changed while this process was
            # waiting for a sibling builder. Refresh both fingerprints before
            # deciding whether that sibling produced a reusable artifact.
            source_fingerprint = _native_build_fingerprint(tracker_name)
            configuration_fingerprint = _native_build_configuration_fingerprint(build_dir)

            # Re-check after acquiring the file lock: a sibling process may
            # have just finished building the artifact while we waited.
            if not force_rebuild:
                current_candidate = _current_native_candidate(
                    candidates,
                    source_dir=source_dir,
                    build_dir=build_dir,
                    target=target,
                    source_fingerprint=source_fingerprint,
                    configuration_fingerprint=configuration_fingerprint,
                    trust_installed=trust_installed,
                )
                if current_candidate is not None:
                    return current_candidate

            before_build = {
                candidate: _native_artifact_state(candidate)
                for candidate in candidates
                if not _is_installed_native_artifact(candidate, source_dir)
            }

            configure_cmd = [
                "cmake",
                "-S",
                str(source_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ]
            # Stream output live so the user sees progress (CMake configure +
            # build can take a minute or more for OpenCV-heavy trackers).
            rc = run_build_step(
                cmd=configure_cmd,
                label=f"Building {display_name}: configuring...",
            )
            if rc != 0:
                raise RuntimeError(
                    f"Failed to configure native {display_name}.\n"
                    "Requirements: CMake 3.16+, OpenCV 4.x, Eigen3 3.3+.\n"
                    f"Command: {' '.join(configure_cmd)}"
                )

            removed_candidates = _remove_stale_native_candidates(
                candidates,
                source_dir=source_dir,
                build_dir=build_dir,
            )
            for candidate in removed_candidates:
                # The target must recreate a removed artifact even on a
                # filesystem whose timestamp resolution is too coarse to
                # distinguish the old and new file.
                before_build[candidate] = None

            build_cmd = [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                target,
                "--parallel",
            ]
            rc = run_build_step(
                cmd=build_cmd,
                label=f"Building {display_name}: compiling...",
            )
            if rc != 0:
                raise RuntimeError(
                    f"Failed to build native {display_name}.\n"
                    "Requirements: C++17 compiler, OpenCV 4.x, Eigen3 3.3+.\n"
                    f"Command: {' '.join(build_cmd)}"
                )

            built_candidate = _select_built_candidate(
                candidates,
                source_dir=source_dir,
                build_dir=build_dir,
                before_build=before_build,
            )
            if built_candidate is not None:
                # Configure populates CMakeCache.txt and compiler/dependency
                # state, so stamp the post-configure fingerprint. Computing it
                # only before the build would force one redundant rebuild.
                configuration_fingerprint = _native_build_configuration_fingerprint(build_dir)
                _write_native_build_stamp(
                    build_dir,
                    target,
                    source_fingerprint,
                    configuration_fingerprint,
                    built_candidate,
                )
                return built_candidate

            if trust_installed:
                for candidate in candidates:
                    if candidate.exists() and _is_installed_native_artifact(candidate, source_dir):
                        return candidate

            raise RuntimeError(not_found_message)


# ---------------------------------------------------------------------------
# dets_n_embs cache layout
# ---------------------------------------------------------------------------


def dets_n_embs_root(project_root: str | Path, dataset_name: str | None = None, split: str | None = None) -> Path:
    """Return the canonical ``dets_n_embs`` cache root for a project.

    Mirrors the layout used by :mod:`boxmot.engine.eval.cache` and the native
    replay binaries: ``<project_root>/dets_n_embs[/<dataset>][/<split>]``.
    """
    root = Path(project_root) / "dets_n_embs"
    if dataset_name:
        root = root / dataset_name
    if split:
        root = root / split
    return root


def cached_embedding_path(
    project_root: str | Path,
    detector_name: str,
    reid_name: str,
    sequence_name: str,
    *,
    dataset_name: str | None = None,
    split: str | None = None,
    preprocess_name: str | None = None,
    tracker_backend: str | None = None,
) -> Path:
    """Return the expected path of a cached embedding ``.npy`` for a sequence.

    The canonical bucket and preprocessing names come from
    :func:`boxmot.data.cache.reid_cache_key` and
    :func:`boxmot.data.cache.reid_preprocess_cache_key`.
    """
    from boxmot.data.cache import reid_cache_key, reid_preprocess_cache_key

    detector_key = _stem_key(detector_name)
    preprocess_key = reid_preprocess_cache_key(preprocess_name)
    embs_root = dets_n_embs_root(project_root, dataset_name, split=split) / detector_key / "embs"

    canonical_key = reid_cache_key(reid_name, tracker_backend=tracker_backend)
    return embs_root / canonical_key / preprocess_key / f"{sequence_name}.npy"


def resolve_embedding_cache_location(
    project_root: str | Path,
    detector_name: str | Path,
    reid_name: str | Path,
    sequence_name: str,
    *,
    dataset_name: str | None = None,
    split: str | None = None,
    preprocess_name: str | None = None,
    tracker_backend: str | None = None,
    embedding_cache_dir: str | Path | None = None,
) -> tuple[str, str, Path, Path]:
    """Resolve native replay cache arguments and the selected sequence files.

    Native replay accepts the model bucket and preprocessing bucket as separate
    command-line values. ``embedding_cache_dir`` is the authoritative directory
    selected by the evaluation cache planner and may point at either the current
    layout or a trusted older layout.
    """
    from boxmot.data.cache import reid_cache_key, reid_preprocess_cache_key

    detector_key = _stem_key(detector_name)
    detector_root = dets_n_embs_root(project_root, dataset_name, split=split) / detector_key
    embeddings_root = detector_root / "embs"

    if embedding_cache_dir is None:
        reid_key = reid_cache_key(reid_name, tracker_backend=tracker_backend)
        preprocess_key = reid_preprocess_cache_key(preprocess_name)
        selected_dir = embeddings_root / reid_key / preprocess_key
    else:
        selected_dir = Path(embedding_cache_dir)
        try:
            reid_relative = selected_dir.parent.resolve().relative_to(embeddings_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Embedding cache directory must be under {embeddings_root}: {selected_dir}") from exc
        if reid_relative == Path(".") or not selected_dir.name:
            raise ValueError(f"Embedding cache directory is missing model/preprocess components: {selected_dir}")
        reid_key = reid_relative.as_posix()
        preprocess_key = selected_dir.name

    filename = f"{Path(sequence_name).stem}.npy"
    return (
        str(reid_key),
        str(preprocess_key),
        selected_dir / filename,
        detector_root / "dets" / filename,
    )


def embedding_cache_is_complete(embedding_path: str | Path, detection_path: str | Path) -> bool:
    """Return whether a numeric embedding cache is row-aligned with detections."""
    embedding_path = Path(embedding_path)
    detection_path = Path(detection_path)
    if not embedding_path.is_file() or not detection_path.is_file():
        return False

    try:
        import numpy as np

        embeddings = np.load(embedding_path, mmap_mode="r")
        detections = np.load(detection_path, mmap_mode="r")
    except Exception:  # noqa: BLE001 - corrupt cache files are treated as misses
        return False
    return (
        embeddings.ndim == 2
        and detections.ndim == 2
        and embeddings.shape[0] == detections.shape[0]
        and (embeddings.shape[0] == 0 or embeddings.shape[1] > 0)
    )


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
    """Path of the ONNX cache produced from a ``.pt`` file.

    The native cpp ReID path uses a plain ``<stem>.onnx`` sibling next to the
    ``.pt`` weights so the cache is interoperable with any ONNX consumer.
    """
    return weights.with_suffix(".onnx")


def resolve_reid_model_ref(reid_weights: str | Path | None) -> Path | None:
    """Resolve a user-provided ReID weight reference to a concrete file path.

    Lookup precedence used by the native trackers: prefer a sibling ``*.onnx``
    cache when one is available, otherwise fall back to the original ONNX or
    PyTorch weights.
    """
    if reid_weights is None:
        return None

    path = Path(reid_weights)
    if path.suffix.lower() == ".onnx":
        return resolve_model_path(path)

    if path.suffix.lower() == ".pt":
        resolved_pt = resolve_model_path(path)
        onnx_candidate = native_onnx_cache_path(resolved_pt)
        if onnx_candidate.exists():
            return onnx_candidate
        return resolved_pt

    if not path.suffix:
        pt_candidate = resolve_model_path(path.with_suffix(".pt"))
        onnx_candidate = native_onnx_cache_path(pt_candidate)
        if onnx_candidate.exists():
            return onnx_candidate
        explicit_onnx = resolve_model_path(path.with_suffix(".onnx"))
        if explicit_onnx.exists():
            return explicit_onnx
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
    import torch

    from boxmot.engine.reid.export import setup_model

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
        raise RuntimeError(f"Failed to export native {display_name} ReID model to ONNX: {weights}")
    return onnx_path


def _download_reid_pt_weights(weights: Path, *, display_name: str = "ReID") -> None:
    """Auto-download a known ReID ``.pt`` checkpoint into ``weights``.

    Mirrors the lazy download performed by ``BaseModelBackend.download_model``
    so the native cpp ReID path matches the Python path's UX (e.g. CI runners
    that have never cached the weights locally).
    """
    try:
        import gdown  # noqa: WPS433 (runtime import to avoid hard dep at import-time)
        from filelock import SoftFileLock  # noqa: WPS433

        from boxmot.reid.core.registry import ReIDModelRegistry  # noqa: WPS433
        from boxmot.utils import logger as LOGGER  # noqa: WPS433
    except Exception:  # pragma: no cover - if optional deps missing, fall through
        return

    weights.parent.mkdir(parents=True, exist_ok=True)
    model_url = ReIDModelRegistry.get_model_url(weights)
    if not model_url:
        return

    lock = SoftFileLock(str(weights) + ".lock", timeout=300)
    with lock:
        if weights.exists():
            return
        LOGGER.info(f"[PID {os.getpid()}] Downloading native {display_name} weights from {model_url} -> {weights}")
        gdown.download(model_url, str(weights), quiet=False)


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
        _download_reid_pt_weights(resolved, display_name=display_name)
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
    raise RuntimeError(f"Failed to parse native {display_name} summary JSON from stdout:\n{text}")
