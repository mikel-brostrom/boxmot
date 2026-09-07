"""Build helpers for native trackers and the standalone native ReID encoder."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

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
    from multiple worker subprocesses.
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


def native_component_source_dir(name: str) -> Path:
    """Return the CMake source directory for a low-level native component."""

    if name == "reid":
        return package_native_root() / "cpp" / "reid"
    return tracker_source_dir(name)


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
    src = native_component_source_dir(name)
    return [src / lib_filename, src / "lib" / lib_filename]


def build_library_candidates(name: str, lib_filename: str) -> list[Path]:
    """Editable-install fallback locations for the shared library."""
    bd = tracker_build_dir(name)
    return [bd / lib_filename, bd / "Release" / lib_filename, bd / "Debug" / lib_filename]


# ---------------------------------------------------------------------------
# Platform-aware filename + candidate helpers (per-tracker convenience)
# ---------------------------------------------------------------------------


def library_filename(tracker_name: str) -> str:
    """Return the C-API shared library filename for a tracker on the current OS.

    Convention: ``<tracker>_capi`` with the platform's shared-library suffix.
    """
    if os.name == "nt":
        return f"{tracker_name}_capi.dll"
    if sys.platform == "darwin":
        return f"{tracker_name}_capi.dylib"
    return f"{tracker_name}_capi.so"


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
    roots = [
        native_component_source_dir(tracker_name),
        native_cpp_root / "include",
        native_cpp_root / "cmake",
    ]
    if tracker_name != "reid":
        roots.append(tracker_source_dir("base"))
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
    sibling C API artifacts) and prevents CMake from accepting a
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
        source_dir = native_component_source_dir(tracker_name)
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
        # worker subprocesses trampling
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
