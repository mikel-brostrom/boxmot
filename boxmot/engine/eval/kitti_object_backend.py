"""Offline adapter for the user-installed, unmodified KITTI object devkit.

Only our standalone harness ships with BoxMOT. Explicit installation compiles
it against a verified official devkit and records the build provenance. Eval
uses that binary without a compiler, source checkout, or network connection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

_SOURCE_HASHES = {
    "evaluate_object.cpp": "b16410ba8914a732405ec937a16d1cd039b6e3bf7d9e8edb6dd3ec79c15892b4",
    "mail.h": "cd757cc4804dfa1f78c083d31b02be0f7a6c39e34baad605cca884be305c9209",
}
_DEVKIT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip"
_INSTALL_HINT = "Run 'boxmot install --kitti-devkit /path/to/devkit_object' with the official KITTI object devkit."
_HARNESS = Path(__file__).with_name("kitti_object_harness.cpp")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_METRICS = ("2d", "3d")
_CLASSES = ("car", "pedestrian")
_DIFFICULTIES = ("easy", "moderate", "hard")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cache_root() -> Path:
    from platformdirs import user_cache_path

    return user_cache_path("boxmot") / "kitti-object-evaluator"


def _write_json(path: Path, value: dict) -> None:
    """Publish one complete JSON document, including when replacing the pointer."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compiler() -> tuple[list[str], str]:
    command = shlex.split(os.environ.get("CXX", ""))
    if not command:
        command = [next((item for item in ("c++", "clang++", "g++") if shutil.which(item)), "c++")]
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError("KITTI devkit setup requires a C++17 compiler. Install Clang or GCC, or set CXX.")
    command[0] = str(Path(executable).resolve())
    try:
        result = subprocess.run([*command, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot run the KITTI devkit compiler: {exc}") from exc
    return command, result.stdout.strip()


def _boost() -> tuple[Path, dict]:
    """Identify the installed header-only dependency without installing anything."""
    candidates = []
    if os.environ.get("BOOST_INCLUDEDIR"):
        candidates.append(Path(os.environ["BOOST_INCLUDEDIR"]))
    if os.environ.get("BOOST_ROOT"):
        candidates.extend((Path(os.environ["BOOST_ROOT"]) / "include", Path(os.environ["BOOST_ROOT"])))
    candidates.extend(Path(item) for item in ("/opt/homebrew/include", "/usr/local/include", "/usr/include"))
    required = ("boost/geometry.hpp", "boost/numeric/ublas/matrix.hpp", "boost/version.hpp")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            candidate = candidate.resolve()
            version = (candidate / "boost/version.hpp").read_text()
            match = re.search(r'^#define BOOST_LIB_VERSION "([^"]+)"', version, re.MULTILINE)
            # Installation runs once. Hash the header tree so patched Boost
            # installations cannot collide merely because versions agree.
            headers = {
                str(path.relative_to(candidate)): _sha256(path)
                for path in sorted((candidate / "boost").rglob("*"))
                if path.is_file()
            }
            return candidate, {
                "include": str(candidate),
                "version": match.group(1) if match else "unknown",
                "headers_sha256": _digest(headers),
            }
    raise RuntimeError(
        "KITTI devkit setup requires Boost headers. Install Boost (macOS: brew install boost; "
        "Debian/Ubuntu: apt install libboost-dev), or set BOOST_INCLUDEDIR."
    )


def _read_backend(entry: Path) -> tuple[Path, dict]:
    manifest = json.loads((entry / "manifest.json").read_text())
    identity = manifest["identity"]
    binary = entry / "evaluate-kitti-objects"
    if (
        identity["schema"] != 1
        or identity["platform"] != [platform.system(), platform.machine()]
        or identity["sources"] != _SOURCE_HASHES
        or identity["harness_sha256"] != _sha256(_HARNESS)
        or _digest(identity) != entry.name
        or not os.access(binary, os.X_OK)
        or manifest["binary_sha256"] != _sha256(binary)
    ):
        raise ValueError("Cached KITTI object evaluator is stale or corrupt")
    return binary, manifest


def resolve_kitti_object_backend() -> Path:
    """Find a verified installed evaluator; never compile or install during eval."""
    try:
        key = json.loads((_cache_root() / "current.json").read_text())["key"]
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Invalid evaluator cache pointer")
        binary, _ = _read_backend(_cache_root() / key)
        return binary
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"KITTI object evaluator is not installed or needs rebuilding. {_INSTALL_HINT}") from exc


def install_kitti_object_backend(devkit: Path) -> Path:
    """Compile our harness against the pinned official source supplied by the user."""
    from filelock import FileLock

    source = Path(devkit).expanduser().resolve()
    if (source / "cpp/evaluate_object.cpp").is_file():
        source /= "cpp"
    for name, expected in _SOURCE_HASHES.items():
        if not (source / name).is_file() or _sha256(source / name) != expected:
            raise ValueError(
                f"KITTI devkit file '{source / name}' is missing or differs from the supported official release. "
                f"Extract the official object devkit from {_DEVKIT_URL}."
            )
    compiler, compiler_version = _compiler()
    boost_include, boost_identity = _boost()
    identity = {
        "schema": 1,
        "platform": [platform.system(), platform.machine()],
        "sources": _SOURCE_HASHES,
        "source_directory": str(source),
        "harness_sha256": _sha256(_HARNESS),
        "compiler": compiler,
        "compiler_version": compiler_version,
        "boost": boost_identity,
        "flags": ["-std=c++17", "-O2", "-DNDEBUG"],
    }
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    entry = root / _digest(identity)
    with FileLock(str(root / ".install.lock")):
        try:
            binary, _ = _read_backend(entry)
        except (OSError, ValueError, KeyError, TypeError):
            with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as temporary:
                staging = Path(temporary)
                binary = staging / "evaluate-kitti-objects"
                command = [
                    *compiler,
                    *identity["flags"],
                    f"-I{source}",
                    f"-I{boost_include}",
                    str(_HARNESS),
                    "-o",
                    str(binary),
                ]
                try:
                    subprocess.run(command, capture_output=True, text=True, check=True)
                    subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True)
                except (OSError, subprocess.CalledProcessError) as exc:
                    detail = (exc.stderr or str(exc)) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
                    raise RuntimeError(f"KITTI devkit compilation failed: {detail.strip()}") from exc
                manifest = {"identity": identity, "binary_sha256": _sha256(binary), "source_url": _DEVKIT_URL}
                _write_json(staging / "manifest.json", manifest)
                if entry.exists():
                    shutil.rmtree(entry)
                staging.rename(entry)
            binary = entry / "evaluate-kitti-objects"
        _write_json(root / "current.json", {"key": entry.name})
    return binary


def _snapshot_labels(source: Path, destination: Path, *, prediction: bool) -> str:
    """Validate before the official fscanf loader and preserve the exact scored bytes."""
    try:
        raw = source.read_bytes()
        text = raw.decode("ascii")
        # Python also treats ASCII record separators as whitespace; fscanf
        # does not. Reject these before handing a token stream to the devkit.
        if any(ord(character) < 32 and character not in "\t\n\v\f\r" for character in text):
            raise ValueError(f"Invalid control character in KITTI object labels '{source}'.")
        lines = text.splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read KITTI object labels '{source}': {exc}") from exc
    expected = 16 if prediction else 15
    for index, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        valid = (
            len(fields) == expected
            and re.fullmatch(r"[A-Za-z_]{1,254}", fields[0]) is not None
            and all(_NUMBER.fullmatch(value) and math.isfinite(float(value)) for value in fields[1:])
        )
        if valid and not prediction:
            valid = re.fullmatch(r"[+-]?[0-9]+", fields[2]) is not None and -(2**31) <= int(fields[2]) < 2**31
        if not valid:
            label_type = "prediction" if prediction else "ground truth"
            raise ValueError(f"Invalid KITTI object {label_type} at {source}:{index}.")
    destination.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _validate_result(result: dict) -> None:
    """Reject truncated, nonfinite, or inconsistent native output before publishing."""
    for metric in _METRICS:
        for cls in _CLASSES:
            for difficulty in _DIFFICULTIES:
                score = result["metrics"][metric][cls][difficulty]
                curve = result["curves"][metric][cls][difficulty]
                eligible = curve["eligible_ground_truth"]
                precision = curve["precision"]
                if (
                    type(eligible) is not int
                    or eligible < 0
                    or curve["recall_samples"] != [index / 40 for index in range(41)]
                    or len(precision) != 41
                    or any(type(value) not in (float, int) or not 0 <= value <= 1 for value in precision)
                ):
                    raise ValueError("Invalid precision/recall curve")
                if eligible == 0:
                    if score is not None:
                        raise ValueError("AP must be null when no ground truth is eligible")
                elif (
                    type(score) not in (float, int)
                    or not 0 <= score <= 100
                    or not math.isclose(score, 100 * sum(precision[1:]) / 40, abs_tol=1e-9)
                ):
                    raise ValueError("Invalid AP_R40 score")


def evaluate_kitti_objects(
    ground_truth_dir: Path,
    prediction_dir: Path,
    frame_ids: Sequence[str],
    output: Path,
) -> dict:
    """Return percent AP_R40 and persist curves/provenance for the selected frames.

    No eligible ground truth gives ``None``; eligible GT without detections gives
    zero. Difficulty rules and 2D/3D matching are the official devkit's unchanged
    implementation. These are local split results, not benchmark submissions.
    """
    binary = resolve_kitti_object_backend()
    frames = tuple(frame_ids)
    if (
        not frames
        or any(not isinstance(frame, str) or re.fullmatch(r"[0-9]+", frame) is None for frame in frames)
        or len(set(frames)) != len(frames)
    ):
        raise ValueError("KITTI object evaluation requires distinct nonempty decimal frame IDs.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    with tempfile.TemporaryDirectory(prefix=".kitti-object-", dir=output) as temporary:
        staging = Path(temporary)
        gt = staging / "ground_truth"
        predictions = staging / "predictions"
        gt.mkdir()
        predictions.mkdir()
        for frame in frames:
            name = f"{frame}.txt"
            sources.append(
                {
                    "frame_id": frame,
                    "ground_truth_sha256": _snapshot_labels(Path(ground_truth_dir) / name, gt / name, prediction=False),
                    "prediction_sha256": _snapshot_labels(
                        Path(prediction_dir) / name, predictions / name, prediction=True
                    ),
                }
            )
        selected = staging / "frames.txt"
        selected.write_text("\n".join(frames) + "\n")
        native_output = staging / "result.json"
        try:
            subprocess.run(
                [str(binary), str(gt), str(predictions), str(selected), str(native_output)],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = (
                (exc.stderr or exc.stdout or str(exc)) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            )
            raise RuntimeError(f"KITTI object evaluation failed: {detail.strip()}") from exc
        try:
            result = json.loads(native_output.read_text())
            _validate_result(result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"KITTI object evaluator returned invalid output: {exc}") from exc
    result["provenance"] = {
        "protocol": "KITTI object AP_R40",
        "ap_units": "percent",
        "scope": "selected dataset frames; not official benchmark split parity",
        "backend": json.loads((binary.parent / "manifest.json").read_text()),
        "ground_truth_directory": str(Path(ground_truth_dir).resolve()),
        "prediction_directory": str(Path(prediction_dir).resolve()),
        "frames": sources,
    }
    _write_json(output / "object_evaluation.json", result)
    return result["metrics"]
