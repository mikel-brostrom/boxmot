from __future__ import annotations

import contextlib
import json
import os
import threading

import pytest

from boxmot.native import _common


def _make_native_sources(tmp_path):
    native_root = tmp_path / "native"
    tracker_root = native_root / "cpp" / "trackers" / "demo"
    base_root = native_root / "cpp" / "trackers" / "base"
    cmake_root = native_root / "cpp" / "cmake"
    for root in (tracker_root, base_root, cmake_root):
        root.mkdir(parents=True)

    tracker_source = tracker_root / "src" / "tracker.cpp"
    tracker_source.parent.mkdir()
    tracker_source.write_text("int tracker_version = 1;\n", encoding="utf-8")
    (tracker_root / "CMakeLists.txt").write_text("project(demo)\n", encoding="utf-8")
    (base_root / "base.cpp").write_text("int base_version = 1;\n", encoding="utf-8")
    cmake_helper = cmake_root / "BoxMOTNative.cmake"
    cmake_helper.write_text("set(BOXMOT_NATIVE_VERSION 1)\n", encoding="utf-8")
    return native_root, tracker_root, tracker_source, cmake_helper


def test_build_native_target_rebuilds_when_sources_or_cmake_change(monkeypatch, tmp_path):
    native_root, _, tracker_source, cmake_helper = _make_native_sources(tmp_path)
    build_dir = tmp_path / "build" / "demo"
    build_dir.mkdir(parents=True)
    artifact = build_dir / "demo_replay"
    artifact.write_text("existing binary\n", encoding="utf-8")

    monkeypatch.setattr(_common, "package_native_root", lambda: native_root)
    monkeypatch.setattr(_common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(_common, "_is_native_source_checkout", lambda: True)
    monkeypatch.setattr(_common, "_cross_process_build_lock", lambda _build_dir: contextlib.nullcontext())
    build_calls = []

    def fake_build_step(**kwargs):
        build_calls.append(kwargs["cmd"])
        if "--build" in kwargs["cmd"]:
            artifact.write_text(f"rebuilt artifact {len(build_calls)}\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(_common, "run_build_step", fake_build_step)

    kwargs = {
        "tracker_name": "demo",
        "display_name": "Demo",
        "target": "demo_replay",
        "candidates": [artifact],
        "force_rebuild": False,
        "not_found_message": "missing demo artifact",
        "build_lock": threading.Lock(),
    }

    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 2
    stamp_path = _common._native_build_stamp_path(build_dir, "demo_replay")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    assert stamp["artifact_sha256"] == _common._sha256_file(artifact)
    assert stamp["configuration_sha256"]

    # An unchanged source tree reuses the artifact without another CMake call.
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 2

    artifact.write_text("externally replaced binary\n", encoding="utf-8")
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 4

    tracker_source.write_text("int tracker_version = 2;\n", encoding="utf-8")
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 6

    cmake_helper.write_text("set(BOXMOT_NATIVE_VERSION 2)\n", encoding="utf-8")
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 8


def test_build_native_target_hashes_artifact_contents(monkeypatch, tmp_path):
    native_root, _, _, _ = _make_native_sources(tmp_path)
    build_dir = tmp_path / "build" / "demo"
    build_dir.mkdir(parents=True)
    artifact = build_dir / "demo_replay"
    artifact.write_bytes(b"unbuilt-native-artifact")
    build_calls = []

    def fake_build_step(**kwargs):
        build_calls.append(kwargs["cmd"])
        if "--build" in kwargs["cmd"]:
            artifact.write_bytes(b"original-native-artifact")
        return 0

    monkeypatch.setattr(_common, "package_native_root", lambda: native_root)
    monkeypatch.setattr(_common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(_common, "_is_native_source_checkout", lambda: True)
    monkeypatch.setattr(_common, "_cross_process_build_lock", lambda _build_dir: contextlib.nullcontext())
    monkeypatch.setattr(_common, "run_build_step", fake_build_step)

    kwargs = {
        "tracker_name": "demo",
        "display_name": "Demo",
        "target": "demo_replay",
        "candidates": [artifact],
        "force_rebuild": False,
        "not_found_message": "missing demo artifact",
        "build_lock": threading.Lock(),
    }
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 2

    original_stat = artifact.stat()
    artifact.write_bytes(b"tampered-native-artifact")
    assert artifact.stat().st_size == original_stat.st_size
    os.utime(artifact, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 4
    assert artifact.read_bytes() == b"original-native-artifact"


def test_build_native_target_relinks_requested_artifact_without_cleaning_siblings(monkeypatch, tmp_path):
    native_root, _, _, _ = _make_native_sources(tmp_path)
    build_dir = tmp_path / "build" / "demo"
    build_dir.mkdir(parents=True)
    artifact = build_dir / "demo_replay"
    artifact.write_text("stale artifact\n", encoding="utf-8")
    sibling = build_dir / "demo_capi.so"
    sibling.write_text("sibling artifact\n", encoding="utf-8")
    build_calls = []

    monkeypatch.setattr(_common, "package_native_root", lambda: native_root)
    monkeypatch.setattr(_common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(_common, "_is_native_source_checkout", lambda: True)
    monkeypatch.setattr(_common, "_cross_process_build_lock", lambda _build_dir: contextlib.nullcontext())

    def fake_build_step(**kwargs):
        build_calls.append(kwargs["cmd"])
        assert sibling.read_text(encoding="utf-8") == "sibling artifact\n"
        if "--build" in kwargs["cmd"]:
            assert not artifact.exists()
            artifact.write_text("rebuilt artifact\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(_common, "run_build_step", fake_build_step)

    result = _common.build_native_target(
        tracker_name="demo",
        display_name="Demo",
        target="demo_replay",
        candidates=[artifact],
        force_rebuild=True,
        not_found_message="missing demo artifact",
        build_lock=threading.Lock(),
    )

    assert result == artifact
    assert len(build_calls) == 2
    assert "--clean-first" not in build_calls[1]
    assert artifact.read_text(encoding="utf-8") == "rebuilt artifact\n"
    assert sibling.read_text(encoding="utf-8") == "sibling artifact\n"
    stamp = json.loads(_common._native_build_stamp_path(build_dir, "demo_replay").read_text(encoding="utf-8"))
    assert stamp["artifact_sha256"] == _common._sha256_file(artifact)


def test_build_native_target_rejects_success_without_recreated_artifact(monkeypatch, tmp_path):
    native_root, _, _, _ = _make_native_sources(tmp_path)
    build_dir = tmp_path / "build" / "demo"
    build_dir.mkdir(parents=True)
    artifact = build_dir / "demo_replay"
    artifact.write_text("tampered artifact\n", encoding="utf-8")

    monkeypatch.setattr(_common, "package_native_root", lambda: native_root)
    monkeypatch.setattr(_common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(_common, "_is_native_source_checkout", lambda: True)
    monkeypatch.setattr(_common, "_cross_process_build_lock", lambda _build_dir: contextlib.nullcontext())
    monkeypatch.setattr(_common, "run_build_step", lambda **_kwargs: 0)

    with pytest.raises(RuntimeError, match="missing demo artifact"):
        _common.build_native_target(
            tracker_name="demo",
            display_name="Demo",
            target="demo_replay",
            candidates=[artifact],
            force_rebuild=True,
            not_found_message="missing demo artifact",
            build_lock=threading.Lock(),
        )

    assert not artifact.exists()


def test_build_native_target_rebuilds_for_cmake_and_dependency_configuration(monkeypatch, tmp_path):
    native_root, _, _, _ = _make_native_sources(tmp_path)
    build_dir = tmp_path / "build" / "demo"
    build_dir.mkdir(parents=True)
    artifact = build_dir / "demo_replay"
    dependency_dir = tmp_path / "onnxruntime" / "lib" / "cmake" / "onnxruntime"
    dependency_dir.mkdir(parents=True)
    dependency_config = dependency_dir / "onnxruntimeConfigVersion.cmake"
    dependency_config.write_text('set(PACKAGE_VERSION "1.0")\n', encoding="utf-8")
    cache = build_dir / "CMakeCache.txt"
    cache.write_text(
        f"BOXMOT_REID_ONNXRUNTIME:BOOL=ON\nCMAKE_GENERATOR:INTERNAL=Ninja\nonnxruntime_DIR:PATH={dependency_dir}\n",
        encoding="utf-8",
    )
    build_calls = []

    def fake_build_step(**kwargs):
        build_calls.append(kwargs["cmd"])
        if "--build" in kwargs["cmd"]:
            artifact.write_text(f"build {len(build_calls)}\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(_common, "package_native_root", lambda: native_root)
    monkeypatch.setattr(_common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(_common, "_is_native_source_checkout", lambda: True)
    monkeypatch.setattr(_common, "_cross_process_build_lock", lambda _build_dir: contextlib.nullcontext())
    monkeypatch.setattr(_common, "run_build_step", fake_build_step)

    kwargs = {
        "tracker_name": "demo",
        "display_name": "Demo",
        "target": "demo_replay",
        "candidates": [artifact],
        "force_rebuild": False,
        "not_found_message": "missing demo artifact",
        "build_lock": threading.Lock(),
    }
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 2
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 2

    cache.write_text(cache.read_text(encoding="utf-8").replace("Ninja", "Xcode"), encoding="utf-8")
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 4

    dependency_config.write_text('set(PACKAGE_VERSION "2.0")\n', encoding="utf-8")
    assert _common.build_native_target(**kwargs) == artifact
    assert len(build_calls) == 6


def test_build_configuration_fingerprint_tracks_effective_toolchain_and_dependency_versions(monkeypatch, tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    compiler = tmp_path / "c++"
    compiler.write_text("compiler version 1\n", encoding="utf-8")

    dependency_dirs = {}
    for name in ("eigen", "opencv", "onnxruntime"):
        dependency_dir = tmp_path / name
        dependency_dir.mkdir()
        (dependency_dir / f"{name}ConfigVersion.cmake").write_text(
            'set(PACKAGE_VERSION "1.0")\n',
            encoding="utf-8",
        )
        dependency_dirs[name] = dependency_dir

    cache = build_dir / "CMakeCache.txt"
    cache.write_text(
        "BOXMOT_REID_ONNXRUNTIME:BOOL=ON\n"
        f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n"
        "CMAKE_GENERATOR:INTERNAL=Ninja\n"
        f"Eigen3_DIR:PATH={dependency_dirs['eigen']}\n"
        f"OpenCV_DIR:PATH={dependency_dirs['opencv']}\n"
        f"onnxruntime_DIR:PATH={dependency_dirs['onnxruntime']}\n",
        encoding="utf-8",
    )

    fingerprint = _common._native_build_configuration_fingerprint(build_dir)

    # CMake ignores initial-only compiler/generator environment variables once
    # this cache exists, so they must not bless an old-config artifact as if
    # those inputs had been applied.
    monkeypatch.setenv("CXX", str(tmp_path / "ignored-c++"))
    monkeypatch.setenv("CMAKE_GENERATOR", "Ignored Generator")
    assert _common._native_build_configuration_fingerprint(build_dir) == fingerprint

    compiler.write_text("compiler version 2\n", encoding="utf-8")
    compiler_fingerprint = _common._native_build_configuration_fingerprint(build_dir)
    assert compiler_fingerprint != fingerprint

    cache.write_text(cache.read_text(encoding="utf-8").replace("Ninja", "Xcode"), encoding="utf-8")
    generator_fingerprint = _common._native_build_configuration_fingerprint(build_dir)
    assert generator_fingerprint != compiler_fingerprint

    for name in ("eigen", "opencv", "onnxruntime"):
        version_file = dependency_dirs[name] / f"{name}ConfigVersion.cmake"
        version_file.write_text('set(PACKAGE_VERSION "2.0")\n', encoding="utf-8")
        dependency_fingerprint = _common._native_build_configuration_fingerprint(build_dir)
        assert dependency_fingerprint != generator_fingerprint
        generator_fingerprint = dependency_fingerprint

    cache.write_text(cache.read_text(encoding="utf-8").replace("BOOL=ON", "BOOL=OFF"), encoding="utf-8")
    assert _common._native_build_configuration_fingerprint(build_dir) != generator_fingerprint


def test_build_native_target_selects_produced_multi_config_artifact(monkeypatch, tmp_path):
    native_root, _, _, _ = _make_native_sources(tmp_path)
    build_dir = tmp_path / "build" / "demo"
    release_dir = build_dir / "Release"
    release_dir.mkdir(parents=True)
    stale_root_artifact = build_dir / "demo_replay"
    release_artifact = release_dir / "demo_replay"
    stale_root_artifact.write_text("stale single-config artifact\n", encoding="utf-8")
    release_artifact.write_text("old release artifact\n", encoding="utf-8")
    (build_dir / "CMakeCache.txt").write_text(
        "CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release\nCMAKE_GENERATOR:INTERNAL=Xcode\n",
        encoding="utf-8",
    )

    def fake_build_step(**kwargs):
        if "--build" in kwargs["cmd"]:
            release_artifact.write_text("new release artifact\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(_common, "package_native_root", lambda: native_root)
    monkeypatch.setattr(_common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(_common, "_is_native_source_checkout", lambda: True)
    monkeypatch.setattr(_common, "_cross_process_build_lock", lambda _build_dir: contextlib.nullcontext())
    monkeypatch.setattr(_common, "run_build_step", fake_build_step)

    result = _common.build_native_target(
        tracker_name="demo",
        display_name="Demo",
        target="demo_replay",
        candidates=[stale_root_artifact, release_artifact],
        force_rebuild=False,
        not_found_message="missing demo artifact",
        build_lock=threading.Lock(),
    )

    assert result == release_artifact
    stamp = json.loads(_common._native_build_stamp_path(build_dir, "demo_replay").read_text(encoding="utf-8"))
    assert stamp["artifact"] == str(release_artifact.resolve())
    assert stamp["artifact_sha256"] == _common._sha256_file(release_artifact)


def test_build_native_target_trusts_packaged_artifact(monkeypatch, tmp_path):
    native_root, tracker_root, _, _ = _make_native_sources(tmp_path)
    build_dir = tmp_path / "build" / "demo"
    artifact = tracker_root / "demo_replay"
    artifact.write_text("packaged binary\n", encoding="utf-8")

    monkeypatch.setattr(_common, "package_native_root", lambda: native_root)
    monkeypatch.setattr(_common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(_common, "_is_native_source_checkout", lambda: False)
    monkeypatch.setattr(
        _common,
        "run_build_step",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("packaged artifacts must not rebuild")),
    )

    result = _common.build_native_target(
        tracker_name="demo",
        display_name="Demo",
        target="demo_replay",
        candidates=[artifact],
        force_rebuild=False,
        not_found_message="missing demo artifact",
        build_lock=threading.Lock(),
    )

    assert result == artifact
    assert not build_dir.exists()


def test_build_native_target_ignores_packaged_artifact_in_source_checkout(monkeypatch, tmp_path):
    native_root, tracker_root, _, _ = _make_native_sources(tmp_path)
    build_dir = tmp_path / "build" / "demo"
    packaged_artifact = tracker_root / "demo_replay"
    packaged_artifact.write_text("stale packaged binary\n", encoding="utf-8")
    built_artifact = build_dir / "demo_replay"
    build_calls = []

    def fake_build_step(**kwargs):
        build_calls.append(kwargs["cmd"])
        if "--build" in kwargs["cmd"]:
            built_artifact.write_text("fresh editable binary\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(_common, "package_native_root", lambda: native_root)
    monkeypatch.setattr(_common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(_common, "_is_native_source_checkout", lambda: True)
    monkeypatch.setattr(_common, "_cross_process_build_lock", lambda _build_dir: contextlib.nullcontext())
    monkeypatch.setattr(_common, "run_build_step", fake_build_step)

    result = _common.build_native_target(
        tracker_name="demo",
        display_name="Demo",
        target="demo_replay",
        candidates=[packaged_artifact, built_artifact],
        force_rebuild=False,
        not_found_message="missing demo artifact",
        build_lock=threading.Lock(),
    )

    assert result == built_artifact
    assert len(build_calls) == 2
