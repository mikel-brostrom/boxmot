"""Offline installation, native devkit equivalence, and result integrity."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from boxmot.engine.eval import kitti_object_backend as backend

_GT = "Car 0 0 0 0 0 100 100 2 2 4 0 2 20 0"


@pytest.fixture
def fake_install(tmp_path, monkeypatch):
    """Exercise installation publication without requiring third-party source in CI."""
    source = tmp_path / "devkit" / "cpp"
    source.mkdir(parents=True)
    hashes = {}
    for name in ("evaluate_object.cpp", "mail.h"):
        (source / name).write_text(name)
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    monkeypatch.setattr(backend, "_SOURCE_HASHES", hashes)
    monkeypatch.setattr(backend, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(backend, "_compiler", lambda: (["c++"], "test compiler"))
    monkeypatch.setattr(backend, "_boost", lambda: (tmp_path, {"version": "test", "headers_sha256": "headers"}))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "-o" in command:
            binary = Path(command[command.index("-o") + 1])
            binary.write_bytes(b"test binary")
            binary.chmod(0o755)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(backend.subprocess, "run", run)
    return source, commands


def test_missing_backend_does_not_install_or_compile(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(backend.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected subprocess"))
    with pytest.raises(RuntimeError, match="boxmot install --kitti-devkit"):
        backend.resolve_kitti_object_backend()
    assert list(tmp_path.iterdir()) == []


def test_install_reuses_verified_binary_and_rejects_corruption(fake_install):
    source, commands = fake_install
    binary = backend.install_kitti_object_backend(source.parent)
    assert backend.resolve_kitti_object_backend() == binary
    assert backend.install_kitti_object_backend(source) == binary
    assert sum("-o" in command for command in commands) == 1
    manifest = json.loads((binary.parent / "manifest.json").read_text())
    assert manifest["identity"]["compiler_version"] == "test compiler"
    assert manifest["identity"]["boost"]["headers_sha256"] == "headers"
    assert not (binary.parent / "evaluate_object.cpp").exists()

    binary.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="needs rebuilding"):
        backend.resolve_kitti_object_backend()
    assert backend.install_kitti_object_backend(source) == binary
    assert sum("-o" in command for command in commands) == 2


def test_install_requires_exact_official_sources_before_compiling(fake_install):
    source, commands = fake_install
    (source / "evaluate_object.cpp").write_text("changed")
    with pytest.raises(ValueError, match="differs from the supported official release"):
        backend.install_kitti_object_backend(source)
    assert commands == []


def test_compilation_failure_does_not_publish_an_install(fake_install, monkeypatch):
    source, _ = fake_install

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="missing Boost header")

    monkeypatch.setattr(backend.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="compilation failed: missing Boost header"):
        backend.install_kitti_object_backend(source)
    assert not (backend._cache_root() / "current.json").exists()
    assert list(backend._cache_root().glob(".build-*")) == []


def test_changed_compiler_boost_or_harness_creates_new_install(fake_install, monkeypatch, tmp_path):
    source, _ = fake_install
    first = backend.install_kitti_object_backend(source)
    monkeypatch.setattr(backend, "_compiler", lambda: (["c++"], "different compiler"))
    second = backend.install_kitti_object_backend(source)
    assert second != first
    monkeypatch.setattr(backend, "_boost", lambda: (tmp_path, {"version": "test", "headers_sha256": "different"}))
    third = backend.install_kitti_object_backend(source)
    assert third != second
    harness = tmp_path / "harness.cpp"
    harness.write_text("changed harness")
    monkeypatch.setattr(backend, "_HARNESS", harness)
    with pytest.raises(RuntimeError, match="needs rebuilding"):
        backend.resolve_kitti_object_backend()
    assert backend.install_kitti_object_backend(source) != third


@pytest.mark.parametrize("frames", [[], ["0", "0"], ["../0"], ["０"], [0]])
def test_frame_ids_are_checked_before_native_execution(fake_install, tmp_path, frames):
    source, commands = fake_install
    backend.install_kitti_object_backend(source)
    count = len(commands)
    with pytest.raises(ValueError, match="decimal frame IDs"):
        backend.evaluate_kitti_objects(tmp_path, tmp_path, frames, tmp_path / "output")
    assert len(commands) == count


@pytest.mark.parametrize("replacement", ["nan", "1_0", "1e999", "oops"])
def test_malformed_numbers_never_reach_official_fscanf(fake_install, tmp_path, replacement):
    source, commands = fake_install
    backend.install_kitti_object_backend(source)
    fields = _GT.split()
    fields[4] = replacement
    (tmp_path / "0.txt").write_text(" ".join(fields))
    count = len(commands)
    with pytest.raises(ValueError, match="Invalid KITTI object ground truth"):
        backend.evaluate_kitti_objects(tmp_path, tmp_path, ["0"], tmp_path / "output")
    assert len(commands) == count
    assert list((tmp_path / "output").iterdir()) == []


def test_native_failure_is_actionable_and_cleans_staging(fake_install, monkeypatch, tmp_path):
    source, _ = fake_install
    backend.install_kitti_object_backend(source)
    gt, predictions = tmp_path / "gt", tmp_path / "predictions"
    gt.mkdir()
    predictions.mkdir()
    (gt / "0.txt").write_text(_GT)
    (predictions / "0.txt").write_text(_GT + " 0.9")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(2, command, stderr="native scorer failed")

    monkeypatch.setattr(backend.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="evaluation failed: native scorer failed"):
        backend.evaluate_kitti_objects(gt, predictions, ["0"], tmp_path / "output")
    assert list((tmp_path / "output").iterdir()) == []


@pytest.mark.parametrize("native_output", [None, "broken JSON", '{"metrics": {}}', "null"])
def test_invalid_native_output_is_not_published(fake_install, monkeypatch, tmp_path, native_output):
    source, _ = fake_install
    backend.install_kitti_object_backend(source)
    gt, predictions = tmp_path / "gt", tmp_path / "predictions"
    gt.mkdir()
    predictions.mkdir()
    (gt / "0.txt").write_text(_GT)
    (predictions / "0.txt").write_text(_GT + " 0.9")

    def run(command, **kwargs):
        if native_output is not None:
            Path(command[-1]).write_text(native_output)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(backend.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="returned invalid output"):
        backend.evaluate_kitti_objects(gt, predictions, ["0"], tmp_path / "output")
    assert list((tmp_path / "output").iterdir()) == []


@pytest.fixture(scope="module")
def native_cache(tmp_path_factory):
    """Opt-in real compilation: official source cannot be redistributed in the tests."""
    devkit = os.environ.get("BOXMOT_TEST_KITTI_DEVKIT")
    if devkit is None:
        pytest.skip("Set BOXMOT_TEST_KITTI_DEVKIT to test the external official C++ devkit")
    cache = tmp_path_factory.mktemp("kitti-devkit-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend, "_cache_root", lambda: cache)
        backend.install_kitti_object_backend(Path(devkit))
    return cache


def _native_evaluate(tmp_path, monkeypatch, native_cache, gt_text, prediction_text):
    """Use 41 eligible objects so all official recall samples are populated."""
    monkeypatch.setattr(backend, "_cache_root", lambda: native_cache)
    gt, predictions = tmp_path / "gt", tmp_path / "predictions"
    gt.mkdir()
    predictions.mkdir()
    frames = [f"{index:06d}" for index in range(41)]
    for frame in frames:
        (gt / f"{frame}.txt").write_text(gt_text + "\n")
        (predictions / f"{frame}.txt").write_text(prediction_text + "\n")
    scores = backend.evaluate_kitti_objects(gt, predictions, frames, tmp_path / "output")
    report = json.loads((tmp_path / "output/object_evaluation.json").read_text())
    assert report["provenance"]["ap_units"] == "percent"
    assert len(report["provenance"]["frames"]) == 41
    assert report["metrics"] == scores
    assert list((tmp_path / "output").iterdir()) == [tmp_path / "output/object_evaluation.json"]
    return scores, report


@pytest.mark.parametrize("prediction,expected_2d,expected_3d", [(_GT + " 0.9", 100, 100), ("", 0, 0)])
def test_native_perfect_and_empty_predictions(
    tmp_path, monkeypatch, native_cache, prediction, expected_2d, expected_3d
):
    scores, report = _native_evaluate(tmp_path, monkeypatch, native_cache, _GT, prediction)
    for metric, expected in (("2d", expected_2d), ("3d", expected_3d)):
        assert scores[metric]["car"] == dict.fromkeys(("easy", "moderate", "hard"), expected)
        assert scores[metric]["pedestrian"] == dict.fromkeys(("easy", "moderate", "hard"), None)
        assert report["curves"][metric]["car"]["easy"]["eligible_ground_truth"] == 41


def test_native_3d_overlap_is_independent_of_image_overlap(tmp_path, monkeypatch, native_cache):
    prediction = _GT.replace("0 2 20 0", "0 2 200 0") + " 0.9"
    scores, _ = _native_evaluate(tmp_path, monkeypatch, native_cache, _GT, prediction)
    assert scores["2d"]["car"]["moderate"] == 100
    assert scores["3d"]["car"]["moderate"] == 0


@pytest.mark.parametrize(
    "ground_truth,expected",
    [(_GT.replace("Car 0 0", "Car 0.2 0"), [None, 100, 100]),
     (_GT.replace("Car 0 0", "Car 0 2"), [None, None, 100]),
     (_GT.replace("0 0 100 100", "0 0 100 40"), [None, 100, 100])],
)
def test_native_difficulty_rules_include_fractional_truncation_and_height_boundary(
    tmp_path, monkeypatch, native_cache, ground_truth, expected
):
    scores, _ = _native_evaluate(tmp_path, monkeypatch, native_cache, ground_truth, ground_truth + " 0.9")
    for metric in ("2d", "3d"):
        assert list(scores[metric]["car"].values()) == expected


def test_native_dontcare_uses_the_official_metric_specific_overlap(tmp_path, monkeypatch, native_cache):
    dontcare = "DontCare -1 -1 -10 200 0 300 100 -1 -1 -1 -1000 -1000 -1000 -10"
    false_positive = "Car -1 -1 0 200 0 300 100 2 2 4 20 2 20 0 0.95"
    scores, _ = _native_evaluate(
        tmp_path, monkeypatch, native_cache, _GT + "\n" + dontcare, _GT + " 0.9\n" + false_positive
    )
    assert scores["2d"]["car"]["moderate"] == 100
    assert scores["3d"]["car"]["moderate"] == 50
