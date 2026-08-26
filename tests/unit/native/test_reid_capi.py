"""Unit tests for the native C++ ReID ABI wrapper.

Skipped automatically when the optional ``reid_capi`` shared library or its
ONNX/OpenCV runtime dependencies are unavailable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests._paths import REPO_ROOT as ROOT


def _write_tiny_reid_onnx(
    path: Path,
    *,
    batch: int | str = "batch",
    height: int = 20,
    width: int = 10,
) -> Path:
    """Create a pixel-sensitive ONNX descriptor with a controlled NCHW input."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ONNX test dependency unavailable: {exc}")

    input_shape = [batch, 3, height, width]
    output_shape = [batch, 3 * height * width]
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["images"], ["features"], axis=1)],
        "tiny_reid",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
        producer_name="boxmot-test",
    )
    # Accepted by both current ONNX Runtime and the older ONNX reader bundled
    # with some OpenCV 4 builds used by the native fallback.
    model.ir_version = 8
    onnx.save(model, path)
    return path


def _write_batch_centered_reid_onnx(path: Path, *, height: int = 20, width: int = 10) -> Path:
    """Create a descriptor whose output proves whether dynamic N ran together."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ONNX test dependency unavailable: {exc}")

    feature_dim = 3 * height * width
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["images"], ["flat"], axis=1),
            helper.make_node("ReduceMean", ["flat"], ["batch_mean"], axes=[0], keepdims=1),
            helper.make_node("Sub", ["flat", "batch_mean"], ["features"]),
        ],
        "batch_centered_reid",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, height, width])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", feature_dim])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
        producer_name="boxmot-test",
    )
    model.ir_version = 8
    onnx.save(model, path)
    return path


def _model_or_skip() -> Path:
    candidate = ROOT / "models" / "lmbn_n_duke.onnx"
    if not candidate.exists():
        pytest.skip(f"Required ONNX ReID model not found: {candidate}")
    return candidate


def _image_or_skip() -> np.ndarray:
    import cv2

    img_path = ROOT / "assets" / "MOT17-mini" / "train" / "MOT17-02-FRCNN" / "img1" / "000001.jpg"
    if not img_path.exists():
        pytest.skip(f"Required test image not found: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        pytest.skip(f"Failed to read test image: {img_path}")
    return image


def _load_adapter():
    try:
        from boxmot.native.reid import CppOnnxReID
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Native ReID C ABI unavailable: {exc}")
    return CppOnnxReID


@pytest.mark.parametrize("force_rebuild", [False, True])
def test_reid_capi_ensure_uses_shared_freshness_builder(monkeypatch, tmp_path, force_rebuild):
    from boxmot.native.reid import capi

    stale_library = tmp_path / capi._library_name()
    stale_library.write_bytes(b"stale native library")
    fresh_library = tmp_path / f"fresh-{capi._library_name()}"
    captured = {}

    def fake_build_native_target(**kwargs):
        captured.update(kwargs)
        return fresh_library

    monkeypatch.setattr(capi, "_candidate_libraries", lambda: [stale_library])
    monkeypatch.setattr(capi._common, "build_native_target", fake_build_native_target)

    assert capi.ensure_reid_capi_library(force_rebuild=force_rebuild) == fresh_library
    assert captured == {
        "tracker_name": "base",
        "display_name": "ReID C ABI",
        "target": "reid_capi",
        "candidates": [stale_library],
        "force_rebuild": force_rebuild,
        "not_found_message": "Native ReID C ABI build succeeded but the shared library was not found.",
        "build_lock": capi._BUILD_LOCK,
    }


def test_reid_capi_ensure_trusts_packaged_wheel_library(monkeypatch, tmp_path):
    from boxmot.native.reid import capi

    source_dir = tmp_path / "site-packages" / "boxmot" / "native" / "cpp" / "trackers" / "base"
    source_dir.mkdir(parents=True)
    packaged_library = source_dir / capi._library_name()
    packaged_library.write_bytes(b"packaged native library")
    build_dir = tmp_path / "build" / "native" / "base"

    monkeypatch.setattr(capi, "_candidate_libraries", lambda: [packaged_library])
    monkeypatch.setattr(capi._common, "tracker_source_dir", lambda _name: source_dir)
    monkeypatch.setattr(capi._common, "tracker_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(capi._common, "_native_build_fingerprint", lambda _name: "source-hash")
    monkeypatch.setattr(capi._common, "_is_native_source_checkout", lambda: False)
    monkeypatch.setattr(
        capi._common,
        "run_build_step",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("packaged library must not rebuild")),
    )

    assert capi.ensure_reid_capi_library() == packaged_library
    assert not build_dir.exists()


def test_cpp_reid_smoke_aabb():
    image = _image_or_skip()
    weights = _model_or_skip()
    CppOnnxReID = _load_adapter()

    reid = CppOnnxReID(weights=weights)
    try:
        boxes = np.array(
            [
                [100, 100, 200, 300],
                [300, 150, 400, 400],
                [50, 80, 150, 250],
            ],
            dtype=np.float32,
        )
        feats = reid.get_features(boxes, image)
        individual = np.concatenate([reid.get_features(box[None, :], image) for box in boxes], axis=0)

        assert feats.dtype == np.float32
        assert feats.ndim == 2
        assert feats.shape[0] == boxes.shape[0]
        assert feats.shape[1] == reid.feature_dim > 0

        # L2 normalised rows
        norms = np.linalg.norm(feats, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)
        np.testing.assert_allclose(feats, individual, atol=5e-5, rtol=5e-5)
    finally:
        reid.close()


def test_cpp_reid_handles_empty_boxes():
    image = _image_or_skip()
    weights = _model_or_skip()
    CppOnnxReID = _load_adapter()

    reid = CppOnnxReID(weights=weights)
    try:
        out = reid.get_features(np.empty((0, 4), dtype=np.float32), image)
        assert isinstance(out, np.ndarray)
        assert out.size == 0
    finally:
        reid.close()


def test_cpp_reid_axis_aligned_obb_matches_aabb_crop():
    image = _image_or_skip()
    weights = _model_or_skip()
    CppOnnxReID = _load_adapter()

    reid = CppOnnxReID(weights=weights)
    try:
        # OBB rows: (cx, cy, w, h, theta). theta=0 ⇒ AABB equivalent (cx-w/2, ...).
        obb = np.array([[200.0, 250.0, 80.0, 200.0, 0.0]], dtype=np.float32)
        aabb = np.array([[160.0, 150.0, 240.0, 350.0]], dtype=np.float32)

        feats_obb = reid.get_features(obb, image)
        feats_aabb = reid.get_features(aabb, image)

        # With theta=0, native OBB rectification must match an explicit AABB crop.
        assert feats_obb.shape == feats_aabb.shape
        np.testing.assert_allclose(feats_obb, feats_aabb, atol=1e-5)
    finally:
        reid.close()


def test_cpp_reid_preserves_obb_rows_for_native_rectification():
    CppOnnxReID = _load_adapter()
    boxes = np.array([[50.0, 60.0, 40.0, 12.0, 0.7]], dtype=np.float32)

    normalized = CppOnnxReID._normalise_boxes(boxes)

    np.testing.assert_array_equal(normalized, boxes)

    detections = np.column_stack([boxes, np.array([[0.9, 2.0]], dtype=np.float32)])
    np.testing.assert_array_equal(CppOnnxReID._normalise_boxes(detections), boxes)


@pytest.mark.parametrize("batch", ["batch", 4])
def test_cpp_reid_reads_onnx_input_spec_and_honours_batch(
    tmp_path: Path,
    batch,
    monkeypatch,
):
    # CoreML compilation of temporary ONNX files depends on host temp-folder
    # permissions. The contract under test is backend-independent graph shape
    # handling, so keep it deterministic on the CPU provider.
    monkeypatch.setenv("BOXMOT_REID_DEVICE", "cpu")
    CppOnnxReID = _load_adapter()
    weights = _write_tiny_reid_onnx(
        tmp_path / "arbitrary_model_name.onnx",
        batch=batch,
        height=20,
        width=10,
    )
    image = np.arange(40 * 30 * 3, dtype=np.uint8).reshape(40, 30, 3)

    reid = CppOnnxReID(weights=weights, preprocess_name="resize")
    try:
        # These values come from the graph, not a model-family filename.
        assert reid.input_shape == (20, 10)
        assert reid.input_batch_size == (None if batch == "batch" else batch)

        boxes = np.array([[2, 3, 18, 32], [8, 5, 27, 38]], dtype=np.float32)
        features = reid.get_features(boxes, image)
        assert features.shape == (2, 600)
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-6)

        individual = np.concatenate([reid.get_features(box[None, :], image) for box in boxes], axis=0)
        np.testing.assert_allclose(features, individual, atol=1e-6, rtol=1e-6)
    finally:
        reid.close()


@pytest.mark.parametrize("requested_backend", ["onnxruntime", "opencv"])
def test_cpp_reid_dynamic_batch_respects_backend_capability(
    tmp_path: Path,
    monkeypatch,
    capfd,
    requested_backend,
):
    monkeypatch.setenv("BOXMOT_REID_BACKEND", requested_backend)
    monkeypatch.setenv("BOXMOT_REID_DEVICE", "cpu")
    CppOnnxReID = _load_adapter()
    weights = _write_batch_centered_reid_onnx(tmp_path / "batch_centered.onnx")
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :20] = np.array([20, 60, 100], dtype=np.uint8)
    image[:, 20:] = np.array([200, 80, 40], dtype=np.uint8)
    boxes = np.array([[0, 0, 20, 40], [20, 0, 40, 40]], dtype=np.float32)

    reid = CppOnnxReID(weights=weights, preprocess_name="resize")
    backend_log = capfd.readouterr().err
    try:
        batched = reid.get_features(boxes, image)
        individual = np.concatenate([reid.get_features(box[None, :], image) for box in boxes], axis=0)
    finally:
        reid.close()

    # A one-row invocation subtracts that row's own batch mean and is exactly
    # zero. With ORT available, the two staged crops must share one Run call,
    # producing opposite, L2-normalized descriptors. Builds without ORT fall
    # back to OpenCV DNN, whose deliberate dynamic-N behavior stays per-crop.
    np.testing.assert_allclose(individual, 0.0, atol=1e-7)
    if requested_backend == "onnxruntime" and "inference backend=onnxruntime" in backend_log:
        np.testing.assert_allclose(np.linalg.norm(batched, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(batched[0], -batched[1], atol=1e-6, rtol=1e-6)
    else:
        assert "inference backend=opencv_dnn" in backend_log
        np.testing.assert_allclose(batched, 0.0, atol=1e-7)


def test_cpp_reid_equivalent_obb_forms_have_identical_crops(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BOXMOT_REID_DEVICE", "cpu")
    CppOnnxReID = _load_adapter()
    weights = _write_tiny_reid_onnx(
        tmp_path / "portrait_descriptor.onnx",
        height=24,
        width=12,
    )
    rng = np.random.default_rng(1234)
    image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)

    theta = 0.37
    equivalent = np.array(
        [
            [48.0, 48.0, 30.0, 62.0, theta],
            [48.0, 48.0, 30.0, 62.0, theta + np.pi],
            [48.0, 48.0, 62.0, 30.0, theta + np.pi / 2.0],
            [48.0, 48.0, 62.0, 30.0, theta - np.pi / 2.0],
        ],
        dtype=np.float32,
    )
    equivalent_squares = np.array(
        [
            [48.0, 48.0, 40.0, 40.0, theta],
            [48.0, 48.0, 40.0, 40.0, theta + np.pi / 2.0],
        ],
        dtype=np.float32,
    )

    reid = CppOnnxReID(weights=weights, preprocess_name="resize")
    try:
        features = reid.get_features(equivalent, image)
        expected = np.repeat(features[:1], len(features), axis=0)
        np.testing.assert_allclose(features, expected, atol=2e-5, rtol=2e-5)

        square_features = reid.get_features(equivalent_squares, image)
        expected_squares = np.repeat(square_features[:1], len(square_features), axis=0)
        np.testing.assert_allclose(square_features, expected_squares, atol=2e-5, rtol=2e-5)
    finally:
        reid.close()


def test_cpp_reid_resize_pad_matches_python_floor_and_mean_padding(tmp_path: Path, monkeypatch):
    import cv2

    from boxmot.reid.core.preprocessing import resize_pad

    monkeypatch.setenv("BOXMOT_REID_DEVICE", "cpu")
    CppOnnxReID = _load_adapter()
    weights = _write_tiny_reid_onnx(
        tmp_path / "resize_pad_descriptor.onnx",
        height=20,
        width=10,
    )
    # 7 * 0.5 == 3.5: Python floors to 3 rows, while the old native path
    # rounded to 4. The padding pixels also distinguish ImageNet mean from 0.
    image = np.arange(7 * 20 * 3, dtype=np.uint8).reshape(7, 20, 3)
    boxes = np.array([[0, 0, 20, 7]], dtype=np.float32)

    reid = CppOnnxReID(weights=weights, preprocess_name="resize_pad")
    try:
        actual = reid.get_features(boxes, image)[0]
    finally:
        reid.close()

    prepared = resize_pad(image, (20, 10))
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    expected = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    expected = expected.transpose(2, 0, 1).reshape(-1)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_cpp_reid_matches_python_embeddings():
    """C++ embeddings must agree with the Python ONNX backend.

    Both stacks share the same ONNX weights and ImageNet preprocessing
    (resize → BGR→RGB → /255 → mean/std → L2 norm). The only place the two
    can diverge is the box-to-crop conversion in
    ``boxmot/native/cpp/trackers/base/src/reid_onnx.cpp::ClampBoxToImage`` (cpp)
    vs ``boxmot/reid/backends/base_backend.py::get_crops`` (Python). This
    test guards against silent regressions in either path.
    """
    weights = _model_or_skip()
    image_path = ROOT / "assets" / "MOT17-mini" / "train" / "MOT17-02-FRCNN" / "img1" / "000001.jpg"
    if not image_path.exists():
        pytest.skip(f"Required test image not found: {image_path}")

    # Loading the Homebrew-linked native OpenCV runtime and pytest's pip cv2
    # extension in one pytest process can segfault on macOS even though the
    # same application code is stable. Exercise the real combined path in a
    # clean interpreter, which also catches C ABI crashes via the return code.
    script = textwrap.dedent(
        """
        import json
        import sys

        import cv2
        import numpy as np

        from boxmot.native.reid import CppOnnxReID
        from boxmot.reid import ReID

        weights, image_path = sys.argv[1:]
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Could not load parity image: {image_path}")
        h, w = image.shape[:2]
        boxes = np.array(
            [
                [100.4, 100.6, 200.5, 300.5],
                [300.0, 150.7, 400.2, 400.9],
                [50.5, 80.0, 150.5, 250.0],
                [w - 120.3, h - 220.7, w - 0.5, h - 0.5],
            ],
            dtype=np.float32,
        )
        obbs = np.array(
            [
                [200.0, 250.0, 80.0, 200.0, 0.35],
                [400.0, 300.0, 160.0, 50.0, -0.7],
            ],
            dtype=np.float32,
        )

        cpp = CppOnnxReID(weights=weights)
        py = ReID(path=weights, device="cpu", half=False)
        try:
            aabb_cpp = cpp.get_features(boxes, image)
            aabb_py = np.asarray(py.model.get_features(boxes, image), dtype=np.float32)
            obb_cpp = cpp.get_features(obbs, image)
            obb_py = np.asarray(py.model.get_features(obbs, image), dtype=np.float32)
            payload = {
                "aabb_shapes": [list(aabb_cpp.shape), list(aabb_py.shape)],
                "aabb_norms": [
                    np.linalg.norm(aabb_cpp, axis=1).tolist(),
                    np.linalg.norm(aabb_py, axis=1).tolist(),
                ],
                "aabb_cos": np.sum(aabb_cpp * aabb_py, axis=1).tolist(),
                "obb_shapes": [list(obb_cpp.shape), list(obb_py.shape)],
                "obb_cos": np.sum(obb_cpp * obb_py, axis=1).tolist(),
            }
            print("BOXMOT_PARITY=" + json.dumps(payload))
        finally:
            cpp.close()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(weights), str(image_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Python/native ReID parity subprocess failed "
            f"with code {completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
        )
    payload_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith("BOXMOT_PARITY=")),
        None,
    )
    assert payload_line is not None, completed.stdout
    payload = json.loads(payload_line.removeprefix("BOXMOT_PARITY="))

    assert payload["aabb_shapes"][0] == payload["aabb_shapes"][1]
    assert payload["obb_shapes"][0] == payload["obb_shapes"][1]
    for norms in payload["aabb_norms"]:
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)
    assert np.all(np.asarray(payload["aabb_cos"]) > 0.99), payload["aabb_cos"]
    assert np.all(np.asarray(payload["obb_cos"]) > 0.99), payload["obb_cos"]
