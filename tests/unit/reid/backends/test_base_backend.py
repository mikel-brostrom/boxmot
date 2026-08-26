import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import boxmot.reid.backends.base_backend as base_backend_module
import boxmot.utils.download as download_module
from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.reid.backends.openvino_backend import OpenVinoBackend
from boxmot.reid.backends.tensorrt_backend import TensorRTBackend
from boxmot.reid.backends.tflite_backend import TFLiteBackend
from boxmot.reid.backends.torchscript_backend import TorchscriptBackend
from boxmot.reid.core.artifacts import write_artifact_metadata
from boxmot.reid.core.crops import boxes_to_xyxy, canonicalize_obb_for_crop, coerce_boxes, crop_obb
from boxmot.reid.core.registry import ReIDModelRegistry


class DummyBackend(BaseModelBackend):
    def __init__(self):
        from boxmot.reid.core.preprocessing import get_preprocess_fn

        self.device = torch.device("cpu")
        self.half = False
        self.input_shape = (16, 8)
        self.mean_array = torch.zeros((1, 3, 1, 1), device=self.device)
        self.std_array = torch.ones((1, 3, 1, 1), device=self.device)
        self.nhwc = False
        self.preprocess_fn = get_preprocess_fn(None)

    def forward(self, im_batch):
        return im_batch

    def load_model(self, w):
        return None


class InitOnlyBackend(BaseModelBackend):
    def forward(self, im_batch):
        return im_batch

    def load_model(self, w):
        self.loaded_weights = Path(w)


class RegistryLoadingBackend(BaseModelBackend):
    def forward(self, im_batch):
        return self.model(im_batch)

    def load_model(self, w):
        self.checkpoint_preprocess = ReIDModelRegistry.get_checkpoint_preprocess(w)
        self.load_report = ReIDModelRegistry.load_deployment_weights(self.model, w)


def test_download_model_does_not_wait_for_stale_or_unavailable_lock_when_weights_exist(
    monkeypatch,
    tmp_path,
):
    weights = tmp_path / "lmbn_n_duke.pt"
    weights.write_bytes(b"complete checkpoint")
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    stale_lock = lock_dir / f"{weights.name}.lock"
    stale_lock.write_text("left by interrupted downloader", encoding="utf-8")

    class UnavailableLock:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("existing weights must be checked before constructing the download lock")

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(lock_dir))
    monkeypatch.setattr(base_backend_module, "FileLock", UnavailableLock)
    monkeypatch.setattr(
        ReIDModelRegistry,
        "get_model_url",
        lambda _weights: pytest.fail("existing weights must not require catalog lookup"),
    )

    BaseModelBackend.download_model(DummyBackend(), weights)

    assert stale_lock.read_text(encoding="utf-8") == "left by interrupted downloader"


def test_download_model_rechecks_weights_after_acquiring_lock(monkeypatch, tmp_path):
    weights = tmp_path / "lmbn_n_duke.pt"

    class CompletingPeerLock:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            weights.write_bytes(b"checkpoint downloaded by peer")
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(base_backend_module, "FileLock", CompletingPeerLock)
    monkeypatch.setattr(ReIDModelRegistry, "get_model_url", lambda _weights: "https://example.test/model.pt")

    BaseModelBackend.download_model(DummyBackend(), weights)

    assert weights.read_bytes() == b"checkpoint downloaded by peer"


def test_missing_weights_acquire_filelock_despite_leftover_marker(monkeypatch, tmp_path):
    weights = tmp_path / "lmbn_n_duke.pt"
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    lock_path = lock_dir / f"{weights.name}.lock"
    lock_path.write_text("left by crashed downloader", encoding="utf-8")
    downloaded = b"complete checkpoint"

    def fake_download(_url, destination):
        destination.write_bytes(downloaded)
        return destination

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(lock_dir))
    monkeypatch.setattr(
        ReIDModelRegistry,
        "get_model_url",
        lambda _weights: "https://example.test/model.pt",
    )
    monkeypatch.setattr(download_module, "download_file", fake_download)

    BaseModelBackend.download_model(DummyBackend(), weights)

    assert weights.read_bytes() == downloaded


def test_backend_construction_deserializes_checkpoint_once_per_scope(monkeypatch, tmp_path):
    weights = tmp_path / "tiny_reid.pt"
    source_model = torch.nn.Linear(2, 2)
    torch.save(
        {
            "model_name": "tiny_reid",
            "num_classes": 1,
            "preprocess": "resize",
            "state_dict": source_model.state_dict(),
        },
        weights,
    )
    monkeypatch.setattr(
        ReIDModelRegistry,
        "build_model",
        lambda *_args, **_kwargs: torch.nn.Linear(2, 2),
    )

    original_torch_load = torch.load
    load_calls = []

    def counted_torch_load(*args, **kwargs):
        load_calls.append(Path(args[0]))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_torch_load)

    backend = RegistryLoadingBackend(weights, torch.device("cpu"), half=False)

    assert backend.checkpoint_preprocess == "resize"
    assert backend.load_report.numel_coverage == 1.0
    assert load_calls == [weights]

    # The cache is construction-scoped, not process-wide: later reads see a
    # fresh file identity and do not retain the checkpoint tensor mapping.
    assert ReIDModelRegistry.get_checkpoint_preprocess(weights) == "resize"
    assert load_calls == [weights, weights]


@pytest.mark.parametrize(
    ("backend_cls", "artifact_name", "is_directory"),
    [
        (TensorRTBackend, "csl_tinyvit_7m_v20.engine", False),
        (OpenVinoBackend, "csl_tinyvit_7m_v20_openvino_model", True),
        (TFLiteBackend, "csl_tinyvit_7m_v20.tflite", False),
        (TorchscriptBackend, "csl_tinyvit_7m_v20.torchscript", False),
    ],
)
def test_compiled_backends_use_metadata_without_building_source_model(
    monkeypatch,
    tmp_path,
    backend_cls,
    artifact_name,
    is_directory,
):
    artifact = tmp_path / artifact_name
    artifact.mkdir() if is_directory else artifact.write_bytes(b"compiled")
    write_artifact_metadata(
        artifact,
        {
            "model_name": "csl_tinyvit_7m_v20",
            "num_classes": 751,
            "model_kwargs_schema_version": 1,
            "model_kwargs": {"img_size": [320, 96]},
        },
    )
    load_calls = []
    monkeypatch.setattr(
        ReIDModelRegistry,
        "build_model",
        lambda *args, **kwargs: pytest.fail("compiled backends must not build a source PyTorch model"),
    )
    monkeypatch.setattr(
        backend_cls,
        "load_model",
        lambda self, path: load_calls.append(path),
    )

    backend = backend_cls(artifact, torch.device("cpu"), half=False)

    assert backend.model is None
    assert backend.model_name == "csl_tinyvit_7m_v20"
    assert backend.input_shape == (320, 96)
    assert load_calls == [artifact]


def test_openvino_preserves_batch_limit_discovered_during_load(monkeypatch, tmp_path):
    artifact = tmp_path / "osnet_x0_25_openvino_model"
    artifact.mkdir()
    monkeypatch.setattr(
        OpenVinoBackend,
        "load_model",
        lambda self, _path: setattr(self, "_max_batch", 8),
    )

    backend = OpenVinoBackend(artifact, torch.device("cpu"), half=False)

    assert backend._max_batch == 8


def test_boxes_to_xyxy_keeps_aabb_boxes():
    boxes = np.array([[10, 20, 30, 40, 0.9, 0]], dtype=np.float32)

    xyxy = boxes_to_xyxy(boxes)

    assert xyxy.shape == (1, 4)
    np.testing.assert_array_equal(xyxy[0], np.array([10, 20, 30, 40], dtype=np.float32))


def test_boxes_to_xyxy_converts_obb_detections():
    boxes = np.array([[32, 24, 20, 10, 0.0, 0.9, 0]], dtype=np.float32)

    xyxy = boxes_to_xyxy(boxes)

    assert xyxy.shape == (1, 4)
    np.testing.assert_allclose(xyxy[0], np.array([22, 19, 42, 29], dtype=np.float32), atol=1e-4)


def test_boxes_to_xyxy_converts_obb_track_outputs():
    boxes = np.array([[32, 24, 20, 10, 0.0, 7, 0.9, 0, 5]], dtype=np.float32)

    xyxy = boxes_to_xyxy(boxes)

    assert xyxy.shape == (1, 4)
    np.testing.assert_allclose(xyxy[0], np.array([22, 19, 42, 29], dtype=np.float32), atol=1e-4)


@pytest.mark.parametrize(
    ("columns", "geometry_columns"),
    [(4, 4), (6, 4), (8, 4), (5, 5), (7, 5), (9, 5)],
)
def test_coerce_boxes_standardizes_aabb_and_obb_geometry(columns, geometry_columns):
    boxes = np.arange(columns, dtype=np.float32).reshape(1, columns)

    assert coerce_boxes(boxes).shape == (1, geometry_columns)
    assert coerce_boxes(np.empty((0, columns), dtype=np.float32)).shape == (0, geometry_columns)


@pytest.mark.parametrize("columns", [0, 1, 2, 3, 10])
def test_coerce_boxes_rejects_noncanonical_row_widths(columns):
    with pytest.raises(ValueError, match="AABB rows with 4/6/8 columns"):
        coerce_boxes(np.empty((0, columns), dtype=np.float32))


def test_get_crops_accepts_obb_boxes():
    backend = DummyBackend()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[19:29, 22:42] = 255
    boxes = np.array([[32, 24, 20, 10, 0.0]], dtype=np.float32)

    crops = backend.get_crops(boxes, img)

    assert tuple(crops.shape) == (1, 3, 16, 8)
    assert torch.count_nonzero(crops) > 0


def test_get_crops_rectifies_rotated_obb_boxes():
    backend = DummyBackend()
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    rect = ((48.0, 48.0), (40.0, 20.0), 35.0)
    corners = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(img, corners, (255, 255, 255))
    box = np.array([[48.0, 48.0, 40.0, 20.0, np.deg2rad(35.0)]], dtype=np.float32)

    crops = backend.get_crops(box, img)
    crop = crops[0].permute(1, 2, 0).cpu().numpy()

    assert tuple(crops.shape) == (1, 3, 16, 8)
    assert crop.mean() > 0.2


def test_crop_obb_excludes_enclosing_aabb_background():
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    box = np.array([60.0, 60.0, 60.0, 18.0, np.deg2rad(35.0)], dtype=np.float32)
    cv2.fillConvexPoly(img, cv2.boxPoints(((60.0, 60.0), (60.0, 18.0), 35.0)).astype(np.int32), (255, 255, 255))

    crop = crop_obb(box, img)

    # DummyBackend uses a portrait ReID input, so the long OBB edge is
    # canonicalized vertically before rectification.
    assert crop.shape[:2] == (60, 18)
    assert crop.mean() > 225.0


def test_crop_obb_is_invariant_to_equivalent_rectangle_forms():
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    texture = np.arange(50 * 20 * 3, dtype=np.uint16).reshape(50, 20, 3)
    img[23:73, 38:58] = (texture % 255).astype(np.uint8)
    equivalent = np.array(
        [
            [48.0, 48.0, 20.0, 50.0, 0.0],
            [48.0, 48.0, 50.0, 20.0, np.pi / 2.0],
            [48.0, 48.0, 20.0, 50.0, np.pi],
            [48.0, 48.0, 50.0, 20.0, -np.pi / 2.0],
        ],
        dtype=np.float32,
    )

    crops = [crop_obb(box, img, input_shape=(16, 8)) for box in equivalent]

    assert {crop.shape for crop in crops} == {(50, 20, 3)}
    for crop in crops[1:]:
        np.testing.assert_array_equal(crop, crops[0])


def test_obb_crop_canonical_angle_is_continuous_across_half_turn_boundary():
    before = canonicalize_obb_for_crop(
        np.array([48, 48, 20, 50, np.pi - 1e-4], dtype=np.float32),
        input_shape=(16, 8),
    )
    after = canonicalize_obb_for_crop(
        np.array([48, 48, 20, 50, np.pi + 1e-4], dtype=np.float32),
        input_shape=(16, 8),
    )

    assert abs(float(after[4] - before[4])) < 1e-3


def test_get_features_keeps_invalid_descriptors_finite():
    backend = DummyBackend()
    backend.forward = lambda crops: np.array(
        [[0.0, 0.0, 0.0, 0.0], [np.nan, np.inf, -np.inf, 0.0]],
        dtype=np.float32,
    )
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    features = backend.get_features(
        np.array([[0, 0, 16, 16], [8, 8, 24, 24]], dtype=np.float32),
        img,
    )

    np.testing.assert_array_equal(features, np.zeros((2, 4), dtype=np.float32))
    assert np.isfinite(features).all()


def test_base_backend_preserves_explicit_export_paths(monkeypatch):
    monkeypatch.setattr(ReIDModelRegistry, "get_model_name", lambda _weights: "osnet_x0_25")
    monkeypatch.setattr(ReIDModelRegistry, "get_nr_classes", lambda _weights: 1)
    monkeypatch.setattr(ReIDModelRegistry, "build_model", lambda *args, **kwargs: object())

    explicit_path = Path("models/osnet_x0_25_msmt17_saved_model/osnet_x0_25_msmt17_float32.tflite")
    backend = InitOnlyBackend(explicit_path, torch.device("cpu"), half=False)

    assert backend.weights == explicit_path
    assert backend.loaded_weights == explicit_path


def test_base_backend_uses_reid_crop_shape_for_mobilenetv4(monkeypatch, tmp_path):
    monkeypatch.setattr(ReIDModelRegistry, "get_model_name", lambda _weights: "mobilenetv4_conv_small")
    monkeypatch.setattr(ReIDModelRegistry, "get_nr_classes", lambda _weights: 1)
    monkeypatch.setattr(ReIDModelRegistry, "build_model", lambda *args, **kwargs: object())

    backend = InitOnlyBackend(tmp_path / "mobilenetv4_conv_small_market1501.pt", torch.device("cpu"), half=False)

    assert backend.input_shape == (384, 128)


def test_base_backend_prefers_checkpoint_img_size(monkeypatch, tmp_path):
    weights = tmp_path / "csl_tinyvit_7m_v20_veri_custom.pt"
    torch.save(
        {
            "model_name": "csl_tinyvit_7m_v20",
            "num_classes": 4,
            "model_kwargs_schema_version": 1,
            "model_kwargs": {"img_size": [320, 96]},
            "state_dict": {},
        },
        weights,
    )
    captured = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ReIDModelRegistry, "build_model", fake_build)

    backend = InitOnlyBackend(weights, torch.device("cpu"), half=False)

    assert backend.input_shape == (320, 96)
    assert captured["img_size"] == (320, 96)
    assert captured["num_classes"] == 1
