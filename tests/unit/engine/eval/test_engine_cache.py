from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import boxmot.engine.eval.cache as cache_module
import boxmot.engine.eval.evaluator as evaluator_module
from boxmot.data.cache import reid_cache_key, reid_preprocess_cache_key


def _cache_args(tmp_path: Path, reid_model: Path, *, resume: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        project=tmp_path,
        cache_project=tmp_path / "cache",
        benchmark="mot17",
        split="ablation",
        detector=[Path("det.pt")],
        reid=[reid_model],
        device="cpu",
        reid_device="cpu",
        imgsz=[640, 640],
        half=False,
        reid_half=False,
        reid_preprocess="resize",
        tracker_backend="python",
        n_threads=1,
        batch_size=1,
        auto_batch=False,
        resume=resume,
        verbose=False,
        show_progress=False,
        iou=0.7,
        agnostic_nms=False,
        classes=None,
        eval_box_type=None,
        ci=True,
    )


def _make_cache_sequence(tmp_path: Path, *, frame_count: int) -> tuple[Path, str]:
    source_root = tmp_path / "source"
    sequence_name = "MOT17-02-FRCNN"
    image_dir = source_root / sequence_name / "img1"
    image_dir.mkdir(parents=True)
    for frame_id in range(1, frame_count + 1):
        (image_dir / f"{frame_id:06d}.jpg").write_bytes(b"")
    return source_root, sequence_name


def _cache_paths(args: SimpleNamespace, sequence_name: str) -> tuple[Path, Path]:
    detector_root = args.cache_project / "dets_n_embs" / args.benchmark / args.split / "det"
    return detector_root / "dets" / f"{sequence_name}.npy", detector_root / "embs"


def _detection_rows(frame_count: int) -> np.ndarray:
    return np.asarray(
        [[frame_id, 1, 2, 10, 12, 0.9, 0] for frame_id in range(1, frame_count + 1)],
        dtype=np.float32,
    )


def _forbid_detector_and_reid_work(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("a complete cache must not run detector or ReID work")

    monkeypatch.setattr(cache_module, "DetectorReIDPipeline", fail)
    monkeypatch.setattr(cache_module, "_run_embeddings_only_fill", fail)
    monkeypatch.setattr(cache_module, "AppendableNpyWriter", fail)


def test_cache_main_runs_generation_pipeline(monkeypatch):
    generated = []
    printed = []

    monkeypatch.setattr(
        cache_module,
        "run_generate_dets_embs",
        lambda args, timing_stats=None, progress_callback=None: generated.append((args, timing_stats)),
    )
    monkeypatch.setattr(cache_module.TimingStats, "print_summary", lambda self: printed.append(self.frames))

    args = SimpleNamespace()
    cache_module.main(args)

    assert generated and generated[0][0] is args
    assert len(printed) == 0


def test_evaluator_reexports_cache_generation_helpers():
    assert evaluator_module.generate_dets_embs_batched is cache_module.generate_dets_embs_batched
    assert evaluator_module.run_generate_dets_embs is cache_module.run_generate_dets_embs


def test_run_generate_dets_embs_logs_only_when_verbose(monkeypatch, tmp_path):
    logged = []
    generated = []

    monkeypatch.setattr(cache_module, "_configure_benchmark_runtime", lambda args: None)
    monkeypatch.setattr(
        cache_module,
        "generate_dets_embs_batched",
        lambda args, detector, source_root, timing_stats=None: generated.append(
            (args.verbose, args.show_progress, detector, source_root)
        ),
    )
    monkeypatch.setattr(cache_module.LOGGER, "info", lambda message: logged.append(message))

    quiet_args = SimpleNamespace(
        project=tmp_path,
        source=tmp_path / "benchmark",
        experiment=None,
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        batch_size=16,
        n_threads=1,
        auto_batch=True,
        resume=True,
        verbose=False,
        show_progress=True,
    )
    cache_module.run_generate_dets_embs(quiet_args)

    assert logged == []
    assert len(generated) == 1
    assert generated[0][0] is False
    assert generated[0][1] is True
    assert generated[0][2].name == "det.pt"
    assert generated[0][3] == tmp_path / "benchmark"

    verbose_args = SimpleNamespace(
        project=tmp_path,
        source=tmp_path / "benchmark",
        experiment=None,
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        batch_size=16,
        n_threads=1,
        auto_batch=True,
        resume=True,
        verbose=True,
        show_progress=True,
    )
    cache_module.run_generate_dets_embs(verbose_args)

    assert logged == ["Generating dets+embs (batched single-process): det.pt"]
    assert generated[-1][0] is True
    assert generated[-1][1] is True
    assert generated[-1][2].name == "det.pt"
    assert generated[-1][3] == tmp_path / "benchmark"


def test_generate_dets_embs_batched_routes_progress_into_callback(tmp_path, monkeypatch):
    source_root = tmp_path / "train"
    for seq_name in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        img_dir = source_root / seq_name / "img1"
        img_dir.mkdir(parents=True)
        (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        cache_project=tmp_path / "cache",
        benchmark="mot17-mini",
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        device="cpu",
        reid_device="cpu",
        imgsz=[640, 640],
        half=False,
        reid_half=False,
        reid_preprocess=None,
        n_threads=1,
        batch_size=1,
        auto_batch=False,
        resume=False,
        verbose=False,
        show_progress=True,
        iou=0.7,
        agnostic_nms=False,
        classes=None,
        eval_box_type=None,
    )

    tqdm_calls = []
    messages = []

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            tqdm_calls.append(kwargs)

        def update(self, _value):
            return None

        def close(self):
            return None

    class FakeWriter:
        def __init__(self, *args, **kwargs):
            self.rows = []

        def append(self, arr):
            self.rows.append(np.asarray(arr))

        def close(self):
            return None

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            return None

        def warmup(self):
            return None

        def autotune_batch_size(self, batch_size):
            return batch_size

        def predict_batch(self, images, conf, iou, agnostic_nms, classes):
            return [SimpleNamespace(dets=np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)) for _ in images]

        def get_all_reid_features(self, det_boxes_np, img):
            return {"reid.pt": np.ones((det_boxes_np.shape[0], 4), dtype=np.float32)}

    monkeypatch.setattr(cache_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(cache_module, "AppendableNpyWriter", FakeWriter)
    monkeypatch.setattr(cache_module, "DetectorReIDPipeline", FakePipeline)
    monkeypatch.setattr(cache_module, "_read_image_cv2", lambda path: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(
        cache_module,
        "_serialize_eval_detections",
        lambda dets, frame_id: (
            np.array([[frame_id, 1, 2, 10, 12, 0.9, 0]], dtype=np.float32),
            np.array([[1, 2, 10, 12]], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda device: None)

    cache_module.generate_dets_embs_batched(
        args,
        Path("det.pt"),
        source_root,
        progress_callback=messages.append,
    )

    assert messages
    assert "Generating detections and embeddings" in messages[-1]
    assert "MOT17-02-FRCNN" in messages[-1]
    assert "MOT17-04-FRCNN" in messages[-1]
    assert all(call["disable"] is True for call in tqdm_calls)


def test_generate_dets_embs_batched_complete_canonical_cache_is_a_noop(tmp_path, monkeypatch):
    reid_model = tmp_path / "models" / "lmbn_n_duke.pt"
    reid_model.parent.mkdir()
    reid_model.write_bytes(b"model weights")
    args = _cache_args(tmp_path, reid_model)
    args.allow_legacy_reid_cache = True
    source_root, sequence_name = _make_cache_sequence(tmp_path, frame_count=2)
    dets_file, embeddings_root = _cache_paths(args, sequence_name)
    canonical_file = (
        embeddings_root
        / reid_cache_key(reid_model, tracker_backend="python")
        / reid_preprocess_cache_key("resize")
        / f"{sequence_name}.npy"
    )
    dets_file.parent.mkdir(parents=True)
    canonical_file.parent.mkdir(parents=True)
    np.save(dets_file, _detection_rows(2))
    np.save(canonical_file, np.ones((2, 4), dtype=np.float32))
    _forbid_detector_and_reid_work(monkeypatch)

    cache_module.generate_dets_embs_batched(args, Path("det.pt"), source_root)

    assert np.load(canonical_file).shape == (2, 4)


def test_generate_dets_embs_batched_reuses_row_aligned_legacy_lmbn_cache(tmp_path, monkeypatch):
    reid_model = tmp_path / "models" / "lmbn_n_duke.pt"
    reid_model.parent.mkdir()
    reid_model.write_bytes(b"model weights")
    args = _cache_args(tmp_path, reid_model)
    args.allow_legacy_reid_cache = True
    source_root, sequence_name = _make_cache_sequence(tmp_path, frame_count=2)
    dets_file, embeddings_root = _cache_paths(args, sequence_name)
    legacy_file = embeddings_root / "lmbn_n_duke" / "resize" / f"{sequence_name}.npy"
    dets_file.parent.mkdir(parents=True)
    legacy_file.parent.mkdir(parents=True)
    np.save(dets_file, _detection_rows(2))
    np.save(legacy_file, np.ones((2, 4), dtype=np.float32))
    _forbid_detector_and_reid_work(monkeypatch)

    cache_module.generate_dets_embs_batched(args, Path("det.pt"), source_root)

    assert np.load(legacy_file).shape == (2, 4)


def test_generate_dets_embs_batched_prefers_complete_legacy_over_partial_canonical(
    tmp_path,
    monkeypatch,
):
    reid_model = tmp_path / "models" / "lmbn_n_duke.pt"
    reid_model.parent.mkdir()
    reid_model.write_bytes(b"model weights")
    args = _cache_args(tmp_path, reid_model)
    args.allow_legacy_reid_cache = True
    source_root, sequence_name = _make_cache_sequence(tmp_path, frame_count=2)
    dets_file, embeddings_root = _cache_paths(args, sequence_name)
    canonical_file = (
        embeddings_root
        / reid_cache_key(reid_model, tracker_backend="python")
        / reid_preprocess_cache_key("resize")
        / f"{sequence_name}.npy"
    )
    legacy_file = embeddings_root / "lmbn_n_duke" / "resize" / f"{sequence_name}.npy"
    dets_file.parent.mkdir(parents=True)
    canonical_file.parent.mkdir(parents=True)
    legacy_file.parent.mkdir(parents=True)
    np.save(dets_file, _detection_rows(2))
    np.save(canonical_file, np.ones((1, 4), dtype=np.float32))
    np.save(legacy_file, np.ones((2, 4), dtype=np.float32))
    _forbid_detector_and_reid_work(monkeypatch)

    cache_module.generate_dets_embs_batched(args, Path("det.pt"), source_root)

    assert np.load(legacy_file).shape == (2, 4)


def test_generate_dets_embs_batched_uses_full_model_path_for_writer_identity(tmp_path, monkeypatch):
    reid_model = tmp_path / "models" / "lmbn_n_duke.pt"
    reid_model.parent.mkdir()
    reid_model.write_bytes(b"model weights")
    args = _cache_args(tmp_path, reid_model, resume=False)
    source_root, sequence_name = _make_cache_sequence(tmp_path, frame_count=1)
    _, embeddings_root = _cache_paths(args, sequence_name)

    class FakePipeline:
        def __init__(self, *, reid_paths, **_kwargs):
            assert reid_paths == [reid_model]

        def warmup(self):
            return None

        def autotune_batch_size(self, batch_size):
            return batch_size

        def predict_batch(self, images, conf, iou, agnostic_nms, classes):
            return [SimpleNamespace(dets=np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)) for _ in images]

        def get_all_reid_features(self, det_boxes_np, _image):
            # The pipeline preserves the resolved model identifier so cache
            # writers retain the content fingerprint from the full path.
            return {str(reid_model): np.ones((det_boxes_np.shape[0], 4), dtype=np.float32)}

    monkeypatch.setattr(cache_module, "DetectorReIDPipeline", FakePipeline)
    monkeypatch.setattr(
        cache_module,
        "_read_image_cv2",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda _device: None)

    cache_module.generate_dets_embs_batched(args, Path("det.pt"), source_root)

    canonical_file = (
        embeddings_root
        / reid_cache_key(reid_model, tracker_backend="python")
        / reid_preprocess_cache_key("resize")
        / f"{sequence_name}.npy"
    )
    assert np.load(canonical_file).shape == (1, 4)


def test_generate_dets_embs_batched_rekeys_writer_after_model_creation(tmp_path, monkeypatch):
    reid_model = tmp_path / "models" / "lmbn_n_duke.pt"
    reid_model.parent.mkdir()
    args = _cache_args(tmp_path, reid_model, resume=False)
    source_root, sequence_name = _make_cache_sequence(tmp_path, frame_count=1)
    _, embeddings_root = _cache_paths(args, sequence_name)
    frozen_key = reid_cache_key(reid_model, tracker_backend="python")

    class FakePipeline:
        def __init__(self, *, reid_paths, **_kwargs):
            assert reid_paths == [reid_model]
            reid_model.write_bytes(b"downloaded model weights")

        def warmup(self):
            return None

        def autotune_batch_size(self, batch_size):
            return batch_size

        def predict_batch(self, images, conf, iou, agnostic_nms, classes):
            return [SimpleNamespace(dets=np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)) for _ in images]

        def get_all_reid_features(self, det_boxes_np, _image):
            return {str(reid_model): np.ones((det_boxes_np.shape[0], 4), dtype=np.float32)}

    monkeypatch.setattr(cache_module, "DetectorReIDPipeline", FakePipeline)
    monkeypatch.setattr(
        cache_module,
        "_read_image_cv2",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda _device: None)

    cache_module.generate_dets_embs_batched(args, Path("det.pt"), source_root)

    actual_key = reid_cache_key(reid_model, tracker_backend="python")
    actual_file = embeddings_root / actual_key / reid_preprocess_cache_key("resize") / f"{sequence_name}.npy"
    frozen_file = embeddings_root / frozen_key / reid_preprocess_cache_key("resize") / f"{sequence_name}.npy"
    assert actual_key != frozen_key
    assert np.load(actual_file).shape == (1, 4)
    assert not frozen_file.exists()


def test_embeddings_only_fill_scopes_models_to_each_sequences_missing_keys(tmp_path, monkeypatch):
    models = [tmp_path / "model_a.pt", tmp_path / "model_b.pt"]
    for index, model in enumerate(models, start=1):
        model.write_bytes(f"model {index}".encode())
    keys = [reid_cache_key(model, tracker_backend="python") for model in models]

    class FakeReIDModel:
        def __init__(self, value):
            self.value = value

        def get_features(self, boxes, _image):
            return np.full((boxes.shape[0], 4), self.value, dtype=np.float32)

    def build_models(reid_paths, **_kwargs):
        return {str(path): FakeReIDModel(index) for index, path in enumerate(reid_paths, start=1)}

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def update(self, _value):
            return None

        def close(self):
            return None

    monkeypatch.setattr(cache_module, "_build_reid_only_models", build_models)
    monkeypatch.setattr(cache_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(
        cache_module,
        "_read_image_cv2",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )

    states = {}
    output_paths = []
    for index, (model_key, sequence_name) in enumerate(
        zip(keys, ("sequence_a", "sequence_b")),
        start=1,
    ):
        frame_path = tmp_path / f"frame_{index}.jpg"
        frame_path.write_bytes(b"")
        dets_path = tmp_path / f"dets_{index}.npy"
        np.save(dets_path, np.array([[1, 1, 2, 10, 12, 0.9, 0]], dtype=np.float32))
        output_path = tmp_path / f"embs_{index}.npy"
        output_paths.append(output_path)
        states[sequence_name] = {
            "frames": [frame_path],
            "dets_path": dets_path,
            "missing_keys": [model_key],
            "emb_paths": {model_key: output_path},
        }

    args = SimpleNamespace(reid=models, device="cpu", reid_device="cpu", half=False, reid_half=False)
    cache_module._run_embeddings_only_fill(
        args,
        states,
        expected_det_cols=7,
        preprocess_name="resize",
        tracker_backend="python",
        progress_callback=None,
        show_progress=False,
        own_terminal_progress=True,
        verbose=False,
    )

    np.testing.assert_array_equal(np.load(output_paths[0]), np.ones((1, 4), dtype=np.float32))
    np.testing.assert_array_equal(np.load(output_paths[1]), np.full((1, 4), 2, dtype=np.float32))


def test_generate_dets_embs_batched_resets_partial_cache(tmp_path, monkeypatch):
    """A previously interrupted run can leave the dets cache ahead of the
    embs cache (e.g. dets file written, embs file missing or shorter). The
    next run must heal the on-disk state so the native C++ replay never sees
    ``det_rows != emb_rows``.

    Since the embeddings-only fill landed, the heal path now keeps the
    detections cache and only regenerates the missing embedding bucket(s),
    saving an unnecessary YOLO pass. This regression guards both that the
    dets cache is preserved and that fresh embeddings are written for it.
    """

    source_root = tmp_path / "train"
    seq_name = "MOT17-02-FRCNN"
    img_dir = source_root / seq_name / "img1"
    img_dir.mkdir(parents=True)
    (img_dir / "000001.jpg").write_bytes(b"")

    # Pre-create a stale dets cache with rows but no matching embs.
    dets_dir = tmp_path / "cache" / "dets_n_embs" / "mot17-mini" / "det" / "dets"
    dets_dir.mkdir(parents=True)
    stale_dets_path = dets_dir / f"{seq_name}.npy"
    np.save(stale_dets_path, np.array([[1, 2, 3, 4, 5, 0.9, 0]], dtype=np.float32))
    assert stale_dets_path.exists()

    args = SimpleNamespace(
        project=tmp_path,
        cache_project=tmp_path / "cache",
        benchmark="mot17-mini",
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        device="cpu",
        reid_device="cpu",
        imgsz=[640, 640],
        half=False,
        reid_half=False,
        reid_preprocess=None,
        n_threads=1,
        batch_size=1,
        auto_batch=False,
        resume=True,
        verbose=False,
        show_progress=False,
        iou=0.7,
        agnostic_nms=False,
        classes=None,
        eval_box_type=None,
    )

    appended_dets: list[np.ndarray] = []
    appended_embs: list[np.ndarray] = []

    class FakeWriter:
        def __init__(self, path, *_, **__):
            self.path = Path(path)

        def append(self, arr):
            # Distinguish writers by the *immediate* parent kind ("dets" vs the
            # embedding subtree) -- ``dets_n_embs`` appears in BOTH paths.
            parts = self.path.parts
            target = appended_dets if "dets" in parts and "embs" not in parts else appended_embs
            target.append(np.asarray(arr))

        def close(self):
            return None

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            return None

        def warmup(self):
            return None

        def autotune_batch_size(self, batch_size):
            return batch_size

        def predict_batch(self, images, conf, iou, agnostic_nms, classes):
            return [SimpleNamespace(dets=np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)) for _ in images]

        def get_all_reid_features(self, det_boxes_np, img):
            return {"reid.pt": np.ones((det_boxes_np.shape[0], 4), dtype=np.float32)}

    class _Tqdm:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, _):
            pass

        def close(self):
            pass

    class FakeReIDModel:
        def get_features(self, boxes, img):
            return np.ones((boxes.shape[0], 4), dtype=np.float32)

    monkeypatch.setattr(cache_module, "tqdm", _Tqdm)
    monkeypatch.setattr(cache_module, "AppendableNpyWriter", FakeWriter)
    monkeypatch.setattr(cache_module, "DetectorReIDPipeline", FakePipeline)
    monkeypatch.setattr(
        cache_module,
        "_build_reid_only_models",
        lambda reid_paths, **kwargs: {str(Path(p)): FakeReIDModel() for p in reid_paths},
    )
    monkeypatch.setattr(cache_module, "_read_image_cv2", lambda path: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(
        cache_module,
        "_serialize_eval_detections",
        lambda dets, frame_id: (
            np.array([[frame_id, 1, 2, 10, 12, 0.9, 0]], dtype=np.float32),
            np.array([[1, 2, 10, 12]], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda device: None)

    cache_module.generate_dets_embs_batched(args, Path("det.pt"), source_root)

    # The cached detections must be preserved -- the new heal path no longer
    # re-runs YOLO when only the embeddings bucket is missing.
    assert stale_dets_path.exists(), "Cached dets must be preserved by the embeddings-only fill"

    # The main YOLO loop did not run for this sequence (no fresh dets append),
    # but the embeddings-only fill produced one row matching the cached dets.
    assert len(appended_dets) == 0, "YOLO/dets path should not have re-run"
    assert len(appended_embs) == 1
    assert appended_embs[0].shape[0] == 1


def test_generate_dets_embs_batched_appends_embs_before_dets(tmp_path, monkeypatch):
    """If ReID raises mid-frame, the dets writer must NOT have been called for
    that frame, otherwise the on-disk det count will exceed the emb count."""

    source_root = tmp_path / "train"
    seq_name = "MOT17-02-FRCNN"
    img_dir = source_root / seq_name / "img1"
    img_dir.mkdir(parents=True)
    (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        cache_project=tmp_path / "cache",
        benchmark="mot17-mini",
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        device="cpu",
        reid_device="cpu",
        imgsz=[640, 640],
        half=False,
        reid_half=False,
        reid_preprocess=None,
        n_threads=1,
        batch_size=1,
        auto_batch=False,
        resume=False,
        verbose=False,
        show_progress=False,
        iou=0.7,
        agnostic_nms=False,
        classes=None,
        eval_box_type=None,
    )

    det_appends = 0
    emb_appends = 0

    class FakeWriter:
        def __init__(self, path, *_, **__):
            self.path = Path(path)

        def append(self, arr):
            nonlocal det_appends, emb_appends
            parts = self.path.parts
            if "dets" in parts and "embs" not in parts:
                det_appends += 1
            else:
                emb_appends += 1

        def close(self):
            return None

    class _Tqdm:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, _):
            pass

        def close(self):
            pass

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            return None

        def warmup(self):
            return None

        def autotune_batch_size(self, batch_size):
            return batch_size

        def predict_batch(self, images, conf, iou, agnostic_nms, classes):
            return [SimpleNamespace(dets=np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)) for _ in images]

        def get_all_reid_features(self, det_boxes_np, img):
            raise RuntimeError("simulated ReID failure")

    monkeypatch.setattr(cache_module, "tqdm", _Tqdm)
    monkeypatch.setattr(cache_module, "AppendableNpyWriter", FakeWriter)
    monkeypatch.setattr(cache_module, "DetectorReIDPipeline", FakePipeline)
    monkeypatch.setattr(cache_module, "_read_image_cv2", lambda path: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(
        cache_module,
        "_serialize_eval_detections",
        lambda dets, frame_id: (
            np.array([[frame_id, 1, 2, 10, 12, 0.9, 0]], dtype=np.float32),
            np.array([[1, 2, 10, 12]], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda device: None)

    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="simulated ReID failure"):
        cache_module.generate_dets_embs_batched(args, Path("det.pt"), source_root)

    # Critical invariant: dets writer must NEVER have been called when ReID
    # failed for that frame.
    assert det_appends == 0
    assert emb_appends == 0
