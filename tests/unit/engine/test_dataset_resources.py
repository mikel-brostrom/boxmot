from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import boxmot.engine.materialization.resources as dataset_resources
import boxmot.engine.materialization.workflow as materialization_workflow
from boxmot.engine.eval import evaluator


def _dataset(resource: dict | None = None) -> dict:
    return {
        "id": "fixture",
        "root": "Fixture",
        "layout": "mot",
        "box_type": "aabb",
        "default_split": "ablation",
        "splits": {
            "train": {"path": "train", "has_ground_truth": True},
            "ablation": {"path": "ablation", "has_ground_truth": True},
        },
        "classes": {"pedestrian": {"id": 1, "evaluation": "target"}},
        "resources": {} if resource is None else {"dataset": resource},
    }


def test_missing_per_split_dataset_downloads_only_the_selected_split(monkeypatch, tmp_path) -> None:
    resource = {
        "type": "per_split",
        "uris": {
            "train": "hf://owner/fixture/train",
            "ablation": "hf://owner/fixture/ablation",
        },
    }
    config = _dataset(resource)
    calls: list[dict] = []
    registered_status = object()

    def download_eval_data(**kwargs) -> None:
        calls.append(kwargs)
        image = kwargs["dataset_dest"] / "ablation" / "sequence" / "frame.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")

    monkeypatch.setattr(dataset_resources, "_download_eval_data", download_eval_data)
    monkeypatch.setattr(dataset_resources, "_get_download_status_fn", lambda: registered_status)

    split_root = dataset_resources.ensure_dataset_split_available(
        config,
        split="ablation",
        data_root=tmp_path,
        status_callback=lambda _message: None,
    )

    assert split_root == (tmp_path / "Fixture" / "ablation").resolve()
    assert calls == [
        {
            "dataset_url": "hf://owner/fixture/ablation",
            "dataset_dest": (tmp_path / "Fixture").resolve(),
            "overwrite": True,
            "status_fn": registered_status,
        }
    ]


def test_populated_dataset_split_is_never_downloaded(monkeypatch, tmp_path) -> None:
    split_root = tmp_path / "Fixture" / "ablation"
    (split_root / ".sequence").mkdir(parents=True)
    (split_root / ".sequence" / "frame.jpg").write_bytes(b"image")
    monkeypatch.setattr(
        dataset_resources,
        "_download_eval_data",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("downloaded populated split")),
    )

    resolved = dataset_resources.ensure_dataset_split_available(
        _dataset(
            {
                "type": "per_split",
                "uris": {"ablation": "hf://owner/fixture/ablation"},
            }
        ),
        split="ablation",
        data_root=tmp_path,
    )

    assert resolved == split_root.resolve()


def test_missing_local_only_dataset_is_left_for_the_catalog_diagnostic(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        dataset_resources,
        "_download_eval_data",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("downloaded local-only dataset")),
    )

    split_root = dataset_resources.ensure_dataset_split_available(
        _dataset(),
        split="ablation",
        data_root=tmp_path,
    )

    assert split_root == (tmp_path / "Fixture" / "ablation").resolve()
    assert not split_root.exists()


def test_archive_resource_is_left_for_explicit_dataset_setup(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        dataset_resources,
        "_download_eval_data",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("downloaded archive implicitly")),
    )

    split_root = dataset_resources.ensure_dataset_split_available(
        _dataset({"type": "archive", "uri": "https://example.test/fixture.zip"}),
        split="ablation",
        data_root=tmp_path,
    )

    assert split_root == (tmp_path / "Fixture" / "ablation").resolve()
    assert not split_root.exists()


def test_empty_sequence_and_os_metadata_do_not_suppress_download(monkeypatch, tmp_path) -> None:
    split_root = tmp_path / "Fixture" / "ablation"
    (split_root / "empty-sequence").mkdir(parents=True)
    (split_root / ".DS_Store").write_bytes(b"metadata")
    (split_root / "empty-sequence" / "._000001.jpg").write_bytes(b"AppleDouble metadata")
    calls: list[dict] = []

    def download_eval_data(**kwargs) -> None:
        calls.append(kwargs)
        image = split_root / "sequence" / "frame.jpg"
        image.parent.mkdir()
        image.write_bytes(b"image")

    monkeypatch.setattr(dataset_resources, "_download_eval_data", download_eval_data)
    monkeypatch.setattr(dataset_resources, "_get_download_status_fn", lambda: None)

    dataset_resources.ensure_dataset_split_available(
        _dataset(
            {
                "type": "per_split",
                "uris": {"ablation": "hf://owner/fixture/ablation"},
            }
        ),
        split="ablation",
        data_root=tmp_path,
    )

    assert len(calls) == 1


def test_default_download_and_read_roots_ignore_stale_dataset_environment(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []

    def download_eval_data(**kwargs) -> None:
        calls.append(kwargs)
        image = kwargs["dataset_dest"] / "ablation" / "sequence" / "frame.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOXMOT_DATASETS_DIR", "/Volumes/Data/MMOT")
    monkeypatch.setattr(dataset_resources, "_download_eval_data", download_eval_data)
    monkeypatch.setattr(dataset_resources, "_get_download_status_fn", lambda: None)

    split_root = dataset_resources.ensure_dataset_split_available(
        _dataset(
            {
                "type": "per_split",
                "uris": {"ablation": "hf://owner/fixture/ablation"},
            }
        ),
        split="ablation",
    )

    expected_root = (tmp_path / "datasets" / "mot" / "Fixture").resolve()
    assert calls[0]["dataset_dest"] == expected_root
    assert split_root == expected_root / "ablation"


def test_configured_download_must_create_the_selected_split(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dataset_resources, "_download_eval_data", lambda **_kwargs: None)
    monkeypatch.setattr(dataset_resources, "_get_download_status_fn", lambda: None)

    with pytest.raises(FileNotFoundError, match="did not create configured split 'ablation'"):
        dataset_resources.ensure_dataset_split_available(
            _dataset(
                {
                    "type": "per_split",
                    "uris": {"ablation": "hf://owner/fixture/ablation"},
                }
            ),
            split="ablation",
            data_root=tmp_path,
        )


def test_interrupted_hf_download_is_resumed_instead_of_accepting_partial_content(
    monkeypatch,
    tmp_path,
) -> None:
    attempts: list[bool] = []

    def download_eval_data(**kwargs) -> None:
        attempts.append(kwargs["overwrite"])
        split_root = kwargs["dataset_dest"] / "ablation"
        sequence = split_root / "sequence"
        sequence.mkdir(parents=True, exist_ok=True)
        if len(attempts) == 1:
            (sequence / "partial.jpg").write_bytes(b"partial")
            raise RuntimeError("interrupted")
        (sequence / "frame.jpg").write_bytes(b"image")

    monkeypatch.setattr(dataset_resources, "_download_eval_data", download_eval_data)
    monkeypatch.setattr(dataset_resources, "_get_download_status_fn", lambda: None)
    config = _dataset(
        {
            "type": "per_split",
            "uris": {"ablation": "hf://owner/fixture/ablation"},
        }
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        dataset_resources.ensure_dataset_split_available(
            config,
            split="ablation",
            data_root=tmp_path,
        )

    dataset_resources.ensure_dataset_split_available(
        config,
        split="ablation",
        data_root=tmp_path,
    )

    assert attempts == [True, True]
    assert list((tmp_path / ".boxmot" / "downloads").glob("*.incomplete")) == []


def test_concurrent_dataset_acquisition_downloads_the_split_once(monkeypatch, tmp_path) -> None:
    download_started = threading.Event()
    release_download = threading.Event()
    calls: list[dict] = []

    def download_eval_data(**kwargs) -> None:
        calls.append(kwargs)
        download_started.set()
        assert release_download.wait(timeout=5)
        image = kwargs["dataset_dest"] / "ablation" / "sequence" / "frame.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")

    monkeypatch.setattr(dataset_resources, "_download_eval_data", download_eval_data)
    monkeypatch.setattr(dataset_resources, "_get_download_status_fn", lambda: None)
    config = _dataset(
        {
            "type": "per_split",
            "uris": {"ablation": "hf://owner/fixture/ablation"},
        }
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            dataset_resources.ensure_dataset_split_available,
            config,
            split="ablation",
            data_root=tmp_path,
        )
        assert download_started.wait(timeout=5)
        second = executor.submit(
            dataset_resources.ensure_dataset_split_available,
            config,
            split="ablation",
            data_root=tmp_path,
        )
        release_download.set()

        assert first.result(timeout=5) == (tmp_path / "Fixture" / "ablation").resolve()
        assert second.result(timeout=5) == first.result()

    assert len(calls) == 1


def test_missing_split_uri_fails_before_starting_a_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        dataset_resources,
        "_download_eval_data",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("download started")),
    )

    with pytest.raises(ValueError, match="has no download URI for split 'ablation'"):
        dataset_resources.ensure_dataset_split_available(
            _dataset({"type": "per_split", "uris": {"train": "hf://owner/fixture/train"}}),
            split="ablation",
            data_root=tmp_path,
        )


def test_materialization_downloads_the_experiment_dataset_before_cataloging(monkeypatch, tmp_path) -> None:
    data_root = tmp_path / "datasets"
    resolved = {
        "id": "fixture-ablation-detector",
        "source_path": tmp_path / "experiment.yaml",
        "dataset": _dataset(
            {
                "type": "per_split",
                "uris": {"ablation": "hf://owner/fixture/ablation"},
            }
        )
        | {"split": "ablation"},
        "detector": {"ref": "fixture-detector", "checkpoint": "default"},
        "segmentor": None,
        "reid": None,
        "evaluation": {
            "classes": [
                {
                    "name": "pedestrian",
                    "dataset_id": 1,
                    "detector_name": "person",
                    "detector_id": 0,
                }
            ]
        },
    }
    calls: list[dict] = []

    def download_eval_data(**kwargs) -> None:
        calls.append(kwargs)
        image_path = kwargs["dataset_dest"] / "ablation" / "sequence" / "img1" / "000001.jpg"
        image_path.parent.mkdir(parents=True)
        assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))

    monkeypatch.setattr(materialization_workflow, "resolve_experiment_config", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(dataset_resources, "_download_eval_data", download_eval_data)
    monkeypatch.setattr(dataset_resources, "_get_download_status_fn", lambda: None)
    monkeypatch.setattr(
        materialization_workflow,
        "default_source_metadata_cache_path",
        lambda _root: tmp_path / "metadata.json",
    )

    inputs = materialization_workflow._resolved_inputs(
        SimpleNamespace(experiment="fixture-ablation-detector", data_root=data_root)
    )

    assert calls[0]["dataset_url"] == "hf://owner/fixture/ablation"
    assert calls[0]["dataset_dest"] == (data_root / "Fixture").resolve()
    assert inputs[2].samples[0].sample_id == "ablation:sequence:0"


def test_evaluation_ensures_the_raw_split_before_cataloging(monkeypatch, tmp_path) -> None:
    config = _dataset()
    config["split"] = "ablation"
    events: list[str] = []

    monkeypatch.setattr(evaluator, "_resolve_selection", lambda _args: (config, {}))
    monkeypatch.setattr(
        evaluator,
        "resolve_build_path",
        lambda *_args, **_kwargs: events.append("build") or tmp_path / "build",
    )
    monkeypatch.setattr(
        evaluator,
        "DatasetManifest",
        SimpleNamespace(load=lambda _path: events.append("manifest") or SimpleNamespace(metadata={})),
    )
    monkeypatch.setattr(
        evaluator,
        "ensure_dataset_split_available",
        lambda *_args, **_kwargs: events.append("ensure"),
    )

    def catalog(*_args, **_kwargs):
        events.append("catalog")
        raise RuntimeError("stop after catalog entry")

    monkeypatch.setattr(evaluator, "catalog_mot_dataset_for_evaluation", catalog)

    with pytest.raises(RuntimeError, match="stop after catalog entry"):
        evaluator.eval_setup(SimpleNamespace(data_root=tmp_path, build="fixture", build_root=None))

    assert events == ["build", "manifest", "ensure", "catalog"]
