"""Focused tests for the offline privileged-cache entrypoint."""

from __future__ import annotations

import json

import pytest
import torch

from boxmot.engine.reid.privileged_cache import (
    build_privileged_cache,
    export_dataset_index,
    load_dataset_index,
    main,
    validate_privileged_cache,
)
from boxmot.reid.datasets import build_dataset
from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.training.trainer_components.privileged_graph import (
    CACHE_VERSION,
    PrivilegedGraphTeacherCache,
    dataset_samples_sha256,
    sha256_file,
)

PART_NAMES = ("upper", "lower")


def _dataset_rows() -> list[dict[str, int | str]]:
    return [
        {"index": 20, "img_path": "train/person_2/../person_2/b.jpg", "pid": 2, "camid": 1},
        {"index": 10, "img_path": "train/person_1/a.jpg", "pid": 1, "camid": 0},
        {"index": 30, "img_path": "train/person_3/c.jpg", "pid": 3, "camid": 2},
        {"index": 40, "img_path": "train/person_4/d.jpg", "pid": 4, "camid": 3},
    ]


def _teacher_tensors(*, include_indices: bool = False) -> dict[str, torch.Tensor]:
    global_descriptors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
        ]
    )
    values = {
        "global_descriptors": global_descriptors,
        "part_descriptors": torch.stack((global_descriptors, global_descriptors.roll(1, dims=1)), dim=1),
        "part_visibility": torch.tensor([[1.0, 0.8], [1.0, 0.7], [0.9, 1.0], [0.7, 0.6]]),
        "part_confidence": torch.tensor([[0.9, 0.7], [0.8, 0.6], [0.8, 0.9], [0.6, 0.5]]),
        "global_confidence": torch.tensor([0.9, 0.8, 0.9, 0.7]),
        "leave_part_out_descriptors": torch.stack(
            (global_descriptors.roll(1, dims=1), global_descriptors.roll(2, dims=1)),
            dim=1,
        ),
    }
    if include_indices:
        values["sample_indices"] = torch.tensor([20, 10, 30, 40])
    return values


def _write_inputs(tmp_path, *, include_indices: bool = False):
    dataset_index = tmp_path / "train-samples.json"
    dataset_index.write_text(
        json.dumps({"schema": "boxmot_reid_dataset_index_v1", "samples": _dataset_rows()}),
        encoding="utf-8",
    )
    teacher_provenance = tmp_path / "teacher.json"
    teacher_provenance.write_text('{"teacher":"offline-pose-parser-v1"}', encoding="utf-8")
    tensor_input = tmp_path / "teacher-signals.pt"
    torch.save(
        {
            "part_names": list(PART_NAMES),
            "tensors": _teacher_tensors(include_indices=include_indices),
        },
        tensor_input,
    )
    return tensor_input, dataset_index, teacher_provenance


def test_dataset_samples_hash_is_canonical_and_matches_in_memory_samples() -> None:
    explicit_rows = _dataset_rows()
    normalized_reordered = [
        {"index": 40, "img_path": "train\\person_4\\d.jpg", "pid": 4, "camid": 3},
        {"index": 30, "img_path": "train/person_3/./c.jpg", "pid": 3, "camid": 2},
        {"index": 10, "img_path": "train/person_1/a.jpg", "pid": 1, "camid": 0},
        {"index": 20, "img_path": "train/person_2/b.jpg", "pid": 2, "camid": 1},
    ]
    in_memory = [
        ReIDSample(img_path="train/person_1/a.jpg", pid=1, camid=0),
        ReIDSample(img_path="train/person_2/b.jpg", pid=2, camid=1),
        ReIDSample(img_path="train/person_3/c.jpg", pid=3, camid=2),
        ReIDSample(img_path="train/person_4/d.jpg", pid=4, camid=3),
    ]
    explicit_contiguous = [
        {"index": index, "img_path": sample.img_path, "pid": sample.pid, "camid": sample.camid}
        for index, sample in enumerate(in_memory)
    ]

    assert dataset_samples_sha256(explicit_rows) == dataset_samples_sha256(normalized_reordered)
    assert dataset_samples_sha256(in_memory) == dataset_samples_sha256(explicit_contiguous)


def test_dataset_samples_hash_rejects_duplicate_stable_indices() -> None:
    rows = _dataset_rows()
    rows[1]["index"] = rows[0]["index"]

    with pytest.raises(ValueError, match="duplicate stable index"):
        dataset_samples_sha256(rows)


def test_index_export_exactly_reproduces_registered_training_rows(tmp_path) -> None:
    dataset_root = tmp_path / "market"
    for split in ("bounding_box_train", "bounding_box_test", "query"):
        (dataset_root / split).mkdir(parents=True)
    for filename in (
        "0007_c2s1_000001_00.jpg",
        "0042_c1s1_000002_00.jpg",
        "0007_c1s1_000003_00.jpg",
    ):
        (dataset_root / "bounding_box_train" / filename).touch()
    output = tmp_path / "train-samples.json"

    result = export_dataset_index(
        dataset_name="market1501",
        data_dir=dataset_root,
        output=output,
    )
    live_samples = build_dataset("market1501", str(dataset_root)).train.samples
    exported = load_dataset_index(output)

    assert result.sample_count == 3
    assert [row.index for row in exported] == [0, 1, 2]
    assert dataset_samples_sha256(exported) == dataset_samples_sha256(live_samples)
    assert result.dataset_sha256 == dataset_samples_sha256(live_samples)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_dataset_index(
            dataset_name="market1501",
            data_dir=dataset_root,
            output=output,
        )


def test_standalone_main_exports_registered_dataset_index(tmp_path, capsys) -> None:
    dataset_root = tmp_path / "market"
    for split in ("bounding_box_train", "bounding_box_test", "query"):
        (dataset_root / split).mkdir(parents=True)
    (dataset_root / "bounding_box_train" / "0007_c2s1_000001_00.jpg").touch()
    output = tmp_path / "train-samples.json"

    assert (
        main(
            [
                "index",
                "--dataset",
                "market1501",
                "--data-dir",
                str(dataset_root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_index"] == str(output)
    assert summary["sample_count"] == 1


def test_builder_emits_trainer_cache_bound_to_semantic_dataset_and_teacher(tmp_path) -> None:
    tensor_input, dataset_index, teacher_provenance = _write_inputs(tmp_path)
    output = tmp_path / "hpgrd-cache.pt"

    result = build_privileged_cache(
        tensor_input=tensor_input,
        dataset_index=dataset_index,
        teacher_provenance=teacher_provenance,
        output=output,
        extra={"teacher_name": "offline-pose-parser-v1"},
    )
    samples = load_dataset_index(dataset_index)
    loaded = PrivilegedGraphTeacherCache.load(
        output,
        expected_dataset_sha256=dataset_samples_sha256(samples),
        expected_teacher_sha256=sha256_file(teacher_provenance),
        expected_manifest_sha256=result.manifest["manifest_sha256"],
    )
    batch = loaded.lookup([10, 40, 20])

    assert result.output_path == output
    assert result.manifest["sample_count"] == 4
    assert result.manifest["part_count"] == 2
    assert result.manifest["version"] == CACHE_VERSION
    assert result.manifest["part_names"] == list(PART_NAMES)
    assert loaded.part_names == PART_NAMES
    assert result.manifest["extra"]["user"]["teacher_name"] == "offline-pose-parser-v1"
    assert result.manifest["extra"]["dataset_index_file_sha256"] == sha256_file(dataset_index)
    assert batch.sample_indices.tolist() == [10, 40, 20]


def test_builder_accepts_explicit_tensor_indices_in_a_different_dataset_row_order(tmp_path) -> None:
    tensor_input, dataset_index, teacher_provenance = _write_inputs(tmp_path, include_indices=True)
    output = tmp_path / "hpgrd-cache.pt"

    build_privileged_cache(
        tensor_input=tensor_input,
        dataset_index=dataset_index,
        teacher_provenance=teacher_provenance,
        output=output,
    )

    assert PrivilegedGraphTeacherCache.load(output).lookup([20, 10]).sample_indices.tolist() == [20, 10]


def test_builder_rejects_tensor_indices_not_present_in_dataset(tmp_path) -> None:
    tensor_input, dataset_index, teacher_provenance = _write_inputs(tmp_path, include_indices=True)
    signals = torch.load(tensor_input, map_location="cpu", weights_only=True)
    signals["tensors"]["sample_indices"][0] = 999
    torch.save(signals, tensor_input)

    with pytest.raises(ValueError, match="do not match the dataset index"):
        build_privileged_cache(
            tensor_input=tensor_input,
            dataset_index=dataset_index,
            teacher_provenance=teacher_provenance,
            output=tmp_path / "hpgrd-cache.pt",
        )


def test_validator_checks_semantic_dataset_and_teacher_provenance(tmp_path) -> None:
    tensor_input, dataset_index, teacher_provenance = _write_inputs(tmp_path)
    output = tmp_path / "hpgrd-cache.pt"
    built = build_privileged_cache(
        tensor_input=tensor_input,
        dataset_index=dataset_index,
        teacher_provenance=teacher_provenance,
        output=output,
    )

    validated = validate_privileged_cache(
        cache_path=output,
        dataset_index=dataset_index,
        teacher_provenance=teacher_provenance,
        expected_manifest_sha256=built.manifest["manifest_sha256"],
        expected_part_names=PART_NAMES,
        require_exact_index_file=True,
    )
    assert validated.summary()["valid"] is True

    rows = _dataset_rows()
    rows[0]["pid"] = 999
    dataset_index.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        validate_privileged_cache(
            cache_path=output,
            dataset_index=dataset_index,
            teacher_provenance=teacher_provenance,
        )


def test_builder_refuses_overwrite_without_explicit_opt_in(tmp_path) -> None:
    tensor_input, dataset_index, teacher_provenance = _write_inputs(tmp_path)
    output = tmp_path / "hpgrd-cache.pt"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_privileged_cache(
            tensor_input=tensor_input,
            dataset_index=dataset_index,
            teacher_provenance=teacher_provenance,
            output=output,
        )


def test_standalone_main_builds_and_validates_without_loading_a_teacher(tmp_path, capsys) -> None:
    tensor_input, dataset_index, teacher_provenance = _write_inputs(tmp_path)
    output = tmp_path / "hpgrd-cache.pt"

    assert (
        main(
            [
                "build",
                "--tensor-input",
                str(tensor_input),
                "--dataset-index",
                str(dataset_index),
                "--teacher-provenance",
                str(teacher_provenance),
                "--part-names",
                *PART_NAMES,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    build_summary = json.loads(capsys.readouterr().out)
    assert build_summary["cache"] == str(output)
    assert build_summary["part_names"] == list(PART_NAMES)

    assert (
        main(
            [
                "validate",
                "--cache",
                str(output),
                "--dataset-index",
                str(dataset_index),
                "--teacher-provenance",
                str(teacher_provenance),
                "--manifest-sha256",
                build_summary["manifest_sha256"],
                "--part-names",
                *PART_NAMES,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_builder_requires_or_checks_exact_ordered_part_names(tmp_path) -> None:
    tensor_input, dataset_index, teacher_provenance = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="do not match"):
        build_privileged_cache(
            tensor_input=tensor_input,
            dataset_index=dataset_index,
            teacher_provenance=teacher_provenance,
            output=tmp_path / "wrong-order.pt",
            part_names=tuple(reversed(PART_NAMES)),
        )

    raw_input = tmp_path / "raw-signals.pt"
    torch.save(_teacher_tensors(), raw_input)
    with pytest.raises(ValueError, match="part names are required"):
        build_privileged_cache(
            tensor_input=raw_input,
            dataset_index=dataset_index,
            teacher_provenance=teacher_provenance,
            output=tmp_path / "unnamed.pt",
        )
    built = build_privileged_cache(
        tensor_input=raw_input,
        dataset_index=dataset_index,
        teacher_provenance=teacher_provenance,
        output=tmp_path / "explicit.pt",
        part_names=PART_NAMES,
    )
    assert built.manifest["part_names"] == list(PART_NAMES)
