from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from boxmot.reid.datasets.base import DatasetSplit, ReIDSample
from boxmot.reid.training.provenance import (
    anatomical_metadata_provenance,
    build_run_provenance,
    checkpoint_pretrained_provenance,
    dataset_manifest,
    model_pretrained_provenance,
    restore_model_pretrained_provenance,
)


def _dataset(tmp_path):
    image = tmp_path / "bounding_box_train" / "0001_c1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image-v1")
    sample = ReIDSample(str(image), pid=0, camid=1, source="market1501")
    return SimpleNamespace(
        root=tmp_path,
        train=DatasetSplit([sample]),
        query=DatasetSplit([]),
        gallery=DatasetSplit([]),
    ), image


def test_dataset_manifest_covers_split_membership_and_file_size(tmp_path) -> None:
    dataset, image = _dataset(tmp_path)
    first = dataset_manifest(dataset)
    assert first["splits"] == {"train": 1, "query": 0, "gallery": 0}
    assert first["schema"] == "reid-split-path-pid-camid-size-v1"
    assert len(first["sha256"]) == 64
    assert dataset_manifest(dataset) == first

    image.write_bytes(b"image-version-two")
    assert dataset_manifest(dataset)["sha256"] != first["sha256"]


def _anatomical_metadata(root, *, manifest_indent=None) -> None:
    person_mask = root / "person" / "0001.png"
    accessory_mask = root / "bags" / "0001.png"
    person_mask.parent.mkdir(parents=True, exist_ok=True)
    accessory_mask.parent.mkdir(parents=True, exist_ok=True)
    person_mask.write_bytes(b"person-mask-v1")
    accessory_mask.write_bytes(b"bag-mask-v1")
    payload = {
        "images": {
            "bounding_box_train/0001.jpg": {
                "keypoints": [[0.0, 0.0, 1.0]] * 17,
                "person_mask": "person/0001.png",
                "bag_mask": "bags/0001.png",
            }
        }
    }
    (root / "metadata.json").write_text(
        json.dumps(payload, indent=manifest_indent),
        encoding="utf-8",
    )


def test_anatomical_metadata_provenance_binds_paths_and_all_asset_bytes(
    tmp_path,
) -> None:
    first_root = tmp_path / "metadata-a"
    second_root = tmp_path / "metadata-b"
    external_masks = tmp_path / "external"
    _anatomical_metadata(first_root)
    _anatomical_metadata(second_root)
    external_masks.mkdir()
    (external_masks / "0001.png").write_bytes(b"external-mask-v1")

    first = anatomical_metadata_provenance(first_root, external_masks)
    repeated = anatomical_metadata_provenance(first_root, external_masks)
    moved = anatomical_metadata_provenance(second_root, external_masks)

    assert repeated == first
    assert first["schema"] == "anatomical-metadata-content-v1"
    assert len(first["sha256"]) == 64
    assert len(first["manifest_sha256"]) == 64
    assert first["manifest_valid"] is True
    assert first["referenced_asset_count"] == 2
    assert first["missing_referenced_asset_count"] == 0
    assert first["external_person_mask_count"] == 1
    assert moved["sha256"] != first["sha256"]

    _anatomical_metadata(first_root, manifest_indent=2)
    manifest_changed = anatomical_metadata_provenance(
        first_root,
        external_masks,
    )
    assert manifest_changed["manifest_sha256"] != first["manifest_sha256"]
    assert manifest_changed["sha256"] != first["sha256"]

    (first_root / "person" / "0001.png").write_bytes(
        b"person-mask-v2"
    )
    referenced_asset_changed = anatomical_metadata_provenance(
        first_root,
        external_masks,
    )
    assert referenced_asset_changed["sha256"] != manifest_changed["sha256"]

    (external_masks / "0001.png").write_bytes(b"external-mask-v2")
    external_asset_changed = anatomical_metadata_provenance(
        first_root,
        external_masks,
    )
    assert (
        external_asset_changed["sha256"]
        != referenced_asset_changed["sha256"]
    )


def test_pretrained_provenance_reports_verified_coverage() -> None:
    model = SimpleNamespace(
        pretrained_url="https://example.test/tinyvit.pth",
        pretrained_sha256="a" * 64,
        pretrained_backbone_required_tensor_count=292,
        pretrained_backbone_matched_tensor_count=292,
        pretrained_backbone_tensor_coverage=1.0,
        pretrained_backbone_required_numel=5_000_000,
        pretrained_backbone_matched_numel=5_000_000,
        pretrained_backbone_numel_coverage=1.0,
    )
    assert model_pretrained_provenance(model) == {
        "url": "https://example.test/tinyvit.pth",
        "sha256": "a" * 64,
        "required_tensor_count": 292,
        "matched_tensor_count": 292,
        "tensor_coverage": 1.0,
        "required_numel": 5_000_000,
        "matched_numel": 5_000_000,
        "numel_coverage": 1.0,
    }


def test_pretrained_provenance_restore_rejects_invalid_or_conflicting_records() -> None:
    model = SimpleNamespace()
    invalid = {
        "url": "https://example.test/tinyvit.pth",
        "sha256": "not-a-digest",
    }
    with pytest.raises(ValueError, match="64 hexadecimal"):
        restore_model_pretrained_provenance(model, invalid)
    assert not hasattr(model, "pretrained_url")

    valid = {
        "url": "https://example.test/tinyvit.pth",
        "sha256": "a" * 64,
        "required_tensor_count": 2,
        "matched_tensor_count": 2,
        "tensor_coverage": 1.0,
        "required_numel": 4,
        "matched_numel": 4,
        "numel_coverage": 1.0,
    }
    with pytest.raises(ValueError, match="conflicting"):
        checkpoint_pretrained_provenance(
            {
                "pretrained": valid,
                "model": {"pretrained": {**valid, "sha256": "b" * 64}},
            }
        )


def test_run_provenance_includes_source_data_and_pretrained(tmp_path) -> None:
    dataset, _ = _dataset(tmp_path)
    anatomical_metadata = {
        "schema": "anatomical-metadata-content-v1",
        "sha256": "b" * 64,
    }
    provenance = build_run_provenance(
        SimpleNamespace(),
        dataset,
        anatomical_metadata=anatomical_metadata,
    )
    assert provenance["pretrained"] is None
    assert provenance["anatomical_metadata"] == anatomical_metadata
    assert provenance["dataset_manifest"]["splits"]["train"] == 1
    assert len(provenance["source"]["reid_code_sha256"]) == 64
    assert len(provenance["source"]["uv_lock_sha256"]) == 64
    assert provenance["source"]["torch"]
    assert provenance["executable"]
