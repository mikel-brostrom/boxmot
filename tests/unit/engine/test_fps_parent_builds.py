"""Workflow discovery of complete, compatible native-FPS perception builds."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from boxmot.datasets import DatasetManifest
from boxmot.datasets.schema import MANIFEST_FILENAME, SUCCESS_FILENAME
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan
from boxmot.engine.materialization.builds import find_fps_parent_build
from boxmot.engine.materialization.catalog import catalog_mot_dataset
from tests.unit.engine.test_dataset_fps_workflow import _materialize
from tests.unit.engine.test_dataset_fps_workflow import fps_case as fps_case


@pytest.fixture
def parent_case(fps_case: SimpleNamespace) -> SimpleNamespace:
    """Publish actual native perception and construct its canonical FPS plan."""

    parent = _materialize(fps_case, None)
    manifest = DatasetManifest.load(parent)
    native_catalog = catalog_mot_dataset(fps_case.dataset, split="ablation", data_root=fps_case.data_root)
    sampled_catalog = catalog_mot_dataset(fps_case.dataset, split="ablation", data_root=fps_case.data_root, fps=5.0)
    stages = tuple(
        StagePlan(
            name=stage.name,
            fingerprint=stage.fingerprint,
            depends_on=stage.inputs,
            batch_size=stage.batch_size,
            config=stage.config,
            component=stage.component,
        )
        for stage in manifest.stages
    )
    plan = BuildPlan.create(
        build_root=parent.parent,
        dataset_name="fixture",
        box_type="aabb",
        source_fingerprint=sampled_catalog.fingerprint,
        publish=PublishOptions(image_references=True, masks=False, embeddings=True),
        stages=stages,
        metadata={**manifest.metadata, **sampled_catalog.metadata},
    )
    return SimpleNamespace(parent=parent, manifest=manifest, catalog=native_catalog, plan=plan, case=fps_case)


@pytest.mark.parametrize("corruption", ["shard", "success_marker", "manifest"])
def test_corrupt_fps_parent_is_skipped_and_workflow_falls_back(parent_case: SimpleNamespace, corruption: str) -> None:
    """Published corruption cannot silently enter the sampled detection cache."""

    parent = parent_case.parent
    if corruption == "shard":
        shard = parent_case.manifest.artifacts_by_name["embeddings"].shards[0]
        with (parent / shard.path).open("ab") as stream:
            stream.write(b"corrupt")
    elif corruption == "success_marker":
        (parent / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")
    else:
        (parent / MANIFEST_FILENAME).write_text("{", encoding="utf-8")

    assert find_fps_parent_build(parent_case.plan, load_catalog=lambda: parent_case.catalog) is None

    parent_case.case.detector.seen.clear()
    parent_case.case.encoder.seen.clear()
    build = _materialize(parent_case.case, 5.0)
    assert build == parent_case.plan.output_root
    assert parent_case.case.detector.seen == [1, 7, 13, 19]
    assert parent_case.case.encoder.seen == [1, 13, 19]
    assert DatasetManifest.load(build).metadata["fps"] == 5.0


def test_unpublished_fps_parent_is_skipped_before_loading_native_catalog(parent_case: SimpleNamespace) -> None:
    """Incomplete native builds incur neither a source scan nor reuse."""

    (parent_case.parent / SUCCESS_FILENAME).unlink()
    load_catalog = Mock(side_effect=AssertionError("Incomplete candidates must not load the native catalog."))

    assert find_fps_parent_build(parent_case.plan, load_catalog=load_catalog) is None

    load_catalog.assert_not_called()


def test_existing_sampled_build_bypasses_native_catalog_discovery(parent_case: SimpleNamespace) -> None:
    """Repeating an exact FPS build does not rescan the native source."""

    sampled = _materialize(parent_case.case, 5.0)
    assert sampled == parent_case.plan.output_root
    load_catalog = Mock(side_effect=AssertionError("An existing sampled build must bypass native discovery."))

    assert find_fps_parent_build(parent_case.plan, load_catalog=load_catalog) is None

    load_catalog.assert_not_called()


def test_fps_parent_selection_uses_path_order_independent_of_mtime(parent_case: SimpleNamespace) -> None:
    """Equivalent perception candidates resolve deterministically across runs."""

    parent_case.case.experiment["id"] = "another-fixture-experiment"
    other = _materialize(parent_case.case, None)
    first, second = sorted((parent_case.parent, other))
    assert first != second
    load_catalog = Mock(return_value=parent_case.catalog)
    for first_time, second_time in ((1.0, 2.0), (2.0, 1.0)):
        os.utime(first, (first_time, first_time))
        os.utime(second, (second_time, second_time))

        result = find_fps_parent_build(parent_case.plan, load_catalog=load_catalog)

        assert result is not None
        assert result[0] == first
        assert result[1] is parent_case.catalog
    assert load_catalog.call_count == 2


def test_native_source_mismatch_prevents_fps_parent_reuse(parent_case: SimpleNamespace) -> None:
    """Even a changed unselected frame invalidates the native source identity."""

    case = parent_case.case
    unselected_image: Path = case.split_root / "sequence" / "img1" / "000002.png"
    assert cv2.imwrite(str(unselected_image), np.full((8, 10, 3), 42, dtype=np.uint8))
    changed_catalog = catalog_mot_dataset(case.dataset, split="ablation", data_root=case.data_root)
    assert changed_catalog.fingerprint != parent_case.catalog.fingerprint

    assert find_fps_parent_build(parent_case.plan, load_catalog=lambda: changed_catalog) is None

    case.detector.seen.clear()
    case.encoder.seen.clear()
    _materialize(case, 5.0)
    assert case.detector.seen == [1, 7, 13, 19]
    assert case.encoder.seen == [1, 13, 19]
