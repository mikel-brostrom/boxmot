"""Reuse complete perception outputs across unrelated package releases."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from boxmot import __version__
from boxmot.datasets import DatasetManifest
from boxmot.datasets.schema import MANIFEST_FILENAME, SUCCESS_FILENAME
from boxmot.detectors import DetectorSpec
from boxmot.engine.materialization import BuildPlan, SourceSample, StagePlan, fingerprint
from boxmot.engine.materialization import builds as builds_module
from boxmot.engine.materialization import plan as plan_module
from boxmot.reid import ReIDEncoderSpec
from tests.unit.engine.materialization.test_detection_cache import (
    CountingDetector,
    CountingEncoder,
    _derived_plan,
    _run_derived_build,
    _samples,
)


@dataclass
class PublishedCase:
    """Actual immutable artifacts plus the equivalent current-release plan."""

    output: Path
    previous: BuildPlan
    requested: BuildPlan
    samples: tuple[SourceSample, ...]
    detector: CountingDetector
    encoder: CountingEncoder
    encoder_fingerprint: str


def _plan_with_version(
    build_root: Path,
    version: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: str | None = None,
    changed_component: str | None = None,
    changed_field: str | None = None,
) -> tuple[BuildPlan, str]:
    """Vary creation provenance while preserving source and perception inputs."""

    with monkeypatch.context() as patch:
        patch.setattr(plan_module, "__version__", version)
        plan, _ = _derived_plan(build_root)
    components = {
        "detector": dict(plan.stages[0].component),
        "reid": dict(plan.stages[1].component),
        "segmentor": None,
    }
    if device is not None:
        for name, spec_type in (("detector", DetectorSpec), ("reid", ReIDEncoderSpec)):
            artifact = {
                "path": f"/shared/{name}.pt",
                "sha256": fingerprint(name),
                "uri": f"https://example.test/{name}.pt",
            }
            spec = asdict(
                spec_type(
                    "fixture",
                    artifact=artifact["path"],
                    artifact_sha256=artifact["sha256"],
                    device=device,
                    precision="fp32",
                    preprocessing="default",
                )
            )
            if name == changed_component:
                if changed_field == "precision":
                    spec["precision"] = "fp16"
                elif changed_field == "preprocessing":
                    spec["preprocessing"] = "different-preprocessing"
                elif changed_field == "model":
                    artifact["sha256"] = fingerprint("different-model")
                    spec["artifact_sha256"] = artifact["sha256"]
            components[name] = {"artifact": artifact, "spec": spec}
    stages: list[StagePlan] = []
    by_name: dict[str, StagePlan] = {}
    for original in plan.stages:
        component = (
            components[{"detect": "detector", "embed": "reid"}[original.name]] if original.name != "finalize" else {}
        )
        stage = StagePlan.create(
            original.name,
            config=original.config,
            component=component,
            upstream_fingerprints=tuple(by_name[name].fingerprint for name in original.depends_on),
            depends_on=original.depends_on,
            batch_size=original.batch_size,
        )
        stages.append(stage)
        by_name[stage.name] = stage
    metadata = {
        **plan.metadata,
        "boxmot_version": version,
        "dataset_id": plan.dataset_name,
        "split": "test",
        "source_catalog_digest": plan.source_fingerprint,
        "class_taxonomy_digest": fingerprint("taxonomy"),
        "class_bridge": [{"dataset_id": 0, "detector_id": 2, "name": "person"}],
        "components": components,
        "component_fingerprints": {
            name: fingerprint(component) if component is not None else None for name, component in components.items()
        },
        "fps": None,
        "source_root_uri": "file:///original/frames",
        "experiment_config": "/original/experiment.yaml",
    }
    with monkeypatch.context() as patch:
        patch.setattr(plan_module, "__version__", version)
        rebuilt = BuildPlan.create(
            build_root=build_root,
            dataset_name=plan.dataset_name,
            box_type=plan.box_type,
            source_fingerprint=plan.source_fingerprint,
            publish=plan.publish,
            stages=tuple(stages),
            metadata=metadata,
        )
    return rebuilt, fingerprint(components["reid"])


@pytest.fixture
def published_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PublishedCase:
    """Publish tiny but fully validated detection and embedding artifacts."""

    root = tmp_path / "materializations"
    previous, encoder_fingerprint = _plan_with_version(root, "0.1.0", monkeypatch)
    samples = _samples(tmp_path)
    detector = CountingDetector()
    encoder = CountingEncoder(1.0)
    output = _run_derived_build(previous, samples, detector, encoder, encoder_fingerprint, use_cache=False)
    requested, _ = _plan_with_version(root, __version__, monkeypatch)
    assert previous.build_id != requested.build_id
    return PublishedCase(output, previous, requested, samples, detector, encoder, encoder_fingerprint)


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Record publication contents and modification times without changing them."""

    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_matching_build_reuses_previous_release_without_rewriting(published_case: PublishedCase) -> None:
    case = published_case
    before = _snapshot(case.output)
    calls = case.detector.calls, case.encoder.calls
    messages: list[str] = []

    output = builds_module.find_matching_build(case.requested, status_callback=messages.append)

    assert output == case.output
    assert _snapshot(case.output) == before
    assert (case.detector.calls, case.encoder.calls) == calls
    assert not case.requested.output_root.exists()
    assert not case.requested.staging_root.exists()
    assert DatasetManifest.load(output).metadata["boxmot_version"] == "0.1.0"
    assert messages


def test_matching_build_ignores_machine_local_provenance(published_case: PublishedCase) -> None:
    case = published_case
    requested = replace(
        case.requested,
        metadata={
            **case.requested.metadata,
            "source_root_uri": "file:///relocated/frames",
            "experiment_config": "/relocated/experiment.yaml",
        },
    )

    assert builds_module.find_matching_build(requested) == case.output


@pytest.mark.parametrize(
    ("published_device", "requested_device"),
    [("cpu", "mps"), ("cpu", "cuda:0"), ("mps", "cpu"), ("mps", "cuda:0"), ("cuda:0", "cpu"), ("cuda:0", "mps")],
)
def test_matching_build_reuses_all_outputs_across_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published_device: str, requested_device: str
) -> None:
    """Real artifacts retain their original device provenance and row identifiers."""

    root = tmp_path / "materializations"
    previous, encoder_fingerprint = _plan_with_version(root, "0.1.0", monkeypatch, device=published_device)
    output = _run_derived_build(
        previous, _samples(tmp_path), CountingDetector(), CountingEncoder(1.0), encoder_fingerprint, use_cache=False
    )
    requested, _ = _plan_with_version(root, __version__, monkeypatch, device=requested_device)
    before = _snapshot(output)
    assert previous.build_id != requested.build_id
    assert all(first.fingerprint != second.fingerprint for first, second in zip(previous.stages, requested.stages))
    assert previous.metadata["component_fingerprints"] != requested.metadata["component_fingerprints"]

    assert builds_module.find_matching_build(requested) == output

    assert _snapshot(output) == before
    assert not requested.output_root.exists()
    manifest = DatasetManifest.load(output)
    assert manifest.counts["embeddings"] == 2
    assert manifest.metadata["components"]["detector"]["spec"]["device"] == published_device
    assert manifest.metadata["components"]["reid"]["spec"]["device"] == published_device


@pytest.mark.parametrize("component", ["detector", "reid"])
@pytest.mark.parametrize("field", ["precision", "preprocessing", "model"])
def test_cross_device_reuse_preserves_model_and_numerical_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str, field: str
) -> None:
    """Device-neutral matching must not hide meaningful model or preprocessing changes."""

    root = tmp_path / "materializations"
    previous, encoder_fingerprint = _plan_with_version(root, "0.1.0", monkeypatch, device="cpu")
    _run_derived_build(
        previous, _samples(tmp_path), CountingDetector(), CountingEncoder(1.0), encoder_fingerprint, use_cache=False
    )
    requested, _ = _plan_with_version(
        root,
        __version__,
        monkeypatch,
        device="mps",
        changed_component=component,
        changed_field=field,
    )

    assert builds_module.find_matching_build(requested) is None


def test_existing_exact_output_remains_authoritative(
    published_case: PublishedCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = published_case
    case.requested.output_root.mkdir()

    def reject_discovery(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("An existing exact output must remain the materializer's responsibility.")

    monkeypatch.setattr(builds_module.DatasetManifest, "load", reject_discovery)

    assert builds_module.find_matching_build(case.requested) is None


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "dataset_name",
        "geometry",
        "images",
        "embeddings",
        "masks",
        "detector",
        "reid",
        "stage_config",
        "stage_fingerprint",
        "batch_size",
        "stage_inputs",
    ],
)
def test_matching_build_rejects_changed_perception_contract(published_case: PublishedCase, change: str) -> None:
    requested = published_case.requested
    if change == "source":
        requested = replace(requested, source_fingerprint=fingerprint("different-source"))
    elif change == "dataset_name":
        requested = replace(requested, dataset_name="different-dataset")
    elif change == "geometry":
        requested = replace(requested, box_type="obb")
    elif change in {"images", "embeddings", "masks"}:
        key = "image_references" if change == "images" else change
        publish = replace(requested.publish, **{key: not getattr(requested.publish, key)})
        requested = replace(requested, publish=publish)
    else:
        stages = list(requested.stages)
        index = 1 if change in {"reid", "stage_inputs"} else 0
        stage = stages[index]
        if change in {"detector", "reid"}:
            stage = replace(stage, component={**stage.component, "model": "different-model"})
        elif change == "stage_config":
            stage = replace(stage, config={**stage.config, "class_id_map": {"2": 1}})
        elif change == "stage_fingerprint":
            stage = replace(stage, fingerprint=fingerprint("different-preprocessing"))
        elif change == "batch_size":
            stage = replace(stage, batch_size=stage.batch_size + 1)
        else:
            stage = replace(stage, depends_on=())
        stages[index] = stage
        requested = replace(requested, stages=tuple(stages))

    assert builds_module.find_matching_build(requested) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("experiment_id", "different-experiment"),
        ("dataset_id", "different-dataset"),
        ("split", "train"),
        ("source_catalog_digest", "1" * 64),
        ("class_taxonomy_digest", "2" * 64),
        ("class_bridge", [{"dataset_id": 1, "detector_id": 2, "name": "person"}]),
        ("component_fingerprints", {"detector": "3" * 64, "reid": "4" * 64, "segmentor": None}),
        ("fps", 5.0),
    ],
)
def test_matching_build_rejects_changed_dataset_and_experiment_metadata(
    published_case: PublishedCase, key: str, value: object
) -> None:
    requested = replace(
        published_case.requested,
        metadata={**published_case.requested.metadata, key: value},
    )

    assert builds_module.find_matching_build(requested) is None


@pytest.mark.parametrize(
    "corruption",
    ["samples", "instances", "embeddings", "manifest", "marker", "unpublished", "wrong_directory"],
)
def test_matching_build_skips_invalid_publications(published_case: PublishedCase, corruption: str) -> None:
    case = published_case
    if corruption in {"samples", "instances", "embeddings"}:
        manifest = DatasetManifest.load(case.output)
        shard = case.output / manifest.artifact(corruption).shards[0].path
        contents = bytearray(shard.read_bytes())
        contents[len(contents) // 2] ^= 1
        shard.write_bytes(contents)
    elif corruption == "manifest":
        (case.output / MANIFEST_FILENAME).write_text("{", encoding="utf-8")
    elif corruption == "marker":
        (case.output / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")
    elif corruption == "unpublished":
        (case.output / SUCCESS_FILENAME).unlink()
    else:
        case.output.rename(case.output.parent / ("0" * 64))

    assert builds_module.find_matching_build(case.requested) is None


def test_matching_build_skips_symlink_candidate(published_case: PublishedCase, tmp_path: Path) -> None:
    case = published_case
    external = tmp_path / "external" / case.output.name
    external.parent.mkdir()
    case.output.rename(external)
    case.output.symlink_to(external, target_is_directory=True)

    assert builds_module.find_matching_build(case.requested) is None


def test_matching_build_uses_path_order_independent_of_mtime(
    published_case: PublishedCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = published_case
    another, encoder_fingerprint = _plan_with_version(case.output.parent, "0.2.0", monkeypatch)
    other = _run_derived_build(
        another, case.samples, CountingDetector(), CountingEncoder(1.0), encoder_fingerprint, use_cache=False
    )
    first, second = sorted((case.output, other))

    for first_time, second_time in ((1.0, 2.0), (2.0, 1.0)):
        os.utime(first, (first_time, first_time))
        os.utime(second, (second_time, second_time))

        assert builds_module.find_matching_build(case.requested) == first


def test_matching_build_does_not_search_outside_explicit_root(
    published_case: PublishedCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = published_case
    selected = tmp_path / "selected"
    selected.mkdir()
    requested = replace(case.requested, build_root=selected)
    monkeypatch.setattr(builds_module, "former_default_build_root", lambda: case.output.parent)

    assert builds_module.find_matching_build(requested) is None
    shutil.copytree(case.output, selected / case.output.name)
    assert builds_module.find_matching_build(requested) == selected / case.output.name
