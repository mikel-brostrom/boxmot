from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

import boxmot.engine.materialization.metadata_cache as metadata_cache_module
import boxmot.engine.materialization.plan as plan_module
import boxmot.engine.materialization.source as source_module
import boxmot.engine.materialization.stages.detect as detect_stage_module
import boxmot.engine.materialization.stages.embed as embed_stage_module
import boxmot.engine.materialization.stages.segment as segment_stage_module
import boxmot.engine.materialization.workflow as workflow
from boxmot import create_tracker
from boxmot.configs import CONFIG_ROOT
from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.datasets.manifest import sha256_file
from boxmot.detectors import DetectorCapabilities, DetectorSpec
from boxmot.engine.config.experiments import resolve_experiment_config
from boxmot.engine.eval.replay import iter_cached_tracks
from boxmot.engine.materialization import SourceSample, StagePlan, fingerprint
from boxmot.engine.materialization.catalog import SourceCatalog
from boxmot.reid import ReIDEncoderSpec
from boxmot.segmentors import SegmentorSpec
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes
from boxmot.trackers import TrackerSpec
from boxmot.utils.devices import resolve_device
from tests.unit.engine.test_dataset_fps_workflow import _materialize as materialize_fixture
from tests.unit.engine.test_dataset_fps_workflow import fps_case as fps_case


class _Detector:
    capabilities = DetectorCapabilities()

    def predict(self, frames):
        return [
            Detections(
                sample_id=frame.sample_id,
                geometry=Boxes(torch.tensor([[1.0, 2.0, 4.0, 6.0]], dtype=torch.float32)),
                scores=torch.tensor([0.75], dtype=torch.float32),
                class_ids=torch.tensor([2], dtype=torch.int64),
            )
            for frame in frames
        ]


@pytest.fixture(autouse=True)
def _isolate_source_metadata_cache(monkeypatch, tmp_path) -> None:
    """Keep workflow cache writes inside each test's temporary directory."""

    monkeypatch.setattr(
        workflow,
        "default_source_metadata_cache_path",
        lambda source: tmp_path / "source-metadata-cache" / f"{fingerprint(str(Path(source).resolve()))}.json",
    )


def _experiment_case(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Create one tiny catalog-backed experiment resolution."""

    data_root = tmp_path / "datasets"
    dataset_root = data_root / "fixture"
    source_path = dataset_root / "test" / "sequence" / "img1" / "000001.jpg"
    source_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(source_path), np.zeros((8, 10, 3), dtype=np.uint8))
    resolved = {
        "id": "fixture-test-detector",
        "source_path": tmp_path / "fixture-experiment.yaml",
        "dataset": {
            "id": "fixture",
            "root": "fixture",
            "split": "test",
            "layout": "mot",
            "box_type": "aabb",
            "splits": {
                "test": {
                    "path": "test",
                    "annotations": None,
                    "has_ground_truth": True,
                }
            },
            "classes": {"vehicle": {"id": 7, "evaluation": "target"}},
        },
        "detector": {"ref": "fixture-detector", "checkpoint": "default"},
        "segmentor": None,
        "reid": None,
        "evaluation": {
            "classes": [
                {
                    "name": "vehicle",
                    "dataset_id": 7,
                    "detector_name": "car",
                    "detector_id": 2,
                }
            ]
        },
    }
    return data_root, source_path, resolved


@pytest.mark.parametrize(
    "spec",
    (
        DetectorSpec(backend="fixture", device="cpu"),
        SegmentorSpec(backend="fixture", device="cpu"),
        ReIDEncoderSpec(backend="fixture", device="cpu"),
    ),
)
def test_device_override_updates_component_spec_and_fingerprint_provenance(spec) -> None:
    original_provenance = {"spec": {"backend": "fixture", "device": "cpu"}, "artifact": None}

    updated, provenance = workflow._with_device(spec, original_provenance, "mps")

    assert spec.device == "cpu"
    assert updated.device == "mps"
    assert provenance["spec"]["device"] == "mps"
    assert original_provenance["spec"]["device"] == "cpu"
    cpu_stage = StagePlan.create("detect", component=original_provenance)
    mps_stage = StagePlan.create("detect", component=provenance)
    assert cpu_stage.fingerprint != mps_stage.fingerprint


@pytest.mark.parametrize(
    "spec",
    (
        DetectorSpec(backend="fixture", device="auto"),
        SegmentorSpec(backend="fixture", device="auto"),
        ReIDEncoderSpec(backend="fixture", device="auto"),
    ),
)
def test_automatic_component_device_resolves_to_command_default(spec) -> None:
    provenance = {"spec": {"backend": "fixture", "device": "auto"}, "artifact": None}

    updated, updated_provenance = workflow._with_device(
        spec,
        provenance,
        None,
        auto_device="cpu",
    )

    assert updated.device == "cpu"
    assert updated_provenance["spec"]["device"] == "cpu"


@pytest.mark.parametrize("spec_type", [DetectorSpec, SegmentorSpec, ReIDEncoderSpec])
@pytest.mark.parametrize(
    ("value", "expected"),
    (("cpu", "cpu"), ("MPS", "mps"), ("0", "cuda:0"), ("cuda", "cuda:0"), ("cuda:02", "cuda:2")),
)
def test_authored_component_devices_are_canonical_in_runtime_and_fingerprints(
    spec_type, value: str, expected: str
) -> None:
    spec = spec_type("fixture", device=value)
    original = {"spec": asdict(spec), "artifact": None}

    updated, provenance = workflow._with_device(spec, original, None)
    canonical = {"spec": asdict(spec_type("fixture", device=expected)), "artifact": None}

    assert updated.device == expected
    assert provenance["spec"]["device"] == expected
    assert fingerprint(provenance) == fingerprint(canonical)
    assert original["spec"]["device"] == value


def test_only_an_explicit_cli_device_overrides_component_configuration() -> None:
    implicit = SimpleNamespace(device="cpu", materialize_explicit_keys=())
    explicit = SimpleNamespace(device="mps", materialize_explicit_keys=("device",))

    assert workflow._device_override(implicit) is None
    assert workflow._device_override(explicit) == "mps"


def test_invalid_device_selector_fails_before_source_cataloging(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "_resolved_inputs", lambda *args, **kwargs: pytest.fail("Unexpected cataloging"))

    with pytest.raises(ValueError, match="Unsupported device"):
        workflow.materialize(SimpleNamespace(device="cuda:bad", materialize_explicit_keys=("device",)))


@pytest.mark.parametrize("device_source", ["explicit", "authored"])
def test_unavailable_device_fails_only_when_missing_outputs_need_inference(
    monkeypatch: pytest.MonkeyPatch,
    fps_case: SimpleNamespace,  # noqa: F811 - imported pytest fixture
    device_source: str,
) -> None:
    """A real cache miss still checks accelerator availability before model execution."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    discovered: list[Path | None] = []
    actual_find = workflow.find_matching_build

    def find_matching(plan, **kwargs):
        result = actual_find(plan, **kwargs)
        discovered.append(result)
        return result

    def construct_detector(spec):
        assert discovered == [None]
        resolve_device(spec.device)
        pytest.fail("An unavailable accelerator must fail before model execution.")

    monkeypatch.setattr(workflow, "find_matching_build", find_matching)
    monkeypatch.setattr(detect_stage_module, "create_detector", construct_detector)
    options: dict[str, object] = {}
    if device_source == "authored":
        authored = DetectorSpec("fixture", device="cuda:2")
        monkeypatch.setattr(
            workflow, "resolve_detector_spec", lambda *args, **kwargs: (authored, {"spec": asdict(authored)})
        )
    else:
        options.update(device="cuda:2", materialize_explicit_keys=("device",))

    with pytest.raises(RuntimeError, match="cuda:2 is unavailable"):
        materialize_fixture(fps_case, None, **options)

    assert discovered == [None]
    assert fps_case.detector.seen == []
    assert fps_case.encoder.seen == []


def test_experiment_catalog_metadata_is_reused_and_identity_is_canonical(monkeypatch, tmp_path) -> None:
    data_root, source_path, resolved = _experiment_case(tmp_path)
    cache_path = tmp_path / "cache" / "metadata.json"
    monkeypatch.setattr(workflow, "default_source_metadata_cache_path", lambda _source: cache_path)
    resolutions = []

    def resolve_experiment(reference, **kwargs):
        resolutions.append((reference, kwargs))
        return resolved

    monkeypatch.setattr(workflow, "resolve_experiment_config", resolve_experiment)
    real_inspect = metadata_cache_module.inspect_catalog_file
    inspected = []

    def inspect(path, include_image_size):
        inspected.append((path, include_image_size))
        return real_inspect(path, include_image_size)

    monkeypatch.setattr(metadata_cache_module, "inspect_catalog_file", inspect)
    args = SimpleNamespace(
        experiment="fixture-test-detector",
        split="must-not-override-experiment",
        data_root=data_root,
    )
    messages = []

    first = workflow._resolved_inputs(args, status_callback=messages.append)
    second = workflow._resolved_inputs(args, status_callback=messages.append)

    dataset_name, geometry, catalog, detector, segmentor, reid, metadata = first
    assert dataset_name == "fixture"
    assert geometry == "aabb"
    assert detector == "fixture-detector/default"
    assert segmentor is None
    assert reid is None
    assert catalog.metadata["source_kind"] == "dataset"
    assert catalog.metadata["dataset_id"] == "fixture"
    assert catalog.samples[0].sample_id == "test:sequence:0"
    assert catalog.samples[0].sequence_id == "sequence"
    assert metadata["experiment_id"] == "fixture-test-detector"
    assert metadata["dataset_id"] == "fixture"
    assert metadata["split"] == "test"
    assert metadata["class_bridge"] == (
        {
            "name": "vehicle",
            "dataset_id": 7,
            "detector_name": "car",
            "detector_id": 2,
        },
    )
    assert first[2].fingerprint == second[2].fingerprint
    assert resolutions == [
        ("fixture-test-detector", {"mode": "materialize"}),
        ("fixture-test-detector", {"mode": "materialize"}),
    ]
    assert inspected == [(source_path, True)]
    assert any("1 cached, 0 refreshed" in message for message in messages)


@pytest.mark.parametrize("mode", ("materialize", "eval", "tune"))
def test_local_kitti_experiment_materializes_its_own_images_and_annotations(tmp_path: Path, mode: str) -> None:
    """A sibling dataset YAML must retain its local root through source cataloging."""
    dataset = yaml.safe_load((CONFIG_ROOT / "datasets/kitti-2d.yaml").read_text())
    dataset["storage"]["root"] = "."
    dataset["modalities"]["images"]["path"] = "frames/{sequence}"
    dataset["modalities"]["ground_truth"]["path"] = "annotations/{sequence}.txt"
    dataset["splits"] = {"val": {"partition": "training", "sequences": ["0002"], "has_ground_truth": True}}
    (tmp_path / "kitti-2d.yaml").write_text(yaml.safe_dump(dataset))
    image_path = tmp_path / "frames/0002/000000.png"
    image_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image_path), np.zeros((24, 32, 3), dtype=np.uint8))
    annotations = tmp_path / "annotations/0002.txt"
    annotations.parent.mkdir()
    annotations.write_text("0 1 Car 0 0 -10 1 2 10 20 -1 -1 -1 -1000 -1000 -1000 -10\n")
    experiment = yaml.safe_load((CONFIG_ROOT / "experiments/kitti-2d/val-yolo26n-osnet.yaml").read_text())
    experiment["dataset"]["ref"] = "kitti-2d.yaml"
    experiment_path = tmp_path / "local-kitti.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment))

    _, geometry, catalog, detector, _, reid, metadata = workflow._resolved_inputs(
        SimpleNamespace(experiment=experiment_path, materialize_mode=mode)
    )

    assert catalog.source_root == tmp_path.resolve()
    assert len(catalog.samples) == 1
    assert catalog.samples[0].source_uri == image_path.as_uri()
    assert catalog.samples[0].image_ref == "frames/0002/000000.png"
    assert geometry == "aabb" and detector == "yolo26n/default"
    assert reid is not None
    assert metadata["class_bridge"][1]["detector_name"] == "person"


def test_eval_owned_materialization_split_is_forwarded_explicitly(monkeypatch, tmp_path) -> None:
    data_root, _source_path, resolved = _experiment_case(tmp_path)
    resolutions = []

    def resolve_experiment(reference, **kwargs):
        resolutions.append((reference, kwargs))
        return resolved

    monkeypatch.setattr(workflow, "resolve_experiment_config", resolve_experiment)

    workflow._resolved_inputs(
        SimpleNamespace(
            experiment="fixture-test-detector",
            materialize_split="ablation",
            materialize_mode="eval",
            data_root=data_root,
        )
    )

    assert resolutions == [
        (
            "fixture-test-detector",
            {"mode": "eval", "split": "ablation"},
        )
    ]


def test_workflow_reuses_experiment_catalog_digest_and_publishes_canonical_ids(monkeypatch, tmp_path) -> None:
    data_root, source_path, resolved = _experiment_case(tmp_path)
    cache_path = tmp_path / "cache" / "metadata.json"
    received_roots = []

    def cache_path_for(root):
        received_roots.append(root)
        return cache_path

    detector_spec = DetectorSpec(backend="fixture", geometry_mode="aabb")
    monkeypatch.setattr(workflow, "default_source_metadata_cache_path", cache_path_for)
    monkeypatch.setattr(workflow, "resolve_experiment_config", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (
            detector_spec,
            {"spec": {"backend": "fixture", "geometry_mode": geometry}, "artifact": None},
        ),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    monkeypatch.setattr(detect_stage_module, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(detect_stage_module, "create_detector", lambda _spec: _Detector())
    monkeypatch.setattr(
        source_module,
        "sha256_file",
        lambda _path: pytest.fail("pending decode must reuse the stat-valid catalog digest"),
    )

    output = workflow.materialize(
        SimpleNamespace(
            experiment="fixture-test-detector",
            split=None,
            data_root=data_root,
            device="cpu",
            materialize_explicit_keys=(),
            build_root=tmp_path / "builds",
            plan_path=None,
            plan_overrides=(),
            publish_image_refs=True,
            publish_masks=False,
            publish_embeddings=False,
            resume=True,
        )
    )

    assert output.is_dir()
    assert received_roots == [data_root / "fixture", data_root / "fixture"]
    manifest = DatasetManifest.load(output)
    dataset = CachedVisionDataset(output)
    assert manifest.metadata["experiment_id"] == "fixture-test-detector"
    assert manifest.metadata["dataset_id"] == "fixture"
    assert manifest.metadata["split"] == "test"
    detect_stage = next(stage for stage in manifest.stages if stage.name == "detect")
    assert detect_stage.config["class_id_map"] == {"2": 7}
    assert dataset[0].sample_id == "test:sequence:0"
    assert dataset[0].sequence_id == "sequence"
    assert dataset[0].detections.class_ids.tolist() == [7]


def test_workflow_reuses_detector_cache_across_derived_experiments(monkeypatch, tmp_path) -> None:
    data_root, _source_path, resolved = _experiment_case(tmp_path)
    experiments = {
        "fixture-test-detector": resolved,
        "fixture-test-detector-alt": {**resolved, "id": "fixture-test-detector-alt"},
    }
    detector_spec = DetectorSpec(backend="fixture", geometry_mode="aabb")
    detector = _Detector()
    predict_calls = 0
    original_predict = detector.predict

    def predict(frames):
        nonlocal predict_calls
        predict_calls += 1
        return original_predict(frames)

    detector.predict = predict
    monkeypatch.setattr(workflow, "resolve_experiment_config", lambda reference, **_kwargs: experiments[reference])
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (
            detector_spec,
            {"spec": {"backend": "fixture", "geometry_mode": geometry}, "artifact": None},
        ),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    monkeypatch.setattr(detect_stage_module, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(detect_stage_module, "create_detector", lambda _spec: detector)

    def materialize(experiment: str) -> Path:
        return workflow.materialize(
            SimpleNamespace(
                experiment=experiment,
                split=None,
                data_root=data_root,
                device="cpu",
                materialize_explicit_keys=(),
                build_root=tmp_path / "runs" / "materializations",
                plan_path=None,
                plan_overrides=(),
                publish_image_refs=True,
                publish_masks=False,
                publish_embeddings=False,
                resume=True,
            )
        )

    first = materialize("fixture-test-detector")
    second = materialize("fixture-test-detector-alt")

    assert first != second
    assert predict_calls == 1
    assert DatasetManifest.load(first).stages[0].fingerprint == DatasetManifest.load(second).stages[0].fingerprint


@pytest.mark.parametrize("publish_masks", [False, True])
def test_kitti_materialization_publishes_masks_only_when_requested(monkeypatch, tmp_path, publish_masks: bool) -> None:
    """KITTI supports both detection-only builds and masks for optional MOTS replay."""
    data_root, source_path, resolved = _experiment_case(tmp_path)
    image_path = source_path.parent.parent / "000000.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    resolved["dataset"]["layout"] = "sequence"
    resolved["dataset"]["fps"] = 10.0
    resolved["dataset"]["default_split"] = "test"
    resolved["dataset"]["modalities"] = {
        "images": {"format": "image-directory", "paths": ["test/{sequence}"], "options": {}}
    }
    resolved["dataset"]["splits"]["test"]["partition"] = "testing"
    resolved["dataset"]["splits"]["test"]["has_ground_truth"] = False
    resolved["dataset"]["classes"] = {"car": {"id": 1, "evaluation": "target"}}
    resolved["evaluation"]["classes"] = [{"name": "car", "dataset_id": 1, "detector_name": "car", "detector_id": 2}]
    expected_mask = torch.zeros((1, 8, 10), dtype=torch.bool)
    expected_mask[0, 2:6, 1] = True
    expected_mask[0, 5, 1:4] = True

    class MaskDetector(_Detector):
        capabilities = DetectorCapabilities(provides_masks=True)

        def predict(self, frames: list[Frame]) -> list[Detections]:
            return [detections.with_masks(MaskBatch(expected_mask.clone())) for detections in super().predict(frames)]

    detector_type = MaskDetector if publish_masks else _Detector
    detector_spec = DetectorSpec(backend="fixture", geometry_mode="aabb")
    monkeypatch.setattr(workflow, "resolve_experiment_config", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (
            detector_spec,
            {"spec": {"backend": "fixture", "geometry_mode": geometry}, "artifact": None},
        ),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: detector_type.capabilities)
    monkeypatch.setattr(detect_stage_module, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(detect_stage_module, "create_detector", lambda _spec: detector_type())
    args = SimpleNamespace(
        experiment="fixture-test-detector",
        data_root=data_root,
        device="cpu",
        materialize_explicit_keys=(),
        build_root=tmp_path / "builds",
        plan_path=None,
        plan_overrides=(),
        publish_image_refs=True,
        publish_masks=publish_masks,
        publish_embeddings=False,
        tracker="bytetrack",
        resume=True,
    )

    output = workflow.materialize(args)

    assert args.publish_masks is publish_masks
    manifest = DatasetManifest.load(output)
    assert manifest.metadata["layout"] == "sequence"
    assert manifest.publish.masks is publish_masks
    dataset = CachedVisionDataset(output, load_masks=publish_masks)
    assert len(dataset) == 1
    assert dataset[0].frame_index == 0
    assert dataset[0].detections.class_ids.tolist() == [1]
    if publish_masks:
        torch.testing.assert_close(dataset[0].detections.masks.values, expected_mask)
    else:
        assert dataset[0].detections.masks is None
    tracker = create_tracker(TrackerSpec(name=args.tracker))
    assert tracker.requirements.masks is False
    replayed = list(iter_cached_tracks(dataset, tracker, output_format="mots" if publish_masks else "mot"))
    assert len(replayed[0].result.tracks) == 1
    assert replayed[0].result.tracks.detection_indices.tolist() == [0]
    if publish_masks:
        torch.testing.assert_close(replayed[0].result.tracks.masks.values, expected_mask)
    else:
        assert replayed[0].result.tracks.masks is None


def test_mmot_materialization_preserves_native_zero_based_class_ids(monkeypatch, tmp_path) -> None:
    resolved = resolve_experiment_config("mmot-obb/test-yolo11l-lmbn.yaml", mode="materialize")
    class_bridge = tuple(resolved["evaluation"]["classes"])
    sample = SourceSample(
        sample_id="test:data23-1:0",
        split="test",
        sequence_id="data23-1",
        frame_index=0,
        timestamp_s=None,
        image_size=(8, 10),
        source_uri=(tmp_path / "000000.npy").as_uri(),
        source_sha256=fingerprint("mmot-frame"),
    )
    catalog = SourceCatalog(
        samples=(sample,),
        fingerprint=fingerprint("mmot-native-class-domain"),
        source_root=tmp_path,
        metadata={"source_catalog_digest": fingerprint("mmot-source")},
    )
    detector_spec = DetectorSpec(backend="fixture", geometry_mode="obb")
    monkeypatch.setattr(
        workflow,
        "_resolved_inputs",
        lambda _args, **_kwargs: (
            "mmot",
            "obb",
            catalog,
            "yolo11l-mmot-obb/default",
            None,
            None,
            {"class_bridge": class_bridge},
        ),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (
            detector_spec,
            {"spec": {"backend": "fixture", "geometry_mode": geometry}, "artifact": None},
        ),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    captured = {}

    class Materializer:
        def __init__(self, plan, stages, *, progress):
            del progress
            captured["plan"] = plan
            captured["stages"] = stages

        def run(self):
            return captured["plan"].output_root

    monkeypatch.setattr(workflow, "DatasetMaterializer", Materializer)

    workflow.materialize(
        SimpleNamespace(
            device="cpu",
            materialize_explicit_keys=(),
            build_root=tmp_path / "builds",
            plan_path=None,
            plan_overrides=(),
            publish_image_refs=True,
            publish_masks=False,
            publish_embeddings=False,
            resume=True,
        )
    )

    expected = {index: index for index in range(8)}
    detect_plan = next(stage for stage in captured["plan"].stages if stage.name == "detect")
    assert detect_plan.config["class_id_map"] == {str(index): index for index in range(8)}
    detect_stage = captured["stages"][0]
    assert dict(detect_stage.class_id_map) == expected

    mapped = detect_stage._map_class_ids(
        Detections(
            sample_id=sample.sample_id,
            geometry=OrientedBoxes(
                torch.tensor(
                    [[3.0, 3.0, 2.0, 4.0, 0.2], [4.0, 4.0, 3.0, 5.0, 0.3], [5.0, 5.0, 4.0, 6.0, 0.4]],
                    dtype=torch.float32,
                )
            ),
            scores=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
            class_ids=torch.tensor([0, 7, 8], dtype=torch.int64),
        )
    )
    assert mapped.class_ids.tolist() == [0, 7]


def test_materialization_reid_reference_has_no_crop_policy() -> None:
    resolved = resolve_experiment_config("mmot-obb/test-yolo11l-lmbn.yaml", mode="materialize")

    reference = workflow._reid_reference(resolved)

    assert reference is not None
    assert "crop_strategy" not in reference


def test_workflow_does_not_plan_masks_for_builtin_reid(monkeypatch, tmp_path) -> None:
    sample = SourceSample(
        sample_id="default:sequence:0",
        split="default",
        sequence_id="sequence",
        frame_index=0,
        timestamp_s=None,
        image_size=(8, 10),
        source_uri=(tmp_path / "frame.jpg").as_uri(),
        source_sha256=fingerprint("frame"),
    )
    catalog = SourceCatalog(
        samples=(sample,),
        fingerprint=fingerprint("catalog"),
        source_root=tmp_path,
        metadata={"source_catalog_digest": fingerprint("source")},
    )
    detector_spec = DetectorSpec("fixture", geometry_mode="aabb")
    encoder_spec = ReIDEncoderSpec(
        "fixture",
        options=(("embedding_dim", 4),),
    )
    monkeypatch.setattr(
        workflow,
        "_resolved_inputs",
        lambda _args, **_kwargs: ("fixture", "aabb", catalog, "detector", None, "reid", {}),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (detector_spec, {"spec": {"backend": "fixture"}}),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_reid_spec",
        lambda _reference: (encoder_spec, {"spec": {"backend": "fixture"}}),
    )
    monkeypatch.setattr(
        workflow,
        "detector_capabilities",
        lambda _spec: DetectorCapabilities(),
    )
    captured = {}

    class Materializer:
        def __init__(self, plan, stages, *, progress):
            del progress
            captured["plan"] = plan
            captured["stages"] = stages

        def run(self):
            return captured["plan"].output_root

    monkeypatch.setattr(workflow, "DatasetMaterializer", Materializer)

    workflow.materialize(
        SimpleNamespace(
            device="cpu",
            materialize_explicit_keys=(),
            build_root=tmp_path / "runs" / "materializations",
            plan_path=None,
            plan_overrides=(),
            publish_image_refs=True,
            publish_masks=False,
            publish_embeddings=True,
            resume=True,
        )
    )

    assert [stage.name for stage in captured["stages"]] == ["detect", "embed", "finalize"]
    assert captured["stages"][0].requires_native_masks is False
    assert captured["stages"][1].use_masks is False
    embed_plan = next(stage for stage in captured["plan"].stages if stage.name == "embed")
    assert embed_plan.depends_on == ("detect",)


def test_materialize_workflow_publishes_loadable_build(monkeypatch, tmp_path) -> None:
    source_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(source_path), np.zeros((8, 10, 3), dtype=np.uint8))
    sample = SourceSample(
        sample_id="default:sequence:0",
        split="default",
        sequence_id="sequence",
        frame_index=0,
        timestamp_s=None,
        image_size=(8, 10),
        source_uri=source_path.as_uri(),
        source_sha256=sha256_file(source_path),
        image_ref="frame.jpg",
    )
    catalog = SourceCatalog(
        samples=(sample,),
        fingerprint=fingerprint({"catalog": "fixture"}),
        source_root=tmp_path,
        metadata={"source_root_uri": tmp_path.as_uri(), "source_catalog_digest": fingerprint("source")},
    )
    detector_spec = DetectorSpec(backend="fixture", geometry_mode="aabb")
    detector_provenance = {"spec": {"backend": "fixture"}, "artifact": None}
    monkeypatch.setattr(
        workflow,
        "_resolved_inputs",
        lambda _args, **_kwargs: ("fixture", "aabb", catalog, "fixture", None, None, {"split": "default"}),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (detector_spec, detector_provenance),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    created_specs = []

    def create_detector(spec):
        created_specs.append(spec)
        return _Detector()

    monkeypatch.setattr(detect_stage_module, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(detect_stage_module, "create_detector", create_detector)

    args = SimpleNamespace(
        device="mps",
        materialize_explicit_keys=("device",),
        build_root=tmp_path / "builds",
        plan_path=None,
        plan_overrides=(),
        publish_image_refs=True,
        publish_masks=False,
        publish_embeddings=False,
        resume=True,
    )
    output = workflow.materialize(args)
    reused = workflow.materialize(args)

    manifest = DatasetManifest.load(output)
    dataset = CachedVisionDataset(output)
    assert reused == output
    assert manifest.complete is True
    assert manifest.schema == "boxmot.dataset/v1"
    assert [spec.device for spec in created_specs] == ["mps"]
    assert manifest.metadata["components"]["detector"]["spec"]["device"] == "mps"
    assert dataset[0].sample_id == sample.sample_id
    assert dataset[0].image_ref == "frame.jpg"
    assert dataset[0].detections.instance_ids == (f"{manifest.build_id}:{sample.sample_id}:0",)


def test_process_stages_receive_one_effective_device_and_change_build_identity(monkeypatch, tmp_path) -> None:
    sample = SourceSample(
        sample_id="default:sequence:0",
        split="default",
        sequence_id="sequence",
        frame_index=0,
        timestamp_s=None,
        image_size=(8, 10),
        source_uri=(tmp_path / "frame.jpg").as_uri(),
        source_sha256=fingerprint("frame"),
    )
    catalog = SourceCatalog(
        samples=(sample,),
        fingerprint=fingerprint("catalog"),
        source_root=tmp_path,
        metadata={"source_catalog_digest": fingerprint("source")},
    )
    detector_spec = DetectorSpec("fixture", device="cpu", geometry_mode="aabb")
    segmentor_spec = SegmentorSpec("fixture", device="cpu", geometry_mode="aabb")
    encoder_spec = ReIDEncoderSpec(
        "fixture",
        device="cpu",
        options=(("embedding_dim", 4),),
    )
    monkeypatch.setattr(
        workflow,
        "_resolved_inputs",
        lambda _args, **_kwargs: (
            "fixture",
            "aabb",
            catalog,
            "detector",
            "segmentor",
            "reid",
            {"split": "default"},
        ),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (
            detector_spec,
            {"spec": {"backend": "fixture", "device": "cpu"}, "artifact": None},
        ),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_segmentor_spec",
        lambda _reference, *, geometry: (
            segmentor_spec,
            {"spec": {"backend": "fixture", "device": "cpu"}, "artifact": None},
        ),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_reid_spec",
        lambda _reference: (
            encoder_spec,
            {"spec": {"backend": "fixture", "device": "cpu"}, "artifact": None},
        ),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    captured = []

    class Materializer:
        def __init__(self, plan, stages, *, progress):
            del progress
            self.plan = plan
            captured.append((plan, stages))

        def run(self):
            return self.plan.output_root

    monkeypatch.setattr(workflow, "DatasetMaterializer", Materializer)

    def run(device: str):
        return workflow.materialize(
            SimpleNamespace(
                device=device,
                materialize_explicit_keys=("device",),
                build_root=tmp_path / "builds",
                plan_path=None,
                plan_overrides=(
                    "detect.executor=process",
                    "segment.executor=process",
                    "embed.executor=process",
                ),
                publish_image_refs=True,
                publish_masks=True,
                publish_embeddings=True,
                resume=True,
            )
        )

    run("cpu")
    run("mps")

    cpu_plan, _cpu_stages = captured[0]
    mps_plan, mps_stages = captured[1]
    assert cpu_plan.build_id != mps_plan.build_id
    assert mps_stages[0].detector.device == "mps"
    assert mps_stages[1].segmentor.device == "mps"
    assert mps_stages[2].encoder.device == "mps"
    assert {name: component["spec"]["device"] for name, component in mps_plan.metadata["components"].items()} == {
        "detector": "mps",
        "segmentor": "mps",
        "reid": "mps",
    }


@pytest.mark.parametrize(
    ("published_device", "requested_device"),
    [("cpu", "mps"), ("mps", "cuda:0"), ("cuda:0", "cpu")],
)
def test_workflow_reuses_prior_release_and_device_without_perception(
    monkeypatch: pytest.MonkeyPatch,
    fps_case: SimpleNamespace,  # noqa: F811 - imported pytest fixture
    published_device: str,
    requested_device: str,
) -> None:
    """A complete build is reused even when the newly requested accelerator is absent."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with monkeypatch.context() as previous_release:
        previous_release.setattr(plan_module, "__version__", "0.1.0")
        previous_release.setattr(workflow, "__version__", "0.1.0")
        output = materialize_fixture(fps_case, None, device=published_device, materialize_explicit_keys=("device",))

    def snapshot() -> dict[str, tuple[bytes, int]]:
        return {
            path.relative_to(output).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in output.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    calls = list(fps_case.detector.seen), list(fps_case.encoder.seen)

    def forbid_perception(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Matching published outputs must be reused before constructing perception stages or models.")

    for name in ("DetectStage", "EmbedStage", "SegmentStage", "DatasetMaterializer"):
        monkeypatch.setattr(workflow, name, forbid_perception)
    monkeypatch.setattr(detect_stage_module, "create_detector", forbid_perception)
    monkeypatch.setattr(embed_stage_module, "create_reid_encoder", forbid_perception)
    monkeypatch.setattr(segment_stage_module, "create_segmentor", forbid_perception)
    monkeypatch.setattr(detect_stage_module, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(embed_stage_module, "_WORKER_ENCODERS", {})

    reused = materialize_fixture(fps_case, None, device=requested_device, materialize_explicit_keys=("device",))

    assert reused == output
    assert snapshot() == before
    assert (fps_case.detector.seen, fps_case.encoder.seen) == calls
    manifest = DatasetManifest.load(output)
    assert manifest.metadata["boxmot_version"] == "0.1.0"
    assert manifest.metadata["components"]["detector"]["spec"]["device"] == published_device
    assert manifest.metadata["components"]["reid"]["spec"]["device"] == published_device
    assert manifest.counts["samples"] == 19
    assert manifest.counts["embeddings"] == manifest.counts["instances"] == 18


def test_published_build_reuse_does_not_construct_perception_models(monkeypatch, tmp_path) -> None:
    sample = SourceSample(
        sample_id="default:sequence:0",
        split="default",
        sequence_id="sequence",
        frame_index=0,
        timestamp_s=None,
        image_size=(8, 10),
        source_uri=(tmp_path / "frame.jpg").as_uri(),
        source_sha256=fingerprint("frame"),
    )
    catalog = SourceCatalog(
        samples=(sample,),
        fingerprint=fingerprint("catalog"),
        source_root=tmp_path,
        metadata={"source_catalog_digest": fingerprint("source")},
    )
    detector_spec = DetectorSpec("fixture", geometry_mode="aabb")
    segmentor_spec = SegmentorSpec("fixture", geometry_mode="aabb")
    encoder_spec = ReIDEncoderSpec(
        "fixture",
        options=(("embedding_dim", 4),),
    )
    monkeypatch.setattr(
        workflow,
        "_resolved_inputs",
        lambda _args, **_kwargs: ("fixture", "aabb", catalog, "detector", "segmentor", "reid", {}),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (detector_spec, {"spec": {"backend": "fixture"}}),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_segmentor_spec",
        lambda _reference, *, geometry: (segmentor_spec, {"spec": {"backend": "fixture"}}),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_reid_spec",
        lambda _reference: (encoder_spec, {"spec": {"backend": "fixture"}}),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    monkeypatch.setattr(
        detect_stage_module,
        "create_detector",
        lambda _spec: pytest.fail("published reuse must not construct a detector"),
    )
    monkeypatch.setattr(
        segment_stage_module,
        "create_segmentor",
        lambda _spec: pytest.fail("published reuse must not construct a segmentor"),
    )
    monkeypatch.setattr(
        embed_stage_module,
        "create_reid_encoder",
        lambda _spec: pytest.fail("published reuse must not construct an encoder"),
    )

    real_create = workflow.BuildPlan.create

    def create_published_plan(_cls, **kwargs):
        plan = real_create(**kwargs)
        plan.output_root.mkdir(parents=True)
        return plan

    monkeypatch.setattr(workflow.BuildPlan, "create", classmethod(create_published_plan))
    captured_stages = []

    class Materializer:
        def __init__(self, plan, stages, *, progress):
            del progress
            self.plan = plan
            captured_stages.extend(stages)

        def run(self):
            return self.plan.output_root

    monkeypatch.setattr(workflow, "DatasetMaterializer", Materializer)

    output = workflow.materialize(
        SimpleNamespace(
            build_root=tmp_path / "builds",
            plan_path=None,
            plan_overrides=(),
            publish_image_refs=True,
            publish_masks=True,
            publish_embeddings=True,
            resume=True,
        )
    )

    assert output.is_dir()
    assert captured_stages[0].detector is detector_spec
    assert captured_stages[1].segmentor is segmentor_spec
    assert captured_stages[2].encoder is encoder_spec


def test_workflow_build_identity_ignores_machine_local_manifest_provenance(monkeypatch, tmp_path) -> None:
    source_fingerprint = fingerprint({"catalog": "portable"})
    catalogs = []
    source_metadata = []
    for checkout in ("checkout-a", "checkout-b"):
        root = tmp_path / checkout / "datasets" / "Fixture"
        sample = SourceSample(
            sample_id="validation:sequence:0",
            split="validation",
            sequence_id="sequence",
            frame_index=0,
            timestamp_s=None,
            image_size=(8, 10),
            source_uri=(root / "sequence" / "000001.jpg").as_uri(),
            source_sha256=fingerprint("frame"),
            image_ref="sequence/000001.jpg",
        )
        catalogs.append(
            SourceCatalog(
                samples=(sample,),
                fingerprint=source_fingerprint,
                source_root=root,
                metadata={
                    "source_root_uri": root.as_uri(),
                    "source_catalog_digest": source_fingerprint,
                },
            )
        )
        source_metadata.append(
            {
                "dataset_id": "fixture",
                "split": "validation",
                "dataset_config": str(tmp_path / checkout / "configs" / "fixture.yaml"),
                "annotation_path": str(root / "sequence" / "gt" / "gt.txt"),
            }
        )

    resolved_inputs = iter(
        (
            ("fixture", "aabb", catalog, "fixture", None, None, metadata)
            for catalog, metadata in zip(catalogs, source_metadata, strict=True)
        )
    )
    detector_spec = DetectorSpec(backend="fixture", geometry_mode="aabb")
    monkeypatch.setattr(workflow, "_resolved_inputs", lambda _args, **_kwargs: next(resolved_inputs))
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (
            detector_spec,
            {"spec": {"backend": "fixture", "geometry_mode": geometry}, "artifact": None},
        ),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    captured = []

    class Materializer:
        def __init__(self, plan, stages, *, progress):
            del stages, progress
            self.plan = plan
            captured.append(plan)

        def run(self):
            return self.plan.output_root

    monkeypatch.setattr(workflow, "DatasetMaterializer", Materializer)

    for checkout in ("checkout-a", "checkout-b"):
        workflow.materialize(
            SimpleNamespace(
                build_root=tmp_path / checkout / "builds",
                plan_path=None,
                plan_overrides=(),
                publish_image_refs=True,
                publish_masks=False,
                publish_embeddings=False,
                resume=True,
            )
        )

    first, second = captured
    assert first.build_id == second.build_id
    assert first.metadata["source_root_uri"] != second.metadata["source_root_uri"]
    assert first.metadata["dataset_config"] != second.metadata["dataset_config"]
    assert first.metadata["annotation_path"] != second.metadata["annotation_path"]


def test_materialize_workflow_consumes_decode_worker_setting(monkeypatch, tmp_path) -> None:
    source_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(source_path), np.zeros((8, 10, 3), dtype=np.uint8))
    sample = SourceSample(
        sample_id="default:sequence:0",
        split="default",
        sequence_id="sequence",
        frame_index=0,
        timestamp_s=None,
        image_size=(8, 10),
        source_uri=source_path.as_uri(),
        source_sha256=sha256_file(source_path),
    )
    catalog = SourceCatalog(
        samples=(sample,),
        fingerprint=fingerprint({"catalog": "decode-workers"}),
        source_root=tmp_path,
        metadata={"source_root_uri": tmp_path.as_uri(), "source_catalog_digest": fingerprint("source")},
    )
    detector_spec = DetectorSpec(backend="fixture", geometry_mode="aabb")
    monkeypatch.setattr(
        workflow,
        "_resolved_inputs",
        lambda _args, **_kwargs: ("fixture", "aabb", catalog, "fixture", None, None, {"split": "default"}),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (detector_spec, {"spec": {"backend": "fixture"}, "artifact": None}),
    )
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    monkeypatch.setattr(detect_stage_module, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(detect_stage_module, "create_detector", lambda _spec: _Detector())
    real_detect_stage = workflow.DetectStage
    received_workers = []

    def recording_detect_stage(*args, **kwargs):
        received_workers.append(kwargs["decode_workers"])
        return real_detect_stage(*args, **kwargs)

    monkeypatch.setattr(workflow, "DetectStage", recording_detect_stage)
    workflow.materialize(
        SimpleNamespace(
            build_root=tmp_path / "builds",
            plan_path=None,
            plan_overrides=("decode.workers=2",),
            publish_image_refs=False,
            publish_masks=False,
            publish_embeddings=False,
            resume=True,
        )
    )

    assert received_workers == [2]
