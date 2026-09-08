from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import IO, Any

import pytest
import yaml

import boxmot.engine.experiment_config as experiment_config
from boxmot.engine.experiment_config import (
    EXPERIMENT_CONFIGS_DIR,
    ConfigurationError,
    resolve_experiment_config,
    resolve_experiment_path,
    resolve_matching_experiment_path,
)
from boxmot.utils.config import load_yaml_mapping


def test_every_built_in_experiment_has_a_materializable_detector() -> None:
    paths = sorted(EXPERIMENT_CONFIGS_DIR.rglob("*.yaml"))

    assert paths
    for path in paths:
        resolved = resolve_experiment_config(path, mode="materialize")
        relative = path.relative_to(EXPERIMENT_CONFIGS_DIR).with_suffix("")
        assert "id" not in load_yaml_mapping(path), path
        assert resolved["id"] == "-".join(relative.parts), path
        assert "detections" not in resolved, path
        assert resolved["detector"]["ref"], path
        assert resolved["detector"]["checkpoint"], path
        assert resolved["detector"]["model"], path
        assert "config_path" not in resolved["dataset"], path
        assert "config_path" not in resolved["detector"], path
        if resolved["reid"] is not None:
            assert "crop_strategy" not in resolved["reid"], path
            assert "config_path" not in resolved["reid"], path


@pytest.mark.parametrize("reference", ("test-yolo11l-lmbn", "test-yolo11l-lmbn.yaml"))
def test_experiment_resolves_unique_bare_filename_or_stem(reference: str) -> None:
    expected = EXPERIMENT_CONFIGS_DIR / "mmot-obb" / "test-yolo11l-lmbn.yaml"

    assert resolve_experiment_path(reference) == expected.resolve()


@pytest.mark.parametrize(
    "reference",
    ("mot17/ablation-yolox-lmbn", "mot17/ablation-yolox-lmbn.yaml"),
)
def test_experiment_resolves_catalog_relative_filename_or_stem(reference: str) -> None:
    expected = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml"

    assert resolve_experiment_path(reference) == expected.resolve()


@pytest.mark.parametrize("detector", ("yolox-x-mot17", "yolox-x-mot17/ablation"))
def test_component_selectors_resolve_the_authored_experiment_and_its_identity(detector: str) -> None:
    matched = resolve_matching_experiment_path(
        dataset="mot17",
        split="ablation",
        detector=detector,
        reid="lmbn-n-duke",
        mode="eval",
    )

    expected = (EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml").resolve()
    assert matched == expected
    resolved = resolve_experiment_config(matched, mode="eval")
    assert resolved == resolve_experiment_config("mot17/ablation-yolox-lmbn.yaml", mode="eval")
    assert resolved["id"] == "mot17-ablation-yolox-lmbn"
    assert resolved["source_path"] == expected


def test_bare_detector_selector_lets_the_unique_authored_experiment_choose_its_checkpoint() -> None:
    matched = resolve_matching_experiment_path(
        dataset="mot17-mini",
        split="train",
        detector="yolox-x-mot17",
        reid="lmbn-n-duke",
        mode="eval",
    )

    assert matched == (EXPERIMENT_CONFIGS_DIR / "mot17-mini" / "train-yolox-lmbn.yaml").resolve()
    assert resolve_experiment_config(matched)["detector"]["checkpoint"] == "ablation"


@pytest.mark.parametrize("component", ("dataset", "detector", "reid"))
def test_component_selectors_do_not_alias_custom_same_id_profiles_to_builtins(
    component: str,
    tmp_path,
) -> None:
    references = {
        "dataset": "mot17",
        "detector": "yolox-x-mot17",
        "reid": "lmbn-n-duke",
    }
    source_paths = {
        "dataset": experiment_config.CONFIG_ROOT / "datasets" / "mot17.yaml",
        "detector": experiment_config.CONFIG_ROOT / "detectors" / "yolox-x-mot17.yaml",
        "reid": experiment_config.CONFIG_ROOT / "reid" / "lmbn-n-duke.yaml",
    }
    custom = load_yaml_mapping(source_paths[component])
    if component == "dataset":
        custom["storage"]["root"] = "CUSTOM-MOT17"
    elif component == "detector":
        custom["checkpoints"]["ablation"]["path"] = "models/custom-yolox.pt"
    else:
        custom["weights"]["path"] = "models/custom-lmbn.pt"
    custom_path = tmp_path / f"custom-{component}.yaml"
    custom_path.write_text(yaml.safe_dump(custom, sort_keys=False), encoding="utf-8")
    references[component] = str(custom_path)

    with pytest.raises(ConfigurationError, match="(?i)no .*experiment"):
        resolve_matching_experiment_path(
            dataset=references["dataset"],
            split="ablation",
            detector=references["detector"],
            reid=references["reid"],
            mode="eval",
        )


def test_component_selectors_reject_a_missing_authored_experiment(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(experiment_config, "EXPERIMENT_CONFIGS_DIR", tmp_path)

    with pytest.raises(ConfigurationError, match="(?i)no .*experiment"):
        resolve_matching_experiment_path(
            dataset="mot17",
            split="ablation",
            detector="yolox-x-mot17",
            reid="lmbn-n-duke",
            mode="eval",
        )


def test_component_selectors_treat_omitted_reid_as_an_exact_no_reid_match() -> None:
    with pytest.raises(ConfigurationError, match='ReID "none"'):
        resolve_matching_experiment_path(
            dataset="mot17",
            split="ablation",
            detector="yolox-x-mot17",
            mode="eval",
        )


def test_component_selectors_select_only_the_matching_reid_presence(monkeypatch, tmp_path) -> None:
    source = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml"
    with_reid = load_yaml_mapping(source)
    without_reid = load_yaml_mapping(source)
    without_reid.pop("reid")
    with_reid_path = tmp_path / "with-reid.yaml"
    without_reid_path = tmp_path / "without-reid.yaml"
    with_reid_path.write_text(yaml.safe_dump(with_reid, sort_keys=False), encoding="utf-8")
    without_reid_path.write_text(yaml.safe_dump(without_reid, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(experiment_config, "EXPERIMENT_CONFIGS_DIR", tmp_path)

    common = {
        "dataset": "mot17",
        "split": "ablation",
        "detector": "yolox-x-mot17",
        "mode": "eval",
    }
    assert resolve_matching_experiment_path(**common) == without_reid_path.resolve()
    assert resolve_matching_experiment_path(**common, reid="lmbn-n-duke") == with_reid_path.resolve()


def test_component_selectors_reject_ambiguous_authored_experiments(monkeypatch, tmp_path) -> None:
    source = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml"
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    contents = source.read_text(encoding="utf-8")
    first.write_text(contents, encoding="utf-8")
    second.write_text(contents.replace("checkpoint: ablation", "checkpoint: test"), encoding="utf-8")
    monkeypatch.setattr(experiment_config, "EXPERIMENT_CONFIGS_DIR", tmp_path)

    with pytest.raises(ConfigurationError, match="(?i)ambiguous") as error:
        resolve_matching_experiment_path(
            dataset="mot17",
            split="ablation",
            detector="yolox-x-mot17",
            reid="lmbn-n-duke",
            mode="eval",
        )

    assert "first.yaml" in str(error.value)
    assert "second.yaml" in str(error.value)
    assert "--experiment" in str(error.value)
    assert (
        resolve_matching_experiment_path(
            dataset="mot17",
            split="ablation",
            detector="yolox-x-mot17/ablation",
            reid="lmbn-n-duke",
            mode="eval",
        )
        == first.resolve()
    )


def test_component_selectors_do_not_override_an_authored_split(monkeypatch, tmp_path) -> None:
    source = EXPERIMENT_CONFIGS_DIR / "mot17" / "test-yolox-lmbn.yaml"
    (tmp_path / "test-yolox-lmbn.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(experiment_config, "EXPERIMENT_CONFIGS_DIR", tmp_path)

    with pytest.raises(ConfigurationError, match="(?i)no .*experiment"):
        resolve_matching_experiment_path(
            dataset="mot17",
            split="ablation",
            detector="yolox-x-mot17/test",
            reid="lmbn-n-duke",
            mode="eval",
        )


def test_component_selection_reads_each_candidate_and_its_profiles_once(monkeypatch, tmp_path) -> None:
    """Identity matching must reuse profiles already read for semantic validation."""

    source = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml"
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(experiment_config, "EXPERIMENT_CONFIGS_DIR", tmp_path)
    reads: Counter[Path] = Counter()
    original_open = Path.open

    def counted_open(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
        """Record reads of authored YAML files and their component profiles."""

        reads[path.resolve()] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)

    assert (
        resolve_matching_experiment_path(
            dataset="mot17",
            split="ablation",
            detector="yolox-x-mot17",
            reid="lmbn-n-duke",
        )
        == candidate.resolve()
    )

    assert reads[candidate.resolve()] == 1
    for directory, filename in (
        ("datasets", "mot17.yaml"),
        ("detectors", "yolox-x-mot17.yaml"),
        ("reid", "lmbn-n-duke.yaml"),
    ):
        # One read validates the explicit selector, one resolves the candidate.
        assert reads[(experiment_config.CONFIG_ROOT / directory / filename).resolve()] == 2


def test_component_selection_still_validates_unrelated_candidates(monkeypatch, tmp_path) -> None:
    """An existing match must not hide invalid authored configurations elsewhere."""

    source = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml"
    (tmp_path / "a-match.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    unrelated = load_yaml_mapping(EXPERIMENT_CONFIGS_DIR / "mmot-obb" / "test-yolo11l-lmbn.yaml")
    unrelated["evaluation"]["class_map"] = {"absent-class": "absent-class"}
    (tmp_path / "z-invalid.yaml").write_text(yaml.safe_dump(unrelated, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(experiment_config, "EXPERIMENT_CONFIGS_DIR", tmp_path)

    with pytest.raises(ConfigurationError, match='Dataset class "absent-class"'):
        resolve_matching_experiment_path(
            dataset="mot17",
            split="ablation",
            detector="yolox-x-mot17",
            reid="lmbn-n-duke",
        )


def test_component_selection_observes_configuration_edits_between_calls(monkeypatch, tmp_path) -> None:
    """Selector reuse must not return stale matches after an authored file changes."""

    source = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml"
    candidate = tmp_path / "candidate.yaml"
    contents = source.read_text(encoding="utf-8")
    candidate.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(experiment_config, "EXPERIMENT_CONFIGS_DIR", tmp_path)
    selectors = {
        "dataset": "mot17",
        "split": "ablation",
        "detector": "yolox-x-mot17/ablation",
        "reid": "lmbn-n-duke",
    }
    assert resolve_matching_experiment_path(**selectors) == candidate.resolve()

    candidate.write_text(contents.replace("checkpoint: ablation", "checkpoint: test"), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="(?i)no .*experiment"):
        resolve_matching_experiment_path(**selectors)


def test_mot17_osnet_experiment_references_component_filenames() -> None:
    path = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-osnet.yaml"
    authored = load_yaml_mapping(path)

    assert authored["dataset"]["ref"] == "mot17.yaml"
    assert authored["detector"]["ref"] == "yolox-x-mot17.yaml"
    assert authored["reid"]["ref"] == "osnet-x0-25-msmt17.yaml"

    resolved = resolve_experiment_config(path, mode="materialize")
    assert resolved["reid"]["id"] == "osnet-x0-25-msmt17"


def test_experiment_rejects_ambiguous_bare_filename() -> None:
    with pytest.raises(ConfigurationError, match='Ambiguous experiment filename "val-yolox-lmbn"'):
        resolve_experiment_path("val-yolox-lmbn")


def test_experiment_does_not_resolve_former_declared_id() -> None:
    with pytest.raises(FileNotFoundError, match="filename"):
        resolve_experiment_path("mmot-obb-test-yolo11l-lmbn")


def test_experiment_does_not_replace_an_unrelated_filename_suffix() -> None:
    with pytest.raises(FileNotFoundError, match="filename"):
        resolve_experiment_path("test-yolo11l-lmbn.json")


def test_external_experiment_identity_comes_from_filename(tmp_path) -> None:
    source = EXPERIMENT_CONFIGS_DIR / "mmot-obb" / "test-yolo11l-lmbn.yaml"
    experiment = tmp_path / "custom-mmot.yaml"
    experiment.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    resolved = resolve_experiment_config(experiment, mode="materialize")

    assert resolved["id"] == "custom-mmot"


def test_experiment_rejects_authored_id(tmp_path) -> None:
    experiment = tmp_path / "custom-mmot.yaml"
    experiment.write_text("id: unrelated-name\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match='must not define "id"'):
        resolve_experiment_config(experiment, mode="materialize")


def test_experiment_rejects_non_kebab_case_filename(tmp_path) -> None:
    experiment = tmp_path / "invalid_name.yaml"
    experiment.write_text("dataset: {}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="filename stem"):
        resolve_experiment_config(experiment, mode="materialize")


def test_experiment_rejects_legacy_detections_key(tmp_path) -> None:
    experiment = tmp_path / "legacy-detections.yaml"
    experiment.write_text("detections: {}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown keys: detections"):
        resolve_experiment_config(experiment, mode="materialize")


def test_experiment_rejects_unknown_detector_key(tmp_path) -> None:
    experiment = tmp_path / "invalid-detector.yaml"
    experiment.write_text(
        """
dataset:
  ref: mmot
  split: test
detector:
  ref: yolo11l-mmot-obb
  checkpoint: default
  source: model
evaluation:
  class_map: auto
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="detector has unknown keys: source"):
        resolve_experiment_config(experiment, mode="materialize")


def test_experiment_preserves_external_detector_reference(tmp_path) -> None:
    detector = tmp_path / "custom-detector.yaml"
    detector.write_text(
        """
id: custom-detector
box_type: aabb
classes:
  0: person
inference:
  image_size: [640, 640]
  confidence_threshold: 0.25
checkpoints:
  default:
    path: custom-detector.pt
""".strip(),
        encoding="utf-8",
    )
    experiment = tmp_path / "external-detector.yaml"
    experiment.write_text(
        f"""
dataset:
  ref: mot17
  split: val
detector:
  ref: {detector}
  checkpoint: default
evaluation:
  class_map:
    pedestrian: person
""".strip(),
        encoding="utf-8",
    )

    resolved = resolve_experiment_config(experiment, mode="materialize")

    assert resolved["detector"]["ref"] == str(detector)
    assert resolved["detector"]["id"] == "custom-detector"
    assert resolved["detector"]["checkpoint"] == "default"


def test_mot17_mini_ci_experiment_uses_the_prepared_yolo26n_profile() -> None:
    resolved = resolve_experiment_config("mot17-mini/train-yolo26n-lmbn", mode="materialize")

    assert "detections" not in resolved
    assert resolved["detector"]["ref"] == "yolo26n"
    assert resolved["detector"]["checkpoint"] == "default"
    assert resolved["detector"]["id"] == "yolo26n"
    assert resolved["detector"]["model"] == "models/yolo26n.pt"
    assert resolved["detector"]["uri"] == (
        "https://github.com/mikel-brostrom/boxmot/releases/download/v22.0.0/yolo26n.pt"
    )
    assert resolved["detector"]["sha256"] == "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
    assert resolved["detector"]["classes"][0] == "person"
    assert resolved["evaluation"]["classes"] == [
        {
            "name": "pedestrian",
            "dataset_id": 1,
            "detector_name": "person",
            "detector_id": 0,
        }
    ]


def test_mmot_experiment_uses_one_native_identity_class_domain() -> None:
    resolved = resolve_experiment_config("test-yolo11l-lmbn", mode="materialize")

    assert [
        (entry["name"], entry["dataset_id"], entry["detector_id"]) for entry in resolved["evaluation"]["classes"]
    ] == [
        ("car", 0, 0),
        ("bike", 1, 1),
        ("pedestrian", 2, 2),
        ("van", 3, 3),
        ("truck", 4, 4),
        ("bus", 5, 5),
        ("tricycle", 6, 6),
        ("awning-bike", 7, 7),
    ]


def test_mmot_osnet_experiment_resolves_osnet_profile() -> None:
    resolved = resolve_experiment_config("mmot-obb/test-yolo11l-osnet.yaml", mode="materialize")

    assert resolved["id"] == "mmot-obb-test-yolo11l-osnet"
    assert resolved["reid"]["id"] == "osnet-x0-25-msmt17"
    assert resolved["reid"]["model"] == "models/osnet_x0_25_msmt17.pt"
    assert resolved["reid"]["precision"] == "fp32"
    assert resolved["reid"]["image_size"] == [256, 128]
    assert "crop_strategy" not in resolved["reid"]


def test_experiment_rejects_reid_crop_strategy(tmp_path) -> None:
    experiment = tmp_path / "invalid-crop.yaml"
    experiment.write_text(
        """
dataset:
  ref: mmot
  split: test
detector:
  ref: yolo11l-mmot-obb
  checkpoint: default
reid:
  ref: lmbn-n-duke
  crop_strategy: perspective
evaluation:
  class_map: auto
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="reid has unknown keys: crop_strategy"):
        resolve_experiment_config(experiment, mode="materialize")
