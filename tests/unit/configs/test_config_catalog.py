from pathlib import Path, PureWindowsPath

import pytest
import yaml

from boxmot.configs import CONFIG_ROOT
from boxmot.data.config import (
    ARTIFACT_CONFIGS_DIR,
    DATASET_CONFIGS_DIR,
    load_dataset_config,
    resolve_artifact_config_path,
    resolve_dataset_config_path,
)
from boxmot.detectors.config import DETECTOR_CONFIGS_DIR, resolve_detector_config_path
from boxmot.engine.experiment import (
    EXPERIMENT_CONFIGS_DIR,
    resolve_experiment_config,
    resolve_experiment_path,
)
from boxmot.reid.config import REID_CONFIGS_DIR, resolve_reid_config_path
from boxmot.utils.config import (
    CONFIG_ID_PATTERN,
    ConfigurationError,
    index_config_ids,
    resolve_config_path,
)

CATALOGS = (
    ("dataset", DATASET_CONFIGS_DIR, resolve_dataset_config_path),
    ("artifact", ARTIFACT_CONFIGS_DIR, resolve_artifact_config_path),
    ("detector", DETECTOR_CONFIGS_DIR, resolve_detector_config_path),
    ("ReID", REID_CONFIGS_DIR, resolve_reid_config_path),
    ("experiment", EXPERIMENT_CONFIGS_DIR, resolve_experiment_path),
)


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


@pytest.mark.parametrize(("label", "directory", "resolver"), CATALOGS)
def test_builtin_catalog_ids_are_unique_valid_and_resolvable(label, directory, resolver):
    indexed = index_config_ids(directory, label)

    assert indexed
    for config_id, path in indexed.items():
        assert CONFIG_ID_PATTERN.fullmatch(config_id)
        assert resolver(config_id) == path
        assert resolver(path.relative_to(directory)) == path
        assert resolver(path) == path


def test_builtin_yaml_filenames_are_kebab_case_and_values_are_portable():
    for path in CONFIG_ROOT.glob("**/*.yaml"):
        assert CONFIG_ID_PATTERN.fullmatch(path.stem), path
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for value in _strings(payload):
            assert not Path(value).is_absolute(), (path, value)
            assert not PureWindowsPath(value).is_absolute(), (path, value)


def test_every_builtin_experiment_composes_in_inference_mode():
    for experiment_id in index_config_ids(EXPERIMENT_CONFIGS_DIR, "experiment"):
        resolved = resolve_experiment_config(experiment_id, mode="inference")

        assert resolved["id"] == experiment_id
        assert resolved["dataset"]["box_type"] in {"aabb", "obb"}
        assert resolved["detections"]["source"] in {"model", "public", "precomputed"}


def test_catalog_rejects_duplicate_and_malformed_ids(tmp_path):
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    (duplicate_dir / "first.yaml").write_text("id: repeated-id\n", encoding="utf-8")
    (duplicate_dir / "second.yaml").write_text("id: repeated-id\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Duplicate test id"):
        index_config_ids(duplicate_dir, "test")

    malformed_dir = tmp_path / "malformed"
    malformed_dir.mkdir()
    (malformed_dir / "invalid.yaml").write_text("id: Not_Safe\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="lowercase kebab-case"):
        index_config_ids(malformed_dir, "test")


def test_resolver_rejects_ambiguous_filename_and_missing_explicit_path(tmp_path):
    catalog = tmp_path / "catalog"
    (catalog / "one").mkdir(parents=True)
    (catalog / "two").mkdir()
    (catalog / "one" / "shared.yaml").write_text("id: first-id\n", encoding="utf-8")
    (catalog / "two" / "shared.yaml").write_text("id: second-id\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Ambiguous test reference"):
        resolve_config_path(catalog, "shared.yaml", "test")

    with pytest.raises(FileNotFoundError, match="path does not exist"):
        resolve_dataset_config_path(tmp_path / "missing" / "mot17.yaml")

    with pytest.raises(FileNotFoundError, match="must not be empty"):
        resolve_dataset_config_path("")


def test_resolver_does_not_treat_existing_model_weights_as_yaml(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"\x80\x02binary checkpoint")

    with pytest.raises(FileNotFoundError, match="path does not exist"):
        resolve_config_path(REID_CONFIGS_DIR, weights, "ReID")


def test_explicit_config_rejects_malformed_id(tmp_path):
    source = yaml.safe_load((DATASET_CONFIGS_DIR / "mot17.yaml").read_text(encoding="utf-8"))
    source["id"] = "Not_Portable"
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="lowercase kebab-case"):
        load_dataset_config(path)
