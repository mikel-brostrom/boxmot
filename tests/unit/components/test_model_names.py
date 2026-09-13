"""Generated autocomplete names stay synchronized with the canonical catalogs."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest
import yaml

from boxmot.detectors._model_names import DetectorName
from boxmot.detectors._ultralytics_models import ultralytics_detector_names, ultralytics_inventory_version
from boxmot.detectors.config import load_detector_profile
from boxmot.reid._model_names import ReIDName
from boxmot.reid.core.catalog import TRAINED_URLS
from boxmot.utils.config import ConfigurationError
from tools import generate_model_names as generator


def _detector(root: Path, identifier: str, *, filename: str | None = None, checkpoints: object = None) -> Path:
    """Write an authored profile whose checkpoint need not exist to generate names."""
    path = root / "boxmot/configs/detectors" / (filename or f"{identifier}.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "id": identifier,
                "box_type": "aabb",
                "classes": {0: "person"},
                "inference": {"image_size": [640, 640], "confidence_threshold": 0.25},
                "checkpoints": {"default": {"path": "not-installed.pt"}} if checkpoints is None else checkpoints,
            }
        ),
        encoding="utf-8",
    )
    return path


def _reid(root: Path, identifier: str, *, filename: str | None = None) -> Path:
    path = root / "boxmot/configs/reid" / (filename or f"{identifier}.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "id": identifier,
                "weights": {"path": "not-installed.pt"},
                "runtime": {"device": "cpu", "precision": "fp32"},
                "preprocessing": {"mode": "resize", "image_size": [256, 128]},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def catalog_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(generator, "ultralytics_detector_names", lambda: ("yolo99n",))
    monkeypatch.setattr(generator, "ultralytics_inventory_version", lambda: "fixture-version")
    monkeypatch.setattr(generator, "TRAINED_URLS", {"fixture_reid.pt": "https://example.test/fixture_reid.pt"})
    _detector(tmp_path, "base-detector")
    _reid(tmp_path, "base-reid")
    return tmp_path


def test_committed_model_names_are_current() -> None:
    stale = generator.generate(check=True)
    assert not stale, f"Regenerate {stale} with: uv run --no-sync python -m tools.generate_model_names"


def test_every_suggested_detector_name_selects_a_profile_or_official_checkpoint() -> None:
    profiles = generator.detector_names(generator.REPO_ROOT / "boxmot/configs/detectors")
    assert set(get_args(DetectorName)) == set(profiles) | set(ultralytics_detector_names())
    for name in profiles:
        profile = load_detector_profile(name)
        assert profile["checkpoint"]
        assert profile["model"]
    assert "yolox/n" in get_args(DetectorName)
    assert "yolox-x-mot17/ablation" in get_args(DetectorName)
    assert "yolox" not in get_args(DetectorName)
    assert "yolox-x-mot17" not in get_args(DetectorName)


def test_detector_alias_records_installed_inventory_version() -> None:
    detector_module = generator.REPO_ROOT / "boxmot/detectors/_model_names.py"
    assert f"Ultralytics {ultralytics_inventory_version()}" in detector_module.read_text(encoding="utf-8")


def test_every_suggested_reid_name_selects_a_profile_or_downloadable_checkpoint() -> None:
    profiles = generator.reid_names(generator.REPO_ROOT / "boxmot/configs/reid")
    assert set(get_args(ReIDName)) == set(profiles) | {Path(filename).stem for filename in TRAINED_URLS}
    assert {f"{name}.pt" for name in generator.reid_checkpoint_names()} == set(TRAINED_URLS)
    assert {"osnet_x0_25_msmt17", "mobilenetv2_x1_4_market1501", "lmbn_n_duke"} <= set(get_args(ReIDName))
    assert not {"osnet_x0_25", "resnet50", "mlfn"}.intersection(get_args(ReIDName))


@pytest.mark.parametrize("kind", ("detectors", "reid"))
def test_nested_declared_ids_follow_catalog_additions_and_removals(catalog_root: Path, kind: str) -> None:
    write_profile = _detector if kind == "detectors" else _reid
    collect_names = generator.detector_names if kind == "detectors" else generator.reid_names
    directory = catalog_root / "boxmot/configs" / kind
    original = collect_names(directory)
    path = write_profile(catalog_root, "extra-profile", filename="nested/different-filename.yml")

    assert collect_names(directory) == tuple(sorted((*original, "extra-profile")))
    assert "different-filename" not in collect_names(directory)
    path.unlink()
    assert collect_names(directory) == original


def test_checkpoint_count_controls_whether_a_selector_needs_a_suffix(catalog_root: Path) -> None:
    directory = catalog_root / "boxmot/configs/detectors"
    _detector(catalog_root, "fixture", checkpoints={"chosen": {"path": "one.pt"}})
    assert generator.detector_names(directory) == ("base-detector", "fixture")

    _detector(catalog_root, "fixture", checkpoints={"z": {"path": "z.pt"}, "a": {"path": "a.pt"}})
    assert generator.detector_names(directory) == ("base-detector", "fixture/a", "fixture/z")

    _detector(catalog_root, "fixture", checkpoints={"a": {"path": "a.pt"}})
    assert generator.detector_names(directory) == ("base-detector", "fixture")


@pytest.mark.parametrize("checkpoints", ({}, [], "model.pt", {"n": "model.pt"}, {"n": {}}))
def test_malformed_checkpoint_shapes_are_rejected(catalog_root: Path, checkpoints: object) -> None:
    _detector(catalog_root, "fixture", checkpoints=checkpoints)
    with pytest.raises(ConfigurationError, match="checkpoint"):
        generator.detector_names(catalog_root / "boxmot/configs/detectors")


def test_missing_checkpoints_are_rejected(catalog_root: Path) -> None:
    path = _detector(catalog_root, "fixture")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["checkpoints"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="checkpoints"):
        generator.detector_names(catalog_root / "boxmot/configs/detectors")


@pytest.mark.parametrize("checkpoint", ("", "a/b", " bad"))
def test_uncallable_checkpoint_names_are_rejected(catalog_root: Path, checkpoint: str) -> None:
    _detector(catalog_root, "fixture", checkpoints={checkpoint: {"path": "a.pt"}, "good": {"path": "b.pt"}})
    with pytest.raises(ConfigurationError, match="unselectable checkpoint"):
        generator.detector_names(catalog_root / "boxmot/configs/detectors")


@pytest.mark.parametrize("kind", ("detectors", "reid"))
def test_duplicate_nested_ids_fail_instead_of_hiding_a_catalog_entry(catalog_root: Path, kind: str) -> None:
    write_profile = _detector if kind == "detectors" else _reid
    collect_names = generator.detector_names if kind == "detectors" else generator.reid_names
    write_profile(catalog_root, "duplicate", filename="first.yaml")
    write_profile(catalog_root, "duplicate", filename="nested/second.yaml")
    with pytest.raises(ConfigurationError, match="Duplicate"):
        collect_names(catalog_root / "boxmot/configs" / kind)


def test_generation_is_deterministic_and_preserves_sources_and_current_outputs(catalog_root: Path) -> None:
    inputs = {path: path.read_bytes() for path in catalog_root.rglob("*.yaml")}
    expected = generator.generated_sources(catalog_root)
    assert generator.generate(catalog_root, check=True) == tuple(expected)
    assert all(not path.exists() for path in expected)
    assert generator.generate(catalog_root) == tuple(expected)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in expected}

    assert generator.generate(catalog_root, check=True) == ()
    assert generator.generate(catalog_root) == ()
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in expected} == before
    assert {path: path.read_bytes() for path in inputs} == inputs
    assert generator.generated_sources(catalog_root, trackers=("z", "a", "z")) == generator.generated_sources(
        catalog_root, trackers=("a", "z")
    )


def test_tracker_alias_follows_manifest_additions_and_removals(
    catalog_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {"first": object(), "second": object()}
    monkeypatch.setattr(generator, "_TRACKER_MANIFEST", manifest)
    output = catalog_root / "boxmot/trackers/common/_model_names.py"
    before = manifest.copy()
    assert '"first"' in generator.generated_sources(catalog_root)[output]
    assert manifest == before
    manifest["added"] = object()
    del manifest["first"]
    source = generator.generated_sources(catalog_root)[output]
    assert '"added"' in source
    assert '"second"' in source
    assert '"first"' not in source


def test_detector_alias_merges_and_tracks_installed_inventory_changes(
    catalog_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ["yolo99n", "base-detector", "yolo99n"]
    monkeypatch.setattr(generator, "ultralytics_detector_names", lambda: names)
    output = catalog_root / "boxmot/detectors/_model_names.py"
    first = generator.generated_sources(catalog_root)[output]
    assert first.count('"base-detector"') == 1
    assert first.count('"yolo99n"') == 1
    assert names == ["yolo99n", "base-detector", "yolo99n"]
    names[:] = ["yolo100n"]
    second = generator.generated_sources(catalog_root)[output]
    assert '"yolo100n"' in second
    assert '"yolo99n"' not in second
    assert '"base-detector"' in second
    monkeypatch.setattr(generator, "ultralytics_inventory_version", lambda: "updated-version")
    assert "Ultralytics updated-version" in generator.generated_sources(catalog_root)[output]


def test_reid_alias_tracks_download_catalog_changes_and_preserves_selectors(
    catalog_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = {
        "osnet_x0_25_dataset.pt": "https://example.test/osnet.pt",
        "base-reid.pt": "https://example.test/profile.pt",
        "encoder.v2.pt": "https://example.test/encoder.pt",
        "exported.onnx": "https://example.test/exported.onnx",
    }
    monkeypatch.setattr(generator, "TRAINED_URLS", catalog)
    output = catalog_root / "boxmot/reid/_model_names.py"
    original = catalog.copy()
    first = generator.generated_sources(catalog_root)[output]
    assert first.count('"base-reid"') == 1
    assert '"osnet_x0_25_dataset"' in first
    assert '"osnet_x0_25_dataset.pt"' not in first
    assert '"encoder.v2.pt"' in first
    assert '"exported.onnx"' in first
    assert catalog == original
    assert generator.generated_sources(catalog_root)[output] == first

    catalog["added_dataset.pt"] = "https://example.test/added.pt"
    del catalog["osnet_x0_25_dataset.pt"]
    updated = generator.generated_sources(catalog_root)[output]
    assert '"added_dataset"' in updated
    assert '"osnet_x0_25_dataset"' not in updated
    assert '"base-reid"' in updated


def test_check_detects_missing_and_modified_outputs_without_repairing_them(catalog_root: Path) -> None:
    detector, reid, _ = generator.generate(catalog_root)
    detector.write_text("stale detector names\n", encoding="utf-8")
    reid.unlink()

    assert generator.generate(catalog_root, check=True) == (detector, reid)
    assert detector.read_text(encoding="utf-8") == "stale detector names\n"
    assert not reid.exists()
    assert generator.generate(catalog_root) == (detector, reid)
    assert generator.generate(catalog_root, check=True) == ()


def test_check_cli_returns_failure_and_actionable_regeneration_command(
    catalog_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    generate = generator.generate
    monkeypatch.setattr(generator, "REPO_ROOT", catalog_root)
    monkeypatch.setattr(generator, "generate", lambda *, check: generate(catalog_root, check=check))

    assert generator.main(["--check"]) == 1
    output = capsys.readouterr().out
    assert "Stale model names: boxmot/detectors/_model_names.py" in output
    assert "uv run --no-sync python -m tools.generate_model_names" in output
    assert not (catalog_root / "boxmot/detectors/_model_names.py").exists()
    assert generator.main([]) == 0
    assert generator.main(["--check"]) == 0


def test_generated_modules_use_static_python310_type_aliases() -> None:
    for source in generator.generated_sources().values():
        module = ast.parse(source, feature_version=(3, 10))
        imports = [node for node in ast.walk(module) if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert len(imports) == 1
        assert isinstance(imports[0], ast.ImportFrom)
        assert imports[0].module == "typing"
        assert {item.name for item in imports[0].names} == {"Literal", "TypeAlias"}
        assert not any(isinstance(node, ast.Call) for node in ast.walk(module))


def test_importing_generated_aliases_does_not_load_configs_or_model_runtimes() -> None:
    probe = """
import json, sys
from boxmot.detectors._model_names import DetectorName
from boxmot.reid._model_names import ReIDName
from boxmot.trackers.common._model_names import TrackerName
print(json.dumps(sorted(name for name in sys.modules if name in {
    'torch', 'numpy', 'cv2', 'yaml', 'ultralytics',
    'boxmot.detectors.config', 'boxmot.reid.config',
} or '.backends.' in name)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=generator.REPO_ROOT, check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == []
