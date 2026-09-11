"""Workflow capability checks and cache identity for authored modality encodings."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from PIL import Image

from boxmot.datasets.config import load_dataset_config
from boxmot.engine.cli import boxmot
from boxmot.engine.materialization.catalog import catalog_mot_dataset
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


@pytest.mark.parametrize("mode", ["eval", "tune"])
@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        ("options", {"class_divisor": 500}, "class_divisor: 1000"),
        ("options", {"background_id": 77}, "background_id: 0"),
        ("options", {"ignore_ids": [1001]}, "ignore_ids: [10000]"),
        ("classes", {"target": {"vehicle": 7, "person": 9}}, "classes.target car: 1 and pedestrian: 2"),
        ("classes", {"target": {"car": 7, "pedestrian": 9}}, "classes.target car: 1 and pedestrian: 2"),
        ("classes", {"ignore": {"excluded": 9}}, "only ignored class ID 10"),
    ],
)
def test_sensor_commands_reject_unsupported_metric_encodings_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    section: str,
    changes: dict[str, Any],
    message: str,
) -> None:
    """Reject valid generic data conventions the current evaluator cannot score."""
    data = sensor_dataset_fixture(tmp_path)
    payload = yaml.safe_load(data.dataset.read_text())
    if section == "options":
        payload["modalities"]["ground_truth"]["options"].update(changes)
    else:
        payload["classes"].update(changes)
    data.dataset.write_text(yaml.safe_dump(payload))
    command = importlib.import_module(f"boxmot.engine.commands.{mode}")

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Unsupported metric encodings must fail before decoding inputs or starting optimization.")

    monkeypatch.setattr(command, "_dispatch_cli_workflow", unexpected)
    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)

    result = CliRunner().invoke(boxmot, [mode, "--dataset", str(data.dataset.parent), "--tracker", "eagermot"])

    assert result.exit_code == 2, (result.output, result.exception)
    assert message in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("changes", [{"class_divisor": 500}, {"background_id": 77}, {"ignore_ids": [1001]}])
def test_catalog_identity_changes_when_annotation_options_change_without_new_bytes(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    """Identical images and labels with different interpretations need distinct builds."""
    data = sensor_dataset_fixture(tmp_path)
    Image.new("RGB", (4, 3)).save(data.reader_paths["images"] / "000000.png")
    Image.fromarray(np.zeros((3, 4), dtype=np.uint16)).save(data.ground_truth / "000000.png")
    original = catalog_mot_dataset(load_dataset_config(data.dataset), split="val")
    payload = yaml.safe_load(data.dataset.read_text())
    payload["modalities"]["ground_truth"]["options"].update(changes)
    data.dataset.write_text(yaml.safe_dump(payload))

    changed = catalog_mot_dataset(load_dataset_config(data.dataset), split="val")

    assert original.samples == changed.samples
    assert original.metadata["ground_truth_digest"] == changed.metadata["ground_truth_digest"]
    assert original.fingerprint != changed.fingerprint
