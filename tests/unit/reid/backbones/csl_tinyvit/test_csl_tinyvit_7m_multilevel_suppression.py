"""Recipe and CLI checks for 7M multilevel classifier-guided suppression."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config import list_training_recipes, load_training_recipe
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
from boxmot.reid.training.presets import TRAINING_RECIPES_DIR
from tests._paths import REPO_ROOT

RECIPE_NAME = "csl_tinyvit_7m_multilevel_suppression"
LAUNCHER = REPO_ROOT / "train_csl_tinyvit_7m_multilevel_suppression.sh"


def _write_launcher_inputs(tmp_path: Path) -> tuple[Path, Path]:
    market_dir = tmp_path / "market1501"
    filenames = {
        "bounding_box_train": "0001_c1s1_000001_00.jpg",
        "query": "0001_c1s1_000002_00.jpg",
        "bounding_box_test": "0001_c2s1_000003_00.jpg",
    }
    for split, filename in filenames.items():
        split_dir = market_dir / split
        split_dir.mkdir(parents=True)
        (split_dir / filename).write_bytes(b"image")

    metadata_dir = tmp_path / "pav"
    mask_path = metadata_dir / "person" / "bounding_box_train" / "0001_c1s1_000001_00.png"
    mask_path.parent.mkdir(parents=True)
    mask_path.write_bytes(b"mask")
    (metadata_dir / "metadata.json").write_text(
        json.dumps(
            {
                "images": {
                    "bounding_box_train/0001_c1s1_000001_00.jpg": {
                        "keypoints": [[1.0, 1.0, 1.0]] * 17,
                        "person_mask": (
                            "person/bounding_box_train/0001_c1s1_000001_00.png"
                        ),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return market_dir, metadata_dir


def test_multilevel_suppression_recipe_is_exact_v20_training_delta() -> None:
    """Keep this a clean V20 ablation with no Stage-3 adapter treatment."""
    baseline = load_training_recipe("csl_tinyvit_7m_v20")
    treatment = load_training_recipe(RECIPE_NAME)
    expected = dict(baseline)
    expected.update(
        {
            "multilevel_suppression": True,
            "multilevel_suppression_ratio": 0.15,
            "multilevel_suppression_loss_weight": 0.2,
            "multilevel_suppression_start_epoch": 20,
            "multilevel_suppression_ramp_end_epoch": 50,
            "multilevel_suppression_decay_start_epoch": 140,
            "multilevel_suppression_decay_end_epoch": 170,
        }
    )

    assert treatment == expected
    assert treatment["reid_adapter_stages"] == []
    assert RECIPE_NAME in list_training_recipes()

    baseline_raw = yaml.safe_load(
        (TRAINING_RECIPES_DIR / "csl_tinyvit_7m_v20.yaml").read_text(
            encoding="utf-8"
        )
    )
    treatment_raw = yaml.safe_load(
        (TRAINING_RECIPES_DIR / f"{RECIPE_NAME}.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected_raw = deepcopy(baseline_raw)
    expected_raw["model"]["head"]["multilevel_suppression"] = {
        "enabled": True,
        "ratio": 0.15,
    }
    expected_raw["losses"]["multilevel_suppression"] = {
        "weight": 0.2,
        "start_epoch": 20,
        "ramp_end_epoch": 50,
        "decay_start_epoch": 140,
        "decay_end_epoch": 170,
    }
    assert treatment_raw == expected_raw
    assert treatment_raw["derived"]["n_params"] == 7_165_011


def test_multilevel_suppression_recipe_resolves_through_train_cli(monkeypatch) -> None:
    """Exercise the same named-recipe path as the user-facing launcher."""
    captured = {}

    def fake_main(args) -> None:
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.reid.trainer",
        SimpleNamespace(main=fake_main),
    )
    result = CliRunner().invoke(
        boxmot,
        ["train", "--recipe", RECIPE_NAME, "--data-dir", "."],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_7m_v20"
    assert args.num_workers == 4
    assert args.reid_adapter_stages == ()
    assert args.multilevel_suppression is True
    assert args.multilevel_suppression_ratio == pytest.approx(0.15)
    assert args.multilevel_suppression_loss_weight == pytest.approx(0.2)
    assert args.multilevel_suppression_start_epoch == 20
    assert args.multilevel_suppression_ramp_end_epoch == 50
    assert args.multilevel_suppression_decay_start_epoch == 140
    assert args.multilevel_suppression_decay_end_epoch == 170
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.feat_dim == args.neck_dim == 384
    assert args.head_parts == (1, 2, 4)
    assert args.scale_balanced_branches is True
    assert args.anatomical_auxiliary is True
    assert args.anatomical_target_type == "learned_pose_concat_ema"

    trainer_values = trainer_kwargs_from_args(args)
    config = ReIDTrainConfig.from_flat_kwargs(**trainer_values)
    assert config.model.multilevel_suppression is True
    assert config.model.multilevel_suppression_ratio == pytest.approx(0.15)
    assert config.loss.multilevel_suppression_loss_weight == pytest.approx(0.2)
    assert config.loss.multilevel_suppression_start_epoch == 20
    assert config.loss.multilevel_suppression_ramp_end_epoch == 50
    assert config.loss.multilevel_suppression_decay_start_epoch == 140
    assert config.loss.multilevel_suppression_decay_end_epoch == 170


def test_multilevel_suppression_launcher_contract() -> None:
    """Keep the launcher executable and pointed at the controlled recipe."""
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert os.access(LAUNCHER, os.X_OK)
    assert f"--recipe {RECIPE_NAME}" in launcher
    assert "runs/csl_tinyvit_7m_multilevel_suppression" in launcher
    assert "class_cam_q15_v2_seed0" in launcher
    assert "MARKET1501_DIR" in launcher
    assert "PAV_METADATA_DIR" in launcher
    assert "VALIDATE_ONLY" in launcher


def test_multilevel_suppression_launcher_validate_only_cannot_train(
    tmp_path: Path,
) -> None:
    """Make launcher validation terminate before invoking the trainer."""
    market_dir, metadata_dir = _write_launcher_inputs(tmp_path)
    project = tmp_path / "runs"
    environment = {
        **os.environ,
        "MARKET1501_DIR": str(market_dir),
        "PAV_METADATA_DIR": str(metadata_dir),
        "MULTILEVEL_SUPPRESSION_PROJECT": str(project),
        "VALIDATE_ONLY": "1",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Validated multilevel-suppression inputs" in result.stdout
    assert not project.exists()


def test_multilevel_suppression_launcher_rejects_incomplete_market1501(
    tmp_path: Path,
) -> None:
    market_dir, metadata_dir = _write_launcher_inputs(tmp_path)
    for path in (market_dir / "query").iterdir():
        path.unlink()

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "MARKET1501_DIR": str(market_dir),
            "PAV_METADATA_DIR": str(metadata_dir),
            "VALIDATE_ONLY": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "query split contains no valid JPEG images" in result.stderr


def test_multilevel_suppression_launcher_rejects_empty_metadata(
    tmp_path: Path,
) -> None:
    market_dir, metadata_dir = _write_launcher_inputs(tmp_path)
    (metadata_dir / "metadata.json").write_text(
        json.dumps({"images": {}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "MARKET1501_DIR": str(market_dir),
            "PAV_METADATA_DIR": str(metadata_dir),
            "VALIDATE_ONLY": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "does not match any Market-1501 training image" in result.stderr
