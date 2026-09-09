"""Recipe and CLI checks for 7M multilevel classifier-guided suppression."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands.reid import train as train_command
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
from boxmot.reid.training.presets import TRAINING_RECIPES_DIR, list_training_recipes, load_training_recipe

RECIPE_NAME = "csl_tinyvit_7m_multilevel_suppression"


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


def test_multilevel_suppression_recipe_resolves_through_train_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve the documented recipe, metadata, and run options without training."""
    market_dir = tmp_path / "market1501"
    metadata_dir = tmp_path / "pav"
    market_dir.mkdir()
    metadata_dir.mkdir()
    project = tmp_path / "runs" / RECIPE_NAME
    captured = {}

    def fake_main(args) -> None:
        captured["args"] = args

    monkeypatch.setattr(train_command, "main", fake_main)
    result = CliRunner().invoke(
        boxmot,
        [
            "train-reid",
            "--recipe",
            RECIPE_NAME,
            "--data-dir",
            str(market_dir),
            "--anatomical-metadata-dir",
            str(metadata_dir),
            "--project",
            str(project),
            "--name",
            "class_cam_q15_v2_seed0",
            "--device",
            "0",
            "--num-workers",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert Path(args.data_dir) == market_dir
    assert Path(args.anatomical_metadata_dir) == metadata_dir
    assert Path(args.project) == project
    assert args.name == "class_cam_q15_v2_seed0"
    assert args.device == "0"
    assert not project.exists()
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
