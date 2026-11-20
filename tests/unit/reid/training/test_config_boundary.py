"""Reusable training settings work without importing command orchestration."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent


def test_training_config_and_saved_hparams_do_not_import_engine(tmp_path) -> None:
    """Round-trip trainer settings and read run metadata in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", dedent("""\
            import json
            import sys
            from pathlib import Path

            from boxmot.reid.training.config import ReIDTrainConfig
            from boxmot.reid.training.hparams import load_train_hparams

            config = ReIDTrainConfig.from_flat_kwargs(
                model_name="osnet_x0_25",
                dataset_name="market1501",
                data_dir="/datasets",
                img_size=(256, 128),
            )
            assert ReIDTrainConfig.from_flat_kwargs(**config.to_trainer_kwargs()) == config

            run = Path(sys.argv[1])
            (run / "hparams.json").write_text(json.dumps({
                "run": {"model_name": "osnet_x0_25"},
                "data": {"dataset": "market1501", "img_size": [256, 128]},
            }), encoding="utf-8")
            saved = load_train_hparams(run)
            assert saved["model_name"] == config.model.model_name
            assert saved["dataset"] == config.data.dataset_name
            assert tuple(saved["img_size"]) == config.data.img_size
            assert not any(name == "boxmot.engine" or name.startswith("boxmot.engine.") for name in sys.modules)
            """), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
