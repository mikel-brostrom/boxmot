import json

from tools.ablation_summary import load_experiments


def _write_run(root, name, *, hparams, last_epoch=200, epochs=200):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    metrics = {
        "model": "csl_tinyvit_5m",
        "dataset": "market1501",
        "epochs": epochs,
        "best_epoch": last_epoch,
        "best_mAP": 0.8,
        "best_rank1": 0.9,
        "train": [{"epoch": last_epoch, "loss": 1.0, "triplet_loss": 0.5}],
        "val": [],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "hparams.json").write_text(json.dumps(hparams))


def test_load_experiments_flags_incomplete_and_hparam_mismatched_runs(tmp_path):
    _write_run(
        tmp_path,
        "1_loss/b_ce_only",
        hparams={"loss_type": "softmax", "center_loss_weight": 0.005},
    )
    _write_run(
        tmp_path,
        "2_lr/b_low_3.5e-4",
        hparams={"lr": 3.5e-4},
    )
    _write_run(
        tmp_path,
        "1_loss/f_multisimilarity_concat_bn",
        hparams={"loss_type": "ms", "center_loss_weight": 0.0, "metric_feature": "raw_mean"},
    )
    _write_run(
        tmp_path,
        "3_reg/c_no_aug",
        hparams={
            "color_jitter": False,
            "gaussian_blur": False,
            "random_grayscale": 0.0,
            "random_erasing": 0.0,
        },
        last_epoch=110,
    )

    results = {result["name"]: result for result in load_experiments(tmp_path)}

    assert results["1_loss/b_ce_only"]["valid"] is False
    assert "center_loss_weight" in results["1_loss/b_ce_only"]["issues"][0]
    assert results["2_lr/b_low_3.5e-4"]["valid"] is True
    assert results["1_loss/f_multisimilarity_concat_bn"]["valid"] is False
    assert "metric_feature" in results["1_loss/f_multisimilarity_concat_bn"]["issues"][0]
    assert results["3_reg/c_no_aug"]["valid"] is False
    assert results["3_reg/c_no_aug"]["issues"] == ["incomplete 110/200"]
