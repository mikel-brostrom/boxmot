import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch import prepare, train


def test_prepare_warms_cache_and_writes_setup_json(tmp_path, monkeypatch):
    calls: list[tuple[str, str, str, Path]] = []
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runs"

    def fake_eval_setup(args):
        calls.append(("setup", args.tracking_method, args.data, Path(args.project)))

    def fake_generate(args):
        calls.append(("generate", args.tracking_method, args.data, Path(args.project)))

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.evaluator",
        SimpleNamespace(
            eval_setup=fake_eval_setup,
            run_generate_dets_embs=fake_generate,
        ),
    )

    args = prepare.build_parser().parse_args(
        [
            "--benchmark",
            "mot17-mini",
            "--tracker",
            "boosttrack",
            "--project",
            str(artifact_root),
            "--runtime-project",
            str(runtime_root),
            "--results-tsv",
            str(tmp_path / "results.tsv"),
        ]
    )

    artifact = prepare.run_prepare(args)
    payload = json.loads(artifact.read_text())

    assert calls == [
        ("setup", "boosttrack", "mot17-mini", runtime_root),
        ("generate", "boosttrack", "mot17-mini", runtime_root),
    ]
    assert payload["tracker"] == "boosttrack"
    assert payload["cache_warmed"] is True
    assert Path(payload["results_tsv"]).exists()
    assert Path(payload["project"]) == artifact_root
    assert Path(payload["runtime_project"]) == runtime_root
    assert Path(payload["artifact_dir"]) == artifact_root / "autoresearch" / "boosttrack" / "mot17-mini"


def test_train_eval_records_summary_json_and_tsv(tmp_path, monkeypatch):
    calls: list[tuple[str, Path]] = []
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runs"

    def fake_eval_setup(args):
        calls.append(("setup", Path(args.project)))
        return None

    def fake_generate(args):
        calls.append(("generate", Path(args.project)))
        return None

    def fake_track(args):
        calls.append(("track", Path(args.project)))
        return None

    def fake_trackeval(args):
        return {
            "HOTA": 67.1,
            "MOTA": 74.2,
            "IDF1": 80.3,
            "AssA": 61.4,
            "AssRe": 82.1,
            "IDSW": 3,
            "IDs": 50,
        }

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.evaluator",
        SimpleNamespace(
            eval_setup=fake_eval_setup,
            run_generate_dets_embs=fake_generate,
            run_generate_mot_results=fake_track,
            run_trackeval=fake_trackeval,
        ),
    )

    args = train.build_parser().parse_args(
        [
            "eval",
            "--benchmark",
            "mot17-mini",
            "--tracker",
            "boosttrack",
            "--project",
            str(artifact_root),
            "--runtime-project",
            str(runtime_root),
            "--results-tsv",
            str(tmp_path / "results.tsv"),
            "--record",
            "--status",
            "candidate",
            "--description",
            "baseline eval",
        ]
    )

    artifact = train.run_eval(args)
    payload = json.loads(artifact.read_text())
    rows = (tmp_path / "results.tsv").read_text().splitlines()

    assert calls == [
        ("setup", runtime_root),
        ("generate", runtime_root),
        ("track", runtime_root),
    ]
    assert payload["summary"]["HOTA"] == pytest.approx(67.1)
    assert payload["summary"]["IDSW_rate"] == pytest.approx(3 / 50)
    assert Path(payload["project"]) == artifact_root
    assert Path(payload["runtime_project"]) == runtime_root
    assert Path(payload["args"]["project"]) == runtime_root
    assert Path(payload["artifact_dir"]) == artifact_root / "autoresearch" / "boosttrack" / "mot17-mini"
    assert "boosttrack" in rows[1]
    assert "candidate" in rows[1]
    assert "baseline eval" in rows[1]


def test_train_eval_dry_run_prints_artifact_path(tmp_path, capsys):
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runs"
    args = train.build_parser().parse_args(
        [
            "eval",
            "--benchmark",
            "mot17-mini",
            "--tracker",
            "bytetrack",
            "--project",
            str(artifact_root),
            "--runtime-project",
            str(runtime_root),
            "--dry-run",
        ]
    )

    artifact = train.run_eval(args)
    out = capsys.readouterr().out

    assert "dry_run:       yes" in out
    assert f"runtime_project: {runtime_root}" in out
    assert "artifact_dir:  " in out
    assert f"summary_json:  {artifact}" in out


def test_train_tune_writes_best_trial_summary(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runs"

    def fake_tuner(args):
        assert Path(args.project) == runtime_root
        return {
            "tracking_method": args.tracking_method,
            "benchmark": args.data,
            "objectives": ["HOTA"],
            "maximize": ["HOTA"],
            "minimize": [],
            "best_trial_id": "trial_001",
            "best_config": {"match_thresh": 0.85},
            "best_metrics": {
                "HOTA": 68.4,
                "MOTA": 75.1,
                "IDF1": 80.6,
                "AssA": 62.0,
                "AssRe": 83.5,
                "IDSW": 2,
                "IDs": 50,
                "IDSW_rate": 0.04,
            },
            "tune_dir": tmp_path / "ray" / "boosttrack_tune",
            "results_csv": tmp_path / "ray" / "boosttrack_tune" / "results.csv",
            "summary_path": tmp_path / "ray" / "boosttrack_tune" / "summary.md",
            "best_yaml": tmp_path / "ray" / "boosttrack_tune" / "best_boosttrack.yaml",
        }

    monkeypatch.setitem(sys.modules, "boxmot.engine.tuner", SimpleNamespace(main=fake_tuner))

    args = train.build_parser().parse_args(
        [
            "tune",
            "--benchmark",
            "mot17-mini",
            "--tracker",
            "boosttrack",
            "--project",
            str(artifact_root),
            "--runtime-project",
            str(runtime_root),
            "--results-tsv",
            str(tmp_path / "results.tsv"),
            "--record",
            "--status",
            "tune",
            "--description",
            "search defaults",
            "--n-trials",
            "4",
            "--objectives",
            "HOTA",
        ]
    )

    artifact = train.run_tune(args)
    payload = json.loads(artifact.read_text())

    assert payload["summary"]["best_trial_id"] == "trial_001"
    assert payload["summary"]["best_metrics"]["HOTA"] == pytest.approx(68.4)
    assert payload["summary"]["best_metrics"]["IDSW_rate"] == pytest.approx(0.04)
    assert Path(payload["project"]) == artifact_root
    assert Path(payload["runtime_project"]) == runtime_root
    assert Path(payload["args"]["project"]) == runtime_root
    assert Path(payload["artifact_dir"]) == artifact_root / "autoresearch" / "boosttrack" / "mot17-mini"


def test_train_tune_dry_run_prints_artifact_path(tmp_path, capsys):
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runs"
    args = train.build_parser().parse_args(
        [
            "tune",
            "--benchmark",
            "mot17-mini",
            "--tracker",
            "boosttrack",
            "--project",
            str(artifact_root),
            "--runtime-project",
            str(runtime_root),
            "--dry-run",
        ]
    )

    artifact = train.run_tune(args)
    out = capsys.readouterr().out

    assert "mode:          tune" in out
    assert f"runtime_project: {runtime_root}" in out
    assert f"summary_json:  {artifact}" in out
