"""Run one BoxMOT autoresearch experiment via evaluator.py or tuner.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from autoresearch.common import (
    DEFAULT_PROJECT,
    DEFAULT_RESULTS_PATH,
    append_results_row,
    build_experiment_args,
    experiment_dir,
    summarize_eval_results,
    summarize_tune_results,
    to_jsonable,
    write_json,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", default="mot17-ablation", required=True, help="Benchmark bundle, e.g. mot17-ablation.")
    parser.add_argument("--tracker", default="bytetrack", help="Tracker under test.")
    parser.add_argument("--detector", default="yolox_x_mot17_ablation", help="Detector weights or model name.")
    parser.add_argument("--reid", default="lmbn_n_duke", help="ReID weights or model name.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Artifact root.")
    parser.add_argument(
        "--runtime-project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="BoxMOT runtime root for dets_n_embs, mot, and ray outputs.",
    )
    parser.add_argument("--device", help="Runtime device override.")
    parser.add_argument("--imgsz", help="Image size override.")
    parser.add_argument("--batch-size", type=int, help="Batch size override.")
    parser.add_argument("--n-threads", type=int, help="Worker thread override.")
    parser.add_argument("--classes", nargs="+", type=int, help="Optional class filter.")
    parser.add_argument("--tracking-backend", default="thread", help="Tracking backend override.")
    parser.add_argument("--conf", type=float, help="Detector confidence override.")
    parser.add_argument("--iou", type=float, help="NMS/association IoU override.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=None, help="Enable detector FP16.")
    parser.add_argument("--results-tsv", type=Path, default=DEFAULT_RESULTS_PATH, help="Shared research ledger.")
    parser.add_argument("--record", action="store_true", help="Append a row to results.tsv.")
    parser.add_argument("--status", choices=("keep", "discard", "crash"), help="Ledger status for this completed run.")
    parser.add_argument("--description", default="", help="Short experiment description for results.tsv.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned run without invoking BoxMOT.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for eval/tune experiments."""
    parser = argparse.ArgumentParser(
        description="Run one autoresearch experiment against BoxMOT's eval/tune pipeline.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run one benchmark evaluation.")
    _add_common_args(eval_parser)

    tune_parser = subparsers.add_parser("tune", help="Run one hyperparameter tuning sweep.")
    _add_common_args(tune_parser)
    tune_parser.add_argument("--n-trials", type=int, default=8, help="Ray Tune sample count.")
    tune_parser.add_argument("--objectives", nargs="+", help="Optimization objectives.")
    tune_parser.add_argument("--maximize", nargs="+", help="Metrics to maximize.")
    tune_parser.add_argument("--minimize", nargs="+", help="Metrics to minimize.")
    return parser


def _should_record(args: argparse.Namespace) -> bool:
    return bool(args.record)


def _record_result(
    args: argparse.Namespace,
    *,
    summary: dict,
) -> None:
    if not _should_record(args):
        return
    if not args.status:
        raise ValueError("--record requires --status {keep,discard,crash}")
    append_results_row(
        args.results_tsv,
        summary=summary,
        status=args.status,
        tracker=args.tracker,
        benchmark=args.benchmark,
        phase=args.mode,
        description=args.description,
    )


def _print_eval_report(summary: dict, artifact: Path) -> None:
    print("---")
    print(f"summary_name:  {summary['summary_name']}")
    print(f"HOTA:          {summary['HOTA']:.4f}")
    print(f"MOTA:          {summary['MOTA']:.4f}")
    print(f"IDF1:          {summary['IDF1']:.4f}")
    print(f"AssA:          {summary['AssA']:.4f}")
    print(f"AssRe:         {summary['AssRe']:.4f}")
    print(f"IDSW:          {summary['IDSW']}")
    print(f"IDs:           {summary['IDs']}")
    print(f"IDSW_rate:     {summary['IDSW_rate']:.6f}")
    print(f"summary_json:  {artifact}")


def _print_tune_report(summary: dict, artifact: Path) -> None:
    metrics = summary["best_metrics"]
    print("---")
    print(f"best_trial_id: {summary['best_trial_id']}")
    print(f"HOTA:          {metrics['HOTA']:.4f}")
    print(f"MOTA:          {metrics['MOTA']:.4f}")
    print(f"IDF1:          {metrics['IDF1']:.4f}")
    print(f"AssA:          {metrics['AssA']:.4f}")
    print(f"AssRe:         {metrics['AssRe']:.4f}")
    print(f"IDSW:          {metrics['IDSW']}")
    print(f"IDs:           {metrics['IDs']}")
    print(f"IDSW_rate:     {metrics['IDSW_rate']:.6f}")
    print(f"best_yaml:     {summary['best_yaml']}")
    print(f"results_csv:   {summary['results_csv']}")
    print(f"summary_md:    {summary['summary_path']}")
    print(f"summary_json:  {artifact}")


def _print_dry_run_report(
    mode: str,
    tracker: str,
    benchmark: str,
    project: Path,
    runtime_project: Path,
    artifact_dir: Path,
    artifact: Path,
) -> None:
    print("---")
    print(f"mode:          {mode}")
    print(f"tracker:       {tracker}")
    print(f"benchmark:     {benchmark}")
    print(f"project:       {project}")
    print(f"runtime_project: {runtime_project}")
    print(f"artifact_dir:  {artifact_dir}")
    print(f"dry_run:       yes")
    print(f"summary_json:  {artifact}")


def run_eval(args: argparse.Namespace) -> Path:
    """Run evaluator.py against one tracker experiment."""
    artifact_root = Path(args.project).resolve()
    runtime_project = Path(args.runtime_project).resolve()
    eval_args = build_experiment_args(
        "eval",
        benchmark=args.benchmark,
        tracker=args.tracker,
        detector=args.detector,
        reid=args.reid,
        project=runtime_project,
        device=args.device,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        n_threads=args.n_threads,
        classes=args.classes,
        tracking_backend=args.tracking_backend,
        half=args.half,
        conf=args.conf,
        iou=args.iou,
    )

    run_dir = experiment_dir(artifact_root, eval_args.tracking_method, args.benchmark)
    artifact = run_dir / "last_eval.json"
    if args.dry_run:
        payload = {
            "mode": "eval",
            "benchmark": args.benchmark,
            "tracker": eval_args.tracking_method,
            "project": artifact_root,
            "runtime_project": runtime_project,
            "artifact_dir": run_dir,
            "dry_run": True,
            "args": vars(eval_args),
        }
        artifact = write_json(artifact, payload)
        _print_dry_run_report(
            "eval",
            eval_args.tracking_method,
            args.benchmark,
            artifact_root,
            runtime_project,
            run_dir,
            artifact,
        )
        return artifact

    from boxmot.engine.evaluator import (
        eval_setup,
        run_generate_dets_embs,
        run_generate_mot_results,
        run_trackeval,
    )

    eval_setup(eval_args)
    run_generate_dets_embs(eval_args)
    run_generate_mot_results(eval_args)
    results = run_trackeval(eval_args)
    summary = summarize_eval_results(results)
    payload = {
        "mode": "eval",
        "benchmark": args.benchmark,
        "tracker": eval_args.tracking_method,
        "project": artifact_root,
        "runtime_project": runtime_project,
        "artifact_dir": run_dir,
        "args": to_jsonable(vars(eval_args)),
        "summary": summary,
        "raw_results": results,
    }
    artifact = write_json(artifact, payload)
    _record_result(args, summary=summary)
    _print_eval_report(summary, artifact)
    return artifact


def run_tune(args: argparse.Namespace) -> Path:
    """Run tuner.py against one tracker experiment."""
    artifact_root = Path(args.project).resolve()
    runtime_project = Path(args.runtime_project).resolve()
    tune_args = build_experiment_args(
        "tune",
        benchmark=args.benchmark,
        tracker=args.tracker,
        detector=args.detector,
        reid=args.reid,
        project=runtime_project,
        device=args.device,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        n_threads=args.n_threads,
        classes=args.classes,
        tracking_backend=args.tracking_backend,
        half=args.half,
        conf=args.conf,
        iou=args.iou,
        n_trials=args.n_trials,
        objectives=args.objectives,
        maximize=args.maximize,
        minimize=args.minimize,
    )

    run_dir = experiment_dir(artifact_root, tune_args.tracking_method, args.benchmark)
    artifact = run_dir / "last_tune.json"
    if args.dry_run:
        payload = {
            "mode": "tune",
            "benchmark": args.benchmark,
            "tracker": tune_args.tracking_method,
            "project": artifact_root,
            "runtime_project": runtime_project,
            "artifact_dir": run_dir,
            "dry_run": True,
            "args": vars(tune_args),
        }
        artifact = write_json(artifact, payload)
        _print_dry_run_report(
            "tune",
            tune_args.tracking_method,
            args.benchmark,
            artifact_root,
            runtime_project,
            run_dir,
            artifact,
        )
        return artifact

    from boxmot.engine.tuner import main as run_tuner

    results = run_tuner(tune_args)
    summary = summarize_tune_results(results)
    payload = {
        "mode": "tune",
        "benchmark": args.benchmark,
        "tracker": tune_args.tracking_method,
        "project": artifact_root,
        "runtime_project": runtime_project,
        "artifact_dir": run_dir,
        "args": to_jsonable(vars(tune_args)),
        "summary": summary,
        "raw_results": to_jsonable(results),
    }
    artifact = write_json(artifact, payload)
    _record_result(args, summary=summary["best_metrics"])
    _print_tune_report(summary, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m autoresearch.train``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "eval":
        run_eval(args)
    else:
        run_tune(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
