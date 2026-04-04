"""Append one keep/discard/crash row to the autoresearch ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from autoresearch.common import (
    DEFAULT_RESULTS_PATH,
    RESULT_STATUSES,
    append_results_row,
    load_logged_metrics_from_artifact,
    summarize_logged_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for explicit experiment logging."""
    parser = argparse.ArgumentParser(
        description="Append one completed experiment result to autoresearch/results.tsv.",
    )
    parser.add_argument("--artifact", type=Path, help="Eval/tune JSON artifact to summarize.")
    parser.add_argument("--results-tsv", type=Path, default=DEFAULT_RESULTS_PATH, help="Shared research ledger.")
    parser.add_argument("--status", required=True, choices=RESULT_STATUSES, help="Experiment outcome.")
    parser.add_argument("--description", default="", help="Short experiment description for results.tsv.")
    parser.add_argument("--dry-run", action="store_true", help="Print the row without writing it.")
    return parser


def run_log(args: argparse.Namespace) -> Path:
    """Append one explicit keep/discard/crash row to the shared TSV."""
    if args.status == "crash":
        artifact_payload = {}
        summary = None
    else:
        if args.artifact is None:
            raise ValueError("--artifact is required unless --status crash")
        artifact_payload = load_logged_metrics_from_artifact(args.artifact)
        summary = artifact_payload["summary"]

    logged_metrics = summarize_logged_metrics(summary, status=args.status)
    results_path = Path(args.results_tsv).resolve()
    if not args.dry_run:
        results_path = append_results_row(
            results_path,
            summary=summary,
            status=args.status,
            tracker=artifact_payload.get("tracker", ""),
            benchmark=artifact_payload.get("benchmark", ""),
            phase=artifact_payload.get("phase", ""),
            description=args.description,
        )

    print("---")
    print(f"status:       {args.status}")
    print(f"HOTA:         {logged_metrics['HOTA']:.6f}")
    print(f"MOTA:         {logged_metrics['MOTA']:.6f}")
    print(f"IDF1:         {logged_metrics['IDF1']:.6f}")
    print(f"AssA:         {logged_metrics['AssA']:.6f}")
    print(f"AssRe:        {logged_metrics['AssRe']:.6f}")
    print(f"IDSW:         {logged_metrics['IDSW']}")
    print(f"IDs:          {logged_metrics['IDs']}")
    print(f"IDSW_rate:    {logged_metrics['IDSW_rate']:.6f}")
    if artifact_payload:
        print(f"tracker:      {artifact_payload.get('tracker', '')}")
        print(f"benchmark:    {artifact_payload.get('benchmark', '')}")
        print(f"phase:        {artifact_payload.get('phase', '')}")
    print(f"results_tsv:  {results_path}")
    if args.artifact is not None:
        print(f"artifact:     {Path(args.artifact).resolve()}")
    print(f"dry_run:      {'yes' if args.dry_run else 'no'}")
    return results_path


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m autoresearch.log``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_log(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
