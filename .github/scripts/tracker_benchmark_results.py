#!/usr/bin/env python3
"""Record, validate, and publish tracker benchmark results.

The benchmark workflow intentionally keeps benchmark, tracker, and backend in
the result identity. This avoids mixing MOT17 AABB and MMOT OBB results when
the same tracker/backend pair is evaluated in both jobs.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

METRICS = ("HOTA", "MOTA", "IDF1")
BENCHMARKS = ("mot17-aabb", "mmot-obb")
TRACKERS = (
    "occluboost",
    "botsort",
    "boosttrack",
    "strongsort",
    "deepocsort",
    "bytetrack",
    "hybridsort",
    "ocsort",
    "sfsort",
)
NATIVE_TRACKERS = frozenset({"occluboost", "botsort", "bytetrack", "ocsort", "sfsort"})
BACKENDS = ("python", "cpp")
README_CELLS = {
    "mot17-aabb": (2, 3, 4),
    "mmot-obb": (8, 9, 10),
}

ResultKey = tuple[str, str, str]
ResultMap = dict[ResultKey, dict[str, float]]

_README_START_MARKER = "<!-- START TRACKER TABLE -->"
_README_END_MARKER = "<!-- END TRACKER TABLE -->"
_README_ROW_PATTERN = re.compile(r"(<tr>\s*\n(?:\s*<td[^>]*>.*?</td>\s*\n){12}\s*</tr>)", re.DOTALL)
_README_CELL_PATTERN = re.compile(r"(<td[^>]*><sub>)(.*?)(</sub></td>)")
_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class BenchmarkError(ValueError):
    """Raised when benchmark inputs cannot produce a trustworthy comparison."""


def _metric_value(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{context} must be a numeric percentage, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise BenchmarkError(f"{context} must be finite, got {value!r}")
    return value


def extract_metrics(payload: Mapping[str, Any], benchmark: str) -> dict[str, float]:
    """Extract the README metrics from an evaluator JSON payload."""
    if benchmark not in BENCHMARKS:
        raise BenchmarkError(f"unknown benchmark {benchmark!r}; expected one of {BENCHMARKS}")

    section: Mapping[str, Any]
    if benchmark == "mmot-obb":
        candidate = payload.get("cls_comb_cls_av")
        if not isinstance(candidate, Mapping):
            raise BenchmarkError("MMOT OBB output is missing TrackEval Class Avg (Cls) metrics at 'cls_comb_cls_av'")
        section = candidate
    else:
        section = payload

    extracted: dict[str, float] = {}
    for metric in METRICS:
        if metric not in section:
            raise BenchmarkError(f"{benchmark} output is missing metric {metric!r}")
        extracted[metric] = _metric_value(section[metric], context=f"{benchmark} metric {metric}")
    return extracted


def _validate_identity(benchmark: str, tracker: str, backend: str) -> None:
    if benchmark not in BENCHMARKS:
        raise BenchmarkError(f"unknown benchmark {benchmark!r}; expected one of {BENCHMARKS}")
    if tracker not in TRACKERS:
        raise BenchmarkError(f"unknown tracker {tracker!r}; expected one of {TRACKERS}")
    if backend not in BACKENDS:
        raise BenchmarkError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "cpp" and tracker not in NATIVE_TRACKERS:
        raise BenchmarkError(f"tracker {tracker!r} does not have a native C++ benchmark path")


def write_record(
    *,
    benchmark: str,
    tracker: str,
    backend: str,
    metrics_path: Path,
    output_path: Path,
) -> None:
    """Convert evaluator output into one self-identifying workflow artifact."""
    _validate_identity(benchmark, tracker, backend)
    try:
        payload = json.loads(metrics_path.read_text())
    except FileNotFoundError as exc:
        raise BenchmarkError(f"evaluator did not create {metrics_path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"evaluator output {metrics_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BenchmarkError(f"evaluator output {metrics_path} must be a JSON object")

    record = {
        "benchmark": benchmark,
        "tracker": tracker,
        "backend": backend,
        "metrics": extract_metrics(payload, benchmark),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def load_records(results_dir: Path) -> ResultMap:
    """Load benchmark artifacts while rejecting collisions and partial records."""
    records: ResultMap = {}
    for path in sorted(results_dir.rglob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"result artifact {path} is not valid JSON: {exc}") from exc
        if not isinstance(record, Mapping):
            raise BenchmarkError(f"result artifact {path} must be a JSON object")

        try:
            benchmark = str(record["benchmark"])
            tracker = str(record["tracker"])
            backend = str(record["backend"])
            metric_payload = record["metrics"]
        except KeyError as exc:
            raise BenchmarkError(f"result artifact {path} is missing {exc.args[0]!r}") from exc
        _validate_identity(benchmark, tracker, backend)
        if not isinstance(metric_payload, Mapping):
            raise BenchmarkError(f"result artifact {path} has a non-object 'metrics' value")

        missing_metrics = [metric for metric in METRICS if metric not in metric_payload]
        if missing_metrics:
            raise BenchmarkError(f"result artifact {path} is missing metric(s): {', '.join(missing_metrics)}")
        metrics = {
            metric: _metric_value(
                metric_payload[metric],
                context=f"{path}: {benchmark}/{tracker}/{backend}/{metric}",
            )
            for metric in METRICS
        }
        key = (benchmark, tracker, backend)
        if key in records:
            raise BenchmarkError(f"duplicate result for {'/'.join(key)} (latest artifact: {path})")
        records[key] = metrics

    return records


def _normalize_benchmarks(benchmarks: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and de-duplicate a requested benchmark subset."""
    selected = tuple(dict.fromkeys(benchmarks or BENCHMARKS))
    if not selected:
        raise BenchmarkError("at least one benchmark must be selected")
    unknown = sorted(set(selected) - set(BENCHMARKS))
    if unknown:
        raise BenchmarkError(f"unknown benchmark(s): {', '.join(unknown)}")
    return selected


def expected_keys(benchmarks: Sequence[str] | None = None) -> set[ResultKey]:
    """Return all result cells required for the requested benchmark subset."""
    selected = _normalize_benchmarks(benchmarks)
    python_keys = {(benchmark, tracker, "python") for benchmark in selected for tracker in TRACKERS}
    cpp_keys = {(benchmark, tracker, "cpp") for benchmark in selected for tracker in NATIVE_TRACKERS}
    return python_keys | cpp_keys


def validate_records(
    records: ResultMap,
    *,
    tolerance: float = 0.25,
    benchmarks: Sequence[str] | None = None,
) -> list[str]:
    """Require a complete run and enforce Python/C++ native metric parity."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise BenchmarkError(f"parity tolerance must be non-negative, got {tolerance!r}")

    selected = _normalize_benchmarks(benchmarks)
    required = expected_keys(selected)
    missing = sorted(required - records.keys())
    if missing:
        formatted = ", ".join("/".join(key) for key in missing)
        raise BenchmarkError(f"missing required benchmark result(s): {formatted}")
    unexpected = sorted(records.keys() - required)
    if unexpected:
        formatted = ", ".join("/".join(key) for key in unexpected)
        raise BenchmarkError(f"unexpected benchmark result(s) for this run: {formatted}")

    violations: list[str] = []
    comparisons: list[str] = []
    for benchmark in selected:
        for tracker in TRACKERS:
            if tracker not in NATIVE_TRACKERS:
                continue
            python_metrics = records[(benchmark, tracker, "python")]
            cpp_metrics = records[(benchmark, tracker, "cpp")]
            for metric in METRICS:
                delta = abs(python_metrics[metric] - cpp_metrics[metric])
                comparison = f"{benchmark}/{tracker}/{metric}: {delta:.4f} pp"
                comparisons.append(comparison)
                if delta > tolerance + 1e-12:
                    violations.append(comparison)

    if violations:
        details = "; ".join(violations)
        raise BenchmarkError(f"Python/C++ parity exceeded {tolerance:.2f} percentage points: {details}")
    return comparisons


def _readme_tracker_cells(readme: str) -> dict[str, list[str]]:
    """Extract tracker table cell contents without interpreting metric values."""
    start_idx = readme.find(_README_START_MARKER)
    end_idx = readme.find(_README_END_MARKER)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise BenchmarkError("README tracker table markers were not found in order")

    table = readme[start_idx : end_idx + len(_README_END_MARKER)]
    rows: dict[str, list[str]] = {}
    for row_match in _README_ROW_PATTERN.finditer(table):
        cells = _README_CELL_PATTERN.findall(row_match.group(0))
        if len(cells) != 12:
            continue
        tracker = re.sub(r"<[^>]+>", "", cells[0][1]).strip()
        if tracker not in TRACKERS:
            continue
        if tracker in rows:
            raise BenchmarkError(f"README tracker row is duplicated: {tracker}")
        rows[tracker] = [content for _, content, _ in cells]
    return rows


def _parse_published_metric_pair(cell: str, *, context: str) -> tuple[float, float]:
    """Parse one README ``Python<br>(C++)`` metric cell."""
    with_break = re.sub(r"<br\s*/?>", "\n", cell, flags=re.IGNORECASE)
    text = html.unescape(re.sub(r"<[^>]+>", "", with_break)).replace("\xa0", " ").strip()
    match = re.fullmatch(rf"\s*({_NUMBER_PATTERN})\s*\n\s*\(\s*([^()]*)\s*\)\s*", text)
    if match is None:
        raise BenchmarkError(f"{context} must use a populated Python<br>(C++) metric pair, got {text!r}")

    cpp_text = match.group(2).strip()
    if cpp_text in {"—", "–", "-"}:
        raise BenchmarkError(f"{context} is missing its published C++ metric")
    if re.fullmatch(_NUMBER_PATTERN, cpp_text) is None:
        raise BenchmarkError(f"{context} has an invalid published C++ metric: {cpp_text!r}")

    return (
        _metric_value(float(match.group(1)), context=f"{context} Python metric"),
        _metric_value(float(cpp_text), context=f"{context} C++ metric"),
    )


def published_readme_records(readme: str) -> ResultMap:
    """Load all five native trackers' published AABB and OBB metric pairs."""
    rows = _readme_tracker_cells(readme)
    missing_rows = [tracker for tracker in TRACKERS if tracker in NATIVE_TRACKERS and tracker not in rows]
    if missing_rows:
        raise BenchmarkError(f"README native tracker row(s) not found: {', '.join(missing_rows)}")

    records: ResultMap = {}
    for benchmark in BENCHMARKS:
        cell_indices = README_CELLS[benchmark]
        for tracker in TRACKERS:
            if tracker not in NATIVE_TRACKERS:
                continue
            python_metrics: dict[str, float] = {}
            cpp_metrics: dict[str, float] = {}
            for metric, cell_index in zip(METRICS, cell_indices, strict=True):
                python_value, cpp_value = _parse_published_metric_pair(
                    rows[tracker][cell_index],
                    context=f"README {benchmark}/{tracker}/{metric}",
                )
                python_metrics[metric] = python_value
                cpp_metrics[metric] = cpp_value
            records[(benchmark, tracker, "python")] = python_metrics
            records[(benchmark, tracker, "cpp")] = cpp_metrics
    return records


def validate_published_readme(readme: str, *, tolerance: float = 0.25) -> list[str]:
    """Enforce completeness and parity for every native pair published in README."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise BenchmarkError(f"parity tolerance must be non-negative, got {tolerance!r}")

    records = published_readme_records(readme)
    comparisons: list[str] = []
    violations: list[str] = []
    for benchmark in BENCHMARKS:
        for tracker in TRACKERS:
            if tracker not in NATIVE_TRACKERS:
                continue
            python_metrics = records[(benchmark, tracker, "python")]
            cpp_metrics = records[(benchmark, tracker, "cpp")]
            for metric in METRICS:
                delta = abs(python_metrics[metric] - cpp_metrics[metric])
                comparison = f"{benchmark}/{tracker}/{metric}: {delta:.4f} pp"
                comparisons.append(comparison)
                if delta > tolerance + 1e-12:
                    violations.append(comparison)

    if violations:
        details = "; ".join(violations)
        raise BenchmarkError(
            f"Published README Python/C++ parity exceeded {tolerance:.2f} percentage points: {details}"
        )
    return comparisons


def _format_metric(value: float, *, bold: bool) -> str:
    formatted = f"{value:.2f}"
    return f"<b>{formatted}</b>" if bold else formatted


def update_readme(
    readme: str,
    records: ResultMap,
    *,
    benchmarks: Sequence[str] | None = None,
) -> str:
    """Update the selected README benchmark groups after validation."""
    selected = _normalize_benchmarks(benchmarks)
    start_idx = readme.find(_README_START_MARKER)
    end_idx = readme.find(_README_END_MARKER)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise BenchmarkError("README tracker table markers were not found in order")

    maxima = {
        (benchmark, metric): max(records[(benchmark, tracker, "python")][metric] for tracker in TRACKERS)
        for benchmark in selected
        for metric in METRICS
    }
    table = readme[start_idx : end_idx + len(_README_END_MARKER)]
    updated_trackers: set[str] = set()

    def update_row(match: re.Match[str]) -> str:
        row = match.group(0)
        cells = _README_CELL_PATTERN.findall(row)
        if len(cells) != 12:
            return row
        tracker = re.sub(r"<[^>]+>", "", cells[0][1]).strip()
        if tracker not in TRACKERS:
            return row

        replacements: dict[int, str] = {1: "✅"}
        for benchmark in selected:
            cell_indices = README_CELLS[benchmark]
            python_metrics = records[(benchmark, tracker, "python")]
            cpp_metrics = records[(benchmark, tracker, "cpp")] if tracker in NATIVE_TRACKERS else None
            for metric, cell_index in zip(METRICS, cell_indices, strict=True):
                python_value = python_metrics[metric]
                python_text = _format_metric(
                    python_value,
                    bold=math.isclose(python_value, maxima[(benchmark, metric)], abs_tol=1e-12),
                )
                cpp_text = f"{cpp_metrics[metric]:.2f}" if cpp_metrics else "—"
                replacements[cell_index] = f"{python_text}<br>({cpp_text})"

        index = 0

        def replace_cell(cell_match: re.Match[str]) -> str:
            nonlocal index
            replacement = replacements.get(index)
            index += 1
            if replacement is None:
                return cell_match.group(0)
            return f"{cell_match.group(1)}{replacement}{cell_match.group(3)}"

        updated_trackers.add(tracker)
        return _README_CELL_PATTERN.sub(replace_cell, row)

    updated_table = _README_ROW_PATTERN.sub(update_row, table)
    missing_rows = sorted(set(TRACKERS) - updated_trackers)
    if missing_rows:
        raise BenchmarkError(f"README tracker row(s) not found: {', '.join(missing_rows)}")
    return readme[:start_idx] + updated_table + readme[end_idx + len(_README_END_MARKER) :]


def write_csv(path: Path, records: ResultMap) -> None:
    """Write an auditable combined result file with all identity dimensions."""
    benchmark_order = {name: index for index, name in enumerate(BENCHMARKS)}
    tracker_order = {name: index for index, name in enumerate(TRACKERS)}
    backend_order = {name: index for index, name in enumerate(BACKENDS)}
    ordered = sorted(
        records,
        key=lambda key: (benchmark_order[key[0]], tracker_order[key[1]], backend_order[key[2]]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("benchmark", "tracker", "backend", *METRICS))
        for benchmark, tracker, backend in ordered:
            metrics = records[(benchmark, tracker, backend)]
            writer.writerow((benchmark, tracker, backend, *(f"{metrics[m]:.6f}" for m in METRICS)))


def _record_command(args: argparse.Namespace) -> None:
    write_record(
        benchmark=args.benchmark,
        tracker=args.tracker,
        backend=args.backend,
        metrics_path=args.metrics_file,
        output_path=args.output,
    )


def _combine_command(args: argparse.Namespace) -> None:
    benchmarks = _normalize_benchmarks(args.benchmarks)
    records = load_records(args.results_dir)
    comparisons = validate_records(records, tolerance=args.tolerance, benchmarks=benchmarks)
    write_csv(args.csv, records)
    args.readme.write_text(update_readme(args.readme.read_text(), records, benchmarks=benchmarks))
    print(
        f"Validated {len(records)} benchmark records and {len(comparisons)} "
        f"Python/C++ metric comparisons (<= {args.tolerance:.2f} pp)."
    )


def _verify_readme_command(args: argparse.Namespace) -> None:
    comparisons = validate_published_readme(args.readme.read_text(), tolerance=args.tolerance)
    print(f"Validated {len(comparisons)} published Python/C++ metric comparisons (<= {args.tolerance:.2f} pp).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="record one evaluator JSON result")
    record.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    record.add_argument("--tracker", choices=TRACKERS, required=True)
    record.add_argument("--backend", choices=BACKENDS, required=True)
    record.add_argument("--metrics-file", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.set_defaults(func=_record_command)

    combine = subparsers.add_parser("combine", help="validate and publish all results")
    combine.add_argument(
        "--benchmark",
        dest="benchmarks",
        action="append",
        choices=BENCHMARKS,
        help="benchmark group to validate/update; repeat to select multiple (default: all)",
    )
    combine.add_argument("--results-dir", type=Path, required=True)
    combine.add_argument("--readme", type=Path, required=True)
    combine.add_argument("--csv", type=Path, required=True)
    combine.add_argument("--tolerance", type=float, default=0.25)
    combine.set_defaults(func=_combine_command)

    verify_readme = subparsers.add_parser(
        "verify-readme",
        help="verify all published native Python/C++ README metric pairs",
    )
    verify_readme.add_argument("--readme", type=Path, required=True)
    verify_readme.add_argument("--tolerance", type=float, default=0.25)
    verify_readme.set_defaults(func=_verify_readme_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BenchmarkError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
