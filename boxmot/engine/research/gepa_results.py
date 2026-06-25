#!/usr/bin/env python3
"""Summarize and plot GEPA research runs."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CandidateMetrics:
    """Combined benchmark metrics for a GEPA candidate."""

    idx: int
    score: float
    hota: float
    idf1: float
    mota: float
    delta_hota: float
    delta_idf1: float
    delta_mota: float


@dataclass(frozen=True)
class AcceptedTransition:
    """Accepted candidate transition extracted from a GEPA run."""

    iteration: int
    parent_idx: int
    child_idx: int
    changed_files: tuple[str, ...]
    added_lines: int
    removed_lines: int
    headline: str


@dataclass(frozen=True)
class GEPARunReport:
    """Structured summary of a GEPA run directory."""

    run_dir: Path
    accepted_candidate_indices: list[int]
    candidate_metrics: list[CandidateMetrics]
    accepted_transitions: list[AcceptedTransition]

    @property
    def best_candidate_idx(self) -> int:
        return max(self.candidate_metrics, key=lambda row: row.hota).idx


_FUNCTION_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_HEADLINE_RULES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"_rescue_unmatched_tracks"}), "Added recovery pass for unmatched and recently lost tracks"),
    (frozenset({"_apply_ambiguity_suppression"}), "Added ambiguity suppression for dense overlap cases"),
    (frozenset({"_detection_duplicate_penalties"}), "Added duplicate-detection and low-confidence penalties"),
    (
        frozenset({"_normalized_center_distance", "_velocity_direction_consistency"}),
        "Added motion-aware gating and velocity consistency shaping",
    ),
    (
        frozenset({"_apply_association_priors", "_track_is_confirmed"}),
        "Added lifecycle-aware association priors and delayed confirmation",
    ),
)


def _candidate_hash(candidate: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve_cache_file(run_dir: Path, candidate: dict[str, str]) -> Path:
    candidate_hash = _candidate_hash(candidate)
    matches = sorted((run_dir / "fitness_cache").glob(f"{candidate_hash[:16]}_*.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one fitness cache file for candidate {candidate_hash[:16]}, found {len(matches)}"
        )
    return matches[0]


def _parse_cache_metrics(cache_path: Path, idx: int) -> CandidateMetrics:
    cached = pickle.loads(cache_path.read_bytes())
    result = cached["result"]
    if not isinstance(result, tuple) or len(result) < 3:
        raise ValueError(f"Unsupported cache payload in {cache_path}")

    score = float(result[0])
    payload = result[2]
    combined = payload["Combined Metrics"]
    delta = payload.get("Combined Delta vs Baseline", {})

    return CandidateMetrics(
        idx=idx,
        score=score,
        hota=float(combined["HOTA"]),
        idf1=float(combined["IDF1"]),
        mota=float(combined["MOTA"]),
        delta_hota=float(delta.get("HOTA", 0.0)),
        delta_idf1=float(delta.get("IDF1", 0.0)),
        delta_mota=float(delta.get("MOTA", 0.0)),
    )


def _changed_files(parent: dict[str, str], child: dict[str, str]) -> tuple[str, ...]:
    changed = []
    for path in sorted(set(parent) | set(child)):
        if parent.get(path) != child.get(path):
            changed.append(path)
    return tuple(changed)


def _count_changed_lines(parent_text: str, child_text: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in difflib.unified_diff(parent_text.splitlines(), child_text.splitlines(), lineterm=""):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _added_function_names(parent_text: str, child_text: str) -> set[str]:
    before = {match.group(1) for match in map(_FUNCTION_RE.match, parent_text.splitlines()) if match}
    after = {match.group(1) for match in map(_FUNCTION_RE.match, child_text.splitlines()) if match}
    return after - before


def _summarize_transition(parent: dict[str, str], child: dict[str, str]) -> tuple[str, int, int]:
    changed = _changed_files(parent, child)
    total_added = 0
    total_removed = 0
    added_functions: set[str] = set()

    for path in changed:
        before = parent.get(path, "")
        after = child.get(path, "")
        added, removed = _count_changed_lines(before, after)
        total_added += added
        total_removed += removed
        if path.endswith(".py"):
            added_functions.update(_added_function_names(before, after))

    for required_funcs, headline in _HEADLINE_RULES:
        if required_funcs.issubset(added_functions):
            return headline, total_added, total_removed

    if added_functions:
        top_funcs = ", ".join(sorted(added_functions)[:3])
        return f"Added new helper logic ({top_funcs})", total_added, total_removed

    if changed:
        paths = ", ".join(Path(path).name for path in changed[:2])
        return f"Adjusted {paths}", total_added, total_removed

    return "No file changes recorded", total_added, total_removed


def build_gepa_report(run_dir: str | Path) -> GEPARunReport:
    """Build a structured report from a GEPA run directory."""

    run_dir = Path(run_dir).resolve()
    candidates = _load_json(run_dir / "candidates.json")
    run_log = _load_json(run_dir / "run_log.json")

    candidate_metrics = [
        _parse_cache_metrics(_resolve_cache_file(run_dir, candidate), idx)
        for idx, candidate in enumerate(candidates)
    ]

    accepted_indices = [0]
    transitions: list[AcceptedTransition] = []
    for row in run_log:
        if "new_program_idx" not in row:
            continue

        parent_idx = int(row["selected_program_candidate"])
        child_idx = int(row["new_program_idx"])
        headline, added_lines, removed_lines = _summarize_transition(
            candidates[parent_idx],
            candidates[child_idx],
        )
        transitions.append(
            AcceptedTransition(
                iteration=int(row["i"]) + 1,
                parent_idx=parent_idx,
                child_idx=child_idx,
                changed_files=_changed_files(candidates[parent_idx], candidates[child_idx]),
                added_lines=added_lines,
                removed_lines=removed_lines,
                headline=headline,
            )
        )
        accepted_indices.append(child_idx)

    return GEPARunReport(
        run_dir=run_dir,
        accepted_candidate_indices=accepted_indices,
        candidate_metrics=candidate_metrics,
        accepted_transitions=transitions,
    )


def plot_hota_evolution(report: GEPARunReport, output_path: str | Path) -> Path:
    """Plot HOTA over accepted candidates and save it to disk."""

    output_path = Path(output_path)
    accepted_metrics = [report.candidate_metrics[idx] for idx in report.accepted_candidate_indices]
    x = list(range(len(accepted_metrics)))
    hota = [row.hota for row in accepted_metrics]
    labels = ["baseline"] + [f"iter {row.iteration}" for row in report.accepted_transitions]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, hota, marker="o", linewidth=2.2, color="#005f73")
    ax.fill_between(x, hota, [min(hota)] * len(hota), alpha=0.08, color="#0a9396")

    ax.set_xticks(x)
    ax.set_xticklabels([f"c{row.idx}" for row in accepted_metrics])
    ax.set_xlabel("Accepted candidate")
    ax.set_ylabel("HOTA")
    ax.set_title("GEPA accepted-candidate HOTA evolution")
    ax.grid(True, linestyle="--", alpha=0.35)

    for xpos, metric, label in zip(x, accepted_metrics, labels):
        ax.annotate(
            f"{label}\n{metric.hota:.3f}",
            (xpos, metric.hota),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _summary_lines(report: GEPARunReport) -> list[str]:
    baseline = report.candidate_metrics[0]
    best = report.candidate_metrics[report.best_candidate_idx]
    lines = [
        f"Run dir: {report.run_dir}",
        f"Baseline candidate c0: HOTA {baseline.hota:.3f}, IDF1 {baseline.idf1:.3f}, MOTA {baseline.mota:.3f}",
        (
            f"Best candidate c{best.idx}: HOTA {best.hota:.3f} "
            f"(delta {best.delta_hota:+.3f}), IDF1 {best.idf1:.3f}, MOTA {best.mota:.3f}"
        ),
        "",
        "Accepted transitions:",
    ]

    metrics_by_idx = {row.idx: row for row in report.candidate_metrics}
    for transition in report.accepted_transitions:
        child = metrics_by_idx[transition.child_idx]
        changed_files = ", ".join(transition.changed_files)
        lines.extend(
            [
                (
                    f"- Iteration {transition.iteration}: c{transition.parent_idx} -> c{transition.child_idx} | "
                    f"HOTA {child.hota:.3f} ({child.delta_hota:+.3f} vs baseline) | "
                    f"IDF1 {child.idf1:.3f} ({child.delta_idf1:+.3f}) | "
                    f"MOTA {child.mota:.3f} ({child.delta_mota:+.3f})"
                ),
                f"  Main change: {transition.headline}",
                f"  Diff size: +{transition.added_lines} / -{transition.removed_lines} lines",
                f"  Changed files: {changed_files}",
            ]
        )

    return lines


def write_report_artifacts(report: GEPARunReport, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Write plot, text summary, and JSON summary for a GEPA report."""

    output_dir = report.run_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_hota_evolution(report, output_dir / "accepted_hota_evolution.png")
    summary_path = output_dir / "accepted_hota_summary.txt"
    json_path = output_dir / "accepted_hota_summary.json"

    summary_path.write_text("\n".join(_summary_lines(report)) + "\n")
    json_path.write_text(
        json.dumps(
            {
                "run_dir": str(report.run_dir),
                "accepted_candidate_indices": report.accepted_candidate_indices,
                "candidate_metrics": [asdict(row) for row in report.candidate_metrics],
                "accepted_transitions": [asdict(row) for row in report.accepted_transitions],
                "best_candidate_idx": report.best_candidate_idx,
            },
            indent=2,
        )
        + "\n"
    )

    return {
        "plot": plot_path,
        "summary": summary_path,
        "json": json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HOTA evolution and summarize accepted GEPA candidates.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a GEPA run directory containing candidates.json, run_log.json, and fitness_cache/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the plot and summaries will be written. Defaults to --run-dir.",
    )
    args = parser.parse_args()

    report = build_gepa_report(args.run_dir)
    outputs = write_report_artifacts(report, args.output_dir)

    print(f"Wrote plot: {outputs['plot']}")
    print(f"Wrote text summary: {outputs['summary']}")
    print(f"Wrote JSON summary: {outputs['json']}")


if __name__ == "__main__":
    main()
