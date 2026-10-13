"""Export exact MOT metric comparisons from the local ECC/SOF ablation.

Run after the benchmark writes ``summary.json`` and its own trusted
``metric_bundles.pkl``::

    uv run --no-sync python -m tests.performance.trackers.motion.cmc_study_analysis \
        runs/cmc-study

The paired bootstrap resamples complete sequences and recomputes the standard
aggregate metrics. It does not average sequence-level HOTA percentages.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import pickle
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from boxmot.engine.eval.motmetrics import _combine_bundles, _summary_from_bundle

METRICS = ("HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDSW")
SUBSTANTIAL_HOTA_POINTS = 1.0
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


def _profile_identity(name: str, profile: Mapping[str, Any]) -> tuple[str, str]:
    """Identify a method and budget tier without relying on display labels."""
    metadata = profile.get("metadata", {})
    tokens = name.lower().replace("-", "_").split("_")
    method = str(metadata.get("method") or next((item for item in tokens if item in {"ecc", "sof"}), "off"))
    tier = str(metadata.get("tier") or next((item for item in tokens if item in {"fast", "balanced", "slow"}), "off"))
    return method, tier


def _summary(bundles: Mapping[str, Any], sequences: Sequence[str]) -> dict[str, Any]:
    """Use evaluator sufficient statistics for the exact selected aggregate."""
    if not sequences:
        raise ValueError("Cannot summarize an empty sequence subset.")
    return _summary_from_bundle(_combine_bundles({name: bundles[name] for name in sequences}))


def _bootstrap_metrics(
    bundles: Mapping[str, Any], sequences: Sequence[str], weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Reaggregate sufficient statistics for many sequence bootstrap draws.

    These formulas are the evaluator's count-weighted sequence aggregation,
    vectorized over bootstrap weights. The ordinary point estimates continue
    to use ``_combine_bundles`` directly.
    """
    values = [bundles[name] for name in sequences]

    def total(group: str, field: str) -> np.ndarray:
        return weights @ np.asarray([value[group][field] for value in values], dtype=float)

    tp = total("HOTA", "HOTA_TP")
    fn = total("HOTA", "HOTA_FN")
    fp = total("HOTA", "HOTA_FP")
    association_numerator = weights @ np.asarray(
        [value["HOTA"]["AssA"] * value["HOTA"]["HOTA_TP"] for value in values], dtype=float
    )
    association = association_numerator / np.maximum(1e-10, tp)
    detection = tp / np.maximum(1.0, tp + fn + fp)
    idtp, idfn, idfp = (total("Identity", name) for name in ("IDTP", "IDFN", "IDFP"))
    clear_tp, clear_fn, clear_fp = (total("CLEAR", name) for name in ("CLR_TP", "CLR_FN", "CLR_FP"))
    switches = total("CLEAR", "IDSW")
    return {
        "HOTA": 100.0 * np.mean(np.sqrt(detection * association), axis=1),
        "AssA": 100.0 * np.mean(association, axis=1),
        "DetA": 100.0 * np.mean(detection, axis=1),
        "IDF1": 100.0 * idtp / np.maximum(1.0, idtp + 0.5 * idfn + 0.5 * idfp),
        "MOTA": 100.0 * (clear_tp - clear_fp - switches) / np.maximum(1.0, clear_tp + clear_fn),
        "IDSW": switches,
    }


def _bootstrap_delta(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], sequences: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Use identical sequence draws for both sides of every paired comparison."""
    size = len(sequences)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, size, size=(BOOTSTRAP_DRAWS, size))
    weights = np.zeros((BOOTSTRAP_DRAWS, size), dtype=float)
    np.add.at(weights, (np.arange(BOOTSTRAP_DRAWS)[:, None], indices), 1.0)
    left = _bootstrap_metrics(reference, sequences, weights)
    right = _bootstrap_metrics(candidate, sequences, weights)
    return {
        metric: tuple(float(value) for value in np.percentile(right[metric] - left[metric], (2.5, 97.5)))
        for metric in METRICS
    }


def _timing(
    profile: Mapping[str, Any],
    sequences: Sequence[str],
    counts: Mapping[str, int],
    *,
    overall: bool,
    warmup: int = 0,
) -> tuple[float | None, float | None]:
    """Return aggregate latency, without incorrectly averaging p95 values."""
    timing = profile.get("timing", {})
    if overall:
        return timing.get("mean_ms"), timing.get("p95_ms")
    per_sequence = profile.get("per_sequence", {})
    if not all(name in per_sequence and "mean_ms" in per_sequence[name].get("timing", {}) for name in sequences):
        return None, None
    measured_counts = {}
    for name in sequences:
        entry = per_sequence[name]
        fallback = counts[name] - min(warmup, counts[name] - 1)
        count = int(entry.get("timing_frames", entry["timing"].get("timing_frames", fallback)))
        if count <= 0 or count > counts[name]:
            raise ValueError(f"Invalid measured timing frame count for {name}: {count}.")
        measured_counts[name] = count
    mean = sum(float(per_sequence[name]["timing"]["mean_ms"]) * measured_counts[name] for name in sequences)
    return mean / sum(measured_counts.values()), None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write flat numeric results suitable for external statistical analysis."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Export a conventional, standalone HOTA versus CMC latency figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = [row for row in rows if row["subset"] == "all"]
    trackers = sorted({str(row["tracker"]) for row in overall})
    figure, axes = plt.subplots(1, len(trackers), figsize=(5.1 * len(trackers), 4.5), squeeze=False)
    tier_order = {"fast": 0, "balanced": 1, "slow": 2}
    for axis, tracker in zip(axes[0], trackers):
        tracker_rows = [row for row in overall if row["tracker"] == tracker]
        for method, color, marker in (("ecc", "#0072B2", "o"), ("sof", "#D55E00", "s")):
            selected = sorted(
                (row for row in tracker_rows if row["method"] == method and row["cmc_mean_ms"] is not None),
                key=lambda row: tier_order.get(str(row["tier"]), 99),
            )
            if not selected:
                continue
            axis.plot(
                [row["cmc_mean_ms"] for row in selected],
                [row["HOTA"] for row in selected],
                marker=marker,
                color=color,
                label=method.upper(),
                linewidth=1.5,
            )
            for row in selected:
                axis.annotate(
                    str(row["tier"]),
                    (row["cmc_mean_ms"], row["HOTA"]),
                    xytext=(4, 5 if method == "ecc" else -14),
                    textcoords="offset points",
                    fontsize=8,
                )
        baseline = next((row for row in tracker_rows if row["method"] == "off"), None)
        if baseline is not None:
            axis.axhline(float(baseline["HOTA"]), color="#555555", linestyle=":", label="CMC off")
        axis.axvline(1.0, color="#777777", alpha=0.6, linestyle="--", label="1 ms mean")
        axis.set(title=tracker, xlabel="CMC mean latency (ms/frame)", ylabel="HOTA (points)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("ECC / SOF budget ablation · fixed tracker settings", fontsize=13)
    figure.tight_layout()
    figure.savefig(output / "speed_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _format(value: Any) -> str:
    """Escape strings and format numbers consistently for report tables."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value))


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Render a small HTML table with explicit accessible column headings."""
    header = "".join(f"<th scope='col'>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_format(row.get(column))}</td>" for column in columns) + "</tr>" for row in rows
    )
    return f"<div class='table'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def _reproduction_commands(source: Mapping[str, Any], output: Path) -> tuple[str, str]:
    """Render shell-safe commands with the recorded study configuration."""
    environment = source.get("environment", {})
    command = ["uv", "run", "--no-sync", "python", "-m", "tests.performance.trackers.motion.benchmark_cmc_accuracy"]
    options = [
        ["--build", str(source["build"])],
        ["--output", "<new-output-dir>"],
        ["--workers", str(environment.get("replay_workers", source.get("workers", 3)))],
        ["--warmup", str(source.get("warmup_frames_excluded_from_latency_per_sequence", 0))],
        ["--mask-min-conf", str(source.get("mask_min_conf", 0.1))],
    ]
    if source.get("gt_root"):
        options.append(["--gt-root", str(source["gt_root"])])
    for flag, key in (("--trackers", "trackers"), ("--profiles", "profile_names"), ("--sequences", "sequences")):
        if source.get(key):
            options.append([flag, *map(str, source[key])])
    reproduce = " \\\n    ".join(shlex.join(part) for part in [command, *options])
    analyze_again = shlex.join(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "tests.performance.trackers.motion.cmc_study_analysis",
            str(output.resolve()),
        ]
    )
    return reproduce, analyze_again


def _moving_findings(results: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> str:
    """Describe practically interesting camera-specific gains and their cost."""
    candidates = [
        row for row in comparisons if row["subset"] == "moving" and row["delta_HOTA"] >= SUBSTANTIAL_HOTA_POINTS
    ]
    if not candidates:
        return ""
    descriptions = [
        f"{row['tracker']} {row['candidate']}: {row['delta_HOTA']:+.3f} HOTA "
        f"(95% interval {row['delta_HOTA_ci_low']:+.3f} to {row['delta_HOTA_ci_high']:+.3f})"
        for row in candidates
    ]
    text = "The moving-camera subset contains an exception: " + "; ".join(descriptions) + "."
    if all(row["delta_HOTA_ci_low"] < SUBSTANTIAL_HOTA_POINTS for row in candidates):
        text += (
            " These intervals still include gains below +1 HOTA; camera-specific benefits need independent validation."
        )
    paragraphs = [text]
    overall = {(row["tracker"], row["profile"]): row for row in results if row["subset"] == "all"}
    for balanced in candidates:
        balanced_result = overall[balanced["tracker"], balanced["candidate"]]
        if balanced_result["tier"] != "balanced":
            continue
        historic = next(
            (
                row
                for row in candidates
                if row["tracker"] == balanced["tracker"]
                and row["method"] == balanced["method"]
                and overall[row["tracker"], row["candidate"]]["tier"] == "slow"
            ),
            None,
        )
        if historic is None or historic["delta_HOTA"] <= 0:
            continue
        fraction = balanced["delta_HOTA"] / historic["delta_HOTA"]
        balanced_ms = balanced_result["cmc_mean_ms"]
        historic_ms = overall[historic["tracker"], historic["candidate"]]["cmc_mean_ms"]
        if (
            0.75 <= fraction <= 1.0
            and balanced_ms is not None
            and historic_ms is not None
            and balanced_ms < historic_ms
        ):
            paragraphs.append(
                f"For {balanced['tracker']} on moving cameras, {balanced['candidate']} captured "
                f"{100 * fraction:.0f}% of {historic['candidate']}'s observed HOTA gain, with overall mean CMC "
                f"latency {balanced_ms:.3f} ms versus {historic_ms:.3f} ms. The balanced profile is the "
                "more economical option to validate for that use case."
            )
    alternate_baseline = []
    by_key = {(row["tracker"], row["profile"], row["subset"]): row for row in results}
    for candidate in candidates:
        if candidate["method"] != "ecc":
            continue
        fast_sof = next(
            (
                row
                for row in results
                if row["tracker"] == candidate["tracker"]
                and row["method"] == "sof"
                and row["tier"] == "fast"
                and row["subset"] == "all"
            ),
            None,
        )
        if fast_sof is None:
            continue
        delta_all = overall[candidate["tracker"], candidate["candidate"]]["HOTA"] - fast_sof["HOTA"]
        delta_moving = (
            by_key[candidate["tracker"], candidate["candidate"], "moving"]["HOTA"]
            - by_key[candidate["tracker"], fast_sof["profile"], "moving"]["HOTA"]
        )
        alternate_baseline.append(
            f"{candidate['tracker']} {candidate['candidate']} {delta_all:+.3f} overall / {delta_moving:+.3f} moving"
        )
    if alternate_baseline:
        paragraphs.append(
            "Against fast SOF instead, the observed HOTA differences are "
            + "; ".join(alternate_baseline)
            + ". This changes the motion model as well as the budget; these differences are descriptive."
        )
    return "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)


def _baseline_findings(results: Sequence[Mapping[str, Any]]) -> str:
    """Summarize whether fast SOF already improves over disabled compensation."""
    indexed = {(row["tracker"], row["method"], row["tier"], row["subset"]): row for row in results}
    improvements: dict[str, list[float]] = {}
    trackers = {row["tracker"] for row in results}
    for subset in ("all", "moving"):
        deltas = []
        for tracker in sorted(trackers):
            fast = indexed.get((tracker, "sof", "fast", subset))
            off = indexed.get((tracker, "off", "off", subset))
            if fast is not None and off is not None:
                deltas.append(fast["HOTA"] - off["HOTA"])
        if len(deltas) == len(trackers) and all(value > 0 for value in deltas):
            improvements[subset] = deltas
    if "all" not in improvements:
        return ""
    values = improvements["all"]
    text = (
        f"Fast SOF itself improved HOTA over CMC off in all {len(values)} trackers "
        f"({min(values):+.3f} to {max(values):+.3f} overall"
    )
    if "moving" in improvements:
        values = improvements["moving"]
        text += f"; {min(values):+.3f} to {max(values):+.3f} on moving cameras"
    return f"<p>{html.escape(text + ').')}</p>"


def _data_scope(source: Mapping[str, Any]) -> str:
    """Describe frozen perception inputs without implying model inference ran."""
    sequences = source["sequences"]
    counts = source.get("data_counts", {})
    text = f"The study replays {sum(sequences.values()):,} frames across {len(sequences)} sequences"
    if "instances" in counts:
        text += f" from a frozen build containing {int(counts['instances']):,} cached detections"
    text += "."
    components = source.get("data_components", {})
    labels = {"yolox_x_MOT17_ablation.pt": "YOLOX-X MOT17 ablation", "lmbn_n_duke.pt": "LMBN-N Duke"}
    detector = components.get("detector", {})
    detector_path = detector.get("artifact", {}).get("path") if detector else None
    if detector_path:
        filename = Path(detector_path).name
        text += f" Detections come from {labels.get(filename, filename)} ({filename})."
    encoder = components.get("reid", {})
    encoder_path = encoder.get("artifact", {}).get("path") if encoder else None
    if encoder_path and "botsort" in source.get("trackers", ()):
        filename = Path(encoder_path).name
        text += f" BoTSORT alone consumes the cached {labels.get(filename, filename)} embeddings ({filename})."
    if set(source.get("trackers", ())) & {"bytetrack", "ocsort"}:
        text += " ByteTrack and OCSORT use no embeddings."
    text += " No detector or ReID inference is rerun."
    return f"<p>{html.escape(text)}</p>"


def _report(
    output: Path,
    source: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    per_sequence: Sequence[Mapping[str, Any]],
) -> None:
    """Describe measured evidence and its scope in a self-contained report."""
    overall = [row for row in comparisons if row["subset"] == "all"]
    substantial = [row for row in overall if row["delta_HOTA"] >= SUBSTANTIAL_HOTA_POINTS]
    supported = [row for row in substantial if row["delta_HOTA_ci_low"] >= SUBSTANTIAL_HOTA_POINTS]
    evidence = (
        f"{len(substantial)} of {len(overall)} slower-versus-fast comparisons gained at least "
        f"{SUBSTANTIAL_HOTA_POINTS:g} HOTA point overall; {len(supported)} had a paired sequence-bootstrap "
        "95% interval entirely above that threshold."
    )
    if overall and not substantial:
        evidence += " These results do not support spending extra CMC time as a blanket default across trackers."
    if overall:
        best = max(overall, key=lambda row: row["delta_HOTA"])
        evidence += (
            f" The largest observed gain was {best['delta_HOTA']:+.3f} points for {best['tracker']} "
            f"{best['candidate']} versus {best['reference']} "
            f"(95% interval {best['delta_HOTA_ci_low']:+.3f} to {best['delta_HOTA_ci_high']:+.3f})."
        )
    picture = base64.b64encode((output / "speed_accuracy.png").read_bytes()).decode("ascii")
    rows_all = [row for row in results if row["subset"] == "all"]
    display_comparisons = []
    for row in comparisons:
        display_comparisons.append(
            {
                **row,
                "HOTA 95% CI": f"[{row['delta_HOTA_ci_low']:+.3f}, {row['delta_HOTA_ci_high']:+.3f}]",
                "sequence gains": f"{row['sequences_hota_positive']}/{row['sequence_count']}",
            }
        )
    methods = {
        "build": source.get("build"),
        "gt_root": source.get("gt_root"),
        "data_components": source.get("data_components"),
        "data_counts": source.get("data_counts"),
        "sequences": source.get("sequences"),
        "moving_sequences": source.get("moving_sequences"),
        "profiles": source.get("cmc_profiles"),
        "tracker_settings": source.get("tracker_settings", source.get("tracker_configs")),
        "environment": source.get("environment"),
        "mask_policy": source.get("mask_policy"),
        "warmup_frames_excluded_from_latency_per_sequence": source.get(
            "warmup_frames_excluded_from_latency_per_sequence"
        ),
        "motion_strata_source": source.get("motion_strata_source"),
    }
    comparison_table = _table(
        display_comparisons,
        (
            "tracker",
            "reference",
            "candidate",
            "subset",
            "delta_cmc_mean_ms",
            "delta_HOTA",
            "HOTA 95% CI",
            "delta_IDF1",
            "delta_MOTA",
            "delta_AssA",
            "delta_IDSW",
            "sequence gains",
            "assessment",
        ),
    )
    subset_table = _table(
        [row for row in results if row["subset"] != "all"],
        ("tracker", "profile", "subset", "sequence_count", "cmc_mean_ms", *METRICS),
    )
    recorded_methods = html.escape(json.dumps(methods, indent=2, default=str))
    reproduce, analyze_again = (html.escape(command) for command in _reproduction_commands(source, output))
    moving_findings = _moving_findings(results, comparisons)
    baseline_findings = _baseline_findings(results)
    data_scope = _data_scope(source)
    diagnostic_links = [
        f'<a href="{name}">{name}</a>'
        for name in ("warp_diagnostics.csv", "warp_diagnostics.json")
        if (output / name).exists()
    ]
    diagnostics = (
        "<p>Motion diagnostics: "
        + ", ".join(diagnostic_links)
        + ". These describe transform stability and disagreement, not motion accuracy: camera-motion ground truth "
        "is unavailable. Identity outputs may represent static scenes, rejected estimates or reinitialization.</p>"
        if diagnostic_links
        else ""
    )
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECC and SOF speed–tracking accuracy study</title>
<style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:1400px;margin:40px auto;padding:0 24px;color:#172a3a}}
h1,h2{{line-height:1.2}}h2{{margin-top:2em}}p{{max-width:100ch}}img{{max-width:100%;height:auto}}
.table{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;width:100%;margin:16px 0}}
th,td{{padding:7px 10px;border-bottom:1px solid #dce3ea;text-align:right;white-space:nowrap}}
th{{background:#edf3f8}}th:first-child,td:first-child{{text-align:left}}pre{{overflow:auto;font-size:12px}}
.lead{{font-size:19px}}.note{{background:#edf3f8;padding:16px 22px;border-left:4px solid #0072B2}}
</style><body>
<h1>Do slower ECC and SOF improve tracking?</h1>
<p class="lead">{html.escape(evidence)}</p>
{moving_findings}
<p class="note">This is an exploratory study on the available MOT17 ablation sequences, which are tuning data,
not a held-out MOT benchmark. A practical gain was defined before measuring results as at least +1 HOTA point.
The {len(source["sequences"])} sequences provide limited evidence about other cameras or datasets.
Multiple comparisons are not corrected for selection; the best observed result is descriptive.</p>
{data_scope}
<h2>Speed and accuracy</h2>
<p>Latency measures CMC estimation alone, including estimator preprocessing and state updates. It excludes
image decoding, detections, ReID, tracker association and evaluation. It is not end-to-end tracker FPS,
and a mean below 1 ms is not a per-frame latency guarantee.</p>
{baseline_findings}
<img alt="HOTA versus mean CMC latency, one panel per tracker" src="data:image/png;base64,{picture}">
{_table(rows_all, ("tracker", "profile", "cmc_mean_ms", "cmc_p95_ms", *METRICS))}
<h2>Paired comparisons</h2>
<p>Each candidate is compared against the fast version of the same method in the same tracker.
Non-CMC settings, cached detections, cached embeddings and the foreground mask confidence remain fixed.
Every estimator receives identical foreground box inputs. Historical and current implementations may
rasterize masks differently because their preprocessing resolution, rounding and mask application differ.
CMC is enabled on every sequence for these comparisons; no automatic sequence gate, ground-truth-driven
sequence choice, or Kalman refit changes the primary comparison. Positive deltas improve HOTA, IDF1, MOTA,
AssA and DetA; negative IDSW deltas reduce identity switches.</p>
<p>The balanced profiles increase computation budgets within the current implementations. The slow profiles
load the pinned historical implementations and compare the entire prior estimator against the fast version.
Profile names such as <code>ecc-balanced</code> are study labels, not accepted tracker
<code>cmc_method</code> values; use the recorded factories and settings to reproduce them.
These historical comparisons change preprocessing and estimator behavior together, including smoothing or
feature refresh and current-frame masking where applicable. They cannot isolate computation budget alone
as the cause of a quality difference. Recorded profile metadata and source hashes identify these changes.</p>
<p>ECC estimates translation; SOF estimates a similarity transform including rotation and scale, so differences
between methods are not solely budget effects. The historical 0.15 image scale produces a 288-pixel working
width on 1920-pixel images but only 96 pixels on 640-pixel MOT17-05 images, compared with fixed SOF widths
of 192 and 320 for fast and balanced. “Slow” identifies the historical profile, not a guarantee of greater
resolution or latency on every sequence. Changing SOF resolution also changes the full-resolution tolerance
represented by its fixed RANSAC threshold in resized pixels.</p>
<p>Intervals use {BOOTSTRAP_DRAWS:,} paired bootstrap draws of complete sequences, with seed {BOOTSTRAP_SEED}.
Both configurations receive the same draw and duplicate sampled sequences retain their full weight.
HOTA and other metrics are recomputed from sufficient statistics, not averaged as percentages.
Intervals quantify sequence variation, not timing noise. Static and moving subsets use the study's fixed
image-only sequence grouping; neither subset changes CMC activation. Their smaller sample sizes make
intervals particularly unstable, and a one-sequence interval is degenerate.</p>
{comparison_table}
<h2>Moving and static aggregates</h2>
<p>Sequence bundles are recombined using the evaluator's standard detection and association weighting.
Subset latency means are weighted by measured frame counts after excluding each sequence's warmup;
p95 is omitted because per-sequence
percentiles cannot be pooled correctly without the individual timings.</p>
{subset_table}
<h2>Per-sequence tracking metrics</h2>
{_table(per_sequence, ("tracker", "profile", "sequence", "motion_group", *METRICS))}
<h2>Reproduction and artifacts</h2>
<p>From the repository root, using the recorded source versions and dependency environment, rerun the full
study with the command below. Replace the quoted output placeholder with a new directory. The original
materialized build, source images, ground truth and pinned historical Git objects must remain available.</p>
<pre><code>{reproduce}</code></pre>
<p>Regenerate this report and its statistics from the existing study artifacts without rerunning estimators
or trackers:</p>
<pre><code>{analyze_again}</code></pre>
<p>Machine-readable files: <a href="summary.json">summary.json</a>,
<a href="data-manifest.json">data-manifest.json</a>,
<a href="results.csv">results.csv</a>, <a href="comparisons.csv">comparisons.csv</a>,
<a href="per_sequence.csv">per_sequence.csv</a>, <a href="analysis.json">analysis.json</a>.
Validation: <a href="validation.json">validation.json</a>.
Standalone figure: <a href="speed_accuracy.png">speed_accuracy.png</a>.
The benchmark stores raw sequence metric bundles and tracker outputs for exact recomputation.</p>
{diagnostics}
<details><summary>Recorded configuration and provenance</summary><pre>{recorded_methods}</pre></details>
</body></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


def analyze(output_dir: Path) -> dict[str, Any]:
    """Read completed study artifacts and publish numeric and visual analysis."""
    output = Path(output_dir)
    source = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    # The caller owns this benchmark artifact; never load untrusted pickle files.
    with (output / "metric_bundles.pkl").open("rb") as handle:
        bundles = pickle.load(handle)
    counts = {str(name): int(count) for name, count in source["sequences"].items()}
    sequences = tuple(sorted(counts))
    moving = frozenset(source.get("moving_sequences", ()))
    if not moving.issubset(sequences):
        raise ValueError("Moving-sequence IDs must belong to the evaluated sequence set.")
    subsets = {
        "all": sequences,
        "moving": tuple(name for name in sequences if name in moving),
        "static": tuple(name for name in sequences if name not in moving),
    }
    subsets = {name: selected for name, selected in subsets.items() if selected}
    profiles = source["cmc_profiles"]
    warmup = int(source.get("warmup_frames_excluded_from_latency_per_sequence", 0))
    if warmup < 0:
        raise ValueError("Timing warmup must be nonnegative.")
    rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    for run in source["runs"]:
        tracker, profile = str(run["tracker"]), str(run["profile"])
        raw = bundles[tracker, profile]
        if set(raw) != set(sequences):
            raise ValueError(f"Sequence mismatch in metric bundles for {tracker}/{profile}.")
        method, tier = _profile_identity(profile, profiles.get(profile, {}))
        for subset, selected in subsets.items():
            metrics = _summary(raw, selected)
            mean, p95 = _timing(profiles.get(profile, {}), selected, counts, overall=subset == "all", warmup=warmup)
            row = {
                "tracker": tracker,
                "profile": profile,
                "method": method,
                "tier": tier,
                "subset": subset,
                "sequence_count": len(selected),
                "frames": sum(counts[name] for name in selected),
                "cmc_mean_ms": mean,
                "cmc_p95_ms": p95,
                **{metric: metrics[metric] for metric in METRICS},
            }
            rows.append(row)
        for sequence in sequences:
            metrics = _summary_from_bundle(raw[sequence])
            sequence_rows.append(
                {
                    "tracker": tracker,
                    "profile": profile,
                    "sequence": sequence,
                    "motion_group": "moving" if sequence in moving else "static",
                    **{metric: metrics[metric] for metric in METRICS},
                }
            )
    comparisons: list[dict[str, Any]] = []
    for candidate in rows:
        if candidate["method"] not in {"ecc", "sof"} or candidate["tier"] not in {"balanced", "slow"}:
            continue
        reference = next(
            (
                row
                for row in rows
                if row["tracker"] == candidate["tracker"]
                and row["method"] == candidate["method"]
                and row["tier"] == "fast"
                and row["subset"] == candidate["subset"]
            ),
            None,
        )
        if reference is None:
            raise ValueError(f"No matching fast reference for {candidate['tracker']}/{candidate['profile']}.")
        selected = subsets[candidate["subset"]]
        left = bundles[reference["tracker"], reference["profile"]]
        right = bundles[candidate["tracker"], candidate["profile"]]
        intervals = _bootstrap_delta(left, right, selected)
        deltas = {f"delta_{metric}": candidate[metric] - reference[metric] for metric in METRICS}
        assessment = "below +1 HOTA threshold"
        if deltas["delta_HOTA"] >= SUBSTANTIAL_HOTA_POINTS:
            assessment = (
                "gain and interval exceed +1 HOTA"
                if intervals["HOTA"][0] >= SUBSTANTIAL_HOTA_POINTS
                else "estimated gain; uncertainty crosses +1 HOTA"
            )
        mean_left, mean_right = reference["cmc_mean_ms"], candidate["cmc_mean_ms"]
        comparisons.append(
            {
                "tracker": candidate["tracker"],
                "method": candidate["method"],
                "reference": reference["profile"],
                "candidate": candidate["profile"],
                "subset": candidate["subset"],
                "sequence_count": len(selected),
                "delta_cmc_mean_ms": None if mean_left is None or mean_right is None else mean_right - mean_left,
                "cmc_mean_ratio": None if not mean_left or mean_right is None else mean_right / mean_left,
                **deltas,
                **{
                    f"delta_{metric}_ci_{bound}": value
                    for metric, bounds in intervals.items()
                    for bound, value in zip(("low", "high"), bounds)
                },
                "sequences_hota_positive": sum(
                    _summary_from_bundle(right[name])["HOTA"] > _summary_from_bundle(left[name])["HOTA"]
                    for name in selected
                ),
                "assessment": assessment,
            }
        )
    rows.sort(key=lambda row: (row["tracker"], row["subset"], row["profile"]))
    comparisons.sort(key=lambda row: (row["tracker"], row["subset"], row["candidate"]))
    analysis = {
        "substantial_hota_points": SUBSTANTIAL_HOTA_POINTS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "paired sequence",
        "results": rows,
        "comparisons": comparisons,
    }
    _write_csv(output / "results.csv", rows)
    _write_csv(output / "comparisons.csv", comparisons)
    _write_csv(output / "per_sequence.csv", sequence_rows)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    _plot(output, rows)
    _report(output, source, rows, comparisons, sequence_rows)
    return analysis


def main() -> None:
    """Regenerate analysis without rerunning estimators or trackers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
