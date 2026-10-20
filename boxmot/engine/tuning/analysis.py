"""Generate descriptive, conditional-aware reports for saved tuning results."""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import fill
from typing import Any

import numpy as np
import pandas as pd

from boxmot.engine.tuning.analysis_data import NEAR_BEST_DELTA, ParameterSamples, TrialAnalysis, load_tune_analysis

LOGGER = logging.getLogger(__name__)
SAMPLE_COLOR = "#3976a8"
NEAR_COLOR = "#df9b19"
BASELINE_COLOR = "#16836b"
BEST_COLOR = "#c93b42"
IDF1_COLOR = "#8656a8"
CATEGORY_COLORS = ("#3976a8", "#e58a32", "#5b9b60", "#b66ba0", "#7e7e7e", "#53a5ac")


def _grid(count: int, *, columns: int = 3, width: float = 5.2, height: float = 3.9) -> tuple[Any, list[Any]]:
    """Allocate only populated axes, spreading the final row across the figure."""
    import matplotlib.pyplot as plt

    columns = min(columns, count)
    rows = (count + columns - 1) // columns
    fig = plt.figure(figsize=(width * columns, height * rows), layout="constrained")
    grid = fig.add_gridspec(rows, columns)
    axes = []
    for i in range(count):
        row, col = divmod(i, columns)
        stop = columns if i == count - 1 else col + 1
        axes.append(fig.add_subplot(grid[row, col:stop]))
    return fig, axes


def _special_points(ax: Any, data: TrialAnalysis, points: pd.DataFrame, x: str, y: str = "HOTA") -> None:
    """Apply the same near-best outline and selected-trial markers in every plot."""
    near = points.loc[points.index.intersection(data.trials.index[data.near_best])]
    ax.scatter(near[x], near[y], s=44, facecolors="none", edgecolors=NEAR_COLOR, linewidths=1.2, zorder=3)
    selected = [(data.baseline_index, "D", BASELINE_COLOR, 64), (data.best_index, "*", BEST_COLOR, 170)]
    if "IDF1" in data.trials and data.trials["IDF1"].notna().any():
        selected.insert(1, (data.trials["IDF1"].idxmax(), "P", IDF1_COLOR, 85))
    for index, marker, color, size in selected:
        if index in points.index and pd.notna(points.at[index, x]) and pd.notna(points.at[index, y]):
            ax.scatter(
                [points.at[index, x]],
                [points.at[index, y]],
                marker=marker,
                s=size,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )


def _save(fig: Any, path: Path, data: TrialAnalysis, *, footer: str = "", point_legend: bool = True) -> None:
    """Save with shared legends and enough room for labels and provenance notes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], linestyle="none", marker="o", color=SAMPLE_COLOR, label="Trial"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=NEAR_COLOR,
            label=f"Within {NEAR_BEST_DELTA:g} HOTA of best",
        ),
        Line2D([], [], linestyle="none", marker="D", color=BASELINE_COLOR, label=data.baseline_label),
        Line2D([], [], linestyle="none", marker="*", markersize=11, color=BEST_COLOR, label="Best HOTA"),
    ]
    if "IDF1" in data.trials and data.trials["IDF1"].notna().any():
        handles.append(Line2D([], [], linestyle="none", marker="P", color=IDF1_COLOR, label="Best IDF1"))
    if point_legend:
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=len(handles), frameon=False, fontsize=9
        )
    if footer:
        footer = "\n".join(fill(line, width=120) for line in footer.splitlines())
        fig.text(0.5, -0.05 if point_legend else -0.015, footer, ha="center", va="top", fontsize=9, color="#555555")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _score_lines(ax: Any, data: TrialAnalysis) -> None:
    """Draw actual scores and the cumulative incumbent in recorded completion order."""
    trials = data.trials
    x = np.arange(1, len(trials) + 1)
    ax.scatter(x, trials["HOTA"], s=14, alpha=0.35, color=SAMPLE_COLOR)
    ax.step(x, trials["HOTA"].cummax(), where="post", lw=1.7, color=SAMPLE_COLOR)
    ax.axhline(trials.at[data.baseline_index, "HOTA"], color=BASELINE_COLOR, ls=":", lw=1)
    points = trials.assign(completion=x)
    _special_points(ax, data, points, "completion")
    ax.set_xlim(0.5, len(trials) + 0.5)


def _overview(data: TrialAnalysis, directory: Path, tracker: str, n_trials: int | None) -> None:
    """Summarize progress, score concentration, and directly observed metric tradeoffs."""
    trials = data.trials
    tradeoffs = [("DetA", "AssA"), ("HOTA", "IDF1")]
    available = [(x, y) for x, y in tradeoffs if {x, y}.issubset(trials) and trials[[x, y]].dropna().shape[0] > 0]
    omitted = [f"{x} vs {y}: metrics unavailable" for x, y in tradeoffs if (x, y) not in available]
    fig, axes = _grid(2 + len(available), columns=2, width=7, height=4.7)
    best = trials.at[data.best_index, "HOTA"]
    baseline = trials.at[data.baseline_index, "HOTA"]
    count = len(trials)
    title = f"{tracker or 'Tracker'} tuning overview — {count} valid trials"
    if n_trials is not None and n_trials != count:
        title += f" / {n_trials} reported"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    progress, distribution = axes[:2]
    _score_lines(progress, data)
    order_label = "Completed trial" if data.chronological else "CSV row (completion order unavailable)"
    following_label = "later trials" if data.chronological else "following rows"
    progress.set(xlabel=order_label, ylabel="HOTA", title="Search progress")
    progress.text(
        0.03,
        0.04,
        f"Best {best:.4f} at #{data.best_index + 1}\n"
        f"Reference {baseline:.4f} · gain {best - baseline:+.4f}\n"
        f"{count - data.best_index - 1} {following_label} without improvement",
        transform=progress.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    zoom = progress.inset_axes([0.42, 0.16, 0.55, 0.40])
    _score_lines(zoom, data)
    zoom.axhspan(best - NEAR_BEST_DELTA, best, color=NEAR_COLOR, alpha=0.12)
    zoom.set_ylim(best - 1, best + 0.07)
    zoom.set_title("Best one-point range", fontsize=9)
    zoom.tick_params(labelsize=8)

    distribution.hist(trials["HOTA"], bins=min(40, max(5, count // 4)), color=SAMPLE_COLOR, alpha=0.8)
    distribution.axvline(best, color=BEST_COLOR, ls="--", lw=1)
    distribution.axvline(baseline, color=BASELINE_COLOR, ls=":", lw=1)
    distribution.set(xlabel="HOTA", ylabel="Trials", title=f"Score distribution — {data.near_best.sum()} near-best")
    hist_zoom = distribution.inset_axes([0.10, 0.43, 0.55, 0.46])
    close = trials.loc[trials["HOTA"] >= best - 1, "HOTA"]
    hist_zoom.hist(close, bins=np.linspace(best - 1, best + 0.01, 21), color=SAMPLE_COLOR, alpha=0.8)
    hist_zoom.axvspan(best - NEAR_BEST_DELTA, best, color=NEAR_COLOR, alpha=0.2)
    hist_zoom.axvline(best, color=BEST_COLOR, ls="--", lw=1)
    hist_zoom.set(xlim=(best - 1, best + 0.02), title="Best one-point range")
    hist_zoom.title.set_fontsize(9)
    hist_zoom.tick_params(labelsize=8)

    for ax, (x, y) in zip(axes[2:], available):
        points = trials[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(points[x], points[y], color=SAMPLE_COLOR, s=24, alpha=0.55)
        _special_points(ax, data, points, x, y)
        ax.set(xlabel=x, ylabel=y, title=f"{x} vs {y}")
        # Keep the full distribution visible while separating the competitive
        # configurations otherwise compressed into the same corner.
        selected = data.trials.index[data.near_best].union(pd.Index([data.baseline_index]))
        focused = points.loc[points.index.intersection(selected)]
        if len(focused) >= 2 and (points.max() - points.min()).gt(3 * (focused.max() - focused.min())).any():
            inset = ax.inset_axes([0.10, 0.47, 0.52, 0.43])
            inset.scatter(focused[x], focused[y], color=SAMPLE_COLOR, s=19, alpha=0.55)
            _special_points(inset, data, focused, x, y)
            inset.set_title("Near-best trials + reference", fontsize=9)
            inset.tick_params(labelsize=8)
            inset.ticklabel_format(useOffset=False)
    notes = list(dict.fromkeys(data.notes + omitted))
    _save(fig, directory / "analysis.png", data, footer="\n".join(notes))


def _correlation(samples: pd.DataFrame) -> str:
    """Report monotonic association only for sufficient, nonconstant paired data."""
    from scipy.stats import spearmanr

    count = len(samples)
    if count < 11:
        return f"n={count}; too few samples for correlation"
    if samples["value"].nunique() < 2 or samples["HOTA"].nunique() < 2:
        return f"n={count}; correlation undefined (constant values)"
    rho = spearmanr(samples["value"], samples["HOTA"]).statistic
    return f"n={count}; Spearman ρ={rho:+.2f}"


def _numeric_panels(data: TrialAnalysis, directory: Path) -> None:
    """Plot every varying numeric search dimension using only its active samples."""
    parameters = [p for p in data.parameters.values() if p.kind == "numeric" and p.samples["value"].nunique() > 1]
    if not parameters:
        _omit(directory / "analysis_scatter.png", "No varying numeric search dimensions with recorded metadata")
        return
    fig, axes = _grid(len(parameters))
    fig.suptitle("Active numeric search parameters vs HOTA", fontsize=16, fontweight="bold")
    for ax, parameter in zip(axes, parameters):
        samples = parameter.samples
        branches = list(samples["branch"].unique())
        for i, branch in enumerate(branches):
            part = samples.loc[samples["branch"] == branch]
            color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
            label = _correlation(part)
            if branch != "all":
                label = f"{branch}: {label}"
            ax.scatter(part["value"], part["HOTA"], s=19, alpha=0.5, color=color, label=label)
        _special_points(ax, data, samples, "value")
        if parameter.bounds is not None:
            ax.set_xlim(*parameter.bounds)
        ax.set_xscale(parameter.scale)
        ax.set(xlabel=parameter.name, ylabel="HOTA", title=f"{parameter.name} · active n={len(samples)}")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.88, title="Observed correlation", title_fontsize=8)
    _save(
        fig,
        directory / "analysis_scatter.png",
        data,
        footer=(
            "Only active search dimensions are shown. Correlations describe this sampled search; "
            "they are not causal effects."
        ),
    )


def _category_order(parameter: ParameterSamples, branch: str | None = None) -> list[Any]:
    """Show only the selected branch's recorded options, including unsampled ones."""
    samples = parameter.samples
    if branch is not None:
        samples = samples.loc[samples["branch"] == branch]
    observed = samples["value"].dropna().unique().tolist()
    options = list(parameter.branch_options.get(branch, parameter.options))
    return options + [value for value in observed if value not in options]


def _categorical_panels(data: TrialAnalysis, directory: Path) -> None:
    """Compare observed categorical groups without implying controlled ablations."""
    import seaborn as sns

    parameters = [p for p in data.parameters.values() if p.kind == "categorical" and p.samples["value"].nunique() > 1]
    panels = [
        (parameter, branch, samples)
        for parameter in parameters
        for branch, samples in parameter.samples.groupby("branch", sort=True)
    ]
    interaction = all(
        name in data.parameters and not data.parameters[name].samples.empty for name in ("asso_func", "motion_fusion")
    )
    if interaction:
        association, fusion = (data.parameters[name] for name in ("asso_func", "motion_fusion"))
        common = association.samples.index.intersection(fusion.samples.index)
        interaction = len(common) > 0 and (
            association.samples.loc[common, "value"].nunique() > 1 or fusion.samples.loc[common, "value"].nunique() > 1
        )
        if not interaction:
            LOGGER.info("Joint category comparison skipped: no jointly observed varying choices.")
    if not parameters and not interaction:
        _omit(directory / "analysis_categories.png", "No varying categorical search dimensions with recorded metadata")
        return
    fig, axes = _grid(len(panels) + int(interaction), columns=2, width=7, height=4.3)
    fig.suptitle("Categorical choices — observed comparisons", fontsize=16, fontweight="bold")
    rng = np.random.default_rng(0)
    for ax, (parameter, branch, observations) in zip(axes, panels):
        samples = observations.copy()
        order = _category_order(parameter, branch)
        labels = [str(value) for value in order]
        samples["category"] = samples["value"].map(str)
        sns.boxplot(
            data=samples,
            x="category",
            y="HOTA",
            order=labels,
            ax=ax,
            color="#d6e3ed",
            fliersize=0,
            linewidth=0.8,
            width=0.55,
        )
        samples["position"] = samples["category"].map({label: i for i, label in enumerate(labels)})
        samples["position"] = samples["position"] + rng.uniform(-0.16, 0.16, len(samples))
        ax.scatter(samples["position"], samples["HOTA"], color=SAMPLE_COLOR, s=20, alpha=0.45)
        _special_points(ax, data, samples, "position")
        counts = samples["category"].value_counts()
        ax.set_xticks(range(len(labels)), [f"{label}\nn={counts.get(label, 0)}" for label in labels])
        branch_label = f"\n{branch}" if branch != "all" else ""
        ax.set(xlabel="", ylabel="HOTA", title=f"{parameter.name}{branch_label} · active n={len(samples)}")

    if interaction:
        ax = axes[-1]
        association, fusion = (data.parameters[name] for name in ("asso_func", "motion_fusion"))
        pairs = (
            association.samples[["value", "HOTA"]]
            .rename(columns={"value": "association"})
            .join(fusion.samples[["value"]].rename(columns={"value": "fusion"}), how="inner")
        )
        median = pairs.pivot_table(index="association", columns="fusion", values="HOTA", aggfunc="median")
        counts = pairs.pivot_table(index="association", columns="fusion", values="HOTA", aggfunc="count")
        rows, columns = _category_order(association), _category_order(fusion)
        median = median.reindex(index=rows, columns=columns)
        counts = counts.reindex(index=rows, columns=columns).fillna(0).astype(int)
        # Draw missing cells explicitly rather than giving them a synthetic score.
        sns.heatmap(
            median,
            ax=ax,
            cmap="Blues",
            annot=False,
            cbar_kws={"label": "Median HOTA"},
            linewidths=0.5,
            linecolor="white",
        )
        for row in range(len(rows)):
            for col in range(len(columns)):
                score = median.iloc[row, col]
                label = f"{score:.2f}\nn={counts.iloc[row, col]}" if pd.notna(score) else "Not sampled\nn=0"
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
                )
        ax.set(xlabel="motion_fusion", ylabel="asso_func", title="Joint sampling coverage and median score")
    _save(
        fig,
        directory / "analysis_categories.png",
        data,
        footer=(
            "Conditional choices are compared separately within each active branch.\n"
            "Sampling is unequal and other parameters varied simultaneously; these are descriptive comparisons."
        ),
    )


def _candidate_panel(data: TrialAnalysis, directory: Path) -> None:
    """Compare leading configurations against the reference on each recorded sequence."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sequences = data.sequence_hota
    if sequences.empty or data.baseline_index not in sequences.index or sequences.loc[data.baseline_index].isna().all():
        _omit(directory / "analysis_candidates.png", "Per-sequence HOTA or reference sequence metadata unavailable")
        return
    indices = data.trials["HOTA"].nlargest(min(10, len(data.trials)), keep="first").index
    baseline = sequences.loc[data.baseline_index]
    deltas = sequences.reindex(indices).subtract(baseline, axis="columns").dropna(axis="columns", how="all")
    if deltas.empty:
        _omit(directory / "analysis_candidates.png", "No comparable per-sequence HOTA for leading trials")
        return
    trials = data.trials.loc[indices]
    labels = [f"{rank}. {trial_id}" for rank, trial_id in enumerate(trials["trial_id"], 1)]
    deltas.index = labels
    fig = plt.figure(figsize=(max(14, 0.9 * len(deltas.columns) + 7), 7), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(max(7, len(deltas.columns)), 6))
    heatmap, table_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    limit = max(0.01, float(np.nanmax(np.abs(deltas.to_numpy()))))
    sns.heatmap(
        deltas,
        ax=heatmap,
        annot=True,
        fmt="+.2f",
        center=0,
        vmin=-limit,
        vmax=limit,
        cmap="RdBu",
        linewidths=0.5,
        cbar_kws={"label": "HOTA change vs reference", "shrink": 0.7},
    )
    heatmap.set(xlabel="Sequence", ylabel="HOTA rank · trial ID", title="Per-sequence HOTA change")
    heatmap.tick_params(axis="x", rotation=35)
    heatmap.tick_params(axis="y", rotation=0)
    for row, values in enumerate(deltas.to_numpy()):
        for col, value in enumerate(values):
            if np.isnan(value):
                heatmap.text(col + 0.5, row + 0.5, "N/A", ha="center", va="center", color="#777777")
    metrics = [name for name in ("HOTA", "MOTA", "IDF1", "IDSW") if name in trials]
    table_rows = [data.trials.loc[data.baseline_index], *(row for _, row in trials.iterrows())]
    cells = []
    for row in table_rows:
        cells.append(
            [str(row["trial_id"])]
            + [
                "—" if pd.isna(row[metric]) else (f"{row[metric]:.0f}" if metric == "IDSW" else f"{row[metric]:.3f}")
                for metric in metrics
            ]
        )
    table_ax.set_axis_off()
    table_ax.set_title("Reference and top ten (ranked by HOTA)")
    table = table_ax.table(
        cellText=cells,
        colLabels=["Trial", *metrics],
        loc="center",
        cellLoc="center",
        colWidths=[0.29] + [0.71 / len(metrics)] * len(metrics),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#e8eef2")
        elif row == 1:
            cell.set_facecolor("#dcefe8")
        elif row == 2:
            cell.set_facecolor("#f5dddd")
    fig.suptitle(
        f"Leading candidates vs {data.baseline_label.lower()} ({data.trials.at[data.baseline_index, 'trial_id']})",
        fontsize=15,
        fontweight="bold",
    )
    _save(
        fig,
        directory / "analysis_candidates.png",
        data,
        point_legend=False,
        footer=(
            "Table shading: green = reference, red = best HOTA. N/A means unavailable.\n"
            "Sequence changes describe this split; they do not establish repeatability or held-out improvement."
        ),
    )


def _omit(path: Path, reason: str) -> None:
    """Remove a stale generated diagnostic when its required evidence is unavailable."""
    path.unlink(missing_ok=True)
    LOGGER.info("Skipping %s: %s.", path.name, reason)


def generate_tune_analysis(
    tune_dir: Path,
    tracker_name: str = "",
    n_trials: int | None = None,
    *,
    data: TrialAnalysis | None = None,
) -> Path | None:
    """Generate PNG reports from saved CSV, search metadata, and trial JSON.

    The saved schema defines parameter activity and bounds. Missing metadata
    omits the corresponding diagnostics, while score-only reports remain useful.
    ``data`` shares an already loaded snapshot with other report writers.
    Returns the overview path, or None when no valid scored trials are available.
    """
    # Plotting dependencies are optional for installations without tuning reports.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import seaborn as sns
    except ImportError as exc:
        LOGGER.warning("Cannot generate tuning plots (missing optional dependency: %s)", exc)
        return None
    directory = Path(tune_dir)
    if data is None:
        data = load_tune_analysis(directory)
    if data is None or data.trials.empty:
        LOGGER.warning("No valid scored trials available in %s.", directory)
        return None
    for note in data.notes:
        LOGGER.info("Tuning analysis: %s", note)
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook", font_scale=0.85):
        _overview(data, directory, tracker_name, n_trials)
        _numeric_panels(data, directory)
        _categorical_panels(data, directory)
        _candidate_panel(data, directory)
    LOGGER.info("Analysis plots saved to %s", directory)
    return directory / "analysis.png"


def generate_selection_analysis(tune_dir: Path) -> tuple[Path, ...]:
    """Rebuild selection images from reports, independently of completed trials."""
    from boxmot.engine.calibration.cmc_plots import generate_cmc_plot
    from boxmot.engine.calibration.selection_plots import generate_selection_plot

    directory = Path(tune_dir) / "kf-tuning"
    outputs = (
        generate_selection_plot(directory / "selection.json"),
        generate_cmc_plot(directory / "cmc" / "analysis.json"),
    )
    return tuple(path for path in outputs if path is not None)


if __name__ == "__main__":
    import argparse

    from boxmot.engine.tuning.postprocessing import update_analysis_summary

    parser = argparse.ArgumentParser(
        description="Rebuild tuning PNG reports, parameter diagnostics, and saved CMC setup annotations."
    )
    parser.add_argument("tune_dir", type=Path, help="Directory containing results.csv and saved tuning metadata")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    data = load_tune_analysis(args.tune_dir)
    if data is None:
        raise SystemExit(1)
    result = generate_tune_analysis(args.tune_dir, tracker_name=args.tune_dir.name, data=data)
    if result is None:
        raise SystemExit(1)
    update_analysis_summary(args.tune_dir, analysis=data)
