"""Visualize saved image-only CMC eligibility without repeating camera estimation."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)

_ELIGIBLE_COLOR = "#167d9a"
_INELIGIBLE_COLOR = "#94a3b8"
_UNKNOWN_COLOR = "#c4a875"
_THRESHOLD_COLOR = "#b45309"
_REASONS = {
    "reliable_camera_motion": "reliable motion",
    "camera_motion_below_threshold": "motion below threshold",
    "insufficient_accepted_transforms": "too few accepted transforms",
    "no_camera_transitions": "no camera transitions",
}


def _number(value: Any, *, maximum: float | None = None) -> float | None:
    """Preserve absent or unusable nonnegative statistics as missing values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        return None
    return number


def _count(value: Any) -> str:
    """Format transition counts without turning absent observations into zero."""
    number = _number(value)
    return str(int(number)) if number is not None and number.is_integer() else "?"


def _threshold(ax: Axes, value: float | None) -> None:
    """Display the saved gate in the same percentage units as the observations."""
    label = "Threshold unavailable"
    if value is not None:
        ax.axvline(value, color=_THRESHOLD_COLOR, linestyle="--", linewidth=1.3, zorder=3)
        label = f"Threshold ≥ {value:.3g}%"
    ax.text(0, 1.025, label, transform=ax.transAxes, fontsize=9, color=_THRESHOLD_COLOR)


def _draw_cmc_plot(report: Mapping[str, Any]) -> Figure:
    """Build two aligned panels using only summary statistics in the saved report."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    sequences = report.get("sequences")
    settings = report.get("settings", {})
    if not isinstance(sequences, Mapping) or not sequences:
        raise ValueError("the CMC report has no sequence summaries")
    if not isinstance(settings, Mapping) or not all(isinstance(row, Mapping) for row in sequences.values()):
        raise ValueError("the CMC report has invalid settings or sequence summaries")

    names = list(sequences)
    rows = list(sequences.values())
    acceptance = [_number(row.get("accepted_fraction"), maximum=1) for row in rows]
    acceptance = [None if value is None else value * 100 for value in acceptance]
    motion = []
    for row in rows:
        statistics = row.get("grid_motion_fraction_image_diagonal")
        value = _number(statistics.get("median")) if isinstance(statistics, Mapping) else None
        motion.append(None if value is None else value * 100)
    acceptance_threshold = _number(settings.get("minimum_accepted_fraction"), maximum=1)
    motion_threshold = _number(settings.get("minimum_median_motion_fraction_image_diagonal"))
    acceptance_threshold = None if acceptance_threshold is None else acceptance_threshold * 100
    motion_threshold = None if motion_threshold is None else motion_threshold * 100
    colors = [
        _ELIGIBLE_COLOR
        if row.get("enabled") is True
        else (_INELIGIBLE_COLOR if row.get("enabled") is False else _UNKNOWN_COLOR)
        for row in rows
    ]
    eligible = sum(row.get("enabled") is True for row in rows)
    unknown = sum(not isinstance(row.get("enabled"), bool) for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, max(6.0, 3.0 + len(rows) * 0.52)), sharey=True)
    try:
        fig.subplots_adjust(left=0.22, right=0.97, top=0.75, bottom=0.25, wspace=0.18)
        fig.suptitle("Automatic CMC eligibility", fontsize=18, fontweight="bold", x=0.035, ha="left", y=0.975)
        subtitle = f"{str(report.get('tracker', 'Tracker')).upper()} · {eligible}/{len(rows)} sequences eligible"
        if unknown:
            subtitle += f" · {unknown} unknown"
        fig.text(0.035, 0.908, subtitle, fontsize=11, color="#334155")
        handles = [
            Patch(color=_ELIGIBLE_COLOR, label="Eligible under auto"),
            Patch(color=_INELIGIBLE_COLOR, label="Ineligible under auto"),
        ]
        if unknown:
            handles.append(Patch(color=_UNKNOWN_COLOR, label="Decision unavailable"))
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.976, 0.937), frameon=False, ncol=3, fontsize=9)

        accepted_ax, motion_ax = axes
        accepted_ax.set(title="Accepted camera transforms", xlabel="Accepted transitions (%)", xlim=(0, 134))
        accepted_ax.set_xticks([0, 20, 40, 60, 80, 100])
        labels = []
        for name, row in zip(names, rows):
            reason = row.get("reason")
            reason_label = _REASONS.get(reason, str(reason).replace("_", " ") if reason else "decision unavailable")
            labels.append(f"{name}\n{reason_label}")
        accepted_ax.set_yticks(range(len(rows)), labels)
        accepted_ax.set_ylim(len(rows) - 0.45, -0.55)
        for index, (row, value, color) in enumerate(zip(rows, acceptance, colors)):
            if value is None:
                accepted_ax.text(2, index, "Missing", va="center", fontsize=9, color="#64748b")
            else:
                accepted_ax.barh(index, value, height=0.48, color=color)
                accepted_ax.text(102, index, f"{value:.1f}%", va="center", fontsize=9, color="#334155")
            accepted_ax.text(
                132,
                index,
                f"{_count(row.get('accepted_transitions'))}/{_count(row.get('transitions'))}",
                va="center",
                ha="right",
                fontsize=8,
                color="#64748b",
            )
        _threshold(accepted_ax, acceptance_threshold)

        positives = [value for value in [*motion, motion_threshold] if value is not None and value > 0]
        upper = max(positives, default=1.0)
        lower = min(positives, default=1.0)
        use_log = upper / lower >= 50
        if use_log:
            # A linear region near zero preserves true zero values; the remaining
            # axis uses a log scale to make very small camera displacements visible.
            linear_limit = 10 ** math.floor(math.log10(lower))
            motion_ax.set_xscale("symlog", linthresh=linear_limit, linscale=0.5)
            tick_start = math.floor(math.log10(lower))
            tick_end = math.ceil(math.log10(upper))
            ticks = [0, *[10**exponent for exponent in range(tick_start, tick_end + 1)]]
            motion_ax.set_xticks(ticks)
            motion_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
            upper = max(upper, ticks[-1])
        motion_ax.set_xlim(0, upper * (2.2 if use_log else 1.28))
        motion_ax.set_title("Median camera displacement")
        motion_ax.set_xlabel("Image diagonal (%)" + (" · log scale above zero region" if use_log else ""))
        for index, (value, color) in enumerate(zip(motion, colors)):
            if value is None:
                motion_ax.text(
                    0.02,
                    index,
                    "Missing",
                    transform=motion_ax.get_yaxis_transform(),
                    va="center",
                    fontsize=9,
                    color="#64748b",
                )
                continue
            motion_ax.barh(index, value, height=0.48, color=color)
            motion_ax.plot(value, index, "o", color=color, markersize=4, clip_on=False)
            motion_ax.annotate(
                f"{value:.3g}%",
                (value, index),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color="#334155",
            )
        _threshold(motion_ax, motion_threshold)
        for ax in axes:
            title = ax.get_title()
            ax.set_title("")
            ax.set_title(title, fontsize=12, loc="left", pad=30)
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=0.17)
            ax.tick_params(axis="both", labelsize=9)
            ax.tick_params(axis="y", length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#cbd5e1")
        mask_conf = _number(report.get("mask_min_conf"))
        mask_label = f"{mask_conf:.4g}" if mask_conf is not None else "unavailable"
        metadata = (
            f"Method: {str(report.get('method', 'unknown')).upper()} · "
            f"Rule: {report.get('rule', 'unknown')} · Mask confidence: {mask_label} · "
            f"Split: {report.get('split', 'unknown')}"
        )
        fig.text(0.035, 0.13, metadata, fontsize=9, color="#334155")
        fig.text(
            0.035,
            0.085,
            "Image-only eligibility uses full-sequence frames and detections. "
            "The final off/auto policy is selected separately.",
            fontsize=9,
            color="#64748b",
        )
        fig.text(
            0.035,
            0.045,
            "Motion summarizes accepted transitions; unavailable estimates are missing, not zero. "
            "Counts exclude initialization.",
            fontsize=9,
            color="#64748b",
        )
        return fig
    except Exception:
        plt.close(fig)
        raise


def generate_cmc_plot(analysis_path: Path) -> Path | None:
    """Save a CMC eligibility PNG beside its JSON report, without affecting selection.

    Missing reports are expected for runs without camera selection. Plotting is an
    optional artifact: malformed reports or unavailable Matplotlib warn and return
    ``None`` so a successful calibration does not depend on visualization packages.
    """
    analysis_path = Path(analysis_path)
    if not analysis_path.is_file():
        return None
    fig = None
    plt = None
    try:
        report = json.loads(analysis_path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ValueError("the CMC report must be a JSON object")
        # Matplotlib is optional; importing it here keeps calibration usable when
        # only runtime dependencies are installed and permits offline regeneration.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with plt.rc_context({"font.family": "DejaVu Sans", "axes.labelcolor": "#334155", "text.color": "#0f172a"}):
            fig = _draw_cmc_plot(report)
            output = analysis_path.with_suffix(".png")
            fig.savefig(output, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.2)
        return output
    except Exception as exc:
        LOGGER.warning("Could not generate CMC analysis plot from %s: %s", analysis_path, exc)
        return None
    finally:
        if plt is not None and fig is not None:
            plt.close(fig)
