"""Render saved KF/CMC selection reports without repeating calibration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)
_SELECTED_COLOR = "#b45309"
_POLICY_COLOR = "#176b87"


@dataclass(frozen=True)
class _PolicyPanel:
    """A policy's saved scores and winner, preserving selection tie breaks."""

    name: str
    scores: np.ndarray
    labels: np.ndarray
    center_errors: np.ndarray
    center_labels: np.ndarray
    winner: tuple[int, int] | None
    folds: dict[int, float]
    center_folds: dict[int, float]
    selected: bool


def _score(value: Any) -> float:
    """Read a finite IoU without interpreting missing/invalid scores as zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Selection IoU must be a number.")
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Selection IoU must be finite and in [0, 1].")
    return result


def _center_error(value: Any) -> float:
    """Read the optional center-distance ratio; it can exceed one box diagonal."""
    if value is None:
        return np.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Normalized center error must be a number.")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError("Normalized center error must be finite and nonnegative.")
    return result


def _objective_label(report: dict[str, Any]) -> str:
    """Describe the persisted selection objective, never substitute HOTA."""
    objective = report["objective"]
    if objective == "maximize_one_frame_mean_iou":
        return "One-frame held-out motion IoU"
    if objective == "maximize_mean_iou" and not report.get("cmc_candidates"):
        horizons = report["horizons"]
        if not horizons or any(not isinstance(value, int) or value < 1 for value in horizons):
            raise ValueError("Selection horizons must be positive frame counts.")
        return f"Horizon-average held-out motion IoU ({'/'.join(map(str, horizons))} frames)"
    raise ValueError(f"Unsupported selection objective: {objective!r}.")


def _selection_panels(report: dict[str, Any]) -> tuple[list[str], list[str], list[_PolicyPanel]]:
    """Build aligned policy grids from scores and explicitly persisted winners."""
    if report.get("version") not in {3, 4}:
        raise ValueError("Unsupported selection report version.")
    _objective_label(report)
    policies = report.get("cmc_candidates") or {"KF": report}
    selected_policy = report["selected_cmc"] if report.get("cmc_candidates") else "KF"
    if selected_policy not in policies:
        raise ValueError("Selected CMC policy is absent from the report.")
    kinds = list(report["candidate_order"])
    profiles = list(report["profile_order"])
    if not kinds or not profiles or len(set(kinds)) != len(kinds) or len(set(profiles)) != len(profiles):
        raise ValueError("Selection report needs unique KF and noise-profile orders.")
    policy_order = report.get("cmc_candidate_order", list(policies))
    if set(policy_order) != set(policies) or len(policy_order) != len(policies):
        raise ValueError("CMC candidate order does not match the saved policies.")
    panels = []
    for name in policy_order:
        policy = policies[name]
        values = np.full((len(kinds), len(profiles)), np.nan)
        labels = np.full(values.shape, "N/A", dtype=object)
        errors = np.full(values.shape, np.nan)
        error_labels = np.full(values.shape, "N/A", dtype=object)
        winner, folds, center_folds = None, {}, {}
        if policy.get("status") == "invalid":
            labels[:] = "Invalid"
            error_labels[:] = "Invalid"
        else:
            if policy["objective"] != report["objective"]:
                raise ValueError("CMC candidates use different ranking objectives.")
            for row, kind in enumerate(kinds):
                candidate = policy["candidates"].get(kind, {})
                if candidate.get("status") == "invalid":
                    labels[row, :] = "Invalid"
                    error_labels[row, :] = "Invalid"
                    continue
                for col, profile in enumerate(profiles):
                    entry = candidate.get("profiles", {}).get(profile)
                    if entry is None:
                        continue
                    if entry.get("status") != "valid":
                        labels[row, col] = "Invalid"
                        error_labels[row, col] = "Invalid"
                        continue
                    values[row, col] = _score(entry["mean_iou"])
                    labels[row, col] = f"{values[row, col]:.4f}"
                    errors[row, col] = _center_error(entry.get("normalized_center_error"))
                    if np.isfinite(errors[row, col]):
                        error_labels[row, col] = f"{100 * errors[row, col]:.3f}"
            winner = kinds.index(policy["selected_type"]), profiles.index(policy["selected_profile"])
            if not np.isfinite(values[winner]):
                raise ValueError("The saved winner does not have a valid score.")
            entry = policy["candidates"][policy["selected_type"]]["profiles"][policy["selected_profile"]]
            for fold in entry.get("folds", []):
                index = fold["index"]
                if not isinstance(index, int) or index < 0 or index in folds:
                    raise ValueError("Selection fold indices must be unique nonnegative integers.")
                folds[index] = _score(fold["mean_iou"])
                center_folds[index] = _center_error(fold.get("normalized_center_error"))
        if name == selected_policy and winner is None:
            raise ValueError("The selected policy is invalid.")
        panels.append(
            _PolicyPanel(
                name, values, labels, errors, error_labels, winner, folds, center_folds, name == selected_policy
            )
        )
    return kinds, profiles, panels


def _plot_folds(
    ax: Axes, panels: list[_PolicyPanel], kinds: list[str], profiles: list[str], *, center_error: bool = False
) -> None:
    """Pair held-out folds and show exact aggregates for the same IoU-selected winners."""
    valid = [panel for panel in panels if panel.winner is not None]
    fold_indices = sorted({index for panel in valid for index in panel.folds})
    colors = ("#2980b9", "#7e57c2", "#2c9275", "#b36044", "#69747d")
    x = np.arange(len(valid))
    for position, index in enumerate(fold_indices):
        scores = [
            100 * panel.center_folds.get(index, np.nan) if center_error else panel.folds.get(index, np.nan)
            for panel in valid
        ]
        if not np.isfinite(scores).any():
            continue
        ax.plot(
            x,
            scores,
            marker="o",
            markersize=6,
            lw=1.4,
            alpha=0.8,
            color=colors[position % len(colors)],
            label=f"Fold {index + 1}",
        )
    for index, panel in enumerate(valid):
        score = 100 * panel.center_errors[panel.winner] if center_error else panel.scores[panel.winner]
        if not np.isfinite(score):
            continue
        color = _SELECTED_COLOR if panel.selected else _POLICY_COLOR
        ax.scatter(index, score, marker="D", s=78, color=color, edgecolor="white", zorder=5)
        ax.annotate(
            f"{score:.3f}%" if center_error else f"{score:.4f}",
            (index, score),
            xytext=(9 if index == 0 else -9, 8),
            textcoords="offset points",
            ha="left" if index == 0 else "right",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
    if ax.has_data():
        ax.scatter([], [], marker="D", s=50, color="#475569", label="Report aggregate")
    else:
        ax.text(0.5, 0.5, "No center-error scores saved", ha="center", va="center", transform=ax.transAxes)
    labels = []
    for panel in valid:
        row, col = panel.winner
        title = f"CMC {panel.name}" if panel.name != "KF" else "Selected KF"
        labels.append(f"{title}\n{kinds[row]} / {profiles[col]}")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_xlim(-0.4, max(0.4, len(valid) - 0.6))
    if center_error:
        ax.margins(y=0.22)
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
    else:
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.015), top=min(1, ax.get_ylim()[1] + 0.015))
    ax.set_title("Policy winners · held-out folds" if len(panels) > 1 else "Selected KF · held-out folds", pad=18)
    ax.set_ylabel("Center error (% GT box diagonal) ↓" if center_error else "Motion IoU ↑")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    if ax.has_data():
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.43), ncol=2, frameon=False, fontsize=9)


def _draw_selection(fig: Figure, report: dict[str, Any]) -> None:
    """Compare overlap and center distance in aligned rows with independent scales."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    kinds, profiles, panels = _selection_panels(report)
    objective = _objective_label(report)
    selected = next(panel for panel in panels if panel.selected)
    row, col = selected.winner
    title = f"{report.get('tracker', 'Tracker').capitalize()} · KF selection"
    if len(panels) > 1:
        title += " + CMC policy"
    fig.suptitle(title, x=0.065, y=0.975, ha="left", fontsize=20, fontweight="bold", color="#172b40")
    policy_label = f"CMC {selected.name} · " if selected.name != "KF" else ""
    error = selected.center_errors[selected.winner]
    error_label = f"{100 * error:.3f}% center error" if np.isfinite(error) else "center error N/A"
    fig.text(
        0.065,
        0.927,
        f"Selected: {policy_label}{kinds[row]} / {profiles[col]}   |   "
        f"{selected.scores[selected.winner]:.4f} IoU · {error_label}",
        fontsize=12,
        color=_SELECTED_COLOR,
        weight="bold",
    )
    fig.text(0.065, 0.885, f"Selection objective: {objective}", fontsize=11, color="#475569")
    colorbars = []
    for metric_row in range(2):
        center_error = metric_row == 1
        metric_label = "Center error · lower is better ↓" if center_error else "Motion IoU · higher is better ↑"
        grid = fig.add_gridspec(
            1,
            len(panels) + 1,
            left=0.065,
            right=0.97,
            top=0.82 - 0.40 * metric_row,
            bottom=0.605 - 0.40 * metric_row,
            wspace=0.38,
            width_ratios=[1] * len(panels) + [1.1],
        )
        grids = [100 * panel.center_errors if center_error else panel.scores for panel in panels]
        all_scores = np.concatenate([values[np.isfinite(values)] for values in grids])
        if len(all_scores):
            low, high = float(all_scores.min()), float(all_scores.max())
            if low == high:
                margin = max(0.005, abs(low) * 0.05)
                low, high = max(0, low - margin), high + margin
            norm = Normalize(low, high if center_error else min(1, high))
        else:
            norm = Normalize(0, 1)
        cmap = colormaps["YlGnBu_r" if center_error else "YlGnBu"].copy()
        cmap.set_bad("#e8edf1")
        heatmaps = []
        for index, (panel, values) in enumerate(zip(panels, grids)):
            ax = fig.add_subplot(grid[0, index])
            heatmaps.append(ax)
            im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
            labels = panel.center_labels if center_error else panel.labels
            for (r, c), label in np.ndenumerate(labels):
                luminance = np.dot(cmap(norm(values[r, c]))[:3], [0.2126, 0.7152, 0.0722])
                color = "white" if np.isfinite(values[r, c]) and luminance < 0.5 else "#233344"
                ax.text(c, r, label, ha="center", va="center", fontsize=10, color=color)
            if panel.winner is not None:
                r, c = panel.winner
                ax.add_patch(
                    Rectangle(
                        (c - 0.48, r - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=_SELECTED_COLOR if panel.selected else "#58d5c9",
                        linewidth=3.5,
                        linestyle="-" if panel.selected else "--",
                    )
                )
            name = f"CMC {panel.name}" if panel.name != "KF" else "KF × noise profile"
            suffix = " · selected" if panel.selected else " · policy winner"
            ax.set_title(
                name + (suffix if panel.winner is not None else " · invalid") + "\n" + metric_label,
                fontsize=11,
                pad=16,
            )
            ax.set_xticks(range(len(profiles)), profiles, rotation=22, ha="right", fontsize=9)
            ax.set_yticks(range(len(kinds)), kinds, fontsize=10)
            ax.set_xlabel("Noise profile", labelpad=8, fontsize=10)
            ax.spines[:].set_visible(False)
        fold_ax = fig.add_subplot(grid[0, -1])
        _plot_folds(fold_ax, panels, kinds, profiles, center_error=center_error)
        if len(all_scores):
            start, end = heatmaps[0].get_position(), heatmaps[-1].get_position()
            color_label = (
                "Center error (% ground-truth box diagonal) · shared scale"
                if center_error
                else "Saved aggregate motion IoU · shared scale"
            )
            colorbars.append((im, (start.x0, 0.515 - 0.40 * metric_row, end.x1 - start.x0, 0.011), color_label))
    for im, bounds, label in colorbars:
        fig.colorbar(im, cax=fig.add_axes(bounds), orientation="horizontal", label=label)
    fig.text(
        0.065,
        0.020,
        "Center error is predicted-to-ground-truth center distance divided by the ground-truth box diagonal.\n"
        "Outlines mark IoU-selected winners in both rows; cells and diamonds use saved aggregates, not fold averages.",
        fontsize=9,
        color="#64748b",
        linespacing=1.5,
    )


def generate_selection_plot(selection_path: Path) -> Path | None:
    """Save ``selection.png`` from a report; optional plotting must not abort selection."""
    selection_path = Path(selection_path)
    if not selection_path.is_file():
        return None
    fig = None
    try:
        report = json.loads(selection_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Selection report must be a JSON object.")
        _, _, panels = _selection_panels(report)
        # Matplotlib is an optional reporting dependency; import only when requested.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "semibold"}):
            fig = plt.figure(figsize=(5.0 * (len(panels) + 1), 11.4), facecolor="white")
            _draw_selection(fig, report)
            output = selection_path.with_suffix(".png")
            fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.2)
        return output
    except Exception as exc:
        # Reporting is best effort, including optional dependency and backend failures.
        LOGGER.warning("Could not generate KF selection plot from %s: %s", selection_path, exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
