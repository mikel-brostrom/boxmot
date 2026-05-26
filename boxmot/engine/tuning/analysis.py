"""Generate analysis plots for tuning results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

METRIC_COLS = ["HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW"]
BOOL_PARAMS = [
    "use_ecc", "use_dlo_boost", "use_duo_boost", "s_sim_corr",
    "use_rich_s", "use_sb", "use_vt", "with_reid", "use_second_pass",
    "ams_enabled", "gta_enabled", "gta_interpolate",
]
KEY_CONTINUOUS = [
    "det_thresh", "iou_threshold", "lambda_iou", "lambda_mhd",
    "lambda_emb_multiplier", "new_track_thresh", "feat_alpha",
    "recovery_appearance_thresh", "max_age",
]


def generate_tune_analysis(tune_dir: Path, tracker_name: str = "", n_trials: int | None = None) -> Path | None:
    """Generate analysis plots from a tuning results.csv.

    Parameters
    ----------
    tune_dir : Path
        Directory containing results.csv (the tune output folder).
    tracker_name : str
        Tracker name for plot titles.
    n_trials : int | None
        Total number of trials (for title). Inferred from CSV if None.

    Returns
    -------
    Path | None
        Path to the saved analysis.png, or None if generation failed.
    """
    csv_path = tune_dir / "results.csv"
    if not csv_path.exists():
        LOGGER.warning(f"No results.csv found in {tune_dir}, skipping analysis plots.")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import spearmanr
    except ImportError as e:
        LOGGER.warning(f"Cannot generate analysis plots (missing dependency: {e})")
        return None

    # ─── Load & filter ─────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if "HOTA" not in df.columns:
        LOGGER.warning("results.csv missing HOTA column, skipping analysis plots.")
        return None

    n_trials = n_trials or len(df)
    df = df[df["HOTA"] > 0].reset_index(drop=True)
    if len(df) < 5:
        LOGGER.warning("Too few valid trials for analysis plots.")
        return None

    best_idx = df["HOTA"].idxmax()
    best = df.loc[best_idx]

    # ─── Parameter importance (Spearman) ───────────────────────────────────
    numeric_params = [
        c for c in df.columns
        if c not in METRIC_COLS + ["trial_id", "IDs", "IDSW_rate", "cmc_method"]
        and df[c].dtype in [np.float64, np.int64, float, int]
    ]
    # Convert booleans stored as True/False strings
    bool_cols_present = [bp for bp in BOOL_PARAMS if bp in df.columns]
    for col in bool_cols_present:
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(df[col])
        if col not in numeric_params:
            numeric_params.append(col)

    importance = {}
    for col in numeric_params:
        valid = df[[col, "HOTA"]].dropna()
        if len(valid) > 10:
            rho, _ = spearmanr(valid[col], valid["HOTA"])
            importance[col] = rho
    importance = pd.Series(importance).sort_values(key=abs, ascending=False)

    # ─── Main figure (4 panels) ───────────────────────────────────────────
    title = f"{tracker_name or 'Tracker'} Tuning Analysis ({n_trials} trials)"
    sns.set_theme(style="whitegrid", font_scale=0.85)
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3,
                          left=0.08, right=0.97, top=0.93, bottom=0.05)

    # Panel 1: Convergence curve
    ax1 = fig.add_subplot(gs[0, 0])
    cummax = df["HOTA"].cummax()
    ax1.plot(cummax.index, cummax, color="tab:blue", lw=1.5)
    ax1.axhline(best["HOTA"], ls="--", color="tab:red", lw=1, alpha=0.7)
    ax1.set_xlabel("Trial index")
    ax1.set_ylabel("Cumulative-best HOTA")
    ax1.set_title("Convergence Curve")
    ax1.annotate(f"Best: {best['HOTA']:.2f}", xy=(best_idx, best["HOTA"]),
                 xytext=(max(0, best_idx - 80), best["HOTA"] - 1.5),
                 arrowprops=dict(arrowstyle="->", color="tab:red"),
                 fontsize=9, color="tab:red")

    # Panel 2: HOTA distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df["HOTA"], bins=40, kde=True, ax=ax2, color="tab:blue", alpha=0.6)
    ax2.axvline(best["HOTA"], ls="--", color="tab:red", lw=1.5,
                label=f"Best={best['HOTA']:.2f}")
    ax2.axvline(df["HOTA"].median(), ls="--", color="tab:orange", lw=1.5,
                label=f"Median={df['HOTA'].median():.2f}")
    ax2.set_title("HOTA Distribution")
    ax2.legend(fontsize=8)

    # Panel 3: Metric correlation heatmap (top-100 trials)
    ax3 = fig.add_subplot(gs[0, 2])
    available_metrics = [m for m in METRIC_COLS if m in df.columns]
    top100 = df.nlargest(min(100, len(df)), "HOTA")[available_metrics]
    corr = top100.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax3, cbar_kws={"shrink": 0.8})
    ax3.set_title("Metric Correlation (Top-100)")

    # Panel 4: Boolean feature impact box plots
    if bool_cols_present:
        ax4 = fig.add_subplot(gs[1, :])
        bool_data = []
        for bp in bool_cols_present:
            for _, row in df[[bp, "HOTA"]].iterrows():
                bool_data.append({"param": bp, "value": str(bool(row[bp])), "HOTA": row["HOTA"]})
        bool_df = pd.DataFrame(bool_data)
        sns.boxplot(data=bool_df, x="param", y="HOTA", hue="value",
                    ax=ax4, palette={"True": "tab:blue", "False": "tab:orange"},
                    fliersize=2, linewidth=0.8)
        ax4.set_title("Boolean Feature Impact on HOTA")
        ax4.set_xlabel("")
        ax4.tick_params(axis="x", rotation=35)
        ax4.legend(title="Value", fontsize=8, loc="lower left")

    # Panel 5: Parameter importance bar chart
    ax5 = fig.add_subplot(gs[2, :])
    top20 = importance.head(20)
    if len(top20) > 0:
        colors = ["tab:blue" if v > 0 else "tab:red" for v in top20.values]
        ax5.barh(range(len(top20)), top20.values, color=colors, alpha=0.8)
        ax5.set_yticks(range(len(top20)))
        ax5.set_yticklabels(top20.index)
        ax5.set_xlabel("Spearman ρ (correlation with HOTA)")
        ax5.set_title("Parameter Importance (Top-20 by |ρ|)")
        ax5.axvline(0, color="black", lw=0.5)
        ax5.invert_yaxis()

    output_png = tune_dir / "analysis.png"
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    # ─── Scatter figure ────────────────────────────────────────────────────
    available_continuous = [p for p in KEY_CONTINUOUS if p in df.columns]
    if available_continuous:
        n_params = len(available_continuous)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        fig2 = plt.figure(figsize=(16, 4.5 * n_rows), constrained_layout=True)
        fig2.suptitle("Key Continuous Parameters vs HOTA", fontsize=14, fontweight="bold")

        for i, param in enumerate(available_continuous):
            ax = fig2.add_subplot(n_rows, n_cols, i + 1)
            ax.scatter(df[param], df["HOTA"], c=df["HOTA"], cmap="viridis",
                       s=12, alpha=0.6, edgecolors="none")
            ax.axhline(best["HOTA"], ls="--", color="tab:red", lw=0.8, alpha=0.5)
            ax.scatter([best[param]], [best["HOTA"]], c="red", s=60, zorder=5,
                       marker="*", label="Best")
            ax.set_xlabel(param)
            ax.set_ylabel("HOTA")
            ax.set_title(param)

        scatter_png = tune_dir / "analysis_scatter.png"
        fig2.savefig(scatter_png, dpi=150)
        plt.close(fig2)

    LOGGER.info(f"Analysis plots saved to {output_png}")
    return output_png


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    tune_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/ray/occluboost_tune_5")
    result = generate_tune_analysis(tune_dir, tracker_name=tune_dir.name)
    if result:
        print(f"Saved: {result}")
    else:
        print("No plots generated.", file=sys.stderr)
        sys.exit(1)
