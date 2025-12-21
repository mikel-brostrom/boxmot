#!/usr/bin/env python3
"""
ratune.py

Load a Ray Tune experiment directory, print summaries, plot chosen metrics
for each trial by trial index, show Pareto fronts for MOTA vs HOTA (2D),
and interactive 3D scatter (all points + Pareto front) for MOTA vs HOTA vs IDF1 using Plotly.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.ticker import FixedLocator
from ray.tune.analysis import ExperimentAnalysis


def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """
    Find the Pareto-efficient points (maximization problem).
    Args:
        points: an (N, D) array of N points in D dimensions.
    Returns:
        mask: boolean array of length N, True if point is Pareto-efficient.
    """
    N = points.shape[0]
    is_efficient = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_efficient[i]:
            continue
        p = points[i]
        domination = np.any(
            np.all(points >= p, axis=1) & np.any(points > p, axis=1)
        )
        if domination:
            is_efficient[i] = False
    return is_efficient


def plot_metrics_by_trial(df, metrics=("HOTA", "MOTA", "IDF1"), exp_path=None):
    """
    Plot several metrics vs. trial index on one shared figure.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that contains one column per metric plus an
        implicit trial order (row index).
    metrics : list/tuple of str
        The metric column names to plot together.
    """
    # Only keep the metrics that are actually present
    avail = [m for m in metrics if m in df.columns]
    if not avail:
        print("⚠️  None of the requested metrics are in the DataFrame.")
        return

    indices = list(range(len(df)))
    fig, ax = plt.subplots(figsize=(10, 5))

    for metric in avail:
        ax.plot(indices,
                df[metric].values,
                marker="o",
                linestyle="-",
                label=metric)

    ax.xaxis.set_major_locator(FixedLocator(indices))
    ax.set_xticklabels(indices, rotation=90)
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Metric value")
    ax.set_title("Final HOTA, MOTA & IDF1 by Trial Number")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.show()
    fig.savefig(exp_path / "hota_mota_idf1_by_trial.png", dpi=1200)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a Ray Tune experiment directory."
    )
    parser.add_argument(
        "--exp_path",
        default=Path("ray/botsort_tune"),
        help="Path to your Tune experiment folder"
    )
    parser.add_argument(
        "--metric",
        default="HOTA",
        help="Name of the metric to summarize & plot (default: HOTA)",
    )
    parser.add_argument(
        "--mode",
        choices=["max", "min"],
        default="max",
        help="Whether to pick the trial with maximum or minimum final metric (default: max)",
    )
    args = parser.parse_args()

    # Load experiment
    exp_dir = Path(args.exp_path).expanduser().resolve()
    exp_uri = exp_dir.as_uri()
    analysis = ExperimentAnalysis(exp_uri)

    # Summary of experiment
    if getattr(analysis, "errors", None):
        print("⚠️  Some trials errored.")
    else:
        print("✅ No trial errors detected.")
    trial_dfs = analysis.trial_dataframes
    num_runs = len(trial_dfs)
    print(f"Found {num_runs} completed trials.\n")

    # Final metrics table
    results_df = analysis.dataframe().reset_index(drop=True)
    results_df['run_number'] = results_df.index

    # Prepare display of selected metric and time
    display_cols = ['run_number', args.metric, 'time_total_s']
    available = [c for c in display_cols if c in results_df.columns]
    ascending = (args.mode == "min")
    sorted_df = results_df[available].sort_values(by=args.metric, ascending=ascending)

    # Print all runs sorted
    print("Final reported metrics for all trials:")
    print(sorted_df.to_string(index=False))
    print()

    # Best trial
    best = results_df.sort_values(by=args.metric, ascending=ascending).iloc[0]
    print(f"Best trial by '{args.metric}' ({args.mode}):")
    print(best.to_string())
    print()

    # Plot default metric and extras
    plot_metrics_by_trial(results_df, ["HOTA", "MOTA", "IDF1"], exp_path = args.exp_path)

    # 3D scatter for all points + Pareto front using Plotly
    x_metric, y_metric, z_metric = "HOTA", "MOTA", "IDF1"
    if all(m in results_df.columns for m in [x_metric, y_metric, z_metric]):
        df3 = results_df.dropna(subset=[x_metric, y_metric, z_metric]).reset_index(drop=True)
        pts3 = df3[[x_metric, y_metric, z_metric]].values
        mask3 = is_pareto_efficient(pts3)
        others3 = df3[~mask3].copy()
        front3 = df3[mask3].copy()

        # build a hover-text column
        def make_hover(df):
            return (
                "run: " + df['run_number'].astype(str)
                + "<br>HOTA: " + df[x_metric].map("{:.3f}".format)   # x_metric *is* HOTA
                + "<br>MOTA: " + df[y_metric].map("{:.3f}".format)   # y_metric *is* MOTA
                + "<br>IDF1: " + df[z_metric].map("{:.3f}".format)
            )
        others3['hover'] = make_hover(others3)
        front3 ['hover'] = make_hover(front3)

        fig = go.Figure()
        
        # --- Identify the single best-HOTA trial -----------------------------
        best_idx  = df3[x_metric].idxmax()          # x_metric == "HOTA"
        best_row  = df3.loc[[best_idx]].copy()
        best_row['hover'] = make_hover(best_row)    # reuse the same hover builder

        # all other trials
        fig.add_trace(go.Scatter3d(
            x=others3[x_metric],
            y=others3[y_metric],
            z=others3[z_metric],
            mode='markers',
            name='Trials',
            marker=dict(size=4, opacity=0.6),
            hoverinfo='text',
            hovertext=others3['hover']
        ))

        # Pareto front
        fig.add_trace(go.Scatter3d(
            x=front3[x_metric],
            y=front3[y_metric],
            z=front3[z_metric],
            mode='markers',
            name='Pareto Front',
            marker=dict(size=6, symbol='circle', color='red'),
            hoverinfo='text',
            hovertext=front3['hover']
        ))
        
        # ★ Best HOTA trial (highlighted) ★
        fig.add_trace(go.Scatter3d(
            x=best_row[x_metric],
            y=best_row[y_metric],
            z=best_row[z_metric],
            mode='markers',
            name='Best HOTA',                       # new legend entry
            marker=dict(size=9, color='gold',       # any eye-catching colour
                        symbol='diamond'),
            hoverinfo='text',
            hovertext=best_row['hover']
        ))

        fig.update_layout(
            title=f"3D Scatter: all points + Pareto front ({x_metric}, {y_metric}, {z_metric})",
            scene=dict(
                xaxis_title=x_metric,
                yaxis_title=y_metric,
                zaxis_title=z_metric
            ),
            legend=dict(x=0.8, y=0.9)
        )
        fig.show()

        # To export to standalone HTML:
        fig.write_html(args.exp_path / "pareto3d_allpoints.html")
    else:
        print(f"Cannot plot 3D scatter: one or more of '{x_metric}', '{y_metric}', '{z_metric}' not in results.")


if __name__ == "__main__":
    main()
