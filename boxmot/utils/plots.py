import os
import numpy as np
import matplotlib.pyplot as plt

class MetricsPlotter:
    """
    A class for plotting tracking metrics in both radar and FPS-vs-metric line charts.
    Instead of displaying the plots, this version saves them to disk in a specified root folder.

    Attributes
    ----------
    root_folder : str
        The directory where both plots will be saved.

    Methods
    -------
    plot_radar_chart(data, labels, title='Radar Chart',
                     figsize=(6, 6), ylim=(0, 100.0),
                     yticks=None, ytick_labels=None,
                     filename='radar_chart.png')
        Plots a radar chart for multiple methods over a fixed set of metrics, then saves it
        into the root folder under `filename`.

    plot_fps_metrics(fps, data, title=None, figsize=(10, 6),
                     filename='fps_metrics.png')
        Plots metrics (e.g., HOTA, MOTA, IDF1) versus FPS as line curves, then saves it
        into the root folder under `filename`.
    """

    def __init__(self, root_folder: str = '.'):
        """
        Initialize the MetricsPlotter with a root folder for saving plots.

        Parameters
        ----------
        root_folder : str, optional
            The directory in which to save both the radar chart and the FPS-vs-metrics plot.
            Default is the current working directory ('.').
        """
        self.root_folder = root_folder
        # Create the directory if it does not exist
        os.makedirs(self.root_folder, exist_ok=True)

    def plot_radar_chart(self,
                         data: dict,
                         labels: list,
                         title: str = 'Radar Chart',
                         figsize: tuple = (6, 6),
                         ylim: tuple = (0, 100.0),
                         yticks: list = None,
                         ytick_labels: list = None,
                         filename: str = 'radar_chart.png'):
        """
        Plots a radar chart for multiple methods over a fixed set of metrics, then saves it.

        Parameters
        ----------
        data : dict
            Keys are method names (strings), values are lists of numeric scores
            (one score per label). Each list must have the same length as `labels`.
        labels : list of str
            Names of the metrics (e.g. ['HOTA', 'AssA', 'AssR', ...]).
        title : str, optional
            Title of the plot (default: 'Radar Chart').
        figsize : tuple, optional
            Size of the figure in inches (default: (6, 6)).
        ylim : tuple, optional
            The (min, max) range for the radial axis (default: (0, 100.0)).
        yticks : list of float, optional
            Values at which to draw horizontal grid lines (default:
            [0.2 * ylim[1], 0.4 * ylim[1], 0.6 * ylim[1], 0.8 * ylim[1], ylim[1]]).
        ytick_labels : list of str, optional
            String labels for each entry in `yticks` (default: same as numeric ticks).
        filename : str, optional
            Name of the file (with extension) under which to save the radar chart.
            It will be placed inside `root_folder`. Default: 'radar_chart.png'.
        """
        num_vars = len(labels)
        # Compute equally spaced angles around the circle, then close the loop by appending the first angle again
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Prepare figure and polar axes
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        ax.set_ylim(*ylim)

        # Default yticks if none provided
        if yticks is None:
            yticks = [ylim[1] * (0.2 * i) for i in range(1, 6)]
        if ytick_labels is None:
            ytick_labels = [str(t) for t in yticks]

        # Plot each method
        for method, values in data.items():
            if len(values) != num_vars:
                raise ValueError(
                    f"Length of values for '{method}' ({len(values)}) "
                    f"does not match number of labels ({num_vars})."
                )

            # Close the loop by appending the first value at the end:
            vals = values + values[:1]
            ax.plot(angles, vals, label=method, linewidth=2)
            ax.fill(angles, vals, alpha=0.1)

        # Set the category labels at the proper angles
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        # Draw radial grid lines and set tick labels
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        ax.yaxis.grid(True)

        # Place legend below the plot
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        plt.title(title, y=1.08)

        plt.tight_layout()
        # Determine full save path and save
        save_path = os.path.join(self.root_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

    def plot_fps_metrics(self,
                         fps: list,
                         data: dict,
                         title: str = None,
                         figsize: tuple = (10, 6),
                         filename: str = 'fps_metrics.png'):
        """
        Plots tracking metrics (e.g., HOTA, MOTA, IDF1) versus FPS as line curves, then saves it.

        Parameters
        ----------
        fps : list or array-like
            A sequence of FPS values (e.g., [2, 4, 8, 16, 24, 32]).
        data : dict
            Keys are metric names (strings), values are lists of metric scores
            corresponding to each FPS in `fps`. All lists must be the same length as `fps`.
            Example keys: ['BoostTrack HOTA', 'BoostTrack MOTA', 'BoostTrack IDF1'].
        title : str, optional
            Title of the plot. If None, defaults to
            'Metrics vs FPS' or includes metric names automatically.
        figsize : tuple, optional
            Size of the figure in inches (default: (10, 6)).
        filename : str, optional
            Name of the file (with extension) under which to save the FPS-vs-metrics plot.
            It will be placed inside `root_folder`. Default: 'fps_metrics.png'.
        """
        # Validate input lengths
        for metric, values in data.items():
            if len(values) != len(fps):
                raise ValueError(
                    f"Length of values for '{metric}' ({len(values)}) does not match length of fps ({len(fps)})."
                )

        # If no title provided, create a generic one
        if title is None:
            metric_names = ", ".join(data.keys())
            title = f"Metrics vs FPS ({metric_names})"

        fig, ax = plt.subplots(figsize=figsize)
        for metric, values in data.items():
            ax.plot(fps, values, marker='o', label=metric, linewidth=2)

        ax.set_xlabel("FPS")
        ax.set_ylabel("Metric Value")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right')

        plt.tight_layout()
        # Determine full save path and save
        save_path = os.path.join(self.root_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory


# Example usage:
if __name__ == "__main__":
    # Create a MetricsPlotter that saves everything under "plots/" directory
    plotter = MetricsPlotter(root_folder='./')

    # 1) Radar chart example (will be saved as 'plots/radar_chart.png' by default)
    labels = ['HOTA', 'AssA', 'AssR', 'MOTA', 'IDF1']
    radar_data = {
        "BoostTrack": [69.25, 73.859, 77.49, 75.908, 83.199],
        "ByteTrack":  [67.68, 69.145, 75.031, 78.039, 79.157],
        "BoTSORT":    [68.888, 71.15, 76.626, 78.232, 81.331],
        "OCSORT":     [66.441, 69.111, 73.787, 74.548, 77.899],
        "StrongSORT": [68.05, 71.092, 74.983, 76.185, 80.763],
    }
    plotter.plot_radar_chart(
        radar_data,
        labels,
        title='Radar Chart of Method Combinations on DanceTrack',
        ylim=(65, 85),
        yticks=[65, 70, 75, 80, 85],
        ytick_labels=['65', '70', '75', '80', '85']
    )

    # 2) FPS vs Metrics example (will be saved as 'plots/fps_metrics.png' by default)
    fps = [2, 4, 8, 16, 24, 32]
    fps_data = {
        "BoostTrack HOTA": [56.7, 62.7, 66.5, 68.4, 68.9, 68.8],
        "BoostTrack MOTA": [54.1, 65.9, 73.0, 75.2, 75.7, 75.9],
        "BoostTrack IDF1": [69.8, 76.9, 80.1, 81.6, 82.1, 82.2],
    }
    plotter.plot_fps_metrics(
        fps,
        fps_data,
        title='YOLOX-X + BoostTrack MOT17 metrics vs FPS',
        figsize=(12, 6)
    )
