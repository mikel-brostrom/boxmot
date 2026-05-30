
import os
import numpy as np
from .utils import TrackEvalException


def plot_compare_trackers(tracker_folder, tracker_list, cls, output_folder, plots_list=None):
    """Create plots which compare metrics across different trackers."""
    # Define what to plot
    if plots_list is None:
        plots_list = get_default_plots_list()

    # Load data
    data = load_multiple_tracker_summaries(tracker_folder, tracker_list, cls)
    out_loc = os.path.join(output_folder, cls)

    # Plot
    for args in plots_list:
        create_comparison_plot(data, out_loc, *args)


def get_default_plots_list():
    # y_label, x_label, sort_label, bg_label, bg_function
    plots_list = [
        ['AssA', 'DetA', 'HOTA', 'HOTA', 'geometric_mean'],
        ['AssPr', 'AssRe', 'HOTA', 'AssA', 'jaccard'],
        ['DetPr', 'DetRe', 'HOTA', 'DetA', 'jaccard'],
        ['HOTA(0)', 'LocA(0)', 'HOTA', 'HOTALocA(0)', 'multiplication'],
        ['HOTA', 'LocA', 'HOTA', None, None],

        ['HOTA', 'MOTA', 'HOTA', None, None],
        ['HOTA', 'IDF1', 'HOTA', None, None],
        ['IDF1', 'MOTA', 'HOTA', None, None],
    ]
    return plots_list


def load_multiple_tracker_summaries(tracker_folder, tracker_list, cls):
    """Loads summary data for multiple trackers."""
    data = {}
    for tracker in tracker_list:
        with open(os.path.join(tracker_folder, tracker, cls + '_summary.txt')) as f:
            keys = next(f).split(' ')
            done = False
            while not done:
                values = next(f).split(' ')
                if len(values) == len(keys):
                    done = True
            data[tracker] = dict(zip(keys, map(float, values)))
    return data


def create_comparison_plot(data, out_loc, y_label, x_label, sort_label, bg_label=None, bg_function=None, settings=None):
    """ Creates a scatter plot comparing multiple trackers between two metric fields, with one on the x-axis and the
    other on the y axis. Adds pareto optical lines and (optionally) a background contour.

    Inputs:
        data: dict of dicts such that data[tracker_name][metric_field_name] = float
        y_label: the metric_field_name to be plotted on the y-axis
        x_label: the metric_field_name to be plotted on the x-axis
        sort_label: the metric_field_name by which trackers are ordered and ranked
        bg_label: the metric_field_name by which (optional) background contours are plotted
        bg_function: the (optional) function bg_function(x,y) which converts the x_label / y_label values into bg_label.
        settings: dict of plot settings with keys:
            'gap_val': gap between axis ticks and bg curves.
            'num_to_plot': maximum number of trackers to plot
    """

    # Only loaded when run to reduce minimum requirements
    from matplotlib import pyplot as plt

    # Get plot settings
    if settings is None:
        gap_val = 2
        num_to_plot = 20
    else:
        gap_val = settings['gap_val']
        num_to_plot = settings['num_to_plot']

    if (bg_label is None) != (bg_function is None):
        raise TrackEvalException('bg_function and bg_label must either be both given or neither given.')

    # Extract data
    tracker_names = np.array(list(data.keys()))
    sort_index = np.array([data[t][sort_label] for t in tracker_names]).argsort()[::-1]
    x_values = np.array([data[t][x_label] for t in tracker_names])[sort_index][:num_to_plot]
    y_values = np.array([data[t][y_label] for t in tracker_names])[sort_index][:num_to_plot]

    # Print info on what is being plotted
    tracker_names = tracker_names[sort_index][:num_to_plot]
    print('\nPlotting %s vs %s, for the following (ordered) trackers:' % (y_label, x_label))
    for i, name in enumerate(tracker_names):
        print('%i: %s' % (i+1, name))

    # Find best fitting boundaries for data
    boundaries = _get_boundaries(x_values, y_values, round_val=gap_val/2)

    fig = plt.figure()

    # Plot background contour
    if bg_function is not None:
        _plot_bg_contour(bg_function, boundaries, gap_val)

    # Plot pareto optimal lines
    _plot_pareto_optimal_lines(x_values, y_values)

    # Plot data points with number labels
    labels = np.arange(len(y_values)) + 1
    plt.plot(x_values, y_values, 'b.', markersize=15)
    for xx, yy, l in zip(x_values, y_values, labels):
        plt.text(xx, yy, str(l), color="red", fontsize=15)

    # Add extra explanatory text to plots
    plt.text(0, -0.11, 'label order:\nHOTA', horizontalalignment='left', verticalalignment='center',
             transform=fig.axes[0].transAxes, color="red", fontsize=12)
    if bg_label is not None:
        plt.text(1, -0.11, 'curve values:\n' + bg_label, horizontalalignment='right', verticalalignment='center',
                 transform=fig.axes[0].transAxes, color="grey", fontsize=12)

    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    title = y_label + ' vs ' + x_label
    if bg_label is not None:
        title += ' (' + bg_label + ')'
    plt.title(title, fontsize=17)
    plt.xticks(np.arange(0, 100, gap_val))
    plt.yticks(np.arange(0, 100, gap_val))
    min_x, max_x, min_y, max_y = boundaries
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    os.makedirs(out_loc, exist_ok=True)
    filename = os.path.join(out_loc, title.replace(' ', '_'))
    plt.savefig(filename + '.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0.05)


def _get_boundaries(x_values, y_values, round_val):
    x1 = np.min(np.floor((x_values - 0.5) / round_val) * round_val)
    x2 = np.max(np.ceil((x_values + 0.5) / round_val) * round_val)
    y1 = np.min(np.floor((y_values - 0.5) / round_val) * round_val)
    y2 = np.max(np.ceil((y_values + 0.5) / round_val) * round_val)
    x_range = x2 - x1
    y_range = y2 - y1
    max_range = max(x_range, y_range)
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    min_x = max(x_center - max_range / 2, 0)
    max_x = min(x_center + max_range / 2, 100)
    min_y = max(y_center - max_range / 2, 0)
    max_y = min(y_center + max_range / 2, 100)
    return min_x, max_x, min_y, max_y


def geometric_mean(x, y):
    return np.sqrt(x * y)


def jaccard(x, y):
    x = x / 100
    y = y / 100
    return 100 * (x * y) / (x + y - x * y)


def multiplication(x, y):
    return x * y / 100


bg_function_dict = {
    "geometric_mean": geometric_mean,
    "jaccard": jaccard,
    "multiplication": multiplication,
    }


def _plot_bg_contour(bg_function, plot_boundaries, gap_val):
    """ Plot background contour. """

    # Only loaded when run to reduce minimum requirements
    from matplotlib import pyplot as plt

    # Plot background contour
    min_x, max_x, min_y, max_y = plot_boundaries
    x = np.arange(min_x, max_x, 0.1)
    y = np.arange(min_y, max_y, 0.1)
    x_grid, y_grid = np.meshgrid(x, y)
    if bg_function in bg_function_dict.keys():
        z_grid = bg_function_dict[bg_function](x_grid, y_grid)
    else:
        raise TrackEvalException("background plotting function '%s' is not defined." % bg_function)
    levels = np.arange(0, 100, gap_val)
    con = plt.contour(x_grid, y_grid, z_grid, levels, colors='grey')

    def bg_format(val):
        s = '{:1f}'.format(val)
        return '{:.0f}'.format(val) if s[-1] == '0' else s

    con.levels = [bg_format(val) for val in con.levels]
    plt.clabel(con, con.levels, inline=True, fmt='%r', fontsize=8)


def _plot_pareto_optimal_lines(x_values, y_values):
    """ Plot pareto optimal lines """

    # Only loaded when run to reduce minimum requirements
    from matplotlib import pyplot as plt

    # Plot pareto optimal lines
    cxs = x_values
    cys = y_values
    best_y = np.argmax(cys)
    x_pareto = [0, cxs[best_y]]
    y_pareto = [cys[best_y], cys[best_y]]
    t = 2
    remaining = cxs > x_pareto[t - 1]
    cys = cys[remaining]
    cxs = cxs[remaining]
    while len(cxs) > 0 and len(cys) > 0:
        best_y = np.argmax(cys)
        x_pareto += [x_pareto[t - 1], cxs[best_y]]
        y_pareto += [cys[best_y], cys[best_y]]
        t += 2
        remaining = cxs > x_pareto[t - 1]
        cys = cys[remaining]
        cxs = cxs[remaining]
    x_pareto.append(x_pareto[t - 1])
    y_pareto.append(0)
    plt.plot(np.array(x_pareto), np.array(y_pareto), '--r')
