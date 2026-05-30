
""" run_youtube_vis.py
Run example:
run_youtube_vis.py --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL STEm_Seg
Command Line Arguments: Defaults, # Comments
    Eval arguments:
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.
            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': False,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,
            'OUTPUT_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/youtube_vis/youtube_vis_training'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/youtube_vis/youtube_vis_training'),
        # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val'
        'PRINT_CONFIG': True,  # Whether to print current config
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
    Metric arguments:
        'METRICS': ['TrackMAP', 'HOTA', 'CLEAR', 'Identity']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    # print only combined since TrackMAP is undefined for per sequence breakdowns
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_dataset_config = trackeval.datasets.YouTubeVIS.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['TrackMAP', 'HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.YouTubeVIS(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.TrackMAP, trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                   trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            # specify TrackMAP config for YouTubeVIS
            if metric == trackeval.metrics.TrackMAP:
                default_track_map_config = metric.get_default_metric_config()
                default_track_map_config['USE_TIME_RANGES'] = False
                default_track_map_config['AREA_RANGES'] = [[0 ** 2, 128 ** 2],
                                                           [ 128 ** 2, 256 ** 2],
                                                           [256 ** 2, 1e5 ** 2]]
                metrics_list.append(metric(default_track_map_config))
            else:
                metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)