""" run_davis.py

Run example:
run_davis.py --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL ags

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
    '   'GT_FOLDER': os.path.join(code_path, 'data/gt/davis/'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/davis/davis_val'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'SPLIT_TO_EVAL': 'val',  # Valid: 'val', 'train'
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/ImageSets/2017)
        'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/split-to-eval.txt)
        'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
        'GT_LOC_FORMAT': '{gt_folder}/Annotations_unsupervised/480p/{seq}',
        # '{gt_folder}/Annotations_unsupervised/480p/{seq}'
        'MAX_DETECTIONS': 0  # Maximum number of allowed detections per sequence (0 for no threshold)
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'JAndF']
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
    default_dataset_config = trackeval.datasets.DAVIS.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'JAndF']}
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
    dataset_list = [trackeval.datasets.DAVIS(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.JAndF]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)