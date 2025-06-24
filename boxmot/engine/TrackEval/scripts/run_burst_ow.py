""" run_burst_ow.py

The example commands given below expect the following folder structure:

- data
    - gt
        - burst
            - {val,test}
                - all_classes
                    - all_classes.json  (filename is irrelevant)
                - common_classes
                    - common_classes.json  (filename is irrelevant)
                - uncommon_classes.json
                    - uncommon_classes.json  (filename is irrelevant)
    - trackers
        - burst
            - open-world
                - {val,test}
                    - my_tracking_method
                        - data
                            - results.json  (filename is irrelevant)

Run example:

You'll need to run the eval script separately to get the OWTA metric for the three class splits. In the command below,
replace <SPLIT_NAME> with "common", "uncommon" and "all" to get the corresponding results.

run_burst_ow.py --USE_PARALLEL True --GT_FOLDER data/gt/burst/{val,test}/<SPLIT_NAME>_classes --TRACKERS_FOLDER data/trackers/burst/open-world/{val,test}

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
        'GT_FOLDER': os.path.join(code_path, '../data/gt/burst/{val,test}/<SPLIT_NAME>_classes'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, '../data/trackers/burst/open-world/{val,test}),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val'
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        'MAX_DETECTIONS': 300,  # Number of maximal allowed detections per image (0 for unlimited)
        'SUBSET': 'unknown',  # Evaluate on the following subsets ['all', 'known', 'unknown', 'distractor']
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']
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
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.BURST_OW.get_default_dataset_config()
    # default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']}
    default_metrics_config = {'METRICS': ['HOTA']}
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
    dataset_list = [trackeval.datasets.BURST_OW(dataset_config)]
    metrics_list = []
    # for metric in [trackeval.metrics.TrackMAP, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
    #                trackeval.metrics.HOTA]:
    for metric in [trackeval.metrics.HOTA]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list, show_progressbar=True)
