""" run_burst.py

The example commands given below expect the following folder structure:

- data
    - gt
        - burst
            - {val,test}
                - all_classes
                    - all_classes.json (filename is irrelevant)
    - trackers
        - burst
            - exemplar_guided
                - {val,test}
                    - my_tracking_method
                        - data
                            - results.json  (filename is irrelevant)
            - class_guided
                - {val,test}
                    - my_other_tracking_method
                        - data
                            - results.json (filename is irrelevant)

Run example:

1) Exemplar-guided tasks (all three tasks share the same eval logic):
run_burst.py --USE_PARALLEL True --EXEMPLAR_GUIDED True --GT_FOLDER ../data/gt/burst/{val,test}/all_classes --TRACKERS_FOLDER ../data/trackers/burst/exemplar_guided/{val,test}

2) Class-guided tasks (common class and long-tail):
run_burst.py --USE_PARALLEL FTrue --EXEMPLAR_GUIDED False --GT_FOLDER ../data/gt/burst/{val,test}/all_classes --TRACKERS_FOLDER ../data/trackers/burst/class_guided/{val,test}

3) Refer to run_burst_ow.py for open world evaluation

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
        'GT_FOLDER': os.path.join(code_path, 'data/gt/burst/val'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/burst/class-guided/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val'
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        'MAX_DETECTIONS': 300,  # Number of maximal allowed detections per image (0 for unlimited)
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']
"""

import sys
import os
import argparse
from tabulate import tabulate
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402


def main():
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_eval_config['PLOT_CURVES'] = False
    default_eval_config["OUTPUT_DETAILED"] = False
    default_eval_config["PRINT_RESULTS"] = False
    default_eval_config["OUTPUT_SUMMARY"] = False

    default_dataset_config = trackeval.datasets.BURST.get_default_dataset_config()

    # default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']}
    # default_metrics_config = {'METRICS': ['HOTA']}
    default_metrics_config = {'METRICS': ['HOTA', 'TrackMAP']}
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
    dataset_list = [trackeval.datasets.BURST(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.TrackMAP, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                   trackeval.metrics.HOTA]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list, show_progressbar=True)

    class_name_to_id = {x['name']: x['id'] for x in dataset_list[0].gt_data['categories']}
    known_list = [4, 13, 1038, 544, 1057, 34, 35, 36, 41, 45, 58, 60, 579, 1091, 1097, 1099, 78, 79, 81, 91, 1115,
                  1117, 95, 1122, 99, 1132, 621, 1135, 625, 118, 1144, 126, 642, 1155, 133, 1162, 139, 154, 174, 185,
                  699, 1215, 714, 717, 1229, 211, 729, 221, 229, 747, 235, 237, 779, 276, 805, 299, 829, 852, 347,
                  371, 382, 896, 392, 926, 937, 428, 429, 961, 452, 979, 980, 982, 475, 480, 993, 1001, 502, 1018]

    row_labels = ("HOTA", "DetA", "AssA", "AP")
    trackers = list(output_res['BURST'].keys())
    print("\n")

    def average_metric(m):
        return round(100*sum(m) / len(m), 2)

    for tracker in trackers:
        res = output_res['BURST'][tracker]['COMBINED_SEQ']
        all_names = [x for x in res.keys() if (x != 'cls_comb_cls_av') and (x != 'cls_comb_det_av')]

        class_split_names = {
            "All": [x for x in res.keys() if (x != 'cls_comb_cls_av') and (x != 'cls_comb_det_av')],
            "Common": [x for x in all_names if class_name_to_id[x] in known_list],
            "Uncommon": [x for x in all_names if class_name_to_id[x] not in known_list]
        }

        # table columns: 'all', 'common', 'uncommon'
        # table rows: HOTA, AssA, DetA, mAP
        table_data = []

        for row_label in row_labels:
            row = [row_label]
            for split_name in ["All", "Common", "Uncommon"]:
                split_classes = class_split_names[split_name]

                if row_label == "AP":
                    row.append(average_metric([res[c]['TrackMAP']["AP_all"].mean() for c in split_classes]))
                else:
                    row.append(average_metric([res[c]['HOTA'][row_label].mean() for c in split_classes]))

            table_data.append(row)

        print(f"Results for Tracker: {tracker}\n")
        print(tabulate(table_data, ["Metric", "All", "Common", "Uncommon"]))


if __name__ == '__main__':
    main()
