""" Test to ensure that the code is working correctly.
Should test ALL metrics across all datasets and splits currently supported.
Only tests one tracker per dataset/split to give a quick test result.
"""

import sys
import os
import numpy as np
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

# Fixes multiprocessing on windows, does nothing otherwise
if __name__ == '__main__':
    freeze_support()

eval_config = {'USE_PARALLEL': False,
               'NUM_PARALLEL_CORES': 8,
               }
evaluator = trackeval.Evaluator(eval_config)
metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]

tests = [
    {'DATASET': 'Kitti2DBox', 'SPLIT_TO_EVAL': 'training', 'TRACKERS_TO_EVAL': ['CIWT']},
    {'DATASET': 'MotChallenge2DBox', 'BENCHMARK': 'MOT15', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['MPNTrack']},
    {'DATASET': 'MotChallenge2DBox', 'BENCHMARK': 'MOT16', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['MPNTrack']},
    {'DATASET': 'MotChallenge2DBox', 'BENCHMARK': 'MOT17', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['MPNTrack']},
    {'DATASET': 'MotChallenge2DBox', 'BENCHMARK': 'MOT20', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['MPNTrack']},
]

for dataset_config in tests:

    dataset_name = dataset_config.pop('DATASET')
    if dataset_name == 'MotChallenge2DBox':
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        file_loc = os.path.join('mot_challenge', dataset_config['BENCHMARK'] + '-' + dataset_config['SPLIT_TO_EVAL'])
    elif dataset_name == 'Kitti2DBox':
        dataset_list = [trackeval.datasets.Kitti2DBox(dataset_config)]
        file_loc = os.path.join('kitti', 'kitti_2d_box_train')
    else:
        raise Exception('Dataset %s does not exist.' % dataset_name)

    raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)

    classes = dataset_list[0].config['CLASSES_TO_EVAL']
    tracker = dataset_config['TRACKERS_TO_EVAL'][0]
    test_data_loc = os.path.join(os.path.dirname(__file__), '..', 'data', 'tests', file_loc)

    for cls in classes:
        results = {seq: raw_results[dataset_name][tracker][seq][cls] for seq in raw_results[dataset_name][tracker].keys()}
        current_metrics_list = metrics_list + [trackeval.metrics.Count()]
        metric_names = trackeval.utils.validate_metrics_list(current_metrics_list)

        # Load expected results:
        test_data = trackeval.utils.load_detail(os.path.join(test_data_loc, tracker, cls + '_detailed.csv'))

        # Do checks
        for seq in test_data.keys():
            assert len(test_data[seq].keys()) > 250, len(test_data[seq].keys())

            details = []
            for metric, metric_name in zip(current_metrics_list, metric_names):
                table_res = {seq_key: seq_value[metric_name] for seq_key, seq_value in results.items()}
                details.append(metric.detailed_results(table_res))
            res_fields = sum([list(s['COMBINED_SEQ'].keys()) for s in details], [])
            res_values = sum([list(s[seq].values()) for s in details], [])
            res_dict = dict(zip(res_fields, res_values))

            for field in test_data[seq].keys():
                assert np.isclose(res_dict[field], test_data[seq][field]), seq + ': ' + cls + ': ' + field

    print('Tracker %s tests passed' % tracker)
print('All tests passed')

