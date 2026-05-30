""" Test to ensure that the code is working correctly.
Runs all metrics on 14 trackers for the MOT Challenge MOT17 benchmark.
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
test_data_loc = os.path.join(os.path.dirname(__file__), '..', 'data', 'tests', 'mot_challenge', 'MOT17-train')
trackers = [
    'DPMOT',
    'GNNMatch',
    'IA',
    'ISE_MOT17R',
    'Lif_T',
    'Lif_TsimInt',
    'LPC_MOT',
    'MAT',
    'MIFTv2',
    'MPNTrack',
    'SSAT',
    'TracktorCorr',
    'Tracktorv2',
    'UnsupTrack',
]

for tracker in trackers:
    # Run code on tracker
    dataset_config = {'TRACKERS_TO_EVAL': [tracker],
                      'BENCHMARK': 'MOT17'}
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)

    results = {seq: raw_results['MotChallenge2DBox'][tracker][seq]['pedestrian'] for seq in
               raw_results['MotChallenge2DBox'][tracker].keys()}
    current_metrics_list = metrics_list + [trackeval.metrics.Count()]
    metric_names = trackeval.utils.validate_metrics_list(current_metrics_list)

    # Load expected results:
    test_data = trackeval.utils.load_detail(os.path.join(test_data_loc, tracker, 'pedestrian_detailed.csv'))
    assert len(test_data.keys()) == 22, len(test_data.keys())

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
            if not np.isclose(res_dict[field], test_data[seq][field]):
                print(tracker, seq, res_dict[field], test_data[seq][field], field)
                raise AssertionError

    print('Tracker %s tests passed' % tracker)
print('All tests passed')

