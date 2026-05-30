
# python3 scripts\run_rob_mots.py --ROBMOTS_SPLIT val --TRACKERS_TO_EVAL tracker_name (e.g. STP) --USE_PARALLEL True --NUM_PARALLEL_CORES 4

import sys
import os
import csv
import numpy as np
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402
from trackeval import utils
code_path = utils.get_code_path()

if __name__ == '__main__':
    freeze_support()

    script_config = {
        'ROBMOTS_SPLIT': 'train',  # 'train',  # valid: 'train', 'val', 'test', 'test_live', 'test_post', 'test_all'
        'BENCHMARKS': ['kitti_mots', 'davis_unsupervised', 'youtube_vis', 'ovis', 'tao'], # 'bdd_mots' coming soon
        'GT_FOLDER': os.path.join(code_path, 'data/gt/rob_mots'),
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/rob_mots'),
    }

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.RobMOTS.get_default_dataset_config()
    config = {**default_eval_config, **default_dataset_config, **script_config}

    # Command line interface:
    config = utils.update_config(config)

    if config['ROBMOTS_SPLIT'] == 'val':
        config['BENCHMARKS'] = ['kitti_mots', 'bdd_mots', 'davis_unsupervised', 'youtube_vis', 'ovis',
                                       'tao', 'mots_challenge']
        config['SPLIT_TO_EVAL'] = 'val'
    elif config['ROBMOTS_SPLIT'] == 'test' or config['SPLIT_TO_EVAL'] == 'test_live':
        config['BENCHMARKS'] = ['kitti_mots', 'bdd_mots', 'davis_unsupervised', 'youtube_vis', 'ovis', 'tao']
        config['SPLIT_TO_EVAL'] = 'test'
    elif config['ROBMOTS_SPLIT'] == 'test_post':
        config['BENCHMARKS'] = ['mots_challenge', 'waymo']
        config['SPLIT_TO_EVAL'] = 'test'
    elif config['ROBMOTS_SPLIT'] == 'test_all':
        config['BENCHMARKS'] = ['kitti_mots', 'bdd_mots', 'davis_unsupervised', 'youtube_vis', 'ovis',
                                       'tao', 'mots_challenge', 'waymo']
        config['SPLIT_TO_EVAL'] = 'test'
    elif config['ROBMOTS_SPLIT'] == 'train':
        config['BENCHMARKS'] = ['kitti_mots', 'davis_unsupervised', 'youtube_vis', 'ovis', 'tao']  # 'bdd_mots' coming soon
        config['SPLIT_TO_EVAL'] = 'train'

    metrics_config = {'METRICS': ['HOTA']}
    # metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    eval_config = {k: v for k, v in config.items() if k in config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in config.keys()}

    # Run code
    dataset_list = []
    for bench in config['BENCHMARKS']:
        dataset_config['SUB_BENCHMARK'] = bench
        dataset_list.append(trackeval.datasets.RobMOTS(dataset_config))
    evaluator = trackeval.Evaluator(eval_config)
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)


    # For each benchmark, combine the 'all' score with the 'cls_averaged' using geometric mean.
    metrics_to_calc = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA']
    trackers = list(output_res['RobMOTS.' + config['BENCHMARKS'][0]].keys())
    for tracker in trackers:
        # final_results[benchmark][result_type][metric]
        final_results = {}
        res = {bench: output_res['RobMOTS.' + bench][tracker]['COMBINED_SEQ'] for bench in config['BENCHMARKS']}
        for bench in config['BENCHMARKS']:
            final_results[bench] = {'cls_av': {}, 'det_av': {}, 'final': {}}
            for metric in metrics_to_calc:
                final_results[bench]['cls_av'][metric] = np.mean(res[bench]['cls_comb_cls_av']['HOTA'][metric])
                final_results[bench]['det_av'][metric] = np.mean(res[bench]['all']['HOTA'][metric])
                final_results[bench]['final'][metric] = \
                    np.sqrt(final_results[bench]['cls_av'][metric] * final_results[bench]['det_av'][metric])

        # Take the arithmetic mean over all the benchmarks
        final_results['overall'] = {'cls_av': {}, 'det_av': {}, 'final': {}}
        for metric in metrics_to_calc:
            final_results['overall']['cls_av'][metric] = \
                np.mean([final_results[bench]['cls_av'][metric] for bench in config['BENCHMARKS']])
            final_results['overall']['det_av'][metric] = \
                np.mean([final_results[bench]['det_av'][metric] for bench in config['BENCHMARKS']])
            final_results['overall']['final'][metric] = \
                np.mean([final_results[bench]['final'][metric] for bench in config['BENCHMARKS']])

        # Save out result
        headers = [config['SPLIT_TO_EVAL']] + [x + '___' + metric for x in ['f', 'c', 'd'] for metric in metrics_to_calc]

        def rowify(d):
            return [d[x][metric] for x in ['final', 'cls_av', 'det_av'] for metric in metrics_to_calc]

        out_file = os.path.join(script_config['TRACKERS_FOLDER'], script_config['ROBMOTS_SPLIT'], tracker,
                                'final_results.csv')

        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(headers)
            writer.writerow(['overall'] + rowify(final_results['overall']))
            for bench in config['BENCHMARKS']:
                if bench == 'overall':
                    continue
                writer.writerow([bench] + rowify(final_results[bench]))
