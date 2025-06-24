# python3 scripts/run_rob_mots.py --ROBMOTS_SPLIT train --TRACKERS_TO_EVAL STP --USE_PARALLEL True --NUM_PARALLEL_CORES 8

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
        'BENCHMARKS': None,  # If None, use all for each split.
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

    if not config['BENCHMARKS']:
        if config['ROBMOTS_SPLIT'] == 'val':
            config['BENCHMARKS'] = ['kitti_mots', 'bdd_mots', 'davis_unsupervised', 'youtube_vis', 'ovis',
                                    'tao', 'mots_challenge', 'waymo']
            config['SPLIT_TO_EVAL'] = 'val'
        elif config['ROBMOTS_SPLIT'] == 'test' or config['SPLIT_TO_EVAL'] == 'test_live':
            config['BENCHMARKS'] = ['kitti_mots', 'bdd_mots', 'davis_unsupervised', 'youtube_vis', 'tao']
            config['SPLIT_TO_EVAL'] = 'test'
        elif config['ROBMOTS_SPLIT'] == 'test_post':
            config['BENCHMARKS'] = ['mots_challenge', 'waymo', 'ovis']
            config['SPLIT_TO_EVAL'] = 'test'
        elif config['ROBMOTS_SPLIT'] == 'test_all':
            config['BENCHMARKS'] = ['kitti_mots', 'bdd_mots', 'davis_unsupervised', 'youtube_vis', 'ovis',
                                    'tao', 'mots_challenge', 'waymo']
            config['SPLIT_TO_EVAL'] = 'test'
        elif config['ROBMOTS_SPLIT'] == 'train':
            config['BENCHMARKS'] = ['kitti_mots', 'davis_unsupervised', 'youtube_vis', 'ovis', 'tao', 'bdd_mots']
            config['SPLIT_TO_EVAL'] = 'train'
    else:
        config['SPLIT_TO_EVAL'] = config['ROBMOTS_SPLIT']

    metrics_config = {'METRICS': ['HOTA']}
    eval_config = {k: v for k, v in config.items() if k in config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in config.keys()}

    # Run code
    try:
        dataset_list = []
        for bench in config['BENCHMARKS']:
            dataset_config['SUB_BENCHMARK'] = bench
            dataset_list.append(trackeval.datasets.RobMOTS(dataset_config))
        evaluator = trackeval.Evaluator(eval_config)
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                       trackeval.metrics.VACE, trackeval.metrics.JAndF]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric())
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
        output = list(list(output_msg.values())[0].values())[0]

    except Exception as err:
        if type(err) == trackeval.utils.TrackEvalException:
            output = str(err)
        else:
            output = 'Unknown error occurred.'

    success = output == 'Success'
    if not success:
        output = 'ERROR, evaluation failed. \n\nError message: ' + output
        print(output)

    if config['TRACKERS_TO_EVAL']:
        msg = "Thanks you for participating in the RobMOTS benchmark.\n\n"
        msg += "The status of your evaluation is: \n" + output + '\n\n'
        msg += "If your tracking results evaluated successfully on the evaluation server you can see your results here: \n"
        msg += "https://eval.vision.rwth-aachen.de/vision/"
        status_file = os.path.join(config['TRACKERS_FOLDER'], config['ROBMOTS_SPLIT'], config['TRACKERS_TO_EVAL'][0],
                                   'status.txt')
        with open(status_file, 'w', newline='') as f:
            f.write(msg)

    if success:
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
            headers = [config['SPLIT_TO_EVAL']] + [x + '___' + metric for x in ['f', 'c', 'd'] for metric in
                                                   metrics_to_calc]


            def rowify(d):
                return [d[x][metric] for x in ['final', 'cls_av', 'det_av'] for metric in metrics_to_calc]


            out_file = os.path.join(config['TRACKERS_FOLDER'], config['ROBMOTS_SPLIT'], tracker,
                                    'final_results.csv')

            with open(out_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(headers)
                writer.writerow(['overall'] + rowify(final_results['overall']))
                for bench in config['BENCHMARKS']:
                    if bench == 'overall':
                        continue
                    writer.writerow([bench] + rowify(final_results[bench]))
