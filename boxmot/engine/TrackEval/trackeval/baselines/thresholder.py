"""
Thresholder

Author: Jonathon Luiten

Simply reads in a set of detection, thresholds them at a certain score threshold, and writes them out again.
"""

import os
import sys
from multiprocessing.pool import Pool
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from trackeval.baselines import baseline_utils as butils
from trackeval.utils import get_code_path

THRESHOLD = 0.2

code_path = get_code_path()
config = {
    'INPUT_FOL': os.path.join(code_path, 'data/detections/rob_mots/{split}/non_overlap_supplied/data/'),
    'OUTPUT_FOL': os.path.join(code_path, 'data/detections/rob_mots/{split}/threshold_' + str(100*THRESHOLD) + '/data/'),
    'SPLIT': 'train',  # valid: 'train', 'val', 'test'.
    'Benchmarks': None,  # If None, all benchmarks in SPLIT.

    'Num_Parallel_Cores': None,  # If None, run without parallel.

    'DETECTION_THRESHOLD': THRESHOLD,
}


def do_sequence(seq_file):

    # Load input data from file (e.g. provided detections)
    # data format: data['cls'][t] = {'ids', 'scores', 'im_hs', 'im_ws', 'mask_rles'}
    data = butils.load_seq(seq_file)

    # Where to accumulate output data for writing out
    output_data = []

    # Run for each class.
    for cls, cls_data in data.items():

        # Run for each timestep.
        for timestep, t_data in enumerate(cls_data):

            # Threshold detections.
            t_data = butils.threshold(t_data, config['DETECTION_THRESHOLD'])

            # Save result in output format to write to file later.
            # Output Format = [timestep ID class score im_h im_w mask_RLE]
            for i in range(len(t_data['ids'])):
                row = [timestep, int(t_data['ids'][i]), cls, t_data['scores'][i], t_data['im_hs'][i],
                       t_data['im_ws'][i], t_data['mask_rles'][i]]
                output_data.append(row)

    # Write results to file
    out_file = seq_file.replace(config['INPUT_FOL'].format(split=config['SPLIT']),
                                config['OUTPUT_FOL'].format(split=config['SPLIT']))
    butils.write_seq(output_data, out_file)

    print('DONE:', seq_todo)


if __name__ == '__main__':

    # Required to fix bug in multiprocessing on windows.
    freeze_support()

    # Obtain list of sequences to run tracker for.
    if config['Benchmarks']:
        benchmarks = config['Benchmarks']
    else:
        benchmarks = ['davis_unsupervised', 'kitti_mots', 'youtube_vis', 'ovis', 'bdd_mots', 'tao']
        if config['SPLIT'] != 'train':
            benchmarks += ['waymo', 'mots_challenge']
    seqs_todo = []
    for bench in benchmarks:
        bench_fol = os.path.join(config['INPUT_FOL'].format(split=config['SPLIT']), bench)
        seqs_todo += [os.path.join(bench_fol, seq) for seq in os.listdir(bench_fol)]

    # Run in parallel
    if config['Num_Parallel_Cores']:
        with Pool(config['Num_Parallel_Cores']) as pool:
            results = pool.map(do_sequence, seqs_todo)

    # Run in series
    else:
        for seq_todo in seqs_todo:
            do_sequence(seq_todo)

