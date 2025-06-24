"""
Vizualize: Code which converts .txt rle tracking results into a visual .png format.

Author: Jonathon Luiten
"""

import os
import sys
from multiprocessing.pool import Pool
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from trackeval.baselines import baseline_utils as butils
from trackeval.utils import get_code_path
from trackeval.datasets.rob_mots_classmap import cls_id_to_name

code_path = get_code_path()
config = {
    # Tracker format:
    'INPUT_FOL': os.path.join(code_path, 'data/trackers/rob_mots/{split}/STP/data/{bench}'),
    'OUTPUT_FOL': os.path.join(code_path, 'data/viz/rob_mots/{split}/STP/data/{bench}'),
    # GT format:
    # 'INPUT_FOL': os.path.join(code_path, 'data/gt/rob_mots/{split}/{bench}/data/'),
    # 'OUTPUT_FOL': os.path.join(code_path, 'data/gt_viz/rob_mots/{split}/{bench}/'),
    'SPLIT': 'train',  # valid: 'train', 'val', 'test'.
    'Benchmarks': None,  # If None, all benchmarks in SPLIT.
    'Num_Parallel_Cores': None,  # If None, run without parallel.
}


def do_sequence(seq_file):
    # Folder to save resulting visualization in
    out_fol = seq_file.replace(config['INPUT_FOL'].format(split=config['SPLIT'], bench=bench),
                               config['OUTPUT_FOL'].format(split=config['SPLIT'], bench=bench)).replace('.txt', '')

    # Load input data from file (e.g. provided detections)
    # data format: data['cls'][t] = {'ids', 'scores', 'im_hs', 'im_ws', 'mask_rles'}
    data = butils.load_seq(seq_file)

    # Get frame size for visualizing empty frames
    im_h, im_w = butils.get_frame_size(data)

    # First run for each class.
    for cls, cls_data in data.items():

        if cls >= 100:
            continue

        # Run for each timestep.
        for timestep, t_data in enumerate(cls_data):
            # Save out visualization
            out_file = os.path.join(out_fol, cls_id_to_name[cls], str(timestep).zfill(5) + '.png')
            butils.save_as_png(t_data, out_file, im_h, im_w)


    # Then run for all classes combined
    # Converts data from a class-separated to a class-combined format.
    data = butils.combine_classes(data)

    # Run for each timestep.
    for timestep, t_data in enumerate(data):
        # Save out visualization
        out_file = os.path.join(out_fol, 'all_classes', str(timestep).zfill(5) + '.png')
        butils.save_as_png(t_data, out_file, im_h, im_w)

    print('DONE:', seq_file)


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
        bench_fol = config['INPUT_FOL'].format(split=config['SPLIT'], bench=bench)
        seqs_todo += [os.path.join(bench_fol, seq) for seq in os.listdir(bench_fol)]

    # Run in parallel
    if config['Num_Parallel_Cores']:
        with Pool(config['Num_Parallel_Cores']) as pool:
            results = pool.map(do_sequence, seqs_todo)

    # Run in series
    else:
        for seq_todo in seqs_todo:
            do_sequence(seq_todo)
