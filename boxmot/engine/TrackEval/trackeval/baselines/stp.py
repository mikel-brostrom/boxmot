"""
STP: Simplest Tracker Possible

Author: Jonathon Luiten

This simple tracker, simply assigns track IDs which maximise the 'bounding box IoU' between previous tracks and current
detections. It is also able to match detections to tracks at more than one timestep previously.
"""

import os
import sys
import numpy as np
from multiprocessing.pool import Pool
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from trackeval.baselines import baseline_utils as butils
from trackeval.utils import get_code_path

code_path = get_code_path()
config = {
    'INPUT_FOL': os.path.join(code_path, 'data/detections/rob_mots/{split}/non_overlap_supplied/data/'),
    'OUTPUT_FOL': os.path.join(code_path, 'data/trackers/rob_mots/{split}/STP/data/'),
    'SPLIT': 'train',  # valid: 'train', 'val', 'test'.
    'Benchmarks': None,  # If None, all benchmarks in SPLIT.

    'Num_Parallel_Cores': None,  # If None, run without parallel.

    'DETECTION_THRESHOLD': 0.5,
    'ASSOCIATION_THRESHOLD': 1e-10,
    'MAX_FRAMES_SKIP': 7
}


def track_sequence(seq_file):

    # Load input data from file (e.g. provided detections)
    # data format: data['cls'][t] = {'ids', 'scores', 'im_hs', 'im_ws', 'mask_rles'}
    data = butils.load_seq(seq_file)

    # Where to accumulate output data for writing out
    output_data = []

    # To ensure IDs are unique per object across all classes.
    curr_max_id = 0

    # Run tracker for each class.
    for cls, cls_data in data.items():

        # Initialize container for holding previously tracked objects.
        prev = {'boxes': np.empty((0, 4)),
                'ids': np.array([], int),
                'timesteps': np.array([])}

        # Run tracker for each timestep.
        for timestep, t_data in enumerate(cls_data):

            # Threshold detections.
            t_data = butils.threshold(t_data, config['DETECTION_THRESHOLD'])

            # Convert mask dets to bounding boxes.
            boxes = butils.masks2boxes(t_data['mask_rles'], t_data['im_hs'], t_data['im_ws'])

            # Calculate IoU between previous and current frame dets.
            ious = butils.box_iou(prev['boxes'], boxes)

            # Score which decreases quickly for previous dets depending on how many timesteps before they come from.
            prev_timestep_scores = np.power(10, -1 * prev['timesteps'])

            # Matching score is such that it first tries to match 'most recent timesteps',
            # and within each timestep maximised IoU.
            match_scores = prev_timestep_scores[:, np.newaxis] * ious

            # Find best matching between current dets and previous tracks.
            match_rows, match_cols = butils.match(match_scores)

            # Remove matches that have an IoU below a certain threshold.
            actually_matched_mask = ious[match_rows, match_cols] > config['ASSOCIATION_THRESHOLD']
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            # Assign the prev track ID to the current dets if they were matched.
            ids = np.nan * np.ones((len(boxes),), int)
            ids[match_cols] = prev['ids'][match_rows]

            # Create new track IDs for dets that were not matched to previous tracks.
            num_not_matched = len(ids) - len(match_cols)
            new_ids = np.arange(curr_max_id + 1, curr_max_id + num_not_matched + 1)
            ids[np.isnan(ids)] = new_ids

            # Update maximum ID to ensure future added tracks have a unique ID value.
            curr_max_id += num_not_matched

            # Drop tracks from 'previous tracks' if they have not been matched in the last MAX_FRAMES_SKIP frames.
            unmatched_rows = [i for i in range(len(prev['ids'])) if
                              i not in match_rows and (prev['timesteps'][i] + 1 <= config['MAX_FRAMES_SKIP'])]

            # Update the set of previous tracking results to include the newly tracked detections.
            prev['ids'] = np.concatenate((ids, prev['ids'][unmatched_rows]), axis=0)
            prev['boxes'] = np.concatenate((np.atleast_2d(boxes), np.atleast_2d(prev['boxes'][unmatched_rows])), axis=0)
            prev['timesteps'] = np.concatenate((np.zeros((len(ids),)), prev['timesteps'][unmatched_rows] + 1), axis=0)

            # Save result in output format to write to file later.
            # Output Format = [timestep ID class score im_h im_w mask_RLE]
            for i in range(len(t_data['ids'])):
                row = [timestep, int(ids[i]), cls, t_data['scores'][i], t_data['im_hs'][i], t_data['im_ws'][i],
                       t_data['mask_rles'][i]]
                output_data.append(row)

    # Write results to file
    out_file = seq_file.replace(config['INPUT_FOL'].format(split=config['SPLIT']),
                                config['OUTPUT_FOL'].format(split=config['SPLIT']))
    butils.write_seq(output_data, out_file)

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
        bench_fol = os.path.join(config['INPUT_FOL'].format(split=config['SPLIT']), bench)
        seqs_todo += [os.path.join(bench_fol, seq) for seq in os.listdir(bench_fol)]

    # Run in parallel
    if config['Num_Parallel_Cores']:
        with Pool(config['Num_Parallel_Cores']) as pool:
            results = pool.map(track_sequence, seqs_todo)

    # Run in series
    else:
        for seq_todo in seqs_todo:
            track_sequence(seq_todo)

