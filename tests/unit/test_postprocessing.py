import numpy as np

from boxmot.postprocessing.gsi import gaussian_smooth, linear_interpolation
from boxmot.postprocessing.gbrc import gradient_boosting_smooth
from boxmot.postprocessing.gbrc import linear_interpolation as gbrc_linear_interpolation
from boxmot.postprocessing.sct import sct
from pathlib import Path
import shutil


def test_gsi():
    tracking_results = np.array(
        [
            [1, 1, 1475, 419, 75, 169, 0, 0, -1],
            [2, 1, 1475, 419, 75, 169, 0, 0, -1],
            [4, 1, 1475, 419, 75, 169, 0, 0, -1],
            [6, 1, 1475, 419, 75, 169, 0, 0, -1],
        ]
    )
    li = linear_interpolation(tracking_results, interval=20)
    gsi = gaussian_smooth(li, tau=10)
    assert len(gsi) == 6


def test_gbrc():
    tracking_results = np.array(
        [
            [1, 1, 1475, 419, 75, 169, 0, 0, -1],
            [2, 1, 1475, 419, 75, 169, 0, 0, -1],
            [4, 1, 1475, 419, 75, 169, 0, 0, -1],
            [6, 1, 1475, 419, 75, 169, 0, 0, -1],
        ]
    )
    li = gbrc_linear_interpolation(tracking_results, interval=20)
    gbrc = gradient_boosting_smooth(li)
    assert len(gbrc) == 6
    assert gbrc.shape[1] == 10


def test_sct(tmp_path):
    # Create dummy data
    mot_results_folder = tmp_path / "mot"
    mot_results_folder.mkdir()
    dets_folder = tmp_path / "dets"
    dets_folder.mkdir()
    embs_folder = tmp_path / "embs"
    embs_folder.mkdir()

    seq_name = "seq1"
    
    # Create dummy MOT results
    # frame, id, x, y, w, h, conf, -1, -1, -1
    mot_data = np.array([
        [1, 1, 100, 100, 50, 50, 0.9, -1, -1, -1],
        [2, 1, 105, 105, 50, 50, 0.9, -1, -1, -1],
        [3, 2, 110, 110, 50, 50, 0.8, -1, -1, -1], # ID switch
        [4, 2, 115, 115, 50, 50, 0.8, -1, -1, -1],
    ])
    np.savetxt(mot_results_folder / f"{seq_name}.txt", mot_data, fmt='%f', delimiter=',')

    # Create dummy detections
    # frame, x1, y1, x2, y2, conf, cls
    dets_data = np.array([
        [1, 100, 100, 150, 150, 0.9, 0],
        [2, 105, 105, 155, 155, 0.9, 0],
        [3, 110, 110, 160, 160, 0.8, 0],
        [4, 115, 115, 165, 165, 0.8, 0],
    ])
    np.savetxt(dets_folder / f"{seq_name}.txt", dets_data, fmt='%f')

    # Create dummy embeddings (random vectors)
    embs_data = np.random.rand(4, 512)
    # Make embeddings for frame 1,2 similar and 3,4 similar, but 1,2 different from 3,4
    # Actually, for SCT to merge them, they should be similar.
    # Let's make them all similar to test merging.
    embs_data = np.ones((4, 512)) 
    np.savetxt(embs_folder / f"{seq_name}.txt", embs_data, fmt='%f')

    # Run SCT
    sct(mot_results_folder, dets_folder, embs_folder, 
        eps=0.5, min_samples=1, max_k=2, min_len=1, spatial_factor=10.0, merge_dist_thres=0.1)

    # Check results
    # The result file should be overwritten
    result_data = np.loadtxt(mot_results_folder / f"{seq_name}.txt", delimiter=',')
    
    # Since embeddings are identical and spatial constraints are loose, 
    # and we have a small gap (frame 2 to 3), they should be merged into a single track ID.
    # The original IDs were 1 and 2. The new ID should be consistent (e.g., 1).
    
    assert len(np.unique(result_data[:, 1])) == 1
    assert len(result_data) == 4
