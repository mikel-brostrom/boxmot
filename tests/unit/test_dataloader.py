import pytest
import numpy as np
import configparser
import cv2
from pathlib import Path
from boxmot.utils.dataloaders.dataset import (
    read_seq_fps,
    compute_fps_mask,
    MOTDataset,
    MOTSequence,
)


def test_read_seq_fps(tmp_path):
    # create a seqinfo.ini
    seq_dir = tmp_path / "SEQ01"
    seq_dir.mkdir()
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {"frameRate": "30"}
    with open(seq_dir / "seqinfo.ini", "w") as f:
        cfg.write(f)

    assert read_seq_fps(seq_dir) == 30

    # missing file should raise
    with pytest.raises(FileNotFoundError):
        read_seq_fps(tmp_path / "NONEXISTENT")


def test_compute_fps_mask():
    frames = np.arange(1, 7)  # [1,2,3,4,5,6]
    # downsample from 6→3 fps => step=2 → keep [1,3,5]
    mask = compute_fps_mask(frames, orig_fps=6, target_fps=3)
    assert mask.dtype == bool
    assert frames[mask].tolist() == [1, 3, 5]


@pytest.fixture
def simple_sequence(tmp_path):
    """
    Create a minimal MOT sequence structure:
      seq_dir/
        img1/000001.jpg, 000002.jpg
        seqinfo.ini (fps 2)
        gt/gt.txt
      det_emb_root/model/dets/SEQ.txt
      det_emb_root/model/embs/reid/SEQ.txt
    """
    # seq dir & images
    seq_dir = tmp_path / "SEQ"
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir()
    # write two dummy images
    img = np.zeros((8, 8, 3), np.uint8)
    for i in (1, 2):
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), img)

    # seqinfo.ini fps=2
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {"frameRate": "2"}
    with open(seq_dir / "seqinfo.ini", "w") as f:
        cfg.write(f)

    # ground truth with two frames
    gt = np.array([[1, 0, 0, 0, 0, 0], [2, 1, 1, 1, 1, 1]])
    np.savetxt(gt_dir / "gt.txt", gt, delimiter=",")

    # detection + embedding roots
    det_emb_root = tmp_path / "runs"
    model = det_emb_root / "model"
    det_dir = model / "dets"
    emb_dir = model / "embs" / "reid"
    det_dir.mkdir(parents=True)
    emb_dir.mkdir(parents=True)

    # two det rows (frame_id, x,y,w,h,score)
    dets = np.array([[1, 0, 0, 1, 1, 0.9], [2, 0, 0, 1, 1, 0.8]])
    np.savetxt(det_dir / "SEQ.txt", dets, fmt="%f")

    # two 128-d embeddings
    embs = np.vstack([np.arange(128), np.arange(128)])
    np.savetxt(emb_dir / "SEQ.txt", embs, fmt="%f")

    return {
        "mot_root": tmp_path,
        "det_emb_root": det_emb_root,
        "model_name": "model",
        "reid_name": "reid",
        "seq_name": "SEQ",
        "seq_dir": seq_dir,
    }


def test_dataset_indexing_and_iteration(simple_sequence):
    ds = MOTDataset(
        mot_root=str(simple_sequence["mot_root"]),
        det_emb_root=str(simple_sequence["det_emb_root"]),
        model_name=simple_sequence["model_name"],
        reid_name=simple_sequence["reid_name"],
        target_fps=None,
    )
    # sequence_names
    assert simple_sequence["seq_name"] in ds.sequence_names()

    # get_sequence yields 2 frames in order
    seq = ds.get_sequence(simple_sequence["seq_name"])
    out = list(seq)
    assert len(out) == 2
    for idx, frame in enumerate(out, start=1):
        assert frame["frame_id"] == idx
        assert frame["img"].shape == (8, 8, 3)
        # without downsampling, dets and embs should match original
        assert frame["dets"].shape[0] == 1
        assert frame["embs"].shape == (1, 128)


def test_unknown_sequence_raises(simple_sequence):
    ds = MOTDataset(mot_root=str(simple_sequence["mot_root"]))
    with pytest.raises(KeyError):
        _ = ds.get_sequence("DOES_NOT_EXIST")


def test_mismatched_dets_embs_raise(tmp_path, simple_sequence):
    # overwrite embeddings with only one row
    emb_file = (
        tmp_path
        / "runs"
        / "model"
        / "embs"
        / "reid"
        / "SEQ.txt"
    )
    one_emb = np.arange(128)
    np.savetxt(emb_file, one_emb[None, :], fmt="%f")

    with pytest.raises(ValueError):
        MOTDataset(
            mot_root=str(simple_sequence["mot_root"]),
            det_emb_root=str(simple_sequence["det_emb_root"]),
            model_name=simple_sequence["model_name"],
            reid_name=simple_sequence["reid_name"],
            target_fps=None,
        ).get_sequence(simple_sequence["seq_name"])


def test_fps_downsampling_and_gt_temp(tmp_path):
    # manually create minimal sequence as before
    seq_dir = tmp_path / "S"
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir()

    # two dummy images
    img = np.zeros((4, 4, 3), np.uint8)
    for i in (1, 2):
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), img)

    # seqinfo.ini fps=2
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {"frameRate": "2"}
    with open(seq_dir / "seqinfo.ini", "w") as f:
        cfg.write(f)

    # ground truth with two rows
    gt = np.array([[1, 9], [2, 8]])
    np.savetxt(gt_dir / "gt.txt", gt, delimiter=",", fmt="%d")

    # create dets/embs with both frames
    det_emb_root = tmp_path / "R"
    det_dir = det_emb_root / "M" / "dets"
    emb_dir = det_emb_root / "M" / "embs" / "R"
    det_dir.mkdir(parents=True)
    emb_dir.mkdir(parents=True)
    dets = np.array([[1, 0, 0, 1, 1, 0.5], [2, 0, 0, 1, 1, 0.4]])
    embs = np.vstack([np.arange(128), np.arange(128)])
    np.savetxt(det_dir / "S.txt", dets, fmt="%f")
    np.savetxt(emb_dir / "S.txt", embs, fmt="%f")

    # instantiate and trigger downsampling & gt_temp write
    ds = MOTDataset(
        mot_root=str(tmp_path),
        det_emb_root=str(det_emb_root),
        model_name="M",
        reid_name="R",
        target_fps=1,
    )
    _ = ds.get_sequence("S")  # triggers prep

    # load gt_temp.txt (numpy.loadtxt returns 1d for single row)
    gt_temp = np.loadtxt(seq_dir / "gt" / "gt_temp.txt", delimiter=",")

    # ensure only frame 1 remains
    # handle single-row vs 2d output
    if gt_temp.ndim == 1:
        # single row array
        assert gt_temp[0] == 1 and gt_temp[1] == 9
    else:
        assert gt_temp.shape == (1, 2)
        assert gt_temp[0, 0] == 1 and gt_temp[0, 1] == 9

