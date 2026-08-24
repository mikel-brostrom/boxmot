from boxmot.data.benchmark import _ordered_benchmark_eval_class_names
from boxmot.data.dataset import MOTDataset, _collect_seq_info


def test_collect_seq_info_reads_mot_style_sequences(tmp_path):
    seq_dir = tmp_path / "SEQ01" / "img1"
    seq_dir.mkdir(parents=True)
    for frame_id in (1, 2):
        (seq_dir / f"{frame_id:06d}.jpg").write_bytes(b"")

    seq_paths, seq_info = _collect_seq_info(tmp_path)

    assert seq_paths == [seq_dir]
    assert seq_info == {"SEQ01": 2}


def test_collect_seq_info_falls_back_to_seqinfo_when_img_dir_empty(tmp_path):
    seq_root = tmp_path / "SEQ02"
    img_dir = seq_root / "img1"
    img_dir.mkdir(parents=True)
    (seq_root / "seqinfo.ini").write_text(
        "[Sequence]\n"
        "name=SEQ02\n"
        "imDir=img1\n"
        "seqLength=123\n"
    )

    seq_paths, seq_info = _collect_seq_info(tmp_path)

    assert seq_paths == [img_dir]
    assert seq_info == {"SEQ02": 123}


def test_motdataset_indexes_sequence_with_empty_img1_from_seqinfo(tmp_path):
    seq_root = tmp_path / "SEQ03"
    (seq_root / "img1").mkdir(parents=True)
    (seq_root / "seqinfo.ini").write_text(
        "[Sequence]\n"
        "name=SEQ03\n"
        "imDir=img1\n"
        "imWidth=1920\n"
        "imHeight=1080\n"
        "seqLength=5\n"
    )

    dataset = MOTDataset(mot_root=str(tmp_path))

    assert dataset.sequence_names() == ["SEQ03"]


def test_ordered_benchmark_eval_class_names_preserve_multiword_names_direct():
    bench_cfg = {
        "eval_classes": {
            1: "small vehicle",
            2: "large vehicle",
        }
    }

    class_names = _ordered_benchmark_eval_class_names(bench_cfg)

    assert class_names == ["small vehicle", "large vehicle"]
