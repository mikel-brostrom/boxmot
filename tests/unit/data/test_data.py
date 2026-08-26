import pytest

from boxmot.data.benchmark import _ordered_benchmark_eval_class_names
from boxmot.data.dataset import MOTDataset, _collect_seq_info, _list_sequence_frames


def test_collect_seq_info_reads_mot_style_sequences(tmp_path):
    seq_dir = tmp_path / "SEQ01" / "img1"
    seq_dir.mkdir(parents=True)
    for frame_id in (1, 2):
        (seq_dir / f"{frame_id:06d}.jpg").write_bytes(b"")

    seq_paths, seq_info = _collect_seq_info(tmp_path)

    assert seq_paths == [seq_dir]
    assert seq_info == {"SEQ01": 2}


def test_list_sequence_frames_matches_native_extensions(tmp_path):
    expected = ["000001.jpg", "000002.jpeg", "000003.npy", "000004.png"]
    for filename in expected:
        (tmp_path / filename).touch()

    assert [path.name for path in _list_sequence_frames(tmp_path)] == expected


def test_list_sequence_frames_rejects_duplicate_representations(tmp_path):
    first = tmp_path / "000001.jpg"
    second = tmp_path / "000001.npy"
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="Multiple image files found for frame stem '000001'") as exc_info:
        _list_sequence_frames(tmp_path)

    message = str(exc_info.value)
    assert first.name in message
    assert second.name in message


def test_collect_seq_info_falls_back_to_seqinfo_when_img_dir_empty(tmp_path):
    seq_root = tmp_path / "SEQ02"
    img_dir = seq_root / "img1"
    img_dir.mkdir(parents=True)
    (seq_root / "seqinfo.ini").write_text("[Sequence]\nname=SEQ02\nimDir=img1\nseqLength=123\n")

    seq_paths, seq_info = _collect_seq_info(tmp_path)

    assert seq_paths == [img_dir]
    assert seq_info == {"SEQ02": 123}


def test_motdataset_indexes_sequence_with_empty_img1_from_seqinfo(tmp_path):
    seq_root = tmp_path / "SEQ03"
    (seq_root / "img1").mkdir(parents=True)
    (seq_root / "seqinfo.ini").write_text(
        "[Sequence]\nname=SEQ03\nimDir=img1\nimWidth=1920\nimHeight=1080\nseqLength=5\n"
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
