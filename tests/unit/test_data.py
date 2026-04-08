from pathlib import Path

import numpy as np

from boxmot.data.benchmark import _ordered_benchmark_eval_class_names
from boxmot.data.dataset import _collect_seq_info


def test_collect_seq_info_reads_mot_style_sequences(tmp_path):
    seq_dir = tmp_path / "SEQ01" / "img1"
    seq_dir.mkdir(parents=True)
    for frame_id in (1, 2):
        (seq_dir / f"{frame_id:06d}.jpg").write_bytes(b"")

    seq_paths, seq_info = _collect_seq_info(tmp_path)

    assert seq_paths == [seq_dir]
    assert seq_info == {"SEQ01": 2}


def test_ordered_benchmark_eval_class_names_preserve_multiword_names_direct():
    bench_cfg = {
        "eval_classes": {
            1: "small vehicle",
            2: "large vehicle",
        }
    }

    class_names = _ordered_benchmark_eval_class_names(bench_cfg)

    assert class_names == ["small vehicle", "large vehicle"]