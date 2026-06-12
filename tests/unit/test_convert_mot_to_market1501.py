import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def _write_sequence(seq_dir: Path, *, num_pids: int = 6, num_frames: int = 6) -> None:
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "img1").mkdir()
    (seq_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={seq_dir.name}",
                "imDir=img1",
                "imExt=.jpg",
                "",
            ]
        )
    )

    for frame in range(1, num_frames + 1):
        image = Image.new("RGB", (160, 180), color=(frame * 20, frame * 15, frame * 10))
        image.save(seq_dir / "img1" / f"{frame:06d}.jpg")

    rows = []
    for pid in range(1, num_pids + 1):
        for frame in range(1, num_frames + 1):
            x = 8 + pid
            y = 12
            w = 32
            h = 100
            rows.append(f"{frame},{pid},{x},{y},{w},{h},1,1,1")
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")


def _pids(folder: Path) -> set[int]:
    return {int(path.name.split("_", 1)[0]) for path in folder.glob("*.jpg")}


def test_convert_mot_to_market1501_splits_train_and_eval_identities(tmp_path):
    seq_dir = tmp_path / "MOT17-02-FRCNN"
    output = tmp_path / "MOT17-1501"
    _write_sequence(seq_dir)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/convert_mot_to_market1501.py"),
            "--output",
            str(output),
            "--eval-id-ratio",
            "0.5",
            "--seed",
            "7",
            "--min-frame-gap",
            "1",
            "--no-progress",
            str(seq_dir),
        ],
        check=True,
        cwd=ROOT,
    )

    train_pids = _pids(output / "bounding_box_train")
    query_pids = _pids(output / "query")
    gallery_pids = _pids(output / "bounding_box_test")

    assert train_pids
    assert query_pids
    assert gallery_pids
    assert train_pids.isdisjoint(query_pids)
    assert train_pids.isdisjoint(gallery_pids)
    assert query_pids == gallery_pids

    summary = json.loads((output / "conversion_summary.json").read_text())
    assert summary["train_query_pid_overlap"] == 0
    assert summary["train_gallery_pid_overlap"] == 0
    assert summary["query_gallery_pid_overlap"] == len(query_pids)
    assert summary["split_image_counts"]["gt_query"] == summary["split_image_counts"]["query"]
    assert summary["split_image_counts"]["gt_bbox"] == summary["split_image_counts"]["bounding_box_test"]
