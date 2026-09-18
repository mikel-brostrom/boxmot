"""Exercise cached evaluation with real EdgeTAM in a spawned worker.

Run with downloaded weights, for example::

    python -m tests.ci.mask_guidance_eval_smoke --checkpoint /path/to/edgetam.pt --device mps

The fixture validates replay and metric plumbing; its perfect synthetic scores
are not an accuracy benchmark for mask guidance.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from boxmot.datasets.schema import INSTANCES_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.eval.motmetrics import evaluate_motchallenge_hota
from boxmot.engine.eval.replay import replay_build
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint
from boxmot.trackers import TrackerSpec

_SEQUENCES = {
    "sequence-a": (3, (8, 8, 40, 80)),
    "sequence-b": (7, (72, 16, 112, 72)),
}


def _build_fixture(root: Path) -> tuple[Path, dict[str, Path]]:
    """Publish shuffled samples selecting three JPEGs per offset source sequence."""
    detect = StagePlan.create("detect", component={"id": "mask-guidance-eval-smoke"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=root / "builds",
        dataset_name="mask-guidance-eval-smoke",
        box_type="aabb",
        source_fingerprint=fingerprint("mask-guidance-eval-smoke"),
        publish=PublishOptions(image_references=True),
        stages=(detect, finalize),
        metadata={"split": "ablation"},
    )
    samples, instances = [], []
    ground_truth_paths = {}
    for sequence, (offset, box) in _SEQUENCES.items():
        image_dir = plan.staging_root / "images" / sequence
        image_dir.mkdir(parents=True)
        x1, y1, x2, y2 = box
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[y1:y2, x1:x2] = (40, 200, 240)
        ground_truth = []
        # Unselected leading frames must not be loaded as this split's frame 0.
        for index in range(offset):
            assert cv2.imwrite(str(image_dir / f"{index + 1:06d}.jpg"), np.zeros_like(image))
        for local_index in range(3):
            frame_index = offset + local_index
            sample_id = f"{sequence}/{frame_index}"
            filename = f"{frame_index + 1:06d}.jpg"
            assert cv2.imwrite(str(image_dir / filename), image)
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": "ablation",
                    "sequence_id": sequence,
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / 30.0,
                    "image_ref": f"images/{sequence}/{filename}",
                    "height": 96,
                    "width": 128,
                }
            )
            instances.append(
                {
                    "instance_id": f"{plan.build_id}:{sample_id}:0",
                    "sample_id": sample_id,
                    "detection_index": 0,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": 0.95,
                    "class_id": 1,
                }
            )
            ground_truth.append([frame_index + 1, 1, x1, y1, x2 - x1, y2 - y1, 1, 1, 1])
        ground_truth_path = root / f"{sequence}-gt.txt"
        np.savetxt(ground_truth_path, ground_truth, delimiter=",", fmt="%g")
        ground_truth_paths[sequence] = ground_truth_path
    writer = ParquetShardWriter(plan.staging_root, box_type="aabb")
    writer.write(SAMPLES_ARTIFACT, list(reversed(samples)), shard_index=0)
    writer.write(INSTANCES_ARTIFACT, list(reversed(instances)), shard_index=0)
    return finalize_build(plan), ground_truth_paths


def main() -> None:
    """Run actual checkpoint inference, sequence isolation, MOT output, and HOTA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="boxmot-mask-guidance-eval-") as directory:
        root = Path(directory)
        build, ground_truth = _build_fixture(root)
        result = replay_build(
            build,
            TrackerSpec(name="bytetrack"),
            split="ablation",
            output_dir=root / "results",
            workers=1,
            mask_guidance_weights=args.checkpoint,
            mask_guidance_device=args.device,
        )
        assert result.frames == 6
        assert result.track_rows == 6
        assert [path.stem for path in result.sequence_files] == list(_SEQUENCES)
        sequence_files = {}
        for path in result.sequence_files:
            offset, (x1, y1, x2, y2) = _SEQUENCES[path.stem]
            rows = np.loadtxt(path, delimiter=",", ndmin=2)
            assert rows.shape == (3, 9)
            assert np.isfinite(rows).all()
            np.testing.assert_array_equal(rows[:, 0], np.arange(offset + 1, offset + 4))
            np.testing.assert_array_equal(rows[:, 1], np.zeros(3))
            np.testing.assert_allclose(rows[:, 2:6], np.tile([x1, y1, x2 - x1, y2 - y1], (3, 1)))
            np.testing.assert_allclose(rows[:, 6], 0.95)
            np.testing.assert_array_equal(rows[:, 7], np.ones(3))
            sequence_files[path.stem] = (ground_truth[path.stem], path)
        metrics = evaluate_motchallenge_hota(sequence_files)
        for name in ("HOTA", "DetA", "AssA"):
            np.testing.assert_allclose(metrics[name], 1.0)
        assert set(metrics["per_sequence"]) == set(_SEQUENCES)
    print("EdgeTAM evaluation smoke passed: spawned replay, two sequences, source offsets, MOT rows, and HOTA.")


if __name__ == "__main__":
    main()
