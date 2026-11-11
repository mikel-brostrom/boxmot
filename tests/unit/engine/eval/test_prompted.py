"""Ground-truth initialization drives real EdgeTAM updates and full MOT scoring."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config.runtime import build_mode_namespace
from boxmot.engine.eval import catalog_cache, motmetrics, prompted
from boxmot.segmentors.propagation.prompts import EdgeTAMPrompt
from boxmot.structures import Detections, Frame, Tracks
from boxmot.trackers.edgetam.tracker import EdgeTAM


class _StaticPropagator:
    """Keep each prompted mask, exposing unwanted corrections and removals."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[EdgeTAMPrompt, ...]]] = []
        self.masks: dict[int, torch.Tensor] = {}
        self.removed: list[int] = []
        self.resets = 0

    def process(
        self, frame_index: int, frame: np.ndarray, prompts: Sequence[EdgeTAMPrompt] = ()
    ) -> dict[int, torch.Tensor]:
        """Derive masks only from prompts, never from subsequent annotations."""
        del frame_index
        self.calls.append((int(frame[0, 0, 0]), tuple(prompts)))
        for prompt in prompts:
            assert prompt.box is not None
            left, top, right, bottom = map(int, prompt.box)
            mask = torch.zeros(frame.shape[:2], dtype=torch.bool)
            mask[top:bottom, left:right] = True
            self.masks[prompt.object_id] = mask
        return self.masks

    def remove_object(self, object_id: int) -> None:
        """Record any removal so GT disappearance cannot silently drive it."""
        self.removed.append(object_id)
        self.masks.pop(object_id, None)

    def reset(self) -> None:
        """Retain call history while clearing sequence memory."""
        self.resets += 1
        self.masks.clear()


class _RecordingTracker(EdgeTAM):
    """Inspect workflow inputs while retaining the real validated tracker."""

    def __init__(self) -> None:
        self.backend = _StaticPropagator()
        self.updates: list[tuple[int, tuple[int, ...]]] = []
        super().__init__(propagator=self.backend)

    def update(self, detections: Detections, frame: Frame | None = None, **kwargs: Any) -> Tracks:
        """Require detector-free canonical updates on every source frame."""
        assert isinstance(detections, Detections)
        assert len(detections) == 0
        assert isinstance(frame, Frame)
        assert detections.sample_id == frame.sample_id
        result = super().update(detections, frame, **kwargs)
        assert isinstance(result, Tracks)
        self.updates.append((int(frame.image[0, 0, 0]), tuple(result.track_ids.tolist())))
        return result


@pytest.fixture(autouse=True)
def _isolated_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real metrics deterministic and catalog cache writes local."""
    monkeypatch.setattr(motmetrics, "_metric_worker_count", lambda _sequences: 1)
    monkeypatch.setattr(
        catalog_cache, "default_source_metadata_cache_path", lambda _root: tmp_path / "catalog-cache.json"
    )


def _dataset(tmp_path: Path, sequences: dict[str, tuple[int, list[list[float]]]]) -> Path:
    """Write complete MOT sources with frame markers independent of GT rows."""
    for sequence_index, (sequence, (frames, rows)) in enumerate(sequences.items()):
        root = tmp_path / "Fixture" / "train" / sequence
        (root / "img1").mkdir(parents=True)
        (root / "gt").mkdir()
        for frame in range(1, frames + 1):
            image = np.full((40, 60, 3), sequence_index * 10 + frame, dtype=np.uint8)
            assert cv2.imwrite(str(root / "img1" / f"{frame:06d}.png"), image)
        (root / "seqinfo.ini").write_text(
            f"[Sequence]\nname={sequence}\nframeRate=30\nseqLength={frames}\nimWidth=60\nimHeight=40\n",
            encoding="utf-8",
        )
        np.savetxt(root / "gt" / "gt.txt", np.asarray(rows, dtype=float), delimiter=",", fmt="%g")
    path = tmp_path / "fixture.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "prompted-fixture",
                "format": {"layout": "mot", "box_type": "aabb"},
                "storage": {"root": "Fixture"},
                "default_split": "train",
                "splits": {"train": {"path": "train", "has_ground_truth": True}},
                "classes": {"target": {"person": 1}, "ignore": {"distractor": 8}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _arguments(tmp_path: Path, dataset: Path, **options: Any) -> SimpleNamespace:
    """Build the same normalized namespace as the evaluation command."""
    return build_mode_namespace(
        "eval",
        {
            "dataset": dataset,
            "data_root": tmp_path,
            "split": "train",
            "tracker": "edgetam",
            "prompt_source": "first-visible",
            "prompt_min_visibility": 1.0,
            "prompt_weights": "unused-test-checkpoint.pt",
            "sequence_workers": 1,
            "device": "cpu",
            "project": tmp_path / "evaluation",
            **options,
        },
    )


@pytest.mark.parametrize(
    "name, value",
    [
        ("edgetam", True),
        ("mask_guidance_weights", "edgetam.pt"),
        ("mask_guidance_max_objects", 96),
    ],
)
def test_prompted_evaluation_rejects_mask_guidance_before_dataset_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: object
) -> None:
    """Direct Python calls cannot combine detector guidance with prompted tracking."""
    args = _arguments(tmp_path, tmp_path / "unused.yaml", **{name: value})
    monkeypatch.setattr(prompted, "_resolve_selection", lambda *_args: pytest.fail("Dataset accessed"))

    with pytest.raises(ValueError, match=f"does not support --{name.replace('_', '-')}"):
        prompted.run_prompted_evaluation(args)


def _install_tracker(monkeypatch: pytest.MonkeyPatch) -> tuple[_RecordingTracker, list[dict[str, Any]]]:
    """Replace only model construction, retaining replay, output, and metrics."""
    tracker = _RecordingTracker()
    constructions: list[dict[str, Any]] = []

    def create_tracker(*_args: Any, **kwargs: Any) -> _RecordingTracker:
        constructions.append(kwargs)
        return tracker

    monkeypatch.setattr(prompted, "create_tracker", create_tracker)
    return tracker, constructions


def test_first_visible_prompts_once_and_scores_the_full_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-prompt, never-visible, and post-departure errors remain in MOT scores."""
    rows = []
    for frame in range(1, 6):
        if frame < 5:
            rows.append([frame, 10, 2, 3, 8, 12, 1, 1, 0.5 if frame == 1 else 1])
        rows.extend(
            [
                [frame, 20, -1 if frame < 3 else 20, 15, 8, 10, 1, 1, 1],
                [frame, 30, 40, 3, 8, 12, 1, 1, 0.5],
                [frame, 40, 40, 22, 8, 12, 1, 1, 0.5 if frame < 3 else 1],
                [frame, 90, 50, 2, 5, 5, 1, 8, 1],
                [frame, 91, 50, 12, 5, 5, 0, 1, 1],
            ]
        )
    dataset = _dataset(tmp_path, {"SEQ-01": (5, rows)})
    ground_truth = tmp_path / "Fixture/train/SEQ-01/gt/gt.txt"
    original_ground_truth = ground_truth.read_bytes()
    tracker, constructions = _install_tracker(monkeypatch)

    result = prompted.run_prompted_evaluation(_arguments(tmp_path, dataset))

    assert len(constructions) == 1
    assert tracker.updates == [(1, ()), (2, (10,)), (3, (10, 20, 40)), (4, (10, 20, 40)), (5, (10, 20, 40))]
    assert [(frame, tuple(prompt.object_id for prompt in prompts)) for frame, prompts in tracker.backend.calls] == [
        (2, (10,)),
        (3, (20, 40)),
        (4, ()),
        (5, ()),
    ]
    assert tracker.backend.calls[0][1][0].box == (2, 3, 10, 15)
    assert tracker.backend.removed == []
    metrics = result.raw["person"]
    assert metrics["Frames"] == 5
    assert metrics["GT_Dets"] == 19
    assert metrics["GT_IDs"] == 4
    assert metrics["CLR_TP"] == 9
    assert metrics["CLR_FN"] == 10
    assert metrics["CLR_FP"] == 1
    assert metrics["IDSW"] == 0
    predictions = np.loadtxt(result.exp_dir / "SEQ-01.txt", delimiter=",", ndmin=2)
    assert predictions.shape == (10, 9)
    assert set(predictions[:, 0]) == {2, 3, 4, 5}
    assert set(predictions[:, 7]) == {1}
    assert ground_truth.read_bytes() == original_ground_truth
    prompt_table = json.loads((result.exp_dir / "prompts.json").read_text(encoding="utf-8"))
    assert prompt_table == {
        "SEQ-01": [
            {"frame": 2, "object_id": 10, "box_xyxy": [2, 3, 10, 15]},
            {"frame": 3, "object_id": 20, "box_xyxy": [20, 15, 28, 25]},
            {"frame": 3, "object_id": 40, "box_xyxy": [40, 22, 48, 34]},
        ]
    }
    saved_metrics = json.loads((result.exp_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["person"]["CLR_FN"] == 10
    manifest = json.loads((result.exp_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["prompt_counts"] == {"SEQ-01": 3}
    assert manifest["prompt_min_visibility"] == 1.0


def test_sequence_boundaries_reset_memory_and_allow_reused_gt_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identity from one sequence cannot leak into another sequence's frames."""
    dataset = _dataset(
        tmp_path,
        {
            "SEQ-01": (2, [[frame, 10, 2, 3, 8, 12, 1, 1, 1] for frame in (1, 2)]),
            "SEQ-02": (2, [[frame, 10, 40, 3, 8, 12, 1, 1, 0.5 if frame == 1 else 1] for frame in (1, 2)]),
        },
    )
    tracker, _ = _install_tracker(monkeypatch)

    result = prompted.run_prompted_evaluation(_arguments(tmp_path, dataset))

    assert tracker.updates == [(1, (10,)), (2, (10,)), (11, ()), (12, (10,))]
    assert tracker.backend.resets >= 2
    assert [(frame, tuple(prompt.object_id for prompt in prompts)) for frame, prompts in tracker.backend.calls] == [
        (1, (10,)),
        (2, ()),
        (12, (10,)),
    ]
    assert result.raw["person"]["GT_Dets"] == 4
    assert result.raw["person"]["CLR_FN"] == 1
    predictions = np.loadtxt(result.exp_dir / "SEQ-02.txt", delimiter=",", ndmin=2)
    np.testing.assert_array_equal(predictions[:, :6], [[2, 10, 40, 3, 8, 12]])


def test_sequence_selection_does_not_prompt_or_score_unselected_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the requested sequence supplies prompt plans and metric targets."""
    dataset = _dataset(
        tmp_path,
        {
            "SEQ-01": (1, [[1, 10, 2, 3, 8, 12, 1, 1, 1]]),
            "SEQ-02": (1, [[1, 20, 20, 3, 8, 12, 1, 1, 1]]),
        },
    )
    tracker, _ = _install_tracker(monkeypatch)

    result = prompted.run_prompted_evaluation(_arguments(tmp_path, dataset, sequence_names=("SEQ-02",)))

    assert tracker.updates == [(11, (20,))]
    assert result.raw["person"]["GT_Dets"] == 1
    assert result.raw["person"]["HOTA"] == pytest.approx(100)
    assert {path.name for path in result.exp_dir.glob("SEQ-*.txt")} == {"SEQ-02.txt"}


@pytest.mark.parametrize("min_visibility", [0.5, 1.0])
def test_visibility_threshold_can_leave_an_entire_sequence_unprompted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, min_visibility: float
) -> None:
    """The authored threshold reaches prompt selection without trimming scoring."""
    dataset = _dataset(
        tmp_path,
        {"SEQ-01": (2, [[frame, 10, 2, 3, 8, 12, 1, 1, 0.5] for frame in (1, 2)])},
    )
    tracker, _ = _install_tracker(monkeypatch)

    result = prompted.run_prompted_evaluation(_arguments(tmp_path, dataset, prompt_min_visibility=min_visibility))

    prompted_ids = (10,) if min_visibility == 0.5 else ()
    assert tracker.updates == [(1, prompted_ids), (2, prompted_ids)]
    assert result.raw["person"]["GT_Dets"] == 2
    assert result.raw["person"]["CLR_FN"] == (0 if prompted_ids else 2)
    if not prompted_ids:
        assert tracker.backend.calls == []
        assert (result.exp_dir / "SEQ-01.txt").read_text(encoding="utf-8") == ""
        assert json.loads((result.exp_dir / "prompts.json").read_text(encoding="utf-8")) == {"SEQ-01": []}


def test_capacity_is_preflighted_for_all_sequences_before_tracker_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cumulative prompts in a later sequence must fail before model inference."""
    dataset = _dataset(
        tmp_path,
        {
            "SEQ-01": (1, [[1, 10, 2, 3, 8, 12, 1, 1, 1]]),
            "SEQ-02": (2, [[1, 20, 20, 3, 8, 12, 1, 1, 1], [2, 30, 40, 3, 8, 12, 1, 1, 1]]),
        },
    )
    config = tmp_path / "edgetam.yaml"
    config.write_text("max_objects: 1\n", encoding="utf-8")
    monkeypatch.setattr(prompted, "create_tracker", lambda *_args, **_kwargs: pytest.fail("model constructed"))

    with pytest.raises(ValueError, match="max_objects|capacity|at most"):
        prompted.run_prompted_evaluation(_arguments(tmp_path, dataset, tracker_config=config))


def test_missing_visibility_in_later_sequence_fails_before_tracker_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Incomplete MOT annotations cannot trigger a partially executed run."""
    dataset = _dataset(
        tmp_path,
        {
            "SEQ-01": (1, [[1, 10, 2, 3, 8, 12, 1, 1, 1]]),
            "SEQ-02": (1, [[1, 20, 20, 3, 8, 12, 1, 1]]),
        },
    )
    monkeypatch.setattr(prompted, "create_tracker", lambda *_args, **_kwargs: pytest.fail("model constructed"))

    with pytest.raises(ValueError, match="visibility|nine|9"):
        prompted.run_prompted_evaluation(_arguments(tmp_path, dataset))


def test_inference_failure_marks_the_run_failed_and_clears_sequence_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial prediction file must never masquerade as a completed evaluation."""
    dataset = _dataset(
        tmp_path,
        {"SEQ-01": (2, [[frame, 10, 2, 3, 8, 12, 1, 1, 1] for frame in (1, 2)])},
    )
    tracker, _ = _install_tracker(monkeypatch)
    process = tracker.backend.process

    def fail_on_second_frame(
        frame_index: int, frame: np.ndarray, prompts: Sequence[EdgeTAMPrompt] = ()
    ) -> dict[int, torch.Tensor]:
        if int(frame[0, 0, 0]) == 2:
            raise RuntimeError("propagation failed")
        return process(frame_index, frame, prompts)

    monkeypatch.setattr(tracker.backend, "process", fail_on_second_frame)
    args = _arguments(tmp_path, dataset)

    with pytest.raises(RuntimeError, match="propagation failed"):
        prompted.run_prompted_evaluation(args)

    assert tracker.object_ids == ()
    assert tracker.backend.masks == {}
    assert tracker.backend.resets >= 2
    manifest = json.loads((args.exp_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"] == "propagation failed"
    assert not (args.exp_dir / "metrics.json").exists()


def test_cli_runs_real_prompted_workflow_and_reporter_without_loading_weights(tmp_path: Path) -> None:
    """Exercise command dispatch, genuine empty tracking, scoring, and reporting."""
    dataset = _dataset(tmp_path, {"SEQ-01": (1, [[1, 10, 2, 3, 8, 12, 1, 1, 0.5]])})
    project = tmp_path / "cli-evaluation"

    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            str(dataset),
            "--data-root",
            str(tmp_path),
            "--split",
            "train",
            "--tracker",
            "edgetam",
            "--prompt-source",
            "first-visible",
            "--prompt-weights",
            str(tmp_path / "deliberately-absent.pt"),
            "--sequence-workers",
            "1",
            "--project",
            str(project),
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    manifests = list(project.rglob("eval_manifest.json"))
    assert len(manifests) == 1
    output = manifests[0].parent
    assert json.loads(manifests[0].read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((output / "prompts.json").read_text(encoding="utf-8")) == {"SEQ-01": []}
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))["person"]
    assert metrics["GT_Dets"] == metrics["CLR_FN"] == 1
    assert (output / "SEQ-01.txt").read_text(encoding="utf-8") == ""
    assert "Results:" in result.output
