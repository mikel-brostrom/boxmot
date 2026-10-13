"""Selected coordinate-noise profiles construct trackers in every replay adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import yaml

from boxmot import create_tracker
from boxmot.engine.eval import saved_detections, trackrcnn
from boxmot.engine.tracking import workflow
from boxmot.trackers import Tracker, TrackerSpec


class _TrackerConstructed(Exception):
    """Stop an adapter after constructing its real tracker, before dataset replay."""


@pytest.mark.parametrize("adapter", ("live", "saved_detections", "trackrcnn"))
def test_saved_coordinate_profile_constructs_tracker(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YAML lists reach the selected filter unchanged through immutable specs."""
    profile = tmp_path / "selected.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "tracker": "bytetrack",
                "use_cmc": False,
                "kalman": {
                    "type": "xywh",
                    "noise": {"measurement_noise_diagonal": [0.5, 1.0, 1.5, 2.0]},
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        tracker="bytetrack",
        tracker_config=profile,
        dataset="fixture",
        split="train",
        sequence_names=(),
        cmc_policy=None,
    )
    trackers: list[Tracker] = []

    def construct(spec: TrackerSpec) -> NoReturn:
        trackers.append(create_tracker(spec))
        raise _TrackerConstructed

    if adapter == "live":
        trackers.append(create_tracker(workflow._tracker_spec(args, "aabb")))
    else:
        module = saved_detections if adapter == "saved_detections" else trackrcnn
        monkeypatch.setattr(module, "create_tracker", construct)
        if adapter == "saved_detections":
            monkeypatch.setattr(
                module, "load_saved_2d_evaluation_inputs", lambda *args, **kwargs: SimpleNamespace(sequences=())
            )
            run = module.run_saved_detections
        else:
            monkeypatch.setattr(module, "kitti_mots_sequences", lambda *args: ())
            run = module._run
        with pytest.raises(_TrackerConstructed):
            run(args)

    tracker = trackers[0]
    assert tracker.kalman_config.type == "xywh"
    assert tracker.kalman_noise_config.measurement_noise_diagonal == (0.5, 1.0, 1.5, 2.0)
    assert tracker.cmc is None
