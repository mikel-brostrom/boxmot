import os
from pathlib import Path

import torch

from boxmot import BoxMOT, ReIDModel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/MOT17-mini/train/MOT17-02-FRCNN/img1/000001.jpg"
EXPERIMENT = "mot17-mini-train-yolox-lmbn"
METRICS = {"HOTA", "MOTA", "IDF1"}


def test_python_api_smoke(tmp_path, monkeypatch):
    """Exercise the primary public Python workflows with real CPU runtimes."""
    assert torch.version.cuda is None, f"Expected CPU-only PyTorch, got torch {torch.__version__}"
    assert not torch.cuda.is_available()
    assert SOURCE.is_file()
    dataset_link = tmp_path / "assets/MOT17-mini"
    dataset_link.parent.mkdir(parents=True)
    dataset_link.symlink_to(ROOT / "assets/MOT17-mini", target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "runs"
    reid_weights = tmp_path / "osnet_x0_25_msmt17.pt"
    detector_weights = os.environ.get("BOXMOT_CI_DETECTOR", "yolo26n.pt")
    api = BoxMOT(
        detector=detector_weights,
        reid=reid_weights,
        tracker="ocsort",
        classes=[0],
        project=project,
    )

    tracked = api.track(source=SOURCE, imgsz=320, device="cpu", verbose=False)
    assert tracked.summary["frames"] == 1

    evaluated = api.val(
        experiment=EXPERIMENT,
        imgsz=320,
        device="cpu",
        project=project,
        verbose=False,
    )
    assert METRICS <= evaluated.summary.keys()

    # Resolve the downloadable checkpoint into pytest's temporary directory so
    # the export never overwrites a developer's existing model artifact.
    reid = ReIDModel(reid_weights, device="cpu")
    assert reid.path == reid_weights
    del reid

    exported = BoxMOT(reid=reid_weights, project=project).export(
        format="torchscript",
        device="cpu",
        batch_size=1,
        dynamic=False,
    )
    assert Path(exported.files["torchscript"]).is_file()
    assert exported.parity_ok
