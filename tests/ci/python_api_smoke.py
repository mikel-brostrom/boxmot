import shutil
from pathlib import Path

from boxmot import BoxMOT, ReIDModel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/MOT17-mini/train/MOT17-02-FRCNN/img1/000001.jpg"
EXPERIMENT = "mot17-mini-train-yolox-lmbn"
METRICS = {"HOTA", "MOTA", "IDF1"}


def test_python_api_smoke(tmp_path, monkeypatch):
    """Exercise the primary public Python workflows with real CPU runtimes."""
    assert SOURCE.is_file()
    dataset_link = tmp_path / "assets/MOT17-mini"
    dataset_link.parent.mkdir(parents=True)
    dataset_link.symlink_to(ROOT / "assets/MOT17-mini", target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "runs"
    reid_weights = tmp_path / "osnet_x0_25_msmt17.pt"
    api = BoxMOT(
        detector="yolo26n.pt",
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

    # Ray packages its current working directory for workers. Keep that package
    # limited to the miniature benchmark instead of the developer's whole tree.
    ray_workdir = tmp_path / "ray_workdir"
    shutil.copytree(ROOT / "assets/MOT17-mini", ray_workdir / "assets/MOT17-mini")
    monkeypatch.chdir(ray_workdir)

    tuned = api.tune(
        experiment=EXPERIMENT,
        n_trials=1,
        imgsz=320,
        device="cpu",
        project=project,
        verbose=False,
        seed=0,
    )
    assert len(tuned.trials) == 1
    assert tuned.best in tuned.trials
    assert METRICS <= tuned.summary.keys()
    assert tuned.best_yaml.is_file()
