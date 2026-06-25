from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import boxmot.utils.download as download_module


def _write_trackeval_repo(root, *, marker="official", managed=False, branch="master"):
    datasets_dir = root / "trackeval" / "datasets"
    datasets_dir.mkdir(parents=True)
    (root / "trackeval" / "__init__.py").write_text("")
    (root / "trackeval" / "eval.py").write_text(marker)
    (datasets_dir / "__init__.py").write_text("")
    (datasets_dir / "mot_challenge_2d_box.py").write_text(
        "import numpy as np\n"
        "time_data = np.asarray([], dtype=np.float)\n"
        "id_data = np.asarray([], dtype=np.int)\n"
        "flag_data = np.asarray([], dtype=np.bool)\n"
    )
    if managed:
        (root / download_module.TRACKEVAL_SOURCE_MARKER).write_text(
            download_module._trackeval_source_marker_text(branch)
        )


def test_download_trackeval_patches_managed_official_cache_without_downloading(tmp_path):
    dest = tmp_path / "trackeval_root"
    pkg_dir = dest / "trackeval"
    dataset_dir = pkg_dir / "trackeval" / "datasets"
    _write_trackeval_repo(pkg_dir, managed=True)
    dataset_file = dataset_dir / "mot_challenge_2d_box.py"

    download_module.download_trackeval(dest)

    patched = dataset_file.read_text()
    assert "np.float" not in patched
    assert "np.int" not in patched
    assert "np.bool" not in patched
    assert "dtype=float" in patched
    assert "dtype=int" in patched
    assert "dtype=bool" in patched


def test_download_trackeval_uses_official_trackeval_archive(monkeypatch, tmp_path):
    dest = tmp_path / "trackeval_root"
    calls = {}

    def fake_download_file(url, zip_file, overwrite=False):
        calls["url"] = url
        calls["overwrite"] = overwrite
        zip_file.write_bytes(b"zip")
        return zip_file

    def fake_extract_zip(zip_path, extract_to, overwrite=False):
        calls["zip_path"] = zip_path
        calls["extract_to"] = extract_to
        _write_trackeval_repo(extract_to / "TrackEval-master", marker="official")

    monkeypatch.setattr(download_module, "download_file", fake_download_file)
    monkeypatch.setattr(download_module, "extract_zip", fake_extract_zip)

    download_module.download_trackeval(dest)

    assert calls["url"] == "https://github.com/JonathonLuiten/TrackEval/archive/refs/heads/master.zip"
    assert (dest / "trackeval" / "trackeval" / "eval.py").read_text() == "official"
    assert "np.float" not in (dest / "trackeval" / "trackeval" / "datasets" / "mot_challenge_2d_box.py").read_text()
    assert (dest / "trackeval" / download_module.TRACKEVAL_SOURCE_MARKER).read_text() == (
        download_module._trackeval_source_marker_text("master")
    )
    assert not (tmp_path / "TrackEval-master").exists()
    assert not (tmp_path / "trackeval-master.zip").exists()


def test_download_trackeval_refreshes_unmanaged_cache(monkeypatch, tmp_path):
    dest = tmp_path / "trackeval_root"
    _write_trackeval_repo(dest / "trackeval", marker="stale")
    calls = {}

    def fake_download_file(url, zip_file, overwrite=False):
        calls["url"] = url
        zip_file.write_bytes(b"zip")
        return zip_file

    def fake_extract_zip(_zip_path, extract_to, overwrite=False):
        _write_trackeval_repo(extract_to / "TrackEval-master", marker="replacement")

    monkeypatch.setattr(download_module, "download_file", fake_download_file)
    monkeypatch.setattr(download_module, "extract_zip", fake_extract_zip)

    download_module.download_trackeval(dest)

    assert calls["url"] == "https://github.com/JonathonLuiten/TrackEval/archive/refs/heads/master.zip"
    assert (dest / "trackeval" / "trackeval" / "eval.py").read_text() == "replacement"
    assert (dest / "trackeval" / download_module.TRACKEVAL_SOURCE_MARKER).read_text() == (
        download_module._trackeval_source_marker_text("master")
    )


def test_hf_subfolder_workflow_progress_uses_file_units(monkeypatch, tmp_path):
    class RepoFile:
        def __init__(self, size: int) -> None:
            self.size = size
            self.lfs = None

    class FakeHfApi:
        def list_repo_tree(self, **_kwargs):
            return [RepoFile(10), RepoFile(20), RepoFile(30)]

    snapshot_calls = []

    def fake_snapshot_download(*, tqdm_class, **kwargs):
        snapshot_calls.append(kwargs)
        # Hugging Face creates a byte-progress task and a file-fetch task.
        # The subfolder downloader should surface the file task in the workflow.
        byte_progress = tqdm_class(desc="Downloading file", total=60, unit="B", unit_scale=True)
        byte_progress.update(30)
        fetch_progress = tqdm_class(iterable=range(3), desc="Fetching 3 files", total=3)
        for _ in fetch_progress:
            pass

    hf_module = types.ModuleType("huggingface_hub")
    hf_module.HfApi = FakeHfApi
    hf_module.snapshot_download = fake_snapshot_download
    hf_api_module = types.ModuleType("huggingface_hub.hf_api")
    hf_api_module.RepoFile = RepoFile
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.hf_api", hf_api_module)

    class FakeStatus:
        def __init__(self) -> None:
            self.messages: list[str] = []
            self.units: list[str | None] = []
            self.tasks: list[SimpleNamespace] = []

        def __call__(self, message: str) -> None:
            self.messages.append(message)

        @contextmanager
        def tqdm_proxy(self, description: str, *, unit: str | None = None):
            self.units.append(unit)

            class FakeTqdm:
                _task_id = None

                def __init__(inner_self, iterable=None, *args, **kwargs) -> None:
                    inner_self._iterable = iterable
                    inner_self._task_id = len(self.tasks)
                    inner_self.n = int(kwargs.get("initial", 0) or 0)
                    inner_self.total = int(kwargs["total"]) if kwargs.get("total") else 0
                    self.tasks.append(
                        SimpleNamespace(
                            desc=kwargs.get("desc") or description,
                            total=inner_self.total,
                            completed=inner_self.n,
                        )
                    )

                def update(inner_self, n: int = 1) -> None:
                    inner_self.n += int(n)
                    self.tasks[inner_self._task_id].completed += int(n)

                def __iter__(inner_self):
                    for item in inner_self._iterable:
                        yield item
                        inner_self.update(1)

            yield FakeTqdm

    status = FakeStatus()

    download_module.download_hf_dataset_subfolder(
        "user/repo",
        "images/val",
        tmp_path,
        status_fn=status,
    )

    assert status.units == ["files"]
    assert status.tasks == [SimpleNamespace(desc="Fetching 3 files", total=3, completed=3)]
    assert snapshot_calls == [
        {
            "repo_id": "user/repo",
            "repo_type": "dataset",
            "local_dir": str(tmp_path),
            "allow_patterns": ["images/val/**"],
        }
    ]
    assert (tmp_path / "images" / "val" / ".hf_download_complete").exists()


def test_hf_subfolder_skips_populated_target_without_marker(tmp_path):
    target = tmp_path / "images" / "val"
    target.mkdir(parents=True)
    (target / "frame001.jpg").write_bytes(b"image")

    download_module.download_hf_dataset_subfolder("user/repo", "images/val", tmp_path)

    assert (target / ".hf_download_complete").exists()
