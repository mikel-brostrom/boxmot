from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import boxmot.utils.download as download_module


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
