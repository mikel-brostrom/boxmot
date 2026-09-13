from __future__ import annotations

import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import boxmot.resources.download as download_module
import boxmot.utils.dependencies as dependencies
from boxmot.utils.dependencies import MissingDependencyError


class _QuietProgress:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def update(self, _amount: int) -> None:
        pass


class _Response:
    def __init__(self, chunks, *, content_length: int | None = None) -> None:
        self._chunks = chunks
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, *, chunk_size: int):
        del chunk_size
        yield from self._chunks


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def get(self, *_args, **_kwargs) -> _Response:
        return self.response


def _temporary_parts(dest: Path) -> list[Path]:
    return list(dest.parent.glob(f".{dest.name}.*.part"))


def test_concurrent_http_downloads_never_publish_partial_destination(monkeypatch, tmp_path):
    dest = tmp_path / "model.pt"
    payloads = (b"a" * 32, b"b" * 32)
    reached_partial_write = threading.Barrier(3)
    release_downloads = threading.Event()
    assignment_lock = threading.Lock()
    pending_payloads = list(payloads)

    def chunks(payload: bytes):
        yield payload[:8]
        reached_partial_write.wait(timeout=5)
        release_downloads.wait(timeout=5)
        yield payload[8:]

    def fake_session():
        with assignment_lock:
            payload = pending_payloads.pop()
        return _Session(_Response(chunks(payload), content_length=len(payload)))

    monkeypatch.setattr(download_module, "get_http_session", fake_session)
    monkeypatch.setattr(download_module, "tqdm", _QuietProgress)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(download_module.download_file, "https://example.test/model.pt", dest) for _ in range(2)
        ]
        reached_partial_write.wait(timeout=5)
        try:
            assert not dest.exists()
            assert len(_temporary_parts(dest)) == 2
        finally:
            release_downloads.set()
        assert [future.result(timeout=5) for future in futures] == [dest, dest]

    assert dest.read_bytes() in payloads
    assert _temporary_parts(dest) == []


def test_truncated_http_download_preserves_existing_destination_and_cleans_temporary(
    monkeypatch,
    tmp_path,
):
    dest = tmp_path / "model.pt"
    dest.write_bytes(b"previous complete model")
    response = _Response([b"short"], content_length=10)
    monkeypatch.setattr(download_module, "get_http_session", lambda: _Session(response))
    monkeypatch.setattr(download_module, "tqdm", _QuietProgress)

    with pytest.raises(IOError, match="Truncated download"):
        download_module.download_file(
            "https://example.test/model.pt",
            dest,
            overwrite=True,
        )

    assert dest.read_bytes() == b"previous complete model"
    assert _temporary_parts(dest) == []


def test_http_download_error_cleans_temporary_without_publishing(monkeypatch, tmp_path):
    dest = tmp_path / "model.pt"

    def failing_chunks():
        yield b"partial"
        raise RuntimeError("connection dropped")

    monkeypatch.setattr(
        download_module,
        "get_http_session",
        lambda: _Session(_Response(failing_chunks())),
    )
    monkeypatch.setattr(download_module, "tqdm", _QuietProgress)

    with pytest.raises(RuntimeError, match="connection dropped"):
        download_module.download_file("https://example.test/model.pt", dest)

    assert not dest.exists()
    assert _temporary_parts(dest) == []


def test_google_drive_download_publishes_completed_temporary_atomically(monkeypatch, tmp_path):
    dest = tmp_path / "model.pt"
    payload = b"complete google drive model"

    def fake_gdown(*, output: str, **_kwargs):
        temporary = Path(output)
        assert temporary != dest
        assert temporary.parent == dest.parent
        assert not dest.exists()
        temporary.write_bytes(payload)
        return output

    monkeypatch.setattr(download_module.gdown, "download", fake_gdown)

    result = download_module.download_file("https://drive.google.com/file/d/model", dest)

    assert result == dest
    assert dest.read_bytes() == payload
    assert _temporary_parts(dest) == []


def test_google_drive_failure_preserves_destination_and_cleans_temporary(monkeypatch, tmp_path):
    dest = tmp_path / "model.pt"
    dest.write_bytes(b"previous complete model")

    def incomplete_gdown(*, output: str, **_kwargs):
        Path(output).write_bytes(b"partial")
        return None

    monkeypatch.setattr(download_module.gdown, "download", incomplete_gdown)

    with pytest.raises(IOError, match="Google Drive download failed"):
        download_module.download_file(
            "https://drive.google.com/file/d/model",
            dest,
            overwrite=True,
        )

    assert dest.read_bytes() == b"previous complete model"
    assert _temporary_parts(dest) == []


def test_hf_subfolder_workflow_progress_uses_file_units(monkeypatch, tmp_path):
    monkeypatch.setattr(download_module, "require_packages", lambda *args, **kwargs: None)

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


@pytest.mark.parametrize("download_kind", ["dataset", "interrupted-dataset", "subfolder"])
def test_missing_hub_dependency_stops_download_without_installation(monkeypatch, tmp_path, download_kind) -> None:
    """An unavailable Hub package produces install guidance before any network work."""

    def missing_distribution(name: str) -> None:
        raise dependencies.PackageNotFoundError(name)

    def unexpected_operation(*args, **kwargs) -> None:
        pytest.fail("Missing dependencies must not trigger downloads or package installation")

    monkeypatch.setattr(dependencies, "distribution", missing_distribution)
    monkeypatch.setattr(subprocess, "run", unexpected_operation)
    monkeypatch.setattr(subprocess, "check_call", unexpected_operation)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=unexpected_operation, snapshot_download=unexpected_operation),
    )
    target = tmp_path / "dataset"
    if download_kind == "interrupted-dataset":
        incomplete = target / ".cache" / "huggingface" / "download" / "data.incomplete"
        incomplete.parent.mkdir(parents=True)
        incomplete.write_bytes(b"partial")

    with pytest.raises(MissingDependencyError, match="huggingface-hub>=1.7.1") as error:
        if download_kind == "subfolder":
            download_module.snapshot_download_hf_subfolder("user/repo", "images/val", target)
        else:
            download_module.download_hf_dataset("user/repo", target)

    assert "boxmot.engine.cli install --requirement" in str(error.value)


@pytest.mark.parametrize("download_kind", ["dataset", "subfolder"])
def test_complete_hf_dataset_does_not_require_hub(monkeypatch, tmp_path, download_kind) -> None:
    """Completed local datasets remain usable without the downloading dependency."""

    def unexpected_dependency_check(*args, **kwargs) -> None:
        pytest.fail("A cached dataset must not require Hugging Face Hub")

    monkeypatch.setattr(download_module, "require_packages", unexpected_dependency_check)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    target = tmp_path / "images" / "val"
    target.mkdir(parents=True)
    (target / "frame001.jpg").write_bytes(b"image")

    if download_kind == "subfolder":
        download_module.download_hf_dataset_subfolder("user/repo", "images/val", tmp_path)
    else:
        download_module.download_hf_dataset("user/repo", tmp_path)

    assert (target / "frame001.jpg").read_bytes() == b"image"


def test_eval_dataset_download_routes_hf_subfolder_to_dataset_root(monkeypatch, tmp_path):
    calls = []
    messages = []

    def status(message):
        messages.append(message)

    dataset_root = tmp_path / "MOT17"

    def download_subfolder(repo_id, subfolder, dest_root, *, overwrite, status_fn):
        calls.append((repo_id, subfolder, dest_root, overwrite, status_fn))

    monkeypatch.setattr(download_module, "download_hf_dataset_subfolder", download_subfolder)

    download_module.download_eval_data(
        dataset_url="hf://Lekim89/MOT17/ablation",
        dataset_dest=dataset_root,
        status_fn=status,
    )

    assert calls == [("Lekim89/MOT17", "ablation", dataset_root, False, status)]
    assert messages == ["Setting up evaluation data..."]


@pytest.mark.parametrize(
    "subfolder",
    (
        "../outside",
        "images/../../outside",
        ".",
        "/absolute/path",
        r"\absolute\path",
        r"C:\absolute\path",
        r"images\val",
    ),
)
def test_hf_subfolder_rejects_unsafe_repository_paths(subfolder, tmp_path):
    with pytest.raises(ValueError, match="Invalid Hugging Face dataset subfolder"):
        download_module.download_hf_dataset_subfolder("user/repo", subfolder, tmp_path)

    assert not (tmp_path.parent / "outside").exists()
