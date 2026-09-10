"""Dataset resource acquisition shared by engine workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from boxmot.engine.materialization.catalog import (
    STILL_FRAME_EXTENSIONS,
    resolve_dataset_root,
    resolve_dataset_split_root,
)
from boxmot.engine.tracking.sources import is_appledouble_file


def _get_download_status_fn() -> Any:
    """Load the optional download UI bridge only when acquisition is needed."""

    from boxmot.resources.download import get_download_status_fn

    return get_download_status_fn()


def _download_eval_data(**kwargs: Any) -> None:
    """Load the heavyweight HTTP download stack lazily."""

    from boxmot.resources.download import download_eval_data

    download_eval_data(**kwargs)


def _has_dataset_content(path: Path) -> bool:
    """Return whether a split contains at least one candidate MOT frame."""

    if not path.is_dir():
        return False
    for sequence_root in path.iterdir():
        if not sequence_root.is_dir() or is_appledouble_file(sequence_root):
            continue
        image_root = sequence_root / "img1"
        if not image_root.is_dir():
            image_root = sequence_root
        if any(
            child.is_file() and not is_appledouble_file(child) and child.suffix.lower() in STILL_FRAME_EXTENSIONS
            for child in image_root.iterdir()
        ):
            return True
    return False


def _download_state_paths(dataset_root: Path, split: str) -> tuple[Path, Path]:
    """Return stable lock and interrupted-download markers for one split."""

    root_identity = sha256(str(dataset_root).encode("utf-8")).hexdigest()
    split_identity = sha256(f"{dataset_root}\0{split}".encode("utf-8")).hexdigest()
    state_root = dataset_root.parent / ".boxmot" / "downloads"
    return (
        state_root / f"{root_identity}.lock",
        state_root / f"{split_identity}.incomplete",
    )


def _dataset_resource(dataset: Mapping[str, Any]) -> Mapping[str, Any] | None:
    resources = dataset.get("resources")
    if resources in (None, {}):
        return None
    if not isinstance(resources, Mapping):
        raise ValueError("Dataset resources must be a mapping.")
    resource = resources.get("dataset")
    if resource in (None, {}):
        return None
    if not isinstance(resource, Mapping):
        raise ValueError("Dataset resources.dataset must be a mapping.")
    return resource


def _resource_uri(
    resource: Mapping[str, Any],
    *,
    dataset_id: str,
    split: str,
) -> str:
    resource_type = resource.get("type")
    if resource_type != "per_split":
        raise ValueError(f"Dataset {dataset_id!r} resource type must be 'per_split', got {resource_type!r}.")
    uris = resource.get("uris")
    if not isinstance(uris, Mapping):
        raise ValueError(f"Dataset {dataset_id!r} per_split resource must define a uris mapping.")
    uri = uris.get(split)
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError(f"Dataset {dataset_id!r} has no download URI for split {split!r}.")
    uri = uri.strip()
    if not uri.startswith("hf://"):
        raise ValueError(f"Dataset {dataset_id!r} per_split resource must use an hf:// URI, got {uri!r}.")
    return uri


def ensure_dataset_split_available(
    dataset: Mapping[str, Any],
    *,
    split: str,
    data_root: str | Path | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> Path:
    """Download a configured dataset resource when its selected split is absent.

    Existing populated split directories are authoritative and are never
    replaced. Only Hugging Face ``per_split`` resources are acquired here;
    other datasets remain untouched so the catalog can report its normal
    local-path error.
    """

    dataset_root = resolve_dataset_root(dataset, data_root)
    split_root = resolve_dataset_split_root(dataset, split, data_root)
    lock_path, incomplete_path = _download_state_paths(dataset_root, split)
    if _has_dataset_content(split_root) and not incomplete_path.exists():
        return split_root

    resource = _dataset_resource(dataset)
    if resource is None:
        return split_root
    if resource.get("type") != "per_split":
        return split_root

    dataset_id = str(dataset.get("id") or "dataset")
    uri = _resource_uri(resource, dataset_id=dataset_id, split=split)

    # Downloads may run for hours and multiple CLI processes can target the
    # same split. FileLock releases ownership after a crash even though its
    # sentinel remains, while the separate marker tells the next owner to
    # resume a downloader-created partial Hugging Face tree.
    from filelock import FileLock

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        interrupted = incomplete_path.exists()
        if _has_dataset_content(split_root) and not interrupted:
            return split_root

        incomplete_path.touch(exist_ok=True)
        registered_callback = _get_download_status_fn()
        download_status = registered_callback if registered_callback is not None else status_callback
        _download_eval_data(
            dataset_url=uri,
            dataset_dest=dataset_root,
            # This split already failed the candidate-frame readiness check.
            # Bypass the lower-level "any populated target" shortcut; HF's
            # snapshot cache still makes the transfer resumable.
            overwrite=True,
            status_fn=download_status,
        )
        if not _has_dataset_content(split_root):
            raise FileNotFoundError(
                f"Dataset {dataset_id!r} resource {uri!r} did not create configured split {split!r} at {split_root}."
            )
        incomplete_path.unlink(missing_ok=True)
    return split_root


__all__ = ("ensure_dataset_split_available",)
