"""Resolve EdgeTAM inference weights through BoxMOT's model directory."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from boxmot.components.artifacts import resolve_artifact
from boxmot.resources.paths import resolve_model_path

_EDGETAM_URL = "https://huggingface.co/facebook/EdgeTAM/resolve/main/edgetam.pt"
_EDGETAM_SHA256 = "ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df"


def resolve_edgetam_checkpoint(checkpoint: str | Path) -> Path:
    """Reuse a local checkpoint or download official ``edgetam.pt`` into models.

    Bare filenames follow BoxMOT's existing model-path resolution. Explicit
    paths are honored, and only the published checkpoint name is downloadable.
    """
    path = resolve_model_path(Path(checkpoint).expanduser()).resolve()
    if path.is_file():
        return path
    if path.exists():
        raise ValueError(f"EdgeTAM checkpoint must be a file: {path}")
    if path.name.lower() != "edgetam.pt":
        raise FileNotFoundError(
            f"EdgeTAM checkpoint does not exist: {path}. "
            "Use 'edgetam.pt' to download the official weights into the models directory, "
            "or supply an existing checkpoint path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # resolve_artifact verifies after downloading. Keep those bytes private
    # until verification succeeds so retries cannot mistake a failed download
    # for an existing user checkpoint.
    with TemporaryDirectory(prefix=f".{path.name}.", suffix=".part", dir=path.parent) as directory:
        artifact = resolve_artifact(
            Path(directory) / path.name,
            source_uri=_EDGETAM_URL,
            expected_sha256=_EDGETAM_SHA256,
            allow_download=True,
        )
        try:
            # A link publishes the complete file atomically without replacing
            # a checkpoint another caller may have supplied during download.
            path.hardlink_to(artifact.path)
        except FileExistsError:
            if not path.is_file():
                raise ValueError(f"EdgeTAM checkpoint must be a file: {path}") from None
    return path
