"""Content identities for resolved local component artifacts."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

_DIRECTORY_HASH_DOMAIN = b"boxmot-artifact-directory-v1\0"


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """A local, immutable model artifact identity."""

    path: Path
    sha256: str
    source_uri: str | None = None

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        if not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(path)
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("sha256 must be a lowercase 64-character digest")
        if self.source_uri is not None and not self.source_uri.strip():
            raise ValueError("source_uri must be non-empty when provided")
        object.__setattr__(self, "path", path)


def _hash_file(path: Path, *, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a model file or complete local snapshot directory deterministically.

    File identities are their ordinary SHA-256. Directory identities include
    every regular file's relative POSIX path and content digest, so both
    renames and content changes alter the identity. File symlink targets are
    hashed as bytes to support model-cache snapshots; directory symlinks are
    rejected because their trees would escape the enumerated snapshot.
    """

    artifact = Path(path).expanduser().resolve()
    if artifact.is_file():
        return _hash_file(artifact, chunk_size=chunk_size)
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)

    files = sorted(
        (entry for entry in artifact.rglob("*") if entry.is_file() or entry.is_symlink()),
        key=lambda entry: entry.relative_to(artifact).as_posix(),
    )
    if not files:
        raise ValueError(f"Model artifact snapshot is empty: {artifact}")

    digest = hashlib.sha256(_DIRECTORY_HASH_DOMAIN)
    for entry in files:
        relative = entry.relative_to(artifact).as_posix()
        if entry.is_symlink() and not entry.resolve(strict=True).is_file():
            raise ValueError(f"Model artifact snapshots may not contain directory symlinks: {relative}")
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(bytes.fromhex(_hash_file(entry, chunk_size=chunk_size)))
    return digest.hexdigest()


def _download_url(uri: str) -> str:
    if uri.startswith("gdrive://"):
        file_id = uri.removeprefix("gdrive://").strip("/")
        if not file_id:
            raise ValueError("gdrive artifact URI is missing a file ID")
        return f"https://drive.google.com/uc?id={file_id}"
    if uri.startswith(("http://", "https://")):
        return uri
    raise ValueError(f"Unsupported model artifact URI: {uri!r}")


def _huggingface_repo_id(uri: str) -> str:
    """Return the ``owner/repository`` portion of a model snapshot URI."""

    repository = uri.removeprefix("hf://")
    parts = repository.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid Hugging Face model artifact URI {uri!r}; expected 'hf://owner/repository'.")
    return repository


def _download_huggingface_snapshot(uri: str, destination: Path) -> Path:
    """Download and atomically publish one Hugging Face model snapshot."""

    from huggingface_hub import snapshot_download

    repository = _huggingface_repo_id(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    try:
        snapshot_download(
            repo_id=repository,
            repo_type="model",
            local_dir=str(temporary),
        )
        # ``local_dir`` downloads include Hugging Face transport metadata whose
        # timestamps are not part of the model. Exclude only that namespaced
        # metadata so a repository-owned ``.cache`` entry remains intact.
        cache_root = temporary / ".cache"
        shutil.rmtree(cache_root / "huggingface", ignore_errors=True)
        try:
            cache_root.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # The repository owns another entry under ``.cache``.
            pass
        if not any(entry.is_file() or entry.is_symlink() for entry in temporary.rglob("*")):
            raise ValueError(f"Hugging Face model snapshot {repository!r} contains no artifact files.")
        try:
            temporary.replace(destination)
        except OSError:
            # A concurrent resolver may have published the same destination
            # while this process was downloading. Reuse only a complete
            # directory; all other publication failures remain actionable.
            if not destination.is_dir():
                raise
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def resolve_artifact(
    path: str | Path,
    *,
    source_uri: str | None = None,
    expected_sha256: str | None = None,
    allow_download: bool = False,
) -> ResolvedArtifact:
    """Resolve, optionally download, and hash a component artifact."""

    resolved = Path(path).expanduser().resolve()
    if not (resolved.is_file() or resolved.is_dir()):
        if not allow_download or source_uri is None:
            hint = " Supply a resolved local artifact before materialization."
            raise FileNotFoundError(f"Model artifact does not exist: {resolved}.{hint}")
        if source_uri.startswith("hf://"):
            resolved = _download_huggingface_snapshot(source_uri, resolved).resolve()
        else:
            from boxmot.resources.download import download_file

            download_file(_download_url(source_uri), resolved)
    digest = sha256_artifact(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Artifact SHA-256 mismatch for {resolved}: expected {expected_sha256}, got {digest}.")
    return ResolvedArtifact(path=resolved, sha256=digest, source_uri=source_uri)


def verify_artifact_identity(path: str | Path, expected_sha256: str) -> Path:
    """Resolve a local artifact and reject content that differs from its spec."""

    artifact = Path(path).expanduser().resolve()
    actual_sha256 = sha256_artifact(artifact)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Artifact SHA-256 mismatch for {artifact}: expected {expected_sha256}, got {actual_sha256}.")
    return artifact


def require_resolved_artifact(
    path: str | None,
    expected_sha256: str | None,
    *,
    component: str,
) -> Path:
    """Require the complete local content identity used by a component factory."""

    if path is None or expected_sha256 is None:
        raise ValueError(f"{component} requires a resolved local artifact path and its SHA-256 digest.")
    authored = Path(path).expanduser()
    resolved = authored.resolve()
    if not authored.is_absolute() or authored != resolved:
        raise ValueError(f"{component} artifact must be a canonical absolute path, got {path!r}.")
    return verify_artifact_identity(resolved, expected_sha256)


__all__ = (
    "ResolvedArtifact",
    "require_resolved_artifact",
    "resolve_artifact",
    "sha256_artifact",
    "verify_artifact_identity",
)
