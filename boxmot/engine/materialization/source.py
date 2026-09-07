"""Immutable source records and bounded frame decoding for materialization."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from urllib.parse import unquote, urlparse

import cv2

from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.readers.images import NUMPY_IMAGE_EXTENSIONS, read_numpy_bgr_uint8
from boxmot.engine.tracking.sources import frame_from_bgr
from boxmot.structures import Frame

SourceDigestResolver = Callable[[Path], str]


@dataclass(frozen=True, slots=True)
class SourceSample:
    """Metadata-only catalog entry for one finite local source frame.

    ``source_frame_index`` is the zero-based frame number inside a video. It is
    ``None`` for still images. Pixel data is deliberately absent so catalogs
    remain cheap to construct, retain, fingerprint, and use during evaluation.
    """

    sample_id: str
    split: str
    sequence_id: str
    frame_index: int
    timestamp_s: float | None
    image_size: tuple[int, int]
    source_uri: str
    source_sha256: str
    image_ref: str | None = None
    source_frame_index: int | None = None

    def __post_init__(self) -> None:
        for name in ("sample_id", "split", "sequence_id", "source_uri"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"SourceSample.{name} must be a non-empty canonical string.")
        if self.image_ref is not None and (
            not isinstance(self.image_ref, str) or not self.image_ref or self.image_ref != self.image_ref.strip()
        ):
            raise ValueError("SourceSample.image_ref must be a non-empty canonical string or None.")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("SourceSample.source_sha256 must be a full lowercase SHA-256 digest.")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("SourceSample.frame_index must be a non-negative integer.")
        if self.source_frame_index is not None and (
            isinstance(self.source_frame_index, bool)
            or not isinstance(self.source_frame_index, int)
            or self.source_frame_index < 0
        ):
            raise ValueError("SourceSample.source_frame_index must be a non-negative integer or None.")
        if self.timestamp_s is not None and (
            not isinstance(self.timestamp_s, float) or not math.isfinite(self.timestamp_s)
        ):
            raise ValueError("SourceSample.timestamp_s must be a finite float or None.")
        if (
            not isinstance(self.image_size, tuple)
            or len(self.image_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.image_size)
        ):
            raise ValueError("SourceSample.image_size must be a (height, width) tuple of positive integers.")


def _local_path(source_uri: str) -> Path:
    parsed = urlparse(source_uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"Materialization source_uri must be a local file URI, got {source_uri!r}.")
    return Path(unquote(parsed.path))


def _validate_frame(frame: Frame, sample: SourceSample) -> None:
    if frame.sample_id != sample.sample_id:
        raise ValueError(f"Decoded frame identity does not match source sample {sample.sample_id!r}.")
    if frame.sequence_id != sample.sequence_id or frame.frame_index != sample.frame_index:
        raise ValueError(f"Decoded frame sequence identity does not match source sample {sample.sample_id!r}.")
    if frame.timestamp_s != sample.timestamp_s or frame.source_uri != sample.source_uri:
        raise ValueError(f"Decoded frame provenance does not match source sample {sample.sample_id!r}.")
    if frame.image_size != sample.image_size:
        raise ValueError(
            f"Decoded frame {sample.sample_id!r} has size {frame.image_size}, "
            f"but the source catalog records {sample.image_size}."
        )


def _decode_source_sample(sample: SourceSample) -> Frame:
    """Decode one already content-verified catalog record."""

    if not isinstance(sample, SourceSample):
        raise TypeError(f"Expected SourceSample, got {type(sample).__name__}.")
    path = _local_path(sample.source_uri)
    if sample.source_frame_index is None:
        if path.suffix.lower() in NUMPY_IMAGE_EXTENSIONS:
            image = read_numpy_bgr_uint8(path)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode source image: {path}")
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise OSError(f"Could not open source video: {path}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(sample.source_frame_index)):
                raise ValueError(f"Could not seek source video {path} to frame {sample.source_frame_index}.")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"Could not decode source video {path} frame {sample.source_frame_index}.")
        finally:
            capture.release()

    frame = frame_from_bgr(
        image,
        sample_id=sample.sample_id,
        sequence_id=sample.sequence_id,
        frame_index=sample.frame_index,
        timestamp_s=sample.timestamp_s,
        source_uri=sample.source_uri,
    )
    _validate_frame(frame, sample)
    return frame


def decode_source_sample(sample: SourceSample) -> Frame:
    """Verify and decode one catalog record against immutable metadata."""

    if not isinstance(sample, SourceSample):
        raise TypeError(f"Expected SourceSample, got {type(sample).__name__}.")
    path = _local_path(sample.source_uri)
    if sha256_file(path) != sample.source_sha256:
        raise ValueError(f"Source content changed after cataloging: {path}")
    return _decode_source_sample(sample)


class BoundedFrameDecoder:
    """Decode only the current model batch with a bounded thread pool."""

    def __init__(
        self,
        workers: int,
        *,
        digest_resolver: SourceDigestResolver | None = None,
    ) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("decode workers must be a positive integer.")
        if digest_resolver is not None and not callable(digest_resolver):
            raise TypeError("digest_resolver must be callable or None.")
        self.workers = workers
        self._digest_resolver = digest_resolver or sha256_file
        self._executor = None if workers == 1 else ThreadPoolExecutor(max_workers=workers)
        self._verified_sources: dict[str, str] = {}

    def verify(self, samples: Sequence[SourceSample]) -> None:
        """Verify each unique source file once for this decoder lifetime."""

        for sample in samples:
            if not isinstance(sample, SourceSample):
                raise TypeError("BoundedFrameDecoder accepts only SourceSample records.")
            known_digest = self._verified_sources.get(sample.source_uri)
            if known_digest is not None:
                if known_digest != sample.source_sha256:
                    raise ValueError(f"Conflicting source digests for {sample.source_uri!r}.")
                continue
            path = _local_path(sample.source_uri)
            if self._digest_resolver(path) != sample.source_sha256:
                raise ValueError(f"Source content changed after cataloging: {path}")
            self._verified_sources[sample.source_uri] = sample.source_sha256

    def decode(self, samples: Sequence[SourceSample]) -> list[Frame]:
        records = list(samples)
        self.verify(records)
        if self._executor is None:
            return [_decode_source_sample(sample) for sample in records]
        return list(self._executor.map(_decode_source_sample, records))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> BoundedFrameDecoder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = (
    "BoundedFrameDecoder",
    "SourceDigestResolver",
    "SourceSample",
    "decode_source_sample",
)
