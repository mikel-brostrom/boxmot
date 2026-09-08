"""Select dataset frames on a time grid and align their evaluation annotations."""

from __future__ import annotations

import configparser
import io
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from boxmot.engine.materialization.catalog import SourceCatalog
    from boxmot.engine.materialization.source import SourceSample


def validate_dataset_fps(fps: float) -> float:
    """Require a finite, positive dataset sampling rate."""

    if isinstance(fps, bool):
        raise ValueError("Dataset FPS must be a finite, positive number.")
    try:
        value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dataset FPS must be a finite, positive number.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Dataset FPS must be a finite, positive number.")
    return value


def select_sequence_frames(
    timestamps: Sequence[float | None], *, fps: float, sequence_id: str
) -> tuple[int, ...]:
    """Keep the first frame in each occupied target-FPS time bin.

    Bins start at the first capture timestamp. Empty bins never duplicate frames,
    and bin boundaries stay anchored to that timestamp so fractional rate ratios
    do not accumulate stride-rounding error.
    """

    fps = validate_dataset_fps(fps)
    if not timestamps:
        return ()
    if any(value is None or not math.isfinite(value) for value in timestamps):
        raise ValueError(
            f"Dataset FPS sampling requires valid timestamps.csv or a positive, finite "
            f"seqinfo.ini frameRate for sequence {sequence_id!r}."
        )
    origin = float(timestamps[0])
    selected = []
    last_bin = -1
    previous = None
    for index, value in enumerate(timestamps):
        timestamp = float(value)
        if previous is not None and timestamp <= previous:
            raise ValueError(f"Dataset FPS sampling requires increasing timestamps for sequence {sequence_id!r}.")
        position = (timestamp - origin) * fps
        if not math.isfinite(position):
            raise ValueError(f"Dataset FPS sampling exceeds the supported time range for sequence {sequence_id!r}.")
        nearest = round(position)
        # Epoch-based capture times lose sub-microsecond precision when stored
        # as floats. Account for that representation error as well as nominal
        # division error, without snapping a material part of a source interval.
        tolerance = max(1e-9, (math.ulp(timestamp) + math.ulp(origin)) * fps + math.ulp(position))
        tolerance = min(tolerance, 0.25)
        if previous is not None:
            tolerance = min(tolerance, (timestamp - previous) * fps / 4)
        if math.isclose(position, nearest, rel_tol=0.0, abs_tol=tolerance):
            position = nearest
        previous = timestamp
        bin_index = math.floor(position)
        if bin_index > last_bin:
            selected.append(index)
            last_bin = bin_index
    return tuple(selected)


def _write_text(path: Path, text: str) -> None:
    """Atomically publish generated text, preserving an already matching file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sample_annotations(path: Path, frame_map: Mapping[int, int]) -> str:
    """Filter MOT or VisDrone rows and replace only their first frame column."""

    selected = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        comma_separated = "," in line
        columns = line.split(",", 1) if comma_separated else line.split(maxsplit=1)
        try:
            frame_value = float(columns[0])
        except ValueError as exc:
            raise ValueError(f"Ground truth {path} line {line_number} has an invalid frame ID.") from exc
        if not math.isfinite(frame_value) or not frame_value.is_integer() or frame_value <= 0 or len(columns) != 2:
            raise ValueError(f"Ground truth {path} line {line_number} requires a positive integer frame ID and data.")
        target = frame_map.get(int(frame_value))
        if target is not None:
            delimiter = "," if comma_separated else " "
            selected.append(f"{target}{delimiter}{columns[1]}\n")
    return "".join(selected)


def _write_sequence_info(source: Path, target: Path, samples: Sequence[SourceSample], fps: float) -> None:
    """Publish the sampled sequence length and its nominal sampling rate."""

    parser = configparser.ConfigParser()
    if source.is_file():
        parser.read(source)
    if not parser.has_section("Sequence"):
        parser.add_section("Sequence")
    try:
        native_fps = (
            fps
            if (source.parent / "timestamps.csv").is_file()
            else parser.getfloat("Sequence", "frameRate", fallback=fps)
        )
    except ValueError:
        native_fps = fps
    if not math.isfinite(native_fps) or native_fps <= 0.0:
        native_fps = fps
    parser["Sequence"].update(
        {
            "name": samples[0].sequence_id,
            "imDir": "img1",
            "frameRate": f"{min(fps, native_fps):g}",
            "seqLength": str(len(samples)),
            "imHeight": str(samples[0].image_size[0]),
            "imWidth": str(samples[0].image_size[1]),
            "imExt": Path(unquote(urlparse(samples[0].source_uri).path)).suffix,
        }
    )
    text = io.StringIO()
    parser.write(text)
    _write_text(target, text.getvalue())


def materialize_fps_ground_truth(
    config: Mapping[str, Any],
    catalog: SourceCatalog,
    output_root: str | Path,
    *,
    data_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Publish sampled frames and annotations beneath an evaluation-owned root.

    The source catalog supplies the exact original-to-sampled frame mapping.
    Images are symlinked to their original bytes; annotation columns beyond the
    one-based frame ID are preserved for AABB, OBB, and flat annotation layouts.
    Capture timestamps remain unchanged in the generated sequence sidecars.
    """

    from boxmot.engine.materialization.catalog import (
        resolve_dataset_annotation_root,
        resolve_dataset_split_root,
    )

    fps = validate_dataset_fps(catalog.metadata["fps"])
    frame_sampling = catalog.metadata["frame_sampling"]
    split = str(catalog.metadata["split"])
    source_root = resolve_dataset_split_root(config, split, data_root)
    annotation_root = resolve_dataset_annotation_root(config, split, data_root)
    flat_annotations = config["splits"][split].get("annotations") is not None or config["layout"] == "visdrone"
    destination = Path(output_root).expanduser().resolve()
    split_root = destination / "sequences"
    target_annotations = destination / "annotations" if flat_annotations else split_root
    sequences: dict[str, list[SourceSample]] = defaultdict(list)
    for sample in catalog.samples:
        sequences[sample.sequence_id].append(sample)
    for sequence_id, samples in sequences.items():
        original_ids = frame_sampling[sequence_id]
        if len(original_ids) != len(samples):
            raise ValueError(f"FPS frame mapping does not match catalog sequence {sequence_id!r}.")
        frame_map = {
            int(original): sample.frame_index + 1 for original, sample in zip(original_ids, samples, strict=True)
        }
        sequence_root = split_root / sequence_id
        image_root = sequence_root / "img1"
        image_root.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            source = Path(unquote(urlparse(sample.source_uri).path))
            image = image_root / f"{sample.frame_index + 1:06d}{source.suffix}"
            try:
                image.symlink_to(source)
            except FileExistsError:
                if not image.is_symlink() or image.readlink() != source:
                    raise ValueError(f"Sampled image path already exists with different content: {image}.")
        _write_sequence_info(source_root / sequence_id / "seqinfo.ini", sequence_root / "seqinfo.ini", samples, fps)
        _write_text(
            sequence_root / "timestamps.csv",
            "frame_id,timestamp_s\n"
            + "".join(f"{sample.frame_index + 1},{sample.timestamp_s!r}\n" for sample in samples),
        )
        if flat_annotations:
            source = annotation_root / f"{sequence_id}.txt"
            if not source.is_file():
                raise FileNotFoundError(f"Dataset annotation for sequence {sequence_id!r} does not exist: {source}.")
            _write_text(target_annotations / source.name, _sample_annotations(source, frame_map))
        else:
            for source in sorted((annotation_root / sequence_id / "gt").glob("*")):
                if source.is_file() and source.name.startswith("gt") and source.suffix.lower() in {".txt", ".csv"}:
                    _write_text(sequence_root / "gt" / source.name, _sample_annotations(source, frame_map))
    return split_root, target_annotations


__all__ = ("materialize_fps_ground_truth", "select_sequence_frames", "validate_dataset_fps")
