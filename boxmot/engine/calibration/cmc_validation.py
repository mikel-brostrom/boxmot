"""Validate persisted camera evidence for calibration and tracker replay."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from boxmot.engine.calibration.cmc import CMC_ANALYSIS_SCHEMA, cmc_build_signature, cmc_image_signature
from boxmot.engine.config.cmc import SequenceCMCPolicy
from boxmot.trackers.common.motion.cmc.base import BaseCMC


def validate_cmc_artifacts(
    report_bytes: bytes,
    warps: bytes,
    expected_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Require complete evidence for the exact source, frame order and estimator.

    Both calibration masks and runtime tracker masks have saved transforms.
    Validate both before either consumer can reuse a materialized result.
    Source signatures are checked on every load; no mutable object is marked
    permanently valid.
    """
    try:
        report = json.loads(report_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Saved CMC setup analysis is invalid.") from exc
    inputs = dict(expected_inputs)
    selection = inputs.pop("sequences")
    build = inputs.pop("build")
    expected = {
        **inputs,
        "schema": CMC_ANALYSIS_SCHEMA,
        "status": "complete",
        "build_path": build,
        "total_frames": sum(count for _name, count in selection),
        "warps_sha256": hashlib.sha256(warps).hexdigest(),
    }
    if (
        not isinstance(report, dict)
        or report.get("ground_truth_used") is not False
        or any(report.get(name) != value for name, value in expected.items())
    ):
        raise ValueError("Saved CMC setup analysis does not match the current inputs or warp artifact.")
    sequences = report.get("sequences")
    if not isinstance(sequences, Mapping) or set(sequences) != {name for name, _count in selection}:
        raise ValueError("Saved CMC setup analysis does not match the selected sequences.")
    decisions = []
    for name, count in selection:
        stats = sequences[name]
        if not isinstance(stats, Mapping) or stats.get("frames") != count or type(stats.get("enabled")) is not bool:
            raise ValueError("Saved CMC setup analysis has invalid sequence decisions or frame counts.")
        decisions.append((name, stats["enabled"]))
    if report.get("policy") != SequenceCMCPolicy(sequences=tuple(decisions)).to_dict():
        raise ValueError("Saved CMC setup policy disagrees with its sequence analysis.")
    if report.get("source_signature") != cmc_build_signature(Path(build)):
        raise ValueError("Saved CMC source build changed; prepare camera analysis again.")
    arrays = {}
    try:
        with np.load(io.BytesIO(warps), allow_pickle=False) as archive:
            for name, count in selection:
                stats = sequences[name]
                keys = stats["archive_keys"]
                indices = archive[keys["frame_indices"]]
                sample_ids = stats["sample_ids"]
                references = stats["image_refs"]
                if (
                    not isinstance(sample_ids, list)
                    or len(sample_ids) != count
                    or any(not isinstance(value, str) or not value for value in sample_ids)
                    or len(set(sample_ids)) != count
                    or not isinstance(references, list)
                    or len(references) != count
                    or any(not isinstance(value, str) or not value for value in references)
                    or not isinstance(stats["source_image_root"], str)
                    or not stats["source_image_root"]
                    or indices.dtype != np.int64
                    or indices.shape != (count,)
                    or not np.array_equal(indices, np.arange(count))
                ):
                    raise ValueError("Saved CMC sample identities differ from the selected delivered frame order.")
                if cmc_image_signature(tuple(references), stats["source_image_root"]) != stats["image_signature"]:
                    raise ValueError("Saved CMC source images changed; prepare camera analysis again.")
                arrays[keys["frame_indices"]] = indices
                for kind in ("warps", "runtime_warps"):
                    transforms = archive[keys[kind]]
                    if (
                        count < 1
                        or transforms.dtype != np.float32
                        or transforms.shape != (count, 2, 3)
                        or not np.isfinite(transforms).all()
                        or not np.array_equal(transforms[0], np.eye(2, 3, dtype=np.float32))
                        or any(not BaseCMC.is_valid_transform(matrix) for matrix in transforms)
                    ):
                        raise ValueError("Saved CMC transforms are invalid or differ from the selected frame order.")
                    transforms.flags.writeable = False
                    arrays[keys[kind]] = transforms
    except (OSError, EOFError, AttributeError, KeyError, TypeError, ValueError, BadZipFile) as exc:
        raise ValueError(f"Saved CMC replay evidence is invalid: {exc}") from exc
    return report, arrays
