"""Describe saved CMC transforms without rerunning estimators or using motion GT.

Run from the repository root after the accuracy study completes its warp stage::

    uv run --no-sync python -m tests.performance.trackers.motion.cmc_study_diagnostics \
        runs/cmc-study/mot17-ecc-sof

These diagnostics describe estimator outputs; they cannot identify the correct
camera transform. All statistics exclude the first frame of each sequence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _image_sizes(build: Path, sequences: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Read native image dimensions from the measured build's sample metadata."""
    table = pq.read_table(build / "samples", columns=["sequence_id", "width", "height"])
    sizes: dict[str, tuple[int, int]] = {}
    for row in table.to_pylist():
        name = row["sequence_id"]
        if name not in sequences:
            continue
        size = (int(row["width"]), int(row["height"]))
        if name in sizes and sizes[name] != size:
            raise ValueError(f"Sequence {name} changes dimensions; use frame-specific diagnostics.")
        sizes[name] = size
    if set(sizes) != set(sequences):
        raise ValueError("Published samples do not contain every study sequence.")
    return sizes


def _percentiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return descriptive transition quantiles without uncertainty claims."""
    return {
        f"{prefix}_{name}": float(value)
        for name, value in zip(("p50", "p95", "max"), np.percentile(values, (50, 95, 100)))
    }


def _grid(width: int, height: int) -> np.ndarray:
    """Use pixel centers at four image corners and the image center."""
    return np.asarray(
        (
            (0, 0, 1),
            (width - 1, 0, 1),
            (0, height - 1, 1),
            (width - 1, height - 1, 1),
            ((width - 1) / 2, (height - 1) / 2, 1),
        ),
        dtype=np.float64,
    )


def diagnose(output: Path) -> dict[str, Any]:
    """Export small CSV/JSON descriptions from already saved transform arrays."""
    output = Path(output)
    study = json.loads((output / "study.json").read_text())
    profiles = json.loads((output / "cmc_profiles.json").read_text())
    sequences = study["sequences"]
    sizes = _image_sizes(Path(study["build"]), sequences)
    rows = []
    for sequence, count in sequences.items():
        if count < 2:
            raise ValueError(f"Sequence {sequence} has no motion transitions.")
        width, height = sizes[sequence]
        grid = _grid(width, height)
        arrays = {}
        for profile in profiles:
            values = np.load(output / "warps" / profile / f"{sequence}.npy", allow_pickle=False)
            if values.shape != (count, 2, 3) or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved warps for {sequence}/{profile}.")
            expected = profiles[profile]["per_sequence"][sequence]["warp_sha256"]
            if hashlib.sha256(values.tobytes()).hexdigest() != expected:
                raise ValueError(f"Saved warp hash mismatch for {sequence}/{profile}.")
            arrays[profile] = values[1:].astype(np.float64)
        transformed = {name: np.einsum("nij,pj->npi", values, grid) for name, values in arrays.items()}
        for profile, values in arrays.items():
            method = profiles[profile]["metadata"]["method"]
            # Reduce spatial variation within a transition before quantiles
            # across transitions; keep this definition identical for deltas.
            motion = np.median(np.linalg.norm(transformed[profile] - grid[None, :, :2], axis=2), axis=1)
            row = {
                "sequence": sequence,
                "profile": profile,
                "method": method,
                "width": width,
                "height": height,
                "transitions": count - 1,
                "identity_fraction": float(np.isclose(values, np.eye(2, 3), rtol=0, atol=1e-6).all(axis=(1, 2)).mean()),
                **_percentiles(motion, "grid_motion_px"),
            }
            reference = f"{method}-fast"
            if reference in transformed:
                disagreement = np.median(np.linalg.norm(transformed[profile] - transformed[reference], axis=2), axis=1)
                row.update(reference=reference, **_percentiles(disagreement, "fast_disagreement_px"))
            if method == "sof":
                linear = values[:, :, :2]
                scale = np.sqrt(np.abs(np.linalg.det(linear)))
                # The polar rotation stays well-defined after slight unequal
                # x/y resize rounding in the restored similarity transform.
                angle = np.degrees(np.arctan2(linear[:, 1, 0] - linear[:, 0, 1], linear[:, 0, 0] + linear[:, 1, 1]))
                row.update(
                    **_percentiles(np.abs(scale - 1) * 100, "abs_scale_change_percent"),
                    **_percentiles(np.abs(angle), "abs_rotation_deg"),
                    scale_p01=float(np.percentile(scale, 1)),
                    scale_p99=float(np.percentile(scale, 99)),
                    rotation_deg_p01=float(np.percentile(angle, 1)),
                    rotation_deg_p99=float(np.percentile(angle, 99)),
                )
            rows.append(row)
    result = {
        "build": study["build"],
        "manifest_sha256": study["manifest_sha256"],
        "definitions": {
            "scope": "Saved transforms only. No estimator rerun, tracking rerun or camera-motion ground truth.",
            "transitions": "Frame zero excluded; reinitialization and fallback identities retained.",
            "grid": "Native pixel-center coordinates at the four image corners and center.",
            "grid_motion_px": "Median displacement over five grid points per transition, then temporal quantiles.",
            "fast_disagreement_px": "Same grid reduction for candidate minus fast warp, then temporal quantiles.",
            "identity_fraction": "Fraction within absolute tolerance 1e-6 of identity; relative tolerance is zero.",
            "scale": "Square root of absolute determinant of restored 2x2 linear transform.",
            "rotation_deg": "Polar rotation angle of restored 2x2 linear transform, in degrees.",
        },
        "cautions": [
            "Identity outputs include static motion, rejected fits and reinitialization; this is not a failure rate.",
            "Disagreement and large motion do not establish which estimator is correct without motion ground truth.",
            "Scale/rotation variation may represent real camera motion or estimation error.",
            "ECC fits translation only, whereas SOF fits translation, rotation and scale.",
        ],
        "rows": rows,
    }
    (output / "warp_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "warp_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    """Describe every completed profile without modifying benchmark inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.output)
    print(f"Saved {len(result['rows'])} sequence/profile diagnostics in {args.output}.")


if __name__ == "__main__":
    main()
