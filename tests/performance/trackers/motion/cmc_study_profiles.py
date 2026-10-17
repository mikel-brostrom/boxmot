"""Reproducible ECC/SOF settings for the tracking accuracy/latency study.

The slow controls use the complete pre-optimization estimator and BaseCMC
sources from a pinned Git commit. Increasing budgets on the current estimator
alone would not restore its former preprocessing, masking, and refresh logic.
Only the historical estimator's BaseCMC import is rebound to its frozen base;
the production modules and global import table are never replaced.

Callers own OpenCV thread settings and random seeding. Seed each independent
sequence/profile identically and create a fresh estimator for every sequence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from boxmot.trackers.common.motion.cmc.ecc import ECC
from boxmot.trackers.common.motion.cmc.sof import SOF
from boxmot.utils import ROOT

HISTORICAL_REVISION = "975ab9427976c1303bd7bc7c95a37b2bfed37c30"
_SOURCE_DIRECTORY = Path("boxmot/trackers/common/motion/cmc")
_BASE_MODULE = "boxmot.trackers.common.motion.cmc.base"


class MotionEstimator(Protocol):
    """The shared interface of current and historical estimators."""

    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
        """Estimate motion from the preceding frame to this frame."""
        ...

    def reset(self) -> None:
        """Discard temporal state."""
        ...


@dataclass(frozen=True)
class StudyProfile:
    """An independent estimator factory with JSON-serializable provenance."""

    name: str
    factory: Callable[[], MotionEstimator | None]
    metadata: dict[str, Any]

    def create(self) -> MotionEstimator | None:
        """Create fresh sequence state; disabled CMC returns None."""
        return self.factory()


def _load_historical_classes(sources: dict[str, bytes]) -> dict[str, type]:
    """Compile exact historical kernels against their own BaseCMC definition."""
    base_namespace: dict[str, Any] = {"__name__": "cmc_study_historical_base"}
    exec(compile(sources["base"], "cmc_study_historical/base.py", "exec"), base_namespace)
    result: dict[str, type] = {}
    for method in ("ecc", "sof"):
        tree = ast.parse(sources[method], filename=f"cmc_study_historical/{method}.py")
        base_imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == _BASE_MODULE]
        if len(base_imports) != 1 or [(alias.name, alias.asname) for alias in base_imports[0].names] != [
            ("BaseCMC", None)
        ]:
            raise ValueError(f"Unexpected BaseCMC import in historical {method}; review the source before loading.")
        tree.body.remove(base_imports[0])
        namespace: dict[str, Any] = {
            "__name__": f"cmc_study_historical_{method}",
            "BaseCMC": base_namespace["BaseCMC"],
        }
        exec(compile(tree, f"cmc_study_historical/{method}.py", "exec"), namespace)
        result[method] = namespace[method.upper()]
    return result


def _balanced_sof() -> SOF:
    """Increase spatial detail and optical-flow effort with current semantics."""
    estimator = SOF(target_width=320)
    estimator.feature_params["maxCorners"] = 200
    estimator.lk_params.update(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    estimator.feature_refresh_interval = 1
    return estimator


def _settings(estimator: Any, method: str, historical: bool) -> dict[str, Any]:
    """Read the effective runtime settings rather than duplicating defaults."""
    result = {
        "scale": estimator.scale,
        "target_width": estimator.target_width,
        "grayscale": estimator.grayscale,
        "resize_before_grayscale": estimator.resize_before_grayscale,
    }
    if method == "ecc":
        result.update(
            warp_mode=int(estimator.warp_mode),
            motion_model="translation",
            termination_criteria=estimator.termination_criteria,
            min_correlation=estimator.min_correlation,
            gaussian_filter_size=1 if historical else 3,
            align=estimator.align,
        )
    else:
        result.update(
            motion_model="partial_affine_similarity",
            feature_params=estimator.feature_params,
            lk_params=estimator.lk_params,
            min_inliers=estimator.min_inliers,
            min_inlier_ratio=estimator.min_inlier_ratio,
            ransac_reproj_threshold=estimator.ransac_reproj_threshold,
            feature_refresh_interval=1 if historical else estimator.feature_refresh_interval,
            mask_tracked_points_on_current_frame=not historical,
            retain_tracked_points_when_redetection_fails=historical,
        )
    # Normalize tuples and NumPy-compatible scalar values before consumers save it.
    return json.loads(json.dumps(result))


def build_profiles(
    snapshot_dir: Path,
    *,
    repository_root: Path = ROOT,
    historical_revision: str = HISTORICAL_REVISION,
) -> dict[str, StudyProfile]:
    """Snapshot the measured sources and build seven independent CMC profiles.

    Requires the pinned historical commit in the local Git object database.
    Snapshots preserve original bytes for inspection; they are never used as an
    unverified source of executable code on subsequent invocations.
    """
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", f"{historical_revision}^{{commit}}"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_groups: dict[str, dict[str, bytes]] = {"historical": {}, "current": {}}
    provenance: dict[str, dict[str, Any]] = {"historical": {}, "current": {}}
    for method in ("base", "ecc", "sof"):
        source_path = _SOURCE_DIRECTORY / f"{method}.py"
        source_groups["historical"][method] = subprocess.run(
            ["git", "show", f"{revision}:{source_path.as_posix()}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        source_groups["current"][method] = (repository_root / source_path).read_bytes()
        for group, sources in source_groups.items():
            output = snapshot_dir / group / f"{method}.py"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(sources[method])
            provenance[group][method] = {
                "repository_path": source_path.as_posix(),
                "snapshot_path": output.relative_to(snapshot_dir).as_posix(),
                "sha256": hashlib.sha256(sources[method]).hexdigest(),
            }

    historical_classes = _load_historical_classes(source_groups["historical"])
    factories: dict[str, Callable[[], MotionEstimator | None]] = {
        "off": lambda: None,
        "ecc-fast": ECC,
        "ecc-balanced": lambda: ECC(target_width=192, max_iter=30, eps=1e-4),
        "ecc-slow": historical_classes["ecc"],
        "sof-fast": SOF,
        "sof-balanced": _balanced_sof,
        "sof-slow": historical_classes["sof"],
    }
    profiles = {}
    for name, factory in factories.items():
        method = name.split("-")[0]
        historical = name.endswith("-slow")
        group = "historical" if historical else "current"
        metadata: dict[str, Any] = {
            "name": name,
            "method": method,
            "tier": "off" if method == "off" else name.rsplit("-", 1)[1],
            "implementation": "disabled" if method == "off" else group,
        }
        if method != "off":
            metadata.update(
                historical_revision=revision if historical else None,
                sources={key: provenance[group][key] for key in ("base", method)},
                settings=_settings(factory(), method, historical),
            )
        profiles[name] = StudyProfile(name, factory, metadata)
    (snapshot_dir / "profiles.json").write_text(
        json.dumps({name: profile.metadata for name, profile in profiles.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return profiles
