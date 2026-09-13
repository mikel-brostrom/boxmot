"""Generate static model-name Literal aliases from catalogs and Ultralytics.

Run ``uv run --no-sync python -m tools.generate_model_names`` after changing a
detector/ReID profile, tracker manifest, or locked Ultralytics release. Install
the yolo extra before generating. ``--check`` reports stale files without
modifying them. Generated modules need only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from boxmot.detectors._ultralytics_models import ultralytics_detector_names, ultralytics_inventory_version
from boxmot.detectors.config import load_detector_config
from boxmot.reid.config import load_reid_config
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST
from boxmot.utils.config import ConfigurationError, index_config_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
_REGENERATE = "uv run --no-sync python -m tools.generate_model_names"


def detector_names(config_dir: Path) -> tuple[str, ...]:
    """Return callable profile selectors, requiring ambiguous checkpoint names."""
    names = []
    for config_id, path in index_config_ids(config_dir, "detector").items():
        checkpoints = load_detector_config(path)["checkpoints"]
        if len(checkpoints) == 1:
            names.append(config_id)
            continue
        for checkpoint in checkpoints:
            if not checkpoint or checkpoint != checkpoint.strip() or "/" in checkpoint:
                raise ConfigurationError(f'Detector config "{path}" has an unselectable checkpoint {checkpoint!r}.')
            names.append(f"{config_id}/{checkpoint}")
    return tuple(sorted(names))


def reid_names(config_dir: Path) -> tuple[str, ...]:
    """Return runtime profile IDs, independent of unconfigured ReID backbones."""
    indexed = index_config_ids(config_dir, "ReID")
    for path in indexed.values():
        load_reid_config(path)
    return tuple(sorted(indexed))


def render_alias(alias: str, names: Iterable[str], *, provenance: str | None = None) -> str:
    """Render deterministic Python 3.10-compatible source without dynamic lookups."""
    ordered_names = tuple(sorted(set(names)))
    if not ordered_names:
        raise ValueError(f"{alias} requires at least one catalog name.")
    entries = "".join(f"    {json.dumps(name)},\n" for name in ordered_names)
    provenance_line = f"\n{provenance}\n" if provenance else ""
    return (
        f'"""Generated model names for autocomplete. Regenerate with:\n\n{_REGENERATE}\n{provenance_line}"""\n\n'
        "from typing import Literal, TypeAlias\n\n"
        f"{alias}: TypeAlias = Literal[\n{entries}]\n\n"
        f'__all__ = ("{alias}",)\n'
    )


def generated_sources(root: Path = REPO_ROOT, *, trackers: Iterable[str] | None = None) -> dict[Path, str]:
    """Build expected sources without writing catalogs, artifacts, or outputs."""
    configs = root / "boxmot" / "configs"
    detector_selectors = (*detector_names(configs / "detectors"), *ultralytics_detector_names())
    return {
        root / "boxmot/detectors/_model_names.py": render_alias(
            "DetectorName",
            detector_selectors,
            provenance=f"Includes box-producing assets from Ultralytics {ultralytics_inventory_version()}.",
        ),
        root / "boxmot/reid/_model_names.py": render_alias("ReIDName", reid_names(configs / "reid")),
        root / "boxmot/trackers/common/_model_names.py": render_alias(
            "TrackerName", _TRACKER_MANIFEST if trackers is None else trackers
        ),
    }


def generate(root: Path = REPO_ROOT, *, check: bool = False) -> tuple[Path, ...]:
    """Write changed aliases, or return stale paths without writing in check mode."""
    changed = []
    for path, source in generated_sources(root).items():
        expected = source.encode("utf-8")
        if path.is_file() and path.read_bytes() == expected:
            continue
        changed.append(path)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
    return tuple(changed)


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate autocomplete declarations or check their committed freshness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated name aliases need updating.")
    args = parser.parse_args(argv)
    changed = generate(check=args.check)
    if args.check and changed:
        for path in changed:
            print(f"Stale model names: {path.relative_to(REPO_ROOT)}")
        print(f"Regenerate with: {_REGENERATE}")
        return 1
    for path in changed:
        print(f"Generated {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
