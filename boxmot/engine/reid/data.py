"""Resolve ReID training data inputs from dataset names or YAML specs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from boxmot.utils import logger as LOGGER

_YAML_SUFFIXES = {".yaml", ".yml"}


def _split_data_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_values = (value,) if isinstance(value, (str, Path)) else tuple(value)
    items: list[str] = []
    for raw in raw_values:
        if raw is None:
            continue
        for part in str(raw).split(","):
            item = part.strip()
            if item:
                items.append(item)
    return tuple(items)


def _resolve_path(raw: Any, *, base_dir: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _is_yaml_spec(value: str) -> bool:
    path = Path(value).expanduser()
    return path.suffix.lower() in _YAML_SUFFIXES


def _run_download_script(config: dict[str, Any], config_path: Path, root: Path) -> None:
    script = config.get("download")
    if not script:
        return

    yaml_context = dict(config)
    yaml_context["path"] = str(root)
    LOGGER.info(f"Running ReID dataset download script from {config_path}")
    old_cwd = Path.cwd()
    os.chdir(config_path.parent)
    try:
        exec(
            str(script),
            {
                "Path": Path,
                "__file__": str(config_path),
                "__name__": "__boxmot_reid_data_download__",
                "yaml": yaml_context,
            },
        )
    finally:
        os.chdir(old_cwd)


def _root_needs_download(root: Path) -> bool:
    if not root.exists():
        return True
    if not root.is_dir():
        return False
    try:
        next(root.iterdir())
    except StopIteration:
        return True
    return False


def _load_yaml_spec(value: str) -> dict[str, Any]:
    config_path = _resolve_path(value, base_dir=Path.cwd())
    if not config_path.exists():
        raise FileNotFoundError(f"ReID data config not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"ReID data config must be a file: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"ReID data config must contain a YAML mapping: {config_path}")

    root = _resolve_path(config.get("path", config_path.parent), base_dir=config_path.parent)
    if _root_needs_download(root):
        _run_download_script(config, config_path, root)

    name = str(
        config.get("dataset")
        or config.get("name")
        or config.get("id")
        or config_path.stem
    ).strip()
    if not name:
        raise ValueError(f"ReID data config must define a dataset name: {config_path}")

    spec: dict[str, Any] = {
        "name": name,
        "root": str(root),
        "config": str(config_path),
    }
    for key in ("train", "val", "query", "gallery"):
        if key in config and config[key] is not None:
            spec[key] = str(config[key])
    return spec


def _name_spec(name: str, root: str | Path) -> dict[str, Any]:
    return {"name": str(name), "root": str(Path(root).expanduser().resolve())}


def _common_root(specs: tuple[dict[str, Any], ...]) -> str | None:
    roots = [str(Path(spec["root"]).expanduser().resolve()) for spec in specs if spec.get("root")]
    if not roots:
        return None
    try:
        return os.path.commonpath(roots)
    except ValueError:
        return roots[0]


def _mark_data_explicit(args: Any, *, include_data_dir: bool) -> None:
    explicit = set(getattr(args, "train_explicit_keys", ()))
    explicit.update({"data", "dataset", "data_specs"})
    if include_data_dir:
        explicit.add("data_dir")
    args.train_explicit_keys = tuple(sorted(explicit))


def resolve_reid_train_data(args: Any) -> Any:
    """Apply Ultralytics-style ``data=[...]`` inputs to a train namespace.

    ``args.data`` may contain dataset names, YAML paths, comma-separated values,
    or a mix of those. Plain names use the existing shared ``data_dir`` path.
    YAML specs can point each dataset at a different root.
    """

    items = _split_data_values(getattr(args, "data", None))
    if not items:
        return args

    base_data_dir = getattr(args, "data_dir", None)
    names: list[str] = []
    specs: list[dict[str, Any]] = []
    saw_yaml = False

    for item in items:
        if _is_yaml_spec(item):
            spec = _load_yaml_spec(item)
            saw_yaml = True
        else:
            spec = _name_spec(item, base_data_dir) if base_data_dir else None

        names.append(spec["name"] if spec is not None else item)
        if spec is not None:
            specs.append(spec)

    if saw_yaml and len(specs) != len(names):
        raise ValueError("Dataset names mixed with YAML --data entries require --data-dir for the name-only entries")

    args.data = items
    args.dataset = ",".join(names)
    args.data_specs = tuple(specs)
    if specs and not base_data_dir:
        common = _common_root(tuple(specs))
        if common:
            args.data_dir = common

    _mark_data_explicit(args, include_data_dir=bool(getattr(args, "data_dir", None)))
    return args
