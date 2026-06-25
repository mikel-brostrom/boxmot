from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from boxmot.utils import ROOT

from .constants import _PROPOSAL_API_KEY_ENV_BY_PROVIDER
from .paths import _is_relative_to


def _ensure_not_local_gepa_path(path: Path | None) -> None:
    if path is None:
        raise RuntimeError("Unable to determine the installed gepa package path")
    local_checkout = (ROOT / "gepa").resolve()
    resolved = path.resolve()
    if resolved == local_checkout or _is_relative_to(resolved, local_checkout):
        raise RuntimeError(
            "The resolved `gepa` package points at the local `./gepa` checkout. "
            "Install the published pip package instead."
        )


def _import_installed_gepa() -> Any:
    try:
        gepa_dist = distribution("gepa")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "`gepa` is not installed. Install the BoxMOT research extra with "
            "`uv sync --extra research` or `uv pip install '.[research]'`."
        ) from exc

    dist_root = Path(gepa_dist.locate_file("")).resolve()
    _ensure_not_local_gepa_path(dist_root)

    if str(dist_root) not in sys.path:
        sys.path.insert(0, str(dist_root))

    module = importlib.import_module("gepa")
    module_file = getattr(module, "__file__", None)
    if module_file:
        _ensure_not_local_gepa_path(Path(module_file))
    return module


def _load_gepa_litellm_factory() -> Callable[[str], Any] | None:
    """Resolve the published GEPA liteLLM factory when that layout is installed."""
    try:
        module = importlib.import_module("gepa.optimize_anything")
    except ImportError:
        return None

    factory = getattr(module, "make_litellm_lm", None)
    return factory if callable(factory) else None


def _resolve_proposal_api_key_env(model_name: str, configured_env: str | None) -> str | None:
    """Resolve the provider API-key environment variable for a proposal model."""
    if configured_env:
        return str(configured_env).strip() or None

    provider = str(model_name).split("/", 1)[0].strip().lower()
    return _PROPOSAL_API_KEY_ENV_BY_PROVIDER.get(provider)


def _prepare_proposal_model_env(model_name: str, model_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply proposal-model credential settings and return kwargs safe for LM construction."""
    sanitized_kwargs = dict(model_kwargs)
    api_key = sanitized_kwargs.pop("api_key", None)
    api_key_env = sanitized_kwargs.pop("api_key_env", None)

    if api_key is None:
        return sanitized_kwargs

    env_name = _resolve_proposal_api_key_env(model_name, api_key_env)
    if env_name is None:
        raise ValueError(
            f"Cannot infer an API-key environment variable for proposal model '{model_name}'. "
            "Pass --proposal-api-key-env ENV_NAME alongside --proposal-api-key."
        )

    os.environ[env_name] = str(api_key)
    return sanitized_kwargs


def _build_reflection_lm(model_name: str, model_kwargs: Mapping[str, Any]) -> Any:
    """Construct a GEPA-compatible reflection LM across published package layouts."""
    sanitized_kwargs = _prepare_proposal_model_env(model_name, model_kwargs)
    make_litellm_lm = _load_gepa_litellm_factory()
    if make_litellm_lm is not None:
        return make_litellm_lm(model_name)

    from gepa.lm import LM

    return LM(model_name, **sanitized_kwargs)


def _run_instruction_proposal_signature(
    signature_cls: Any,
    *,
    lm: Any,
    input_dict: Mapping[str, Any],
) -> dict[str, str]:
    """Run a GEPA instruction signature across published/newer API variants."""
    runner = getattr(signature_cls, "run_with_metadata", None)
    if callable(runner):
        result, _, _ = runner(lm=lm, input_dict=input_dict)
        return result

    runner = getattr(signature_cls, "run", None)
    if callable(runner):
        return runner(lm=lm, input_dict=input_dict)

    raise AttributeError(f"{signature_cls!r} does not expose a supported run method")

