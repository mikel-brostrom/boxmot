from __future__ import annotations

import os
import signal
import subprocess

from boxmot.configs.benchmark import (
    apply_benchmark_config,
    resolve_required_reid_model,
    resolve_required_yolo_model,
)
from boxmot.utils import ROOT
from boxmot.utils.misc import resolve_model_path
from boxmot.utils.rich.reporters.research import ResearchWorkflowReporter

from . import benchmarks as _benchmarks
from . import proposal as _proposal
from .benchmarks import _discover_sequences, _select_examples, _split_examples
from .candidates import (
    _build_reflection_prompt_templates,
    _candidate_change_summary,
    _candidate_import_modules,
    _inject_validation_feedback,
    _make_checked_candidate_proposer,
    _normalize_editable_files,
    _normalize_proposed_text,
    _proposal_log_summary,
    _ProposalLogText,
    _raw_text,
    _read_candidate,
    _validate_candidate_content,
    _validate_candidate_keys,
)
from .constants import DEFAULT_PROPOSAL_MODEL, MOT_METRIC_GLOSSARY, RESEARCH_METRICS
from .metrics import _aggregate_metrics, _metric_delta, _nested_metric_delta
from .models import RegressionPenalties, ResearchConfig, ResearchResult
from .paths import (
    _count_changed_lines,
    _is_relative_to,
    _json_default,
    _parse_last_json_line,
    _slugify,
    _terminate_subprocess_tree,
    _workspace_copy_ignore,
)
from .proposal import (
    _ensure_not_local_gepa_path,
    _import_installed_gepa,
    _load_gepa_litellm_factory,
    _prepare_proposal_model_env,
    _resolve_proposal_api_key_env,
    _run_instruction_proposal_signature,
)
from .runner import TrackerResearcher, main, run_research


def _resolve_benchmark_runtime(*args, **kwargs):
    old = (
        _benchmarks.apply_benchmark_config,
        _benchmarks.resolve_required_yolo_model,
        _benchmarks.resolve_required_reid_model,
        _benchmarks.resolve_model_path,
    )
    _benchmarks.apply_benchmark_config = apply_benchmark_config
    _benchmarks.resolve_required_yolo_model = resolve_required_yolo_model
    _benchmarks.resolve_required_reid_model = resolve_required_reid_model
    _benchmarks.resolve_model_path = resolve_model_path
    try:
        return _benchmarks._resolve_benchmark_runtime(*args, **kwargs)
    finally:
        (
            _benchmarks.apply_benchmark_config,
            _benchmarks.resolve_required_yolo_model,
            _benchmarks.resolve_required_reid_model,
            _benchmarks.resolve_model_path,
        ) = old


def _build_reflection_lm(*args, **kwargs):
    old = _proposal._load_gepa_litellm_factory
    _proposal._load_gepa_litellm_factory = _load_gepa_litellm_factory
    try:
        return _proposal._build_reflection_lm(*args, **kwargs)
    finally:
        _proposal._load_gepa_litellm_factory = old


__all__ = [
    "DEFAULT_PROPOSAL_MODEL",
    "ROOT",
    "RESEARCH_METRICS",
    "MOT_METRIC_GLOSSARY",
    "RegressionPenalties",
    "ResearchConfig",
    "ResearchResult",
    "ResearchWorkflowReporter",
    "TrackerResearcher",
    "_ProposalLogText",
    "_aggregate_metrics",
    "_build_reflection_lm",
    "_build_reflection_prompt_templates",
    "_candidate_change_summary",
    "_candidate_import_modules",
    "_count_changed_lines",
    "_discover_sequences",
    "_ensure_not_local_gepa_path",
    "_import_installed_gepa",
    "_inject_validation_feedback",
    "_is_relative_to",
    "_json_default",
    "_load_gepa_litellm_factory",
    "_make_checked_candidate_proposer",
    "_metric_delta",
    "_nested_metric_delta",
    "_normalize_editable_files",
    "_normalize_proposed_text",
    "_parse_last_json_line",
    "_prepare_proposal_model_env",
    "_proposal_log_summary",
    "_raw_text",
    "_read_candidate",
    "_resolve_benchmark_runtime",
    "_resolve_proposal_api_key_env",
    "_run_instruction_proposal_signature",
    "_select_examples",
    "_slugify",
    "_split_examples",
    "_terminate_subprocess_tree",
    "_validate_candidate_content",
    "_validate_candidate_keys",
    "_workspace_copy_ignore",
    "apply_benchmark_config",
    "main",
    "os",
    "resolve_model_path",
    "resolve_required_reid_model",
    "resolve_required_yolo_model",
    "run_research",
    "signal",
    "subprocess",
]
