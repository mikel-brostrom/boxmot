from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from boxmot.utils import ROOT
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.rich.reporters.research import ResearchWorkflowReporter
from boxmot.utils.rich.workflow.pipeline import PipelineTracker

from .benchmarks import (
    _discover_sequences,
    _resolve_benchmark_runtime,
    _select_examples,
)
from .candidates import (
    _build_reflection_prompt_templates,
    _candidate_change_summary,
    _candidate_import_modules,
    _make_checked_candidate_proposer,
    _normalize_editable_files,
    _normalize_proposed_text,
    _proposal_log_summary,
    _ProposalLogText,
    _read_candidate,
    _validate_candidate_content,
    _validate_candidate_keys,
)
from .constants import (
    _EVAL_SNIPPET,
    _PREFLIGHT_SNIPPET,
    _RESEARCH_ROOT,
    MOT_METRIC_GLOSSARY,
    RESEARCH_EXTRA,
    RESEARCH_METRICS,
)
from .metrics import _metric_delta, _nested_metric_delta
from .models import ResearchConfig, ResearchResult
from .paths import _json_default, _parse_last_json_line, _slugify, _terminate_subprocess_tree, _workspace_copy_ignore
from .proposal import _build_reflection_lm, _import_installed_gepa, _run_instruction_proposal_signature


class TrackerResearcher:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.cache_project_dir = config.project.resolve()
        run_name = f"{_slugify(config.tracker)}_{_slugify(config.benchmark)}"
        self.run_dir = (config.project / "research" / run_name).resolve()
        self.boxmot_project_dir = self.run_dir / "boxmot_runs"
        self.gepa_run_dir = self.run_dir / "gepa"
        self.workspace_dir: Path | None = None

        (
            self.source_root,
            self.benchmark_id,
            self.detector_path,
            self.reid_path,
            self.benchmark_cfg,
        ) = _resolve_benchmark_runtime(
            config.benchmark,
            source=config.source,
            detector=config.detector,
            reid=config.reid,
        )

        self.editable_files = _normalize_editable_files(config.tracker, config.editable_files)
        self.seed_candidate = _read_candidate(self.editable_files)

        all_examples = _discover_sequences(self.source_root)
        self.all_examples = _select_examples(
            all_examples,
            train_sequences=config.train_sequences,
            val_sequences=config.val_sequences,
        )
        self.selected_sequences = tuple(example["sequence"] for example in self.all_examples)
        self.train_examples = [dict(example) for example in self.all_examples]
        self.val_examples = [dict(example) for example in self.all_examples]
        self.penalties = config.penalties
        self.baseline_summary: dict[str, int | float] | None = None
        self.baseline_summary_label: str = ""
        self.baseline_per_sequence_metrics: dict[str, dict[str, int | float]] = {}
        self.baseline_per_class_metrics: dict[str, dict[str, int | float]] = {}

    def _ensure_dependencies(self) -> None:
        RequirementsChecker().sync_extra(RESEARCH_EXTRA, verbose=False)
        _import_installed_gepa()

    def _reset_gepa_run_dir(self) -> None:
        if self.gepa_run_dir.exists():
            LOGGER.info(f"Resetting stale GEPA state in {self.gepa_run_dir}")
            shutil.rmtree(self.gepa_run_dir, ignore_errors=True)
        self.gepa_run_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_workspace(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.boxmot_project_dir.mkdir(parents=True, exist_ok=True)
        self._reset_gepa_run_dir()

        workspace = Path(
            tempfile.mkdtemp(prefix="workspace_", dir=str(self.run_dir))
        )
        shutil.copytree(ROOT, workspace, dirs_exist_ok=True, ignore=_workspace_copy_ignore)
        self.workspace_dir = workspace
        return workspace

    def _write_candidate_to_workspace(self, candidate: Mapping[str, str]) -> None:
        if self.workspace_dir is None:
            raise RuntimeError("Workspace has not been prepared")
        for file_path, content in candidate.items():
            dst = self.workspace_dir / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")

    def _subset_source_root(self, sequence_names: Sequence[str]) -> Path:
        if self.workspace_dir is None:
            raise RuntimeError("Workspace has not been prepared")

        subset_key = "__".join(sorted(sequence_names))
        container = self.workspace_dir / _RESEARCH_ROOT / "subsets" / _slugify(subset_key)
        data_root = container / "data"
        ann_root = container / "annotations"

        if data_root.exists():
            return data_root

        data_root.mkdir(parents=True, exist_ok=True)
        source_annotations = self.source_root.parent / "annotations"

        by_name = {example["sequence"]: Path(example["sequence_dir"]) for example in self.all_examples}
        for sequence_name in sequence_names:
            seq_dir = by_name[sequence_name]
            target = data_root / sequence_name
            if not target.exists():
                try:
                    target.symlink_to(seq_dir, target_is_directory=True)
                except OSError:
                    shutil.copytree(seq_dir, target, dirs_exist_ok=True)

            if source_annotations.exists():
                ann_file = source_annotations / f"{sequence_name}.txt"
                if ann_file.exists():
                    ann_root.mkdir(parents=True, exist_ok=True)
                    ann_target = ann_root / ann_file.name
                    if not ann_target.exists():
                        try:
                            ann_target.symlink_to(ann_file)
                        except OSError:
                            shutil.copy2(ann_file, ann_target)

        return data_root

    def _score_candidate(self, metrics: Mapping[str, int | float]) -> tuple[float, dict[str, float]]:
        if self.baseline_summary is None:
            raise RuntimeError("Baseline summary is not available for candidate scoring")

        hota = float(metrics.get("HOTA", 0.0))
        baseline_hota = float(self.baseline_summary.get("HOTA", 0.0))
        baseline_idf1 = float(self.baseline_summary.get("IDF1", 0.0))
        baseline_mota = float(self.baseline_summary.get("MOTA", 0.0))
        idf1 = float(metrics.get("IDF1", 0.0))
        mota = float(metrics.get("MOTA", 0.0))

        hota_regression = max(0.0, baseline_hota - hota - float(self.penalties.hota_tolerance))
        idf1_regression = max(0.0, baseline_idf1 - idf1 - float(self.penalties.idf1_tolerance))
        mota_regression = max(0.0, baseline_mota - mota - float(self.penalties.mota_tolerance))
        total_penalty = (
            hota_regression * float(self.penalties.hota_penalty)
            + idf1_regression * float(self.penalties.idf1_penalty)
            + mota_regression * float(self.penalties.mota_penalty)
        )
        score = hota - total_penalty

        return score, {
            "HOTA": hota,
            "baseline_HOTA": baseline_hota,
            "baseline_IDF1": baseline_idf1,
            "baseline_MOTA": baseline_mota,
            "hota_regression": hota_regression,
            "idf1_regression": idf1_regression,
            "mota_regression": mota_regression,
            "hota_tolerance": float(self.penalties.hota_tolerance),
            "idf1_tolerance": float(self.penalties.idf1_tolerance),
            "mota_tolerance": float(self.penalties.mota_tolerance),
            "hota_penalty": float(self.penalties.hota_penalty),
            "idf1_penalty": float(self.penalties.idf1_penalty),
            "mota_penalty": float(self.penalties.mota_penalty),
            "total_penalty": total_penalty,
            "score": score,
        }

    def _build_eval_payload(self, source_root: Path, tag: str) -> dict[str, Any]:
        show_progress = bool(getattr(self.config, "progress_bar", True))
        return {
            "data": None,
            "source": str(source_root),
            "benchmark": self.benchmark_id,
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.benchmark_id,
            "tracker": self.config.tracker,
            "detector": [self.detector_path],
            "reid": [self.reid_path],
            "project": self.boxmot_project_dir,
            "cache_project": self.cache_project_dir,
            "name": tag,
            "verbose": False,
            "show": False,
            "show_progress": show_progress,
            "save": False,
            "save_txt": False,
            "save_crop": False,
            "ci": False,
            "detector_explicit": True,
            "reid_explicit": True,
        }

    def _run_eval_subprocess(self, manifest_path: Path) -> dict[str, Any]:
        stderr_target = None if bool(getattr(self.config, "progress_bar", False)) else subprocess.PIPE
        proc = subprocess.Popen(
            [sys.executable, "-c", _EVAL_SNIPPET, str(manifest_path)],
            cwd=self.workspace_dir,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            text=True,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=self.config.eval_timeout)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _terminate_subprocess_tree(proc, graceful=False)
            return {
                "ok": False,
                "summary": {metric: 0.0 for metric in RESEARCH_METRICS},
                "summary_label": "",
                "per_sequence_metrics": {},
                "per_class_metrics": {},
                "stdout": stdout or exc.stdout or "",
                "stderr": stderr or exc.stderr or "",
                "error": f"Evaluation timed out after {self.config.eval_timeout:.1f}s",
                "traceback": "",
                "static_errors": [],
            }
        except KeyboardInterrupt:
            _terminate_subprocess_tree(proc, graceful=True)
            raise

        try:
            payload_out = _parse_last_json_line(stdout)
        except Exception as exc:
            payload_out = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        ok = bool(proc.returncode == 0 and payload_out.get("ok"))
        stderr_text = stderr or ""
        summary = {metric: 0.0 for metric in RESEARCH_METRICS}
        summary_payload = payload_out.get("summary", {})
        if isinstance(summary_payload, dict):
            for key, value in summary_payload.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    summary[str(key)] = value

        return {
            "ok": ok,
            "summary": summary,
            "summary_label": str(payload_out.get("summary_label", "")),
            "per_sequence_metrics": (
                payload_out.get("per_sequence_metrics", {})
                if isinstance(payload_out.get("per_sequence_metrics", {}), dict)
                else {}
            ),
            "per_class_metrics": (
                payload_out.get("per_class_metrics", {})
                if isinstance(payload_out.get("per_class_metrics", {}), dict)
                else {}
            ),
            "stdout": stdout,
            "stderr": stderr_text,
            "error": None if ok else payload_out.get("error", "Evaluation subprocess failed"),
            "traceback": payload_out.get("traceback", ""),
            "static_errors": [],
        }

    def _run_preflight_subprocess(self, manifest_path: Path) -> dict[str, Any]:
        proc = subprocess.Popen(
            [sys.executable, "-c", _PREFLIGHT_SNIPPET, str(manifest_path)],
            cwd=self.workspace_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        timeout = min(float(self.config.eval_timeout), 30.0)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _terminate_subprocess_tree(proc, graceful=False)
            return {
                "ok": False,
                "error": f"Preflight import timed out after {timeout:.1f}s",
                "traceback": "",
                "stdout": stdout or exc.stdout or "",
                "stderr": stderr or exc.stderr or "",
            }
        except KeyboardInterrupt:
            _terminate_subprocess_tree(proc, graceful=True)
            raise

        try:
            payload_out = _parse_last_json_line(stdout)
        except Exception:
            payload_out = {
                "ok": False,
                "error": "No JSON payload found in preflight subprocess stdout",
                "traceback": traceback.format_exc(),
            }

        ok = bool(proc.returncode == 0 and payload_out.get("ok"))
        return {
            "ok": ok,
            "error": None if ok else payload_out.get("error", "Preflight subprocess failed"),
            "traceback": payload_out.get("traceback", ""),
            "stdout": stdout,
            "stderr": stderr,
        }

    def _preflight_candidate(self, candidate: Mapping[str, str], tag: str) -> list[str]:
        validated = _validate_candidate_keys(candidate, self.editable_files)
        static_errors = _validate_candidate_content(validated)
        if static_errors:
            return static_errors

        self._write_candidate_to_workspace(validated)
        modules = _candidate_import_modules(self.editable_files)
        if not modules:
            return []

        manifest_dir = self.workspace_dir / _RESEARCH_ROOT / "preflight"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{_slugify(tag)}.json"
        manifest_path.write_text(json.dumps({"modules": list(modules)}, indent=2), encoding="utf-8")

        result = self._run_preflight_subprocess(manifest_path)
        if result["ok"]:
            return []

        errors = [str(result["error"])]
        if result.get("traceback"):
            errors.append(str(result["traceback"]))
        return errors

    def _run_candidate_eval(
        self,
        candidate: Mapping[str, str],
        sequence_names: Sequence[str],
        tag: str,
    ) -> dict[str, Any]:
        validated = _validate_candidate_keys(candidate, self.editable_files)
        static_errors = _validate_candidate_content(validated)
        if static_errors:
            return {
                "ok": False,
                "error": "Static validation failed",
                "static_errors": static_errors,
                "summary": {metric: 0.0 for metric in RESEARCH_METRICS},
                "summary_label": "",
                "per_sequence_metrics": {},
                "per_class_metrics": {},
                "stdout": "",
                "stderr": "",
            }

        self._write_candidate_to_workspace(validated)
        subset_root = self._subset_source_root(sequence_names)
        payload = self._build_eval_payload(subset_root, tag)

        manifest_dir = self.workspace_dir / _RESEARCH_ROOT / "payloads"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{_slugify(tag)}.json"
        manifest_path.write_text(json.dumps(payload, default=_json_default, indent=2), encoding="utf-8")
        return self._run_eval_subprocess(manifest_path)

    def _combined_benchmark_eval(self, candidate: Mapping[str, str], tag: str) -> dict[str, Any]:
        result = self._run_candidate_eval(
            candidate,
            self.selected_sequences,
            tag=tag,
        )
        if not result["ok"]:
            raise RuntimeError(f"Combined benchmark evaluation failed for {tag}: {result['error']}")
        return result

    def _combined_benchmark_summary(self, candidate: Mapping[str, str], tag: str) -> dict[str, int | float]:
        return self._combined_benchmark_eval(candidate, tag)["summary"]

    def _candidate_evaluator(self, candidate: Mapping[str, str]) -> tuple[float, dict[str, Any]]:
        validated = _validate_candidate_keys(candidate, self.editable_files)
        result = self._run_candidate_eval(
            validated,
            self.selected_sequences,
            tag="candidate_all_sequences",
        )
        change_summary = _candidate_change_summary(self.seed_candidate, validated)

        if not result["ok"]:
            side_info = {
                "scores": {"score": 0.0},
                "MOT Metrics Summary Label": result.get("summary_label", self.baseline_summary_label),
                "Sequences": list(self.selected_sequences),
                "Combined Metrics": result["summary"],
                "Baseline Metrics": self.baseline_summary or {metric: 0.0 for metric in RESEARCH_METRICS},
                "Combined Delta vs Baseline": _metric_delta(
                    result["summary"],
                    self.baseline_summary or {},
                ),
                "Per-Sequence Metrics": result.get("per_sequence_metrics", {}),
                "Per-Sequence Delta vs Baseline": _nested_metric_delta(
                    result.get("per_sequence_metrics", {}),
                    self.baseline_per_sequence_metrics,
                ),
                "Per-Class Combined Metrics": result.get("per_class_metrics", {}),
                "Per-Class Delta vs Baseline": _nested_metric_delta(
                    result.get("per_class_metrics", {}),
                    self.baseline_per_class_metrics,
                ),
                "Metric Glossary": MOT_METRIC_GLOSSARY,
                "Changed Files": change_summary,
                "Error": result["error"],
                "Validation": result.get("static_errors", []),
                "Traceback": result.get("traceback", ""),
                "Stdout": result.get("stdout", ""),
                "Stderr": result.get("stderr", ""),
            }
            return 0.0, side_info

        summary = result["summary"]
        score, penalty_breakdown = self._score_candidate(summary)

        side_info = {
            "scores": {"score": score},
            "MOT Metrics Summary Label": result.get("summary_label", self.baseline_summary_label),
            "Sequences": list(self.selected_sequences),
            "Combined Metrics": summary,
            "Baseline Metrics": self.baseline_summary,
            "Combined Delta vs Baseline": _metric_delta(summary, self.baseline_summary or {}),
            "Per-Sequence Metrics": result.get("per_sequence_metrics", {}),
            "Per-Sequence Delta vs Baseline": _nested_metric_delta(
                result.get("per_sequence_metrics", {}),
                self.baseline_per_sequence_metrics,
            ),
            "Per-Class Combined Metrics": result.get("per_class_metrics", {}),
            "Per-Class Delta vs Baseline": _nested_metric_delta(
                result.get("per_class_metrics", {}),
                self.baseline_per_class_metrics,
            ),
            "Metric Glossary": MOT_METRIC_GLOSSARY,
            "Penalty Breakdown": penalty_breakdown,
            "Changed Files": change_summary,
            "Stdout": result.get("stdout", ""),
            "Stderr": result.get("stderr", ""),
        }
        return score, side_info

    def _objective(self, baseline_summary: Mapping[str, int | float]) -> str:
        baseline = ", ".join(f"{metric}={baseline_summary.get(metric, 0.0):.2f}" for metric in RESEARCH_METRICS)
        return (
            f"Improve the BoxMOT tracker `{self.config.tracker}` on benchmark `{self.benchmark_id}` while "
            "preserving existing public behavior and file interfaces. Optimize the combined benchmark HOTA "
            "directly, while penalizing regressions in combined IDF1 and MOTA relative to the baseline benchmark "
            f"run. Current combined benchmark baseline: {baseline}."
        )

    def _background(self) -> str:
        editable = "\n".join(f"- {path}" for path in self.editable_files)
        selected_sequences = ", ".join(self.selected_sequences)
        extra_sections: list[str] = []

        for file_path in self.config.extra_context_files:
            abs_path = (ROOT / file_path).resolve()
            if not abs_path.exists():
                continue
            rel_path = abs_path.relative_to(ROOT).as_posix()
            extra_sections.append(
                f"\nContext file: {rel_path}\n```\n{abs_path.read_text(encoding='utf-8')}\n```"
            )

        return (
            f"Benchmark source root: {self.source_root}\n"
            f"Detector: {self.detector_path}\n"
            f"ReID: {self.reid_path}\n"
            "Editable files:\n"
            f"{editable}\n"
            f"Selected benchmark sequences: {selected_sequences}\n"
            "Default research scope is code-first: prioritize tracker implementation changes "
            "over standalone config tuning.\n"
            "Use the existing BoxMOT evaluation pipeline: generate detections/embeddings, run the tracker, "
            "then score with BoxMOT's MOT metrics over the full selected benchmark sequence set on every candidate "
            "evaluation. The scalar search score is combined HOTA minus penalties for any combined IDF1 or "
            "MOTA regression versus the baseline benchmark run. Keep imports minimal, preserve file paths, and "
            "avoid introducing new dependencies."
            + "".join(extra_sections)
        )

    def _save_candidate_snapshot(self, candidate: Mapping[str, str], destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for file_path, content in candidate.items():
            dst = destination / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")

    def run(self, pipeline: PipelineTracker | None = None) -> ResearchResult:
        self._ensure_dependencies()
        if pipeline is not None:
            pipeline.update("Preparing workspace...")
        _import_installed_gepa()
        workspace = self._prepare_workspace()

        LOGGER.info(
            f"Starting tracker research for {self.config.tracker} on {self.benchmark_id} "
            f"with {len(self.selected_sequences)} benchmark sequence(s)"
        )
        LOGGER.info(f"Editable files: {', '.join(self.editable_files)}")
        if pipeline is not None:
            pipeline.update(
                f"Tracker: {self.config.tracker}\n"
                f"Benchmark: {self.benchmark_id}\n"
                f"Sequences: {len(self.selected_sequences)}\n"
                f"Editable files: {', '.join(self.editable_files)}"
            )
            pipeline.advance("Running baseline benchmark evaluation...")

        LOGGER.info("Running baseline benchmark evaluation before GEPA search...")
        baseline_eval = self._combined_benchmark_eval(self.seed_candidate, "baseline_all_sequences")
        baseline_summary = baseline_eval["summary"]
        self.baseline_summary = baseline_summary
        self.baseline_summary_label = str(baseline_eval.get("summary_label", ""))
        self.baseline_per_sequence_metrics = dict(baseline_eval.get("per_sequence_metrics", {}))
        self.baseline_per_class_metrics = dict(baseline_eval.get("per_class_metrics", {}))
        objective = self._objective(baseline_summary)
        background = self._background()

        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        reflection_lm = _build_reflection_lm(
            self.config.proposal_model,
            self.config.proposal_model_kwargs,
        )
        reflection_templates = _build_reflection_prompt_templates(
            self.editable_files,
            objective,
            background,
        )

        def proposal_runner(
            candidate: dict[str, str],
            reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
            components_to_update: list[str],
        ) -> dict[str, str]:
            new_texts: dict[str, str] = {}
            for name in components_to_update:
                if name not in reflective_dataset or not reflective_dataset.get(name):
                    continue
                result = _run_instruction_proposal_signature(
                    InstructionProposalSignature,
                    lm=reflection_lm,
                    input_dict={
                        "current_instruction_doc": candidate[name],
                        "dataset_with_feedback": reflective_dataset[name],
                        "prompt_template": reflection_templates.get(name),
                    },
                )
                new_instruction = _normalize_proposed_text(result["new_instruction"], name)
                new_texts[name] = _ProposalLogText(
                    new_instruction,
                    _proposal_log_summary(name, candidate[name], new_instruction),
                )
            return new_texts

        checked_candidate_proposer = _make_checked_candidate_proposer(
            proposal_runner,
            expected_keys=self.editable_files,
            candidate_checker=lambda candidate: self._preflight_candidate(candidate, "proposal_preflight"),
        )

        gepa_config = GEPAConfig(
            engine=EngineConfig(
                run_dir=str(self.gepa_run_dir),
                frontier_type="instance",
                max_metric_calls=self.config.max_metric_calls,
                display_progress_bar=self.config.progress_bar,
                parallel=False,
                max_workers=1,
                cache_evaluation=True,
                cache_evaluation_storage="disk",
                raise_on_exception=False,
                capture_stdio=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=1,
                reflection_prompt_template=reflection_templates,
                custom_candidate_proposer=checked_candidate_proposer,
            ),
            refiner=None,
            merge=None,
        )

        LOGGER.info("Baseline complete. Starting GEPA optimization...")
        if pipeline is not None:
            pipeline.advance(
                f"Running GEPA search\n"
                f"Max metric calls: {self.config.max_metric_calls}\n"
                f"Proposal model: {self.config.proposal_model}"
            )
        result = optimize_anything(
            seed_candidate=self.seed_candidate,
            evaluator=self._candidate_evaluator,
            config=gepa_config,
        )

        best_candidate = _validate_candidate_keys(result.best_candidate, self.editable_files)
        if pipeline is not None:
            pipeline.advance("Evaluating best candidate on benchmark...")
        best_eval = self._combined_benchmark_eval(best_candidate, "best_all_sequences")
        best_summary = best_eval["summary"]
        baseline_score, baseline_penalty_breakdown = self._score_candidate(baseline_summary)
        best_score, best_penalty_breakdown = self._score_candidate(best_summary)
        delta_summary = {
            metric: float(best_summary.get(metric, 0.0) - baseline_summary.get(metric, 0.0))
            for metric in RESEARCH_METRICS
        }

        best_candidate_dir = self.run_dir / "best_candidate"
        self._save_candidate_snapshot(best_candidate, best_candidate_dir)

        summary_path = self.run_dir / "research_result.json"
        summary_path.write_text(
            json.dumps(
                {
                    "tracker": self.config.tracker,
                    "benchmark": self.benchmark_id,
                    "proposal_model": self.config.proposal_model,
                    "scoring": {
                        "primary_metric": "HOTA",
                        **self.penalties.to_dict(),
                    },
                    "search_signal": "combined_benchmark",
                    "eval_timeout": self.config.eval_timeout,
                    "editable_files": list(self.editable_files),
                    "train_sequences": [example["sequence"] for example in self.train_examples],
                    "val_sequences": [example["sequence"] for example in self.val_examples],
                    "baseline_summary": baseline_summary,
                    "baseline_score": baseline_score,
                    "baseline_penalty_breakdown": baseline_penalty_breakdown,
                    "best_summary": best_summary,
                    "best_score": best_score,
                    "best_penalty_breakdown": best_penalty_breakdown,
                    "delta_summary": delta_summary,
                },
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

        workspace_result = workspace if self.config.keep_workspace else None
        if not self.config.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
            self.workspace_dir = None

        research_result = ResearchResult(
            tracker=self.config.tracker,
            benchmark=self.benchmark_id,
            proposal_model=self.config.proposal_model,
            run_dir=self.run_dir,
            best_candidate_dir=best_candidate_dir,
            editable_files=self.editable_files,
            train_sequences=tuple(example["sequence"] for example in self.train_examples),
            val_sequences=tuple(example["sequence"] for example in self.val_examples),
            baseline_summary=baseline_summary,
            best_summary=best_summary,
            delta_summary=delta_summary,
            workspace_dir=workspace_result,
        )
        if pipeline is not None:
            pipeline.update(research_result.render())
            pipeline.complete_step()
        return research_result


def main(args: argparse.Namespace) -> ResearchResult:
    pipeline = ResearchWorkflowReporter(args).pipeline()
    with pipeline:
        return run_research(args, pipeline=pipeline)


def run_research(
    args: argparse.Namespace,
    *,
    pipeline: PipelineTracker | None = None,
) -> ResearchResult:
    config = args if isinstance(args, ResearchConfig) else ResearchConfig.from_namespace(args)
    return TrackerResearcher(config).run(pipeline=pipeline)
