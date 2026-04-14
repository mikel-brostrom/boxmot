from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import yaml

from boxmot.configs import BOXMOT_DEFAULTS
from boxmot.engine.results import Results
from boxmot.utils.misc import increment_path

from . import _reporting as reporting
from . import _runtime as runtime
from ._results import ExportResult, TrackRunResult, TuneResult, TuneTrialResult, ValidationResult

_UNSET = runtime._UNSET


def track(source, detector, reid=None, tracker=None, *, verbose: bool = True, drawer=None) -> Results:
    """Create a lazy streaming tracking result iterator."""
    if tracker is None:
        raise ValueError("A tracker instance is required.")
    return Results(source, detector, reid, tracker, verbose=verbose, drawer=drawer)


def evaluate(
    data,
    detector=None,
    reid=None,
    tracker=None,
    *,
    metrics: bool = True,
    speed: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Aggregate run metrics over one or more tracking results.

    This helper summarizes execution counts and timing. It does not replace
    TrackEval ground-truth benchmark evaluation.
    """
    runs = runtime.coerce_results(
        data,
        detector=detector,
        reid=reid,
        tracker=tracker,
        verbose=verbose,
        track_fn=track,
    )
    summaries = [run.summary() for run in runs]

    total_frames = sum(summary["frames"] for summary in summaries)
    total_detections = sum(summary["detections"] for summary in summaries)
    total_tracks = sum(summary["tracks"] for summary in summaries)
    total_det_ms = sum(summary["timings_ms"]["det"] for summary in summaries)
    total_reid_ms = sum(summary["timings_ms"]["reid"] for summary in summaries)
    total_track_ms = sum(summary["timings_ms"]["track"] for summary in summaries)
    total_ms = sum(summary["timings_ms"]["total"] for summary in summaries)

    response: dict[str, Any] = {
        "sources": len(summaries),
        "runs": summaries,
    }

    if metrics:
        response["metrics"] = {
            "frames": total_frames,
            "detections": total_detections,
            "tracks": total_tracks,
            "avg_tracks_per_frame": (total_tracks / total_frames) if total_frames else 0.0,
        }

    if speed:
        response["speed"] = {
            "det_ms": total_det_ms,
            "reid_ms": total_reid_ms,
            "track_ms": total_track_ms,
            "total_ms": total_ms,
            "avg_total_ms": (total_ms / total_frames) if total_frames else 0.0,
            "fps": (1000.0 * total_frames / total_ms) if total_ms else 0.0,
        }

    return response


class Boxmot:
    def __init__(
        self,
        detector: Any = _UNSET,
        reid: Any = _UNSET,
        tracker: Any = _UNSET,
        classes: Any = None,
        project: str | Path = BOXMOT_DEFAULTS.track.project,
    ) -> None:
        self._detector_explicit = detector is not _UNSET and detector is not None
        self._reid_explicit = reid is not _UNSET and reid is not None
        self._tracker_explicit = tracker is not _UNSET and tracker is not None

        self.detector = BOXMOT_DEFAULTS.shared.detector if detector is _UNSET else detector
        self.reid = BOXMOT_DEFAULTS.shared.reid if reid is _UNSET else reid
        self.tracker = BOXMOT_DEFAULTS.track.tracker if tracker is _UNSET else tracker
        self.classes = runtime.normalize_classes(classes)
        self.project = Path(project)

    def _detector_path(self, required: bool = True) -> Path | None:
        return runtime.detector_path_from_spec(self.detector, required=required)

    def _reid_path(self, required: bool = True) -> Path | None:
        return runtime.reid_path_from_spec(self.reid, required=required)

    def _tracker_name(self, required: bool = True) -> str | None:
        return runtime.tracker_name_from_spec(self.tracker, required=required)

    def _tracker_config_from_spec(self) -> dict[str, Any] | None:
        return runtime.tracker_config_from_spec(self.tracker)

    def _build_detector(
        self,
        *,
        device: str = BOXMOT_DEFAULTS.track.device,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.track.iou,
    ):
        return runtime.build_detector_from_spec(
            self.detector,
            classes=self.classes,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )

    def _build_reid(
        self,
        *,
        device: str = BOXMOT_DEFAULTS.track.device,
        half: bool = BOXMOT_DEFAULTS.track.half,
    ):
        return runtime.build_reid_from_spec(self.reid, device=device, half=half)

    def _build_tracker(
        self,
        *,
        device: str = BOXMOT_DEFAULTS.track.device,
        half: bool = BOXMOT_DEFAULTS.track.half,
    ):
        return runtime.build_tracker_from_spec(
            self.tracker,
            device=device,
            half=half,
            reid_weights=self._reid_path(required=False),
        )

    def _build_track_reid(
        self,
        tracker: Any,
        *,
        device: str = BOXMOT_DEFAULTS.track.device,
        half: bool = BOXMOT_DEFAULTS.track.half,
    ):
        return runtime.build_tracker_with_reid_spec(
            self.tracker,
            tracker,
            device=device,
            half=half,
            build_reid_fn=self._build_reid,
        )

    def _base_eval_args(
        self,
        benchmark: str | Path,
        *,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        show_progress: bool = True,
        postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
    ):
        return runtime.base_eval_args(
            self,
            benchmark,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            verbose=verbose,
            show_progress=show_progress,
            postprocessing=postprocessing,
        )

    def _run_validation_pipeline(
        self,
        *,
        benchmark: str | Path,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        show_progress: bool = True,
        postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
        evolve_config: dict[str, Any] | None = None,
    ) -> ValidationResult:
        return runtime.run_validation_pipeline(
            self,
            benchmark=benchmark,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            verbose=verbose,
            show_progress=show_progress,
            postprocessing=postprocessing,
            evolve_config=evolve_config,
        )

    def _load_tracker_search_space(self) -> dict[str, Any]:
        return runtime.load_tracker_search_space(self)

    def _default_tracker_config(self) -> dict[str, Any]:
        return runtime.default_tracker_config(self)

    def _sample_param(self, spec: dict[str, Any], rng: random.Random):
        return runtime.sample_param(spec, rng)

    def _iter_tune_configs(self, n_trials: int, rng: random.Random):
        return runtime.iter_tune_configs(self, n_trials, rng)

    @staticmethod
    def _score_summary(
        summary: dict[str, Any],
        maximize: Sequence[str],
        minimize: Sequence[str],
    ) -> tuple[float, ...]:
        return runtime.score_summary(summary, maximize=maximize, minimize=minimize)

    def _resolve_track_output_dir(self, source: Any) -> Path:
        return runtime.resolve_track_output_dir(self.project, source)

    def _resolve_output_fps(self, source: Any, fallback: float = 30.0) -> float:
        return runtime.resolve_output_fps(source, fallback=fallback, cv2_module=cv2)

    def _save_video(self, results: Results, video_path: Path, fps: float) -> Path:
        return runtime.save_video(results, video_path, fps, cv2_module=cv2)

    def _run_export_pipeline(
        self,
        *,
        include: Sequence[str],
        device: str = BOXMOT_DEFAULTS.export.device,
        half: bool = BOXMOT_DEFAULTS.export.half,
        optimize: bool = BOXMOT_DEFAULTS.export.optimize,
        dynamic: bool = BOXMOT_DEFAULTS.export.dynamic,
        simplify: bool = BOXMOT_DEFAULTS.export.simplify,
        opset: int = BOXMOT_DEFAULTS.export.opset,
        workspace: int = BOXMOT_DEFAULTS.export.workspace,
        verbose: bool = False,
        batch_size: int = BOXMOT_DEFAULTS.export.batch_size,
        imgsz=None,
    ) -> ExportResult:
        return runtime.run_export_pipeline(
            self,
            include=include,
            device=device,
            half=half,
            optimize=optimize,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            workspace=workspace,
            verbose=verbose,
            batch_size=batch_size,
            imgsz=imgsz,
        )

    def track(
        self,
        *,
        source: Any,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.track.iou,
        device: str = BOXMOT_DEFAULTS.track.device,
        half: bool = BOXMOT_DEFAULTS.track.half,
        save: bool = BOXMOT_DEFAULTS.track.save,
        save_txt: bool = BOXMOT_DEFAULTS.track.save_txt,
        show: bool = BOXMOT_DEFAULTS.track.show,
        drawer=None,
        verbose: bool = BOXMOT_DEFAULTS.track.verbose,
    ) -> TrackRunResult:
        with runtime.suppress_boxmot_logs(not verbose, level="WARNING"):
            detector = self._build_detector(device=device, imgsz=imgsz, conf=conf, iou=iou)
            tracker = self._build_tracker(device=device, half=half)
            reid = self._build_track_reid(tracker, device=device, half=half)
        run = track(source, detector, reid, tracker, verbose=verbose, drawer=drawer)

        output_dir = self._resolve_track_output_dir(source)
        text_path = output_dir / "tracks.txt" if save_txt else None
        video_path = output_dir / "tracks.mp4" if save else None

        if text_path is not None:
            run.save(text_path)
        if video_path is not None:
            self._save_video(run, video_path, fps=self._resolve_output_fps(source))

        result = TrackRunResult(
            source=source,
            results=run,
            video_path=video_path,
            text_path=text_path,
        )
        if show:
            result.show()
        return result

    def val(
        self,
        *,
        benchmark: str | Path,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
    ) -> ValidationResult:
        return self._run_validation_pipeline(
            benchmark=benchmark,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            verbose=verbose,
            show_progress=True,
            postprocessing=postprocessing,
        )

    def tune(
        self,
        *,
        benchmark: str | Path,
        n_trials: int = BOXMOT_DEFAULTS.tune.n_trials,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        maximize: Sequence[str] = BOXMOT_DEFAULTS.tune.maximize,
        minimize: Sequence[str] = BOXMOT_DEFAULTS.tune.minimize,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        seed: int = 0,
    ) -> TuneResult:
        rng = random.Random(seed)
        tracker_name = self._tracker_name(required=True)
        trials: list[TuneTrialResult] = []
        best: TuneTrialResult | None = None
        progress_width = 0
        last_summary: dict[str, Any] | None = None
        last_was_best = False
        trial_durations: list[float] = []

        for index, config in enumerate(self._iter_tune_configs(n_trials, rng), start=1):
            remaining_seconds = reporting.estimate_tune_remaining(trial_durations, n_trials - (index - 1))
            progress_width = reporting.write_progress_line(
                reporting.format_tune_progress(
                    index - 1,
                    n_trials,
                    summary=last_summary,
                    current_trial=index,
                    is_new_best=last_was_best,
                    remaining_seconds=remaining_seconds,
                ),
                progress_width,
            )

            trial_started = time.perf_counter()
            with runtime.suppress_boxmot_logs(not verbose, level="WARNING"):
                metrics = self._run_validation_pipeline(
                    benchmark=benchmark,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                    half=half,
                    project=project,
                    verbose=False,
                    show_progress=False,
                    evolve_config=config,
                )

            trial_durations.append(time.perf_counter() - trial_started)
            score = self._score_summary(metrics.summary, maximize=maximize, minimize=minimize)
            trial_result = TuneTrialResult(index=index, config=config, metrics=metrics, score=score)
            trials.append(trial_result)

            is_new_best = best is None or runtime.compare_scores(trial_result.score, best.score)
            if is_new_best:
                best = trial_result
            last_summary = metrics.summary
            last_was_best = is_new_best

            remaining_seconds = reporting.estimate_tune_remaining(trial_durations, n_trials - index)
            progress_width = reporting.write_progress_line(
                reporting.format_tune_progress(
                    index,
                    n_trials,
                    metrics.summary,
                    is_new_best=is_new_best,
                    remaining_seconds=remaining_seconds,
                ),
                progress_width,
                final=index == n_trials,
            )

        if best is None:
            raise RuntimeError("Tune did not produce any trials.")

        output_dir = increment_path(Path(project or self.project) / "tune" / f"{benchmark}_{tracker_name}", mkdir=True)
        best_yaml = output_dir / "best.yaml"
        with open(best_yaml, "w", encoding="utf-8") as handle:
            yaml.safe_dump(best.config, handle, sort_keys=False)

        return TuneResult(
            benchmark=str(benchmark),
            tracker=tracker_name,
            trials=trials,
            best=best,
            best_config=best.config,
            best_yaml=best_yaml,
        )

    def export(
        self,
        *,
        include: Sequence[str] = BOXMOT_DEFAULTS.export.include,
        device: str = BOXMOT_DEFAULTS.export.device,
        half: bool = BOXMOT_DEFAULTS.export.half,
        optimize: bool = BOXMOT_DEFAULTS.export.optimize,
        dynamic: bool = BOXMOT_DEFAULTS.export.dynamic,
        simplify: bool = BOXMOT_DEFAULTS.export.simplify,
        opset: int = BOXMOT_DEFAULTS.export.opset,
        workspace: int = BOXMOT_DEFAULTS.export.workspace,
        verbose: bool = False,
        batch_size: int = BOXMOT_DEFAULTS.export.batch_size,
        imgsz=None,
    ) -> ExportResult:
        return self._run_export_pipeline(
            include=include,
            device=device,
            half=half,
            optimize=optimize,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            workspace=workspace,
            verbose=verbose,
            batch_size=batch_size,
            imgsz=imgsz,
        )

