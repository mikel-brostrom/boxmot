from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from boxmot.configs import BOXMOT_DEFAULTS, build_mode_namespace
from boxmot.engine.eval import cache as cache_module
from boxmot.engine.eval import evaluator as evaluator_module
from boxmot.engine.reid import evaluator as reid_evaluator_module
from boxmot.engine.reid import export as export_module
from boxmot.engine.reid import trainer as reid_trainer_module
from boxmot.engine import research as research_module
from boxmot.engine.tracking import tracker as tracker_module
from boxmot.engine.tuning import tuner as tuner_module
from boxmot.engine.workflows import support
from boxmot.engine.tracking.results import Results
from boxmot.engine.workflows.reporting import timing_summary_from_stats
from boxmot.engine.workflows.results import (
    ExportResult,
    GenerateResult,
    TrackRunResult,
    TuneResult,
    ValidationResult,
)

from . import _adapters as adapters

_UNSET = adapters._UNSET


def _is_leaf_source(path: Path) -> bool:
    from boxmot.data import IMAGE_EXTS, VIDEO_EXTS

    if path.is_file():
        return path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    if not path.is_dir():
        return False
    img_dir = path / "img1" if (path / "img1").is_dir() else path
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS for child in img_dir.iterdir())


def _expand_sources(source: Any) -> list[Any]:
    if isinstance(source, (list, tuple)):
        return list(source)

    if not isinstance(source, (str, Path)):
        return [source]

    path = Path(source)
    if not path.is_dir() or _is_leaf_source(path):
        return [source]

    children = [child for child in sorted(path.iterdir()) if _is_leaf_source(child)]
    return children or [source]


def _coerce_results(
    data: Any,
    *,
    detector=None,
    reid=None,
    tracker=None,
    verbose: bool = False,
    track_fn=None,
) -> list[Results]:
    if isinstance(data, Results):
        return [data]

    if isinstance(data, (list, tuple)) and all(isinstance(item, Results) for item in data):
        return list(data)

    if detector is None or tracker is None:
        raise ValueError("Detector and tracker are required when evaluating raw sources.")
    if track_fn is None:
        raise ValueError("A tracking function is required when evaluating raw sources.")

    return [track_fn(source, detector, reid, tracker, verbose=verbose) for source in _expand_sources(data)]


def _cache_dir_from_args(args) -> Path:
    cache_project = Path(getattr(args, "cache_project", getattr(args, "project", "runs")))
    cache_dir = cache_project / "dets_n_embs"
    benchmark = getattr(args, "benchmark", None)
    if benchmark:
        cache_dir = cache_dir / str(benchmark)
    split = getattr(args, "split", None)
    if split:
        cache_dir = cache_dir / str(split)
    return cache_dir


def _validate_generate_inputs(*, benchmark: str | Path | None, source: str | Path | None) -> None:
    has_benchmark = benchmark is not None and str(benchmark) != ""
    has_source = source is not None and str(source) != ""
    if has_benchmark == has_source:
        raise ValueError("Provide exactly one of benchmark=... or source=... when calling Boxmot.generate().")


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
    """Aggregate run metrics over one or more tracking results."""
    runs = _coerce_results(
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
        self.classes = support.normalize_classes(classes)
        self.project = Path(project)

    def _detector_path(self, required: bool = True) -> Path | None:
        return support.detector_path_from_spec(self.detector, required=required)

    def _reid_path(self, required: bool = True) -> Path | None:
        return support.reid_path_from_spec(self.reid, required=required)

    def _tracker_name(self, required: bool = True) -> str | None:
        return support.tracker_name_from_spec(self.tracker, required=required)

    def _tracker_backend(self, required: bool = True) -> str | None:
        return support.tracker_backend_from_spec(self.tracker, required=required)

    def _tracker_config_from_spec(self) -> dict[str, Any] | None:
        return support.tracker_config_from_spec(self.tracker)

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
        tracker_backend: str | None = None,
        tracking_backend: str = "thread",
    ):
        return adapters.build_eval_args(
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
            tracker_backend=tracker_backend,
            tracking_backend=tracking_backend,
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
        show_trajectories: bool = False,
        verbose: bool = BOXMOT_DEFAULTS.track.verbose,
        tracker_backend: str | None = None,
    ) -> TrackRunResult:
        args = adapters.build_track_args(
            self,
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            save=save,
            save_txt=save_txt,
            show=show,
            verbose=verbose,
            tracker_backend=tracker_backend,
        )
        return tracker_module.run_track(
            args,
            detector_spec=self.detector,
            reid_spec=self.reid,
            tracker_spec=self.tracker,
            classes=self.classes,
            drawer=drawer,
            show_trajectories=show_trajectories,
        )

    def generate(
        self,
        *,
        benchmark: str | Path | None = None,
        source: str | Path | None = None,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.generate.iou,
        device: str = BOXMOT_DEFAULTS.generate.device,
        half: bool = BOXMOT_DEFAULTS.generate.half,
        project: str | Path | None = None,
        verbose: bool = BOXMOT_DEFAULTS.generate.verbose,
        batch_size: int = BOXMOT_DEFAULTS.generate.batch_size,
        auto_batch: bool = BOXMOT_DEFAULTS.generate.auto_batch,
        resume: bool = BOXMOT_DEFAULTS.generate.resume,
        n_threads: int = BOXMOT_DEFAULTS.generate.n_threads,
    ) -> GenerateResult:
        _validate_generate_inputs(benchmark=benchmark, source=source)
        args = adapters.build_generate_args(
            self,
            benchmark=benchmark,
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            verbose=verbose,
            batch_size=batch_size,
            auto_batch=auto_batch,
            resume=resume,
            n_threads=n_threads,
        )
        timing_stats = cache_module.run_generate(args)
        return GenerateResult(
            benchmark=str(getattr(args, "benchmark", None) or getattr(args, "data", None) or "") or None,
            source=Path(args.source) if getattr(args, "source", None) else None,
            cache_dir=_cache_dir_from_args(args),
            detectors=tuple(Path(detector) for detector in args.detector),
            reid_models=tuple(Path(reid_model) for reid_model in args.reid),
            timings=timing_summary_from_stats(timing_stats),
            args=args,
        )

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
        tracker_backend: str | None = None,
        tracking_backend: str = "thread",
    ) -> ValidationResult:
        args = self._base_eval_args(
            benchmark,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            verbose=verbose,
            show_progress=True,
            postprocessing=postprocessing,
            tracker_backend=tracker_backend,
            tracking_backend=tracking_backend,
        )
        from boxmot.utils.rich.eval_reporting import EvalWorkflowReporter

        evaluator_module._normalize_eval_models(args)
        pipeline = EvalWorkflowReporter(args).pipeline()
        with pipeline:
            metrics = evaluator_module.run_eval(
                args,
                evolve_config=self._tracker_config_from_spec(),
                pipeline=pipeline,
            )
            metrics.workflow_rendered = True
            return metrics

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
        tracker_backend: str | None = None,
        tracking_backend: str = "thread",
        seed: int = 0,
    ) -> TuneResult:
        args = adapters.build_tune_args(
            self,
            benchmark,
            n_trials=n_trials,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            maximize=maximize,
            minimize=minimize,
            verbose=verbose,
            tracker_backend=tracker_backend,
            tracking_backend=tracking_backend,
            seed=seed,
        )
        args.compare_to_first_trial = True
        tune_results = tuner_module.run_tune(
            args,
            baseline_config=self._tracker_config_from_spec(),
        )
        tune_results.workflow_rendered = True
        return tune_results

    def research(
        self,
        *,
        benchmark: str | Path,
        project: str | Path | None = None,
        verbose: bool = BOXMOT_DEFAULTS.research.verbose,
        proposal_model: str = BOXMOT_DEFAULTS.research.proposal_model,
        proposal_api_key: str | None = BOXMOT_DEFAULTS.research.proposal_api_key,
        proposal_api_key_env: str | None = BOXMOT_DEFAULTS.research.proposal_api_key_env,
        max_metric_calls: int = BOXMOT_DEFAULTS.research.max_metric_calls,
        eval_timeout: float = BOXMOT_DEFAULTS.research.eval_timeout,
        keep_workspace: bool = BOXMOT_DEFAULTS.research.keep_workspace,
        hota_penalty: float = BOXMOT_DEFAULTS.research.hota_penalty,
        idf1_penalty: float = BOXMOT_DEFAULTS.research.idf1_penalty,
        mota_penalty: float = BOXMOT_DEFAULTS.research.mota_penalty,
        hota_tolerance: float = BOXMOT_DEFAULTS.research.hota_tolerance,
        idf1_tolerance: float = BOXMOT_DEFAULTS.research.idf1_tolerance,
        mota_tolerance: float = BOXMOT_DEFAULTS.research.mota_tolerance,
        tracker_backend: str | None = None,
        tracking_backend: str = "thread",
    ) -> research_module.ResearchResult:
        args = adapters.build_research_args(
            self,
            benchmark,
            project=project,
            verbose=verbose,
            proposal_model=proposal_model,
            proposal_api_key=proposal_api_key,
            proposal_api_key_env=proposal_api_key_env,
            max_metric_calls=max_metric_calls,
            eval_timeout=eval_timeout,
            keep_workspace=keep_workspace,
            hota_penalty=hota_penalty,
            idf1_penalty=idf1_penalty,
            mota_penalty=mota_penalty,
            hota_tolerance=hota_tolerance,
            idf1_tolerance=idf1_tolerance,
            mota_tolerance=mota_tolerance,
            tracker_backend=tracker_backend,
            tracking_backend=tracking_backend,
        )
        return research_module.run_research(args)

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
        args = adapters.build_export_args(
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
        return export_module.run_export(args)

    def train(
        self,
        *,
        model: str = BOXMOT_DEFAULTS.train.model,
        dataset: str = BOXMOT_DEFAULTS.train.dataset,
        data_dir: str | Path | None = BOXMOT_DEFAULTS.train.data_dir,
        loss: str = BOXMOT_DEFAULTS.train.loss,
        preprocess: str = BOXMOT_DEFAULTS.train.preprocess,
        imgsz=None,
        batch_size: int = BOXMOT_DEFAULTS.train.batch_size,
        lr: float = BOXMOT_DEFAULTS.train.lr,
        weight_decay: float = BOXMOT_DEFAULTS.train.weight_decay,
        epochs: int = BOXMOT_DEFAULTS.train.epochs,
        warmup_epochs: int = BOXMOT_DEFAULTS.train.warmup_epochs,
        eval_interval: int = BOXMOT_DEFAULTS.train.eval_interval,
        p_ids: int = BOXMOT_DEFAULTS.train.p_ids,
        k_instances: int = BOXMOT_DEFAULTS.train.k_instances,
        margin: float = BOXMOT_DEFAULTS.train.margin,
        label_smooth: float = BOXMOT_DEFAULTS.train.label_smooth,
        center_loss_weight: float = BOXMOT_DEFAULTS.train.center_loss_weight,
        metric_feature: str = BOXMOT_DEFAULTS.train.metric_feature,
        inference_feature: str = BOXMOT_DEFAULTS.train.inference_feature,
        feat_dim: int = BOXMOT_DEFAULTS.train.feat_dim,
        neck_dim: int = BOXMOT_DEFAULTS.train.neck_dim,
        head_pool: str = BOXMOT_DEFAULTS.train.head_pool,
        head_parts: Sequence[int] = BOXMOT_DEFAULTS.train.head_parts,
        branch_aware_metric: bool = BOXMOT_DEFAULTS.train.branch_aware_metric,
        branch_metric_part_weight: float = BOXMOT_DEFAULTS.train.branch_metric_part_weight,
        head_warmup_epochs: int = BOXMOT_DEFAULTS.train.head_warmup_epochs,
        head_warmup_lr_mult: float = BOXMOT_DEFAULTS.train.head_warmup_lr_mult,
        pretrained: bool = BOXMOT_DEFAULTS.train.pretrained,
        device: str = BOXMOT_DEFAULTS.train.device,
        project: str | Path | None = None,
        name: str = BOXMOT_DEFAULTS.train.name,
        num_workers: int = BOXMOT_DEFAULTS.train.num_workers,
        seed: int = BOXMOT_DEFAULTS.train.seed,
        eval_datasets: Sequence[str] = BOXMOT_DEFAULTS.train.eval_datasets,
        ema_decay: float | None = BOXMOT_DEFAULTS.train.ema_decay,
        gaussian_blur: bool = BOXMOT_DEFAULTS.train.gaussian_blur,
        random_grayscale: float = BOXMOT_DEFAULTS.train.random_grayscale,
        color_jitter: bool = BOXMOT_DEFAULTS.train.color_jitter,
        random_erasing: float = BOXMOT_DEFAULTS.train.random_erasing,
        resume: str | Path | None = None,
    ):
        train_project = project if project is not None else BOXMOT_DEFAULTS.train.project
        args = build_mode_namespace(
            "train",
            {
                "model": model,
                "dataset": dataset,
                "data_dir": None if data_dir is None else str(data_dir),
                "loss": loss,
                "preprocess": preprocess,
                "imgsz": imgsz if imgsz is not None else BOXMOT_DEFAULTS.train.imgsz,
                "batch_size": int(batch_size),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "epochs": int(epochs),
                "warmup_epochs": int(warmup_epochs),
                "eval_interval": int(eval_interval),
                "p_ids": int(p_ids),
                "k_instances": int(k_instances),
                "margin": float(margin),
                "label_smooth": float(label_smooth),
                "center_loss_weight": float(center_loss_weight),
                "metric_feature": metric_feature,
                "inference_feature": inference_feature,
                "feat_dim": int(feat_dim),
                "neck_dim": int(neck_dim),
                "head_pool": head_pool,
                "head_parts": tuple(int(part) for part in head_parts),
                "branch_aware_metric": bool(branch_aware_metric),
                "branch_metric_part_weight": float(branch_metric_part_weight),
                "head_warmup_epochs": int(head_warmup_epochs),
                "head_warmup_lr_mult": float(head_warmup_lr_mult),
                "pretrained": bool(pretrained),
                "device": device,
                "project": Path(train_project),
                "name": name,
                "num_workers": int(num_workers),
                "seed": int(seed),
                "eval_datasets": list(eval_datasets),
                "ema_decay": ema_decay,
                "gaussian_blur": bool(gaussian_blur),
                "random_grayscale": float(random_grayscale),
                "color_jitter": bool(color_jitter),
                "random_erasing": float(random_erasing),
                "resume": None if resume is None else str(resume),
            },
        )
        return reid_trainer_module.main(args)

    def eval_reid(
        self,
        *,
        weights: str | Path,
        dataset: str,
        data_dir: str | Path,
        model: str | None = None,
        preprocess: str | None = None,
        imgsz: int | tuple[int, int] | None = None,
        inference_feature: str | None = None,
        flip_tta: bool | None = None,
        device: str = "cpu",
        batch_size: int = 64,
        num_workers: int = 4,
        output: str | Path | None = None,
    ) -> dict[str, Any]:
        args = SimpleNamespace(
            weights=str(weights),
            model=model,
            dataset=dataset,
            data_dir=str(data_dir),
            preprocess=preprocess,
            imgsz=imgsz,
            inference_feature=inference_feature,
            flip_tta=flip_tta,
            device=device,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            output=None if output is None else str(output),
        )
        return reid_evaluator_module.main(args)
