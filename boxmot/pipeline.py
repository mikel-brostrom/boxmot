# BoxMOT AGPL-3.0 license

"""High-level BoxMOT pipeline facade."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence

from boxmot.configs import BOXMOT_DEFAULTS

if TYPE_CHECKING:
    from boxmot.engine.research import ResearchResult
    from boxmot.engine.workflows.results import (
        ExportResult,
        GenerateResult,
        TrackRunResult,
        TuneResult,
        ValidationResult,
    )


class _DefaultArg:
    def __repr__(self) -> str:
        return "DEFAULT"


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
        raise ValueError("Provide exactly one of benchmark=... or source=... when calling BoxMOT.generate().")


def _normalize_classes(classes: Any) -> list[int] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        parts = [part for part in classes.replace(",", " ").split() if part]
        return [int(part) for part in parts]
    if isinstance(classes, int):
        return [int(classes)]
    return [int(value) for value in classes]


def _normalize_reid_data(data: Any) -> tuple[str, ...]:
    if data is None:
        return ()
    if isinstance(data, (str, Path)):
        return (str(data),)
    return tuple(str(value) for value in data)


_MODEL_FILE_SUFFIXES = {
    ".pt",
    ".pth",
    ".torchscript",
    ".onnx",
    ".engine",
    ".xml",
    ".tflite",
}


def _resolve_train_model_spec(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, (str, Path)):
        return None
    spec = str(value).strip()
    if not spec:
        return None

    from boxmot.configs import list_training_recipes
    from boxmot.reid.core.config import MODEL_TYPES

    candidates = [spec]
    path = Path(spec)
    if path.suffix.lower() in _MODEL_FILE_SUFFIXES:
        candidates.append(path.stem)

    model_names = {name.lower(): name for name in MODEL_TYPES}
    recipe_names = {name.lower(): name for name in list_training_recipes()}

    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in model_names:
            return "model", model_names[lowered]
        if lowered in recipe_names:
            return "recipe", recipe_names[lowered]

    if path.suffix.lower() in _MODEL_FILE_SUFFIXES:
        from boxmot.reid.core.registry import ReIDModelRegistry

        model_name = ReIDModelRegistry.get_model_name(path)
        if model_name:
            return "model", model_name
    return None


def _resolve_reid_weight_train_spec(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    if path.suffix.lower() not in _MODEL_FILE_SUFFIXES:
        return None
    return _resolve_train_model_spec(value)


def _normalize_export_include(format_value: Any, include: Sequence[str]) -> tuple[str, ...]:
    if format_value is None:
        return tuple(include)
    if isinstance(format_value, str):
        parts = [part for part in format_value.replace(",", " ").split() if part]
        return tuple(parts)
    if isinstance(format_value, SequenceABC):
        return tuple(str(item) for item in format_value)
    raise TypeError("format must be a string or sequence of strings.")


def _matches_train_default(key: str, value: Any) -> bool:
    if not hasattr(BOXMOT_DEFAULTS.train, key):
        return False
    default = getattr(BOXMOT_DEFAULTS.train, key)
    if key == "imgsz" and isinstance(value, int):
        value = (value, value // 2)
    if key == "imgsz" and isinstance(default, int):
        default = (default, default // 2)
    if key in {"eval_datasets", "head_parts", "imgsz"}:
        return tuple(value or ()) == tuple(default or ())
    return value == default


_UNSET = _DefaultArg()


def _normalize_component_kwargs(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of keyword arguments.")
    return dict(value)


def _reject_kwargs_with_initialized_component(component_name: str, component: Any, kwargs: dict[str, Any]) -> None:
    if not kwargs:
        return
    if component is _UNSET or component is None or isinstance(component, (str, Path, type)):
        return
    raise ValueError(
        f"{component_name}_kwargs cannot be supplied with an initialized {component_name} component."
    )


def _workflow_support():
    from boxmot.engine.workflows import support

    return support


def _api_args():
    from boxmot.api import _args

    return _args


class BoxMOT:
    def __init__(
        self,
        detector: Any = _UNSET,
        reid: Any = _UNSET,
        tracker: Any = _UNSET,
        classes: Any = None,
        project: str | Path = BOXMOT_DEFAULTS.track.project,
        *,
        detector_kwargs: Mapping[str, Any] | None = None,
        reid_kwargs: Mapping[str, Any] | None = None,
        tracker_kwargs: Mapping[str, Any] | None = None,
        model: str | None = None,
        recipe: str | None = None,
    ) -> None:
        self.detector_kwargs = _normalize_component_kwargs(detector_kwargs, "detector_kwargs")
        self.reid_kwargs = _normalize_component_kwargs(reid_kwargs, "reid_kwargs")
        self.tracker_kwargs = _normalize_component_kwargs(tracker_kwargs, "tracker_kwargs")

        _reject_kwargs_with_initialized_component("detector", detector, self.detector_kwargs)
        _reject_kwargs_with_initialized_component("reid", reid, self.reid_kwargs)
        _reject_kwargs_with_initialized_component("tracker", tracker, self.tracker_kwargs)

        train_model = model
        train_recipe = recipe
        if train_model is not None and train_recipe is not None:
            raise ValueError("Provide only one of model=... or recipe=... when selecting a training profile.")

        if (
            detector is not _UNSET
            and reid is _UNSET
            and tracker is _UNSET
            and train_model is None
            and train_recipe is None
        ):
            train_spec = _resolve_reid_weight_train_spec(detector)
            if train_spec is not None:
                train_kind, train_name = train_spec
                reid = detector
                if train_kind == "model":
                    train_model = train_name
                else:
                    train_recipe = train_name
                detector = _UNSET
            else:
                train_spec = _resolve_train_model_spec(detector)
                if train_spec is not None:
                    train_kind, train_name = train_spec
                    if train_kind == "model":
                        train_model = train_name
                    else:
                        train_recipe = train_name
                    detector = _UNSET

        if train_model is not None:
            train_spec = _resolve_train_model_spec(train_model)
            if train_spec is not None:
                train_kind, train_name = train_spec
                if train_kind == "model":
                    train_model = train_name
                else:
                    train_model = None
                    train_recipe = train_name

        if train_recipe is not None:
            train_spec = _resolve_train_model_spec(train_recipe)
            if train_spec is not None:
                train_kind, train_name = train_spec
                if train_kind == "recipe":
                    train_recipe = train_name
                elif train_model is None:
                    train_model = train_name
                    train_recipe = None

        if train_model is None and train_recipe is None and reid is not _UNSET and reid is not None:
            train_spec = _resolve_reid_weight_train_spec(reid)
            if train_spec is not None:
                train_kind, train_name = train_spec
                if train_kind == "model":
                    train_model = train_name
                else:
                    train_recipe = train_name

        self._detector_explicit = detector is not _UNSET and detector is not None
        self._reid_explicit = reid is not _UNSET and reid is not None
        self._tracker_explicit = tracker is not _UNSET and tracker is not None
        self._train_model_explicit = train_model is not None
        self._train_recipe_explicit = train_recipe is not None

        self.detector = BOXMOT_DEFAULTS.shared.detector if detector is _UNSET else detector
        self.reid = BOXMOT_DEFAULTS.shared.reid if reid is _UNSET else reid
        self.tracker = BOXMOT_DEFAULTS.track.tracker if tracker is _UNSET else tracker
        self.train_model = train_model
        self.train_recipe = train_recipe
        self.classes = _normalize_classes(classes)
        self.project = Path(project)

    def _detector_path(self, required: bool = True) -> Path | None:
        return _workflow_support().detector_path_from_spec(self.detector, required=required)

    def _reid_path(self, required: bool = True) -> Path | None:
        return _workflow_support().reid_path_from_spec(self.reid, required=required)

    def _tracker_name(self, required: bool = True) -> str | None:
        return _workflow_support().tracker_name_from_spec(self.tracker, required=required)

    def _tracker_backend(self, required: bool = True) -> str | None:
        return _workflow_support().tracker_backend_from_spec(self.tracker, required=required)

    def _tracker_config_from_spec(self) -> dict[str, Any] | None:
        tracker_config = _workflow_support().tracker_config_from_spec(self.tracker)
        tracker_kwargs = getattr(self, "tracker_kwargs", None)
        if not tracker_kwargs:
            return tracker_config

        if tracker_config is None:
            tracker_config = _workflow_support().default_tracker_config(self.tracker)
        tracker_config.update(tracker_kwargs)
        return tracker_config

    def _base_eval_args(
        self,
        benchmark: str | Path,
        *,
        split: str | None = None,
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
        return _api_args().build_eval_args(
            self,
            benchmark,
            split=split,
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
        from boxmot.engine.tracking import workflow as tracker_module

        args = _api_args().build_track_args(
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
            detector_kwargs=getattr(self, "detector_kwargs", None),
            reid_kwargs=getattr(self, "reid_kwargs", None),
            tracker_kwargs=getattr(self, "tracker_kwargs", None),
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
        from boxmot.engine.eval import cache as cache_module
        from boxmot.engine.workflows.reporting import timing_summary_from_stats
        from boxmot.engine.workflows.results import GenerateResult

        args = _api_args().build_generate_args(
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
        split: str | None = None,
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
            split=split,
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
        from boxmot.engine.eval import evaluator as evaluator_module
        from boxmot.utils.rich.reporters.eval import EvalWorkflowReporter

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
        split: str | None = None,
        n_trials: int = BOXMOT_DEFAULTS.tune.n_trials,
        objectives: Sequence[str] | str | None = None,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        maximize: Sequence[str] | str | None = None,
        minimize: Sequence[str] | str | None = None,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        tracker_backend: str | None = None,
        tracking_backend: str = "thread",
        seed: int = 0,
    ) -> TuneResult:
        from boxmot.engine.tuning import tuner as tuner_module

        args = _api_args().build_tune_args(
            self,
            benchmark,
            split=split,
            n_trials=n_trials,
            objectives=objectives,
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
    ) -> ResearchResult:
        from boxmot.engine import research as research_module

        args = _api_args().build_research_args(
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
        format: str | Sequence[str] | None = None,
        include: Sequence[str] = BOXMOT_DEFAULTS.export.include,
        device: str = BOXMOT_DEFAULTS.export.device,
        half: bool = BOXMOT_DEFAULTS.export.half,
        optimize: bool = BOXMOT_DEFAULTS.export.optimize,
        dynamic: bool = True,
        simplify: bool = BOXMOT_DEFAULTS.export.simplify,
        opset: int = BOXMOT_DEFAULTS.export.opset,
        workspace: int = BOXMOT_DEFAULTS.export.workspace,
        verbose: bool = False,
        batch_size: int = BOXMOT_DEFAULTS.export.batch_size,
        imgsz=None,
        tflite_quantize: str = BOXMOT_DEFAULTS.export.tflite_quantize,
        tflite_calibration_data=None,
        tflite_calibration_samples: int = BOXMOT_DEFAULTS.export.tflite_calibration_samples,
        tflite_calibration_preprocess: str = BOXMOT_DEFAULTS.export.tflite_calibration_preprocess,
        tflite_calibration_seed: int = BOXMOT_DEFAULTS.export.tflite_calibration_seed,
        tflite_calibration_update: str = BOXMOT_DEFAULTS.export.tflite_calibration_update,
        tflite_static_activation_bits: int = BOXMOT_DEFAULTS.export.tflite_static_activation_bits,
    ) -> ExportResult:
        from boxmot.engine.reid import export as export_module

        export_include = _normalize_export_include(format, include)
        args = _api_args().build_export_args(
            self,
            include=export_include,
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
            tflite_quantize=tflite_quantize,
            tflite_calibration_data=tflite_calibration_data,
            tflite_calibration_samples=tflite_calibration_samples,
            tflite_calibration_preprocess=tflite_calibration_preprocess,
            tflite_calibration_seed=tflite_calibration_seed,
            tflite_calibration_update=tflite_calibration_update,
            tflite_static_activation_bits=tflite_static_activation_bits,
        )
        return export_module.run_export(args)

    def embed(
        self,
        *,
        source: str | Path | Any,
        boxes: Any = None,
        device: str = BOXMOT_DEFAULTS.track.device,
        half: bool = BOXMOT_DEFAULTS.track.half,
        preprocess: str | None = None,
    ):
        from boxmot.reid import ReID
        from boxmot.reid.core.preprocessing import get_preprocess_fn

        if isinstance(self.reid, (str, Path)):
            reid = ReID(
                self._reid_path(required=True),
                device=device,
                half=half,
                preprocess_name=preprocess,
            )
        else:
            reid = _workflow_support().build_reid_from_spec(self.reid, device=device, half=half)
            if preprocess is not None:
                reid.preprocess_name = preprocess
                backend = getattr(reid, "model", None)
                if backend is not None:
                    backend._preprocess_name = preprocess
                    backend.preprocess_fn = get_preprocess_fn(preprocess)

        return reid(source, boxes=boxes)

    def train(
        self,
        *,
        cfg: str | Path | None = None,
        recipe: str | None = None,
        model: str = BOXMOT_DEFAULTS.train.model,
        dataset: str = BOXMOT_DEFAULTS.train.dataset,
        data: str | Path | Sequence[str | Path] | None = None,
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
        id_loss_weight: float = BOXMOT_DEFAULTS.train.id_loss_weight,
        metric_loss_weight: float = BOXMOT_DEFAULTS.train.metric_loss_weight,
        early_id_loss_weight: float = BOXMOT_DEFAULTS.train.early_id_loss_weight,
        early_id_loss_epochs: int = BOXMOT_DEFAULTS.train.early_id_loss_epochs,
        center_loss_ramp_start_epoch: int = BOXMOT_DEFAULTS.train.center_loss_ramp_start_epoch,
        center_loss_ramp_end_epoch: int = BOXMOT_DEFAULTS.train.center_loss_ramp_end_epoch,
        metric_feature: str = BOXMOT_DEFAULTS.train.metric_feature,
        inference_feature: str = BOXMOT_DEFAULTS.train.inference_feature,
        feature_fusion: str = BOXMOT_DEFAULTS.train.feature_fusion,
        feat_dim: int = BOXMOT_DEFAULTS.train.feat_dim,
        neck_dim: int = BOXMOT_DEFAULTS.train.neck_dim,
        head_pool: str = BOXMOT_DEFAULTS.train.head_pool,
        head_parts: Sequence[int] = BOXMOT_DEFAULTS.train.head_parts,
        head_type: str = BOXMOT_DEFAULTS.train.head_type,
        part_pooling: str = BOXMOT_DEFAULTS.train.part_pooling,
        num_part_tokens: int = BOXMOT_DEFAULTS.train.num_part_tokens,
        evidence_num_roles: int = BOXMOT_DEFAULTS.train.evidence_num_roles,
        decouple_patterns: bool = BOXMOT_DEFAULTS.train.decouple_patterns,
        pattern_adapter_dim: int = BOXMOT_DEFAULTS.train.pattern_adapter_dim,
        stripe_visibility: bool = BOXMOT_DEFAULTS.train.stripe_visibility,
        branch_aware_metric: bool = BOXMOT_DEFAULTS.train.branch_aware_metric,
        branch_metric_part_weight: float = BOXMOT_DEFAULTS.train.branch_metric_part_weight,
        evidence_alignment_loss_weight: float = BOXMOT_DEFAULTS.train.evidence_alignment_loss_weight,
        evidence_alignment_margin: float = BOXMOT_DEFAULTS.train.evidence_alignment_margin,
        evidence_sinkhorn_iters: int = BOXMOT_DEFAULTS.train.evidence_sinkhorn_iters,
        evidence_sinkhorn_temperature: float = BOXMOT_DEFAULTS.train.evidence_sinkhorn_temperature,
        evidence_rerank_topk: int = BOXMOT_DEFAULTS.train.evidence_rerank_topk,
        evidence_null_loss_weight: float = BOXMOT_DEFAULTS.train.evidence_null_loss_weight,
        evidence_diversity_loss_weight: float = BOXMOT_DEFAULTS.train.evidence_diversity_loss_weight,
        head_warmup_epochs: int = BOXMOT_DEFAULTS.train.head_warmup_epochs,
        head_warmup_lr_mult: float = BOXMOT_DEFAULTS.train.head_warmup_lr_mult,
        gradual_unfreeze: bool = BOXMOT_DEFAULTS.train.gradual_unfreeze,
        gradual_unfreeze_head_epochs: int = BOXMOT_DEFAULTS.train.gradual_unfreeze_head_epochs,
        gradual_unfreeze_stage_epochs: int = BOXMOT_DEFAULTS.train.gradual_unfreeze_stage_epochs,
        gradual_unfreeze_backbone_lr_mult: float = BOXMOT_DEFAULTS.train.gradual_unfreeze_backbone_lr_mult,
        gradual_unfreeze_backbone_lr_epochs: int = BOXMOT_DEFAULTS.train.gradual_unfreeze_backbone_lr_epochs,
        pretrained: bool = BOXMOT_DEFAULTS.train.pretrained,
        device: str = BOXMOT_DEFAULTS.train.device,
        project: str | Path | None = None,
        name: str = BOXMOT_DEFAULTS.train.name,
        num_workers: int = BOXMOT_DEFAULTS.train.num_workers,
        seed: int = BOXMOT_DEFAULTS.train.seed,
        deterministic: bool = BOXMOT_DEFAULTS.train.deterministic,
        eval_datasets: Sequence[str] = BOXMOT_DEFAULTS.train.eval_datasets,
        ema_decay: float | None = BOXMOT_DEFAULTS.train.ema_decay,
        gaussian_blur: bool = BOXMOT_DEFAULTS.train.gaussian_blur,
        random_grayscale: float = BOXMOT_DEFAULTS.train.random_grayscale,
        color_jitter: bool = BOXMOT_DEFAULTS.train.color_jitter,
        random_erasing: float = BOXMOT_DEFAULTS.train.random_erasing,
        resume: str | Path | None = None,
    ):
        train_project = project if project is not None else BOXMOT_DEFAULTS.train.project
        from boxmot.configs import build_mode_namespace
        from boxmot.engine.reid import trainer as reid_trainer_module
        from boxmot.engine.reid.data import resolve_reid_train_data

        train_model = model
        train_recipe = recipe if recipe is not None else self.train_recipe
        if model == BOXMOT_DEFAULTS.train.model and self.train_model is not None:
            train_model = self.train_model
        elif recipe is None:
            train_spec = _resolve_train_model_spec(model)
            if train_spec is not None:
                train_kind, train_name = train_spec
                if train_kind == "model":
                    train_model = train_name
                else:
                    train_model = BOXMOT_DEFAULTS.train.model
                    train_recipe = train_name

        if train_recipe is not None:
            train_spec = _resolve_train_model_spec(train_recipe)
            if train_spec is not None and train_spec[0] == "recipe":
                train_recipe = train_spec[1]

        payload = {
            "cfg": None if cfg is None else str(cfg),
            "recipe": train_recipe,
            "model": train_model,
            "dataset": dataset,
            "data": _normalize_reid_data(data),
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
            "id_loss_weight": float(id_loss_weight),
            "metric_loss_weight": float(metric_loss_weight),
            "early_id_loss_weight": float(early_id_loss_weight),
            "early_id_loss_epochs": int(early_id_loss_epochs),
            "center_loss_ramp_start_epoch": int(center_loss_ramp_start_epoch),
            "center_loss_ramp_end_epoch": int(center_loss_ramp_end_epoch),
            "metric_feature": metric_feature,
            "inference_feature": inference_feature,
            "feature_fusion": feature_fusion,
            "feat_dim": int(feat_dim),
            "neck_dim": int(neck_dim),
            "head_pool": head_pool,
            "head_parts": tuple(int(part) for part in head_parts),
            "head_type": head_type,
            "part_pooling": part_pooling,
            "num_part_tokens": int(num_part_tokens),
            "evidence_num_roles": int(evidence_num_roles),
            "decouple_patterns": bool(decouple_patterns),
            "pattern_adapter_dim": int(pattern_adapter_dim),
            "stripe_visibility": bool(stripe_visibility),
            "branch_aware_metric": bool(branch_aware_metric),
            "branch_metric_part_weight": float(branch_metric_part_weight),
            "evidence_alignment_loss_weight": float(evidence_alignment_loss_weight),
            "evidence_alignment_margin": float(evidence_alignment_margin),
            "evidence_sinkhorn_iters": int(evidence_sinkhorn_iters),
            "evidence_sinkhorn_temperature": float(evidence_sinkhorn_temperature),
            "evidence_rerank_topk": int(evidence_rerank_topk),
            "evidence_null_loss_weight": float(evidence_null_loss_weight),
            "evidence_diversity_loss_weight": float(evidence_diversity_loss_weight),
            "head_warmup_epochs": int(head_warmup_epochs),
            "head_warmup_lr_mult": float(head_warmup_lr_mult),
            "gradual_unfreeze": bool(gradual_unfreeze),
            "gradual_unfreeze_head_epochs": int(gradual_unfreeze_head_epochs),
            "gradual_unfreeze_stage_epochs": int(gradual_unfreeze_stage_epochs),
            "gradual_unfreeze_backbone_lr_mult": float(gradual_unfreeze_backbone_lr_mult),
            "gradual_unfreeze_backbone_lr_epochs": int(gradual_unfreeze_backbone_lr_epochs),
            "pretrained": bool(pretrained),
            "device": device,
            "project": Path(train_project),
            "name": name,
            "num_workers": int(num_workers),
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "eval_datasets": list(eval_datasets),
            "ema_decay": ema_decay,
            "gaussian_blur": bool(gaussian_blur),
            "random_grayscale": float(random_grayscale),
            "color_jitter": bool(color_jitter),
            "random_erasing": float(random_erasing),
            "resume": None if resume is None else str(resume),
        }
        explicit_keys = None
        if cfg is not None:
            explicit_keys = {"cfg"}
            if data is not None:
                explicit_keys.add("data")
            if project is not None:
                explicit_keys.add("project")
            if resume is not None:
                explicit_keys.add("resume")
            for key, value in payload.items():
                if (
                    value is not None
                    and key not in {"cfg", "data", "project", "resume"}
                    and not _matches_train_default(key, value)
                ):
                    explicit_keys.add(key)

        args = build_mode_namespace("train", payload, explicit_keys=explicit_keys)
        args = resolve_reid_train_data(args)
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
        from boxmot.engine.reid import evaluator as reid_evaluator_module

        return reid_evaluator_module.main(args)


__all__ = ("BoxMOT",)
