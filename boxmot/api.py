# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse

import cv2
import yaml

from boxmot.configs import BOXMOT_DEFAULTS, build_mode_namespace
from boxmot.data import IMAGE_EXTS, VIDEO_EXTS
from boxmot.engine.results import Results, Tracks
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.utils.compat import dataclass_slots_kwargs
from boxmot.utils.misc import increment_path, resolve_model_path
from boxmot.utils.timing import TimingStats
from boxmot.utils.torch_utils import select_device

SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
REID_TRACKERS = {"strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"}
TRACKER_CLASS_TO_NAME = {
    class_path.rsplit(".", 1)[-1].lower(): tracker_name
    for tracker_name, class_path in TRACKER_MAPPING.items()
}


@dataclass(**dataclass_slots_kwargs())
class ValidationResult:
    benchmark: str
    raw: dict[str, Any]
    summary_label: str
    summary: dict[str, Any]
    exp_dir: Path | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    args: Any = None


@dataclass(**dataclass_slots_kwargs())
class TuneTrialResult:
    index: int
    config: dict[str, Any]
    metrics: ValidationResult
    score: tuple[float, ...]


@dataclass(**dataclass_slots_kwargs())
class TuneResult:
    benchmark: str
    tracker: str
    trials: list[TuneTrialResult]
    best: TuneTrialResult
    best_config: dict[str, Any]
    best_yaml: Path


@dataclass(**dataclass_slots_kwargs())
class ExportResult:
    weights: Path
    files: dict[str, Any]


@dataclass(**dataclass_slots_kwargs())
class TrackRunResult:
    source: Any
    results: Results
    video_path: Path | None
    text_path: Path | None
    timings: dict[str, Any]
    summary: dict[str, Any]

    def __iter__(self) -> Iterator[Tracks]:
        return iter(self.results.materialize())

    def show(self) -> None:
        self.results.show()


def _normalize_classes(classes: Any) -> list[int] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        parts = [part for part in re.split(r"[\s,]+", classes.strip()) if part]
        return [int(part) for part in parts]
    if isinstance(classes, int):
        return [int(classes)]
    return [int(value) for value in classes]


def _is_leaf_source(path: Path) -> bool:
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


def _coerce_results(data: Any, detector=None, reid=None, tracker=None, verbose: bool = False) -> list[Results]:
    if isinstance(data, Results):
        return [data]

    if isinstance(data, (list, tuple)) and all(isinstance(item, Results) for item in data):
        return list(data)

    if detector is None or tracker is None:
        raise ValueError("Detector and tracker are required when evaluating raw sources.")

    return [track(source, detector, reid, tracker, verbose=verbose) for source in _expand_sources(data)]


def _ensure_model_path(model_ref: str | Path | None) -> Path | None:
    if model_ref is None:
        return None
    path = Path(model_ref)
    if not path.suffix:
        path = path.with_suffix(".pt")
    return resolve_model_path(path)


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned or "run"


def _resolve_output_stem(source: Any) -> str:
    source_str = str(source)
    if source_str.isdigit():
        return f"camera_{source_str}"

    if "://" in source_str:
        parsed = urlparse(source_str)
        pieces = [parsed.scheme, parsed.netloc, parsed.path.strip("/")]
        return _sanitize_name("_".join(piece for piece in pieces if piece))

    path = Path(source_str)
    if path.name == "img1" and path.parent.name:
        return _sanitize_name(path.parent.name)
    if path.suffix:
        return _sanitize_name(path.stem)
    return _sanitize_name(path.name)


def _extract_summary(raw_results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results_module = import_module("boxmot.utils.evaluation.results")
    label, metrics = results_module._select_plot_metrics_data(raw_results)
    if not metrics and raw_results:
        first_value = next(iter(raw_results.values()), {})
        if isinstance(first_value, dict):
            metrics = first_value

    summary = {
        column: metrics.get(column, 0)
        for column in SUMMARY_COLUMNS
        if isinstance(metrics, dict)
    }
    return label, summary


def _timing_summary_from_stats(timing_stats: TimingStats) -> dict[str, Any]:
    totals = dict(timing_stats.totals)
    total_ms = float(totals.get("total", 0.0) or 0.0)
    if total_ms == 0.0:
        total_ms = float(sum(totals.values()))

    frames = int(timing_stats.frames)
    avg_ms = {
        key: (float(value) / frames if frames else 0.0)
        for key, value in totals.items()
    }
    avg_total_ms = total_ms / frames if frames else 0.0
    fps = (1000.0 * frames / total_ms) if total_ms else 0.0

    return {
        "frames": frames,
        "totals_ms": {**{key: float(value) for key, value in totals.items()}, "total": total_ms},
        "avg_ms": {**avg_ms, "total": avg_total_ms},
        "fps": fps,
    }


def _compare_scores(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return left > right


def track(source, detector, reid=None, tracker=None, *, verbose: bool = True, drawer=None) -> Results:
    """Create a lazy streaming tracking result iterator."""
    if tracker is None:
        raise ValueError("A tracker instance is required.")
    return Results(source, detector, reid, tracker, verbose=verbose, drawer=drawer)


def evaluate(data, detector=None, reid=None, tracker=None, *, metrics: bool = True, speed: bool = True, verbose: bool = False) -> dict[str, Any]:
    """Aggregate run metrics over one or more tracking results.

    This helper summarizes execution counts and timing. It does not replace
    TrackEval ground-truth benchmark evaluation.
    """
    runs = _coerce_results(data, detector=detector, reid=reid, tracker=tracker, verbose=verbose)
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
        detector: Any = BOXMOT_DEFAULTS.shared.detector,
        reid: Any = BOXMOT_DEFAULTS.shared.reid,
        tracker: Any = BOXMOT_DEFAULTS.track.tracker,
        classes: Any = None,
        project: str | Path = BOXMOT_DEFAULTS.track.project,
    ) -> None:
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.classes = _normalize_classes(classes)
        self.project = Path(project)

    def _detector_path(self, required: bool = True) -> Path | None:
        spec = self.detector
        if spec is None:
            if required:
                raise ValueError("A detector model path is required for this operation.")
            return None
        if isinstance(spec, (str, Path)):
            return _ensure_model_path(spec)
        path = getattr(spec, "path", None)
        if path is not None:
            return _ensure_model_path(path)
        if required:
            raise ValueError("Detector benchmark workflows require a detector with a resolvable .path.")
        return None

    def _reid_path(self, required: bool = True) -> Path | None:
        spec = self.reid
        if spec is None:
            if required:
                raise ValueError("A ReID model path is required for this operation.")
            return None
        if isinstance(spec, (str, Path)):
            return _ensure_model_path(spec)
        path = getattr(spec, "path", None) or getattr(spec, "weights", None)
        if path is not None:
            return _ensure_model_path(path)
        if required:
            raise ValueError("This operation requires a ReID model with a resolvable .path or .weights.")
        return None

    def _tracker_name(self, required: bool = True) -> str | None:
        spec = self.tracker
        if spec is None:
            if required:
                raise ValueError("A tracker is required.")
            return None
        if isinstance(spec, str):
            name = spec.lower()
            if name in TRACKER_MAPPING:
                return name
        class_name = spec.__class__.__name__.lower() if spec is not None else ""
        if class_name in TRACKER_CLASS_TO_NAME:
            return TRACKER_CLASS_TO_NAME[class_name]
        if required:
            raise ValueError("Could not infer a registered tracker name from the provided tracker spec.")
        return None

    def _tracker_config_from_spec(self) -> dict[str, Any] | None:
        if isinstance(self.tracker, str) or self.tracker is None:
            return None

        tracker_name = self._tracker_name(required=False)
        if tracker_name is None:
            return None

        with open(get_tracker_config(tracker_name), "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        resolved: dict[str, Any] = {}
        for key, details in config.items():
            if hasattr(self.tracker, key):
                resolved[key] = getattr(self.tracker, key)
            else:
                resolved[key] = details.get("default")
        return resolved

    def _build_detector(self, *, device: str = BOXMOT_DEFAULTS.track.device, imgsz=None, conf=None, iou=BOXMOT_DEFAULTS.track.iou):
        from boxmot.detectors import Detector as PublicDetector

        spec = self.detector
        if isinstance(spec, (str, Path)):
            return PublicDetector(
                path=_ensure_model_path(spec),
                device=device,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                classes=self.classes,
            )

        current_device = getattr(spec, "device", None)
        if current_device is not None and str(current_device) != str(device):
            raise ValueError(
                f"Detector instance is already bound to device '{current_device}'. "
                f"Create it on '{device}' or pass a path/string detector spec instead."
            )

        if imgsz is not None and hasattr(spec, "imgsz"):
            spec.imgsz = imgsz
        if conf is not None and hasattr(spec, "conf"):
            spec.conf = float(conf)
        if iou is not None and hasattr(spec, "iou"):
            spec.iou = float(iou)
        if self.classes is not None and hasattr(spec, "classes"):
            spec.classes = self.classes
        return spec

    def _build_reid(self, *, device: str = BOXMOT_DEFAULTS.track.device, half: bool = BOXMOT_DEFAULTS.track.half):
        from boxmot.reid import ReID as PublicReID

        if self.reid is None:
            return None

        spec = self.reid
        if isinstance(spec, (str, Path)):
            return PublicReID(_ensure_model_path(spec), device=device, half=half)

        current_device = getattr(spec, "device", None)
        if current_device is not None and str(current_device) != str(device):
            raise ValueError(
                f"ReID instance is already bound to device '{current_device}'. "
                f"Create it on '{device}' or pass a path/string ReID spec instead."
            )
        return spec

    def _build_tracker(self, *, device: str = BOXMOT_DEFAULTS.track.device, half: bool = BOXMOT_DEFAULTS.track.half):
        if not isinstance(self.tracker, str):
            return self.tracker

        tracker_name = self._tracker_name(required=True)
        reid_path = self._reid_path(required=False)
        return create_tracker(
            tracker_type=tracker_name,
            tracker_config=get_tracker_config(tracker_name),
            reid_weights=reid_path,
            device=select_device(device),
            half=half,
            per_class=False,
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
        postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
    ):
        reid_path = self._reid_path(required=False) or BOXMOT_DEFAULTS.shared.reid
        tracker_spec = self.tracker
        per_class = bool(getattr(tracker_spec, "per_class", False)) if not isinstance(tracker_spec, str) else False

        args = build_mode_namespace(
            "eval",
            {
                "data": str(benchmark),
                "benchmark": str(benchmark),
                "source": None,
                "split": "",
                "yolo_model": [self._detector_path(required=True)],
                "reid_model": [reid_path],
                "device": device,
                "half": bool(half),
                "imgsz": imgsz,
                "conf": conf,
                "iou": float(iou),
                "classes": self.classes,
                "project": Path(project or self.project),
                "name": "python_api",
                "exist_ok": True,
                "ci": True,
                "tracking_method": self._tracker_name(required=True),
                "verbose": bool(verbose),
                "postprocessing": postprocessing,
                "fps": None,
                "show": False,
                "show_trajectories": False,
                "show_kf_preds": False,
                "save": False,
                "save_txt": False,
                "save_crop": False,
                "per_class": per_class,
                "target_id": None,
                "vid_stride": BOXMOT_DEFAULTS.eval.vid_stride,
            },
            explicit_keys={"yolo_model", "reid_model", "device", "half", "tracker"},
        )
        args.reid_device = device
        args.reid_half = bool(half)
        args.dataset_detector_cfg = None
        args.eval_box_type = None
        args.gt_class_remap = None
        args.gt_class_distractor_ids = None
        args.remapped_class_ids = None
        args.remapped_class_names = None
        args.translated_benchmark_class_names = None
        return args

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
        postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
        evolve_config: dict[str, Any] | None = None,
    ) -> ValidationResult:
        evaluator = import_module("boxmot.engine.evaluator")
        replay = import_module("boxmot.engine.replay")
        args = self._base_eval_args(
            benchmark,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            verbose=verbose,
            postprocessing=postprocessing,
        )

        timing_stats = TimingStats()
        evaluator.eval_setup(args)
        evaluator.run_generate_dets_embs(args, timing_stats=timing_stats)
        tracker_config = evolve_config if evolve_config is not None else self._tracker_config_from_spec()
        replay.run_generate_mot_results(
            args,
            evolve_config=tracker_config,
            timing_stats=timing_stats,
            quiet=not verbose,
        )
        raw_results = evaluator.run_trackeval(args, verbose=verbose)
        summary_label, summary = _extract_summary(raw_results)

        return ValidationResult(
            benchmark=str(benchmark),
            raw=raw_results,
            summary_label=summary_label,
            summary=summary,
            exp_dir=getattr(args, "exp_dir", None),
            timings=_timing_summary_from_stats(timing_stats),
            args=args,
        )

    def _load_tracker_search_space(self) -> dict[str, Any]:
        tracker_name = self._tracker_name(required=True)
        with open(get_tracker_config(tracker_name), "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _default_tracker_config(self) -> dict[str, Any]:
        existing = self._tracker_config_from_spec()
        if existing is not None:
            return existing

        search_space = self._load_tracker_search_space()
        return {
            key: details.get("default")
            for key, details in search_space.items()
        }

    def _sample_param(self, spec: dict[str, Any], rng: random.Random):
        param_type = str(spec.get("type", "choice")).lower()

        if param_type == "uniform":
            low, high = spec["range"]
            return float(rng.uniform(float(low), float(high)))

        if param_type == "loguniform":
            low, high = spec["range"]
            return float(math.exp(rng.uniform(math.log(float(low)), math.log(float(high)))))

        if param_type == "randint":
            low, high = spec["range"]
            return int(rng.randint(int(low), int(high)))

        if param_type == "qrandint":
            low, high, step = spec["range"]
            choices = list(range(int(low), int(high), int(step)))
            return int(rng.choice(choices))

        if param_type in {"choice", "grid_search"}:
            options = spec.get("options") or spec.get("values") or []
            if not options:
                return spec.get("default")
            return rng.choice(list(options))

        return spec.get("default")

    def _iter_tune_configs(self, n_trials: int, rng: random.Random) -> Iterator[dict[str, Any]]:
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")

        search_space = self._load_tracker_search_space()
        yield self._default_tracker_config()

        for _ in range(n_trials - 1):
            yield {
                key: self._sample_param(details, rng)
                for key, details in search_space.items()
            }

    @staticmethod
    def _score_summary(
        summary: dict[str, Any],
        maximize: Sequence[str],
        minimize: Sequence[str],
    ) -> tuple[float, ...]:
        score: list[float] = []
        for metric in maximize:
            score.append(float(summary.get(metric, float("-inf"))))
        for metric in minimize:
            score.append(-float(summary.get(metric, float("inf"))))
        return tuple(score)

    def _resolve_track_output_dir(self, source: Any) -> Path:
        base = self.project / "track" / _resolve_output_stem(source)
        return increment_path(base, mkdir=True)

    def _resolve_output_fps(self, source: Any, fallback: float = 30.0) -> float:
        if isinstance(source, (str, Path)):
            source_str = str(source)
            if source_str.isdigit() or "://" in source_str:
                return fallback
            path = Path(source_str)
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                capture = cv2.VideoCapture(str(path))
                try:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                finally:
                    capture.release()
                if fps and fps > 0:
                    return float(fps)
        return fallback

    def _save_video(self, results: Results, video_path: Path, fps: float) -> Path:
        frames = results.materialize()
        if not frames:
            return video_path

        height, width = frames[0].frame.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            for track_result in frames:
                writer.write(track_result.render())
        finally:
            writer.release()
        return video_path

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
        export_module = import_module("boxmot.engine.export")
        weights = self._reid_path(required=True)
        args = build_mode_namespace(
            "export",
            {
                "weights": weights,
                "include": tuple(include),
                "device": device,
                "half": bool(half),
                "optimize": bool(optimize),
                "dynamic": bool(dynamic),
                "simplify": bool(simplify),
                "opset": int(opset),
                "workspace": int(workspace),
                "verbose": bool(verbose),
                "batch_size": int(batch_size),
                "imgsz": imgsz,
            },
            explicit_keys={"weights", "device", "half", "optimize", "dynamic", "simplify", "opset", "workspace", "batch_size", "imgsz", "include"},
        )
        model, dummy_input = export_module.setup_model(args)
        export_tasks = export_module.create_export_tasks(args, model, dummy_input)
        files = export_module.perform_exports(export_tasks)
        return ExportResult(weights=args.weights, files=files)

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
        drawer=None,
        verbose: bool = BOXMOT_DEFAULTS.track.verbose,
    ) -> TrackRunResult:
        detector = self._build_detector(device=device, imgsz=imgsz, conf=conf, iou=iou)
        reid = self._build_reid(device=device, half=half)
        tracker = self._build_tracker(device=device, half=half)
        run = track(source, detector, reid, tracker, verbose=verbose, drawer=drawer)
        run.materialize()

        output_dir = self._resolve_track_output_dir(source)
        text_path = output_dir / "tracks.txt" if save_txt else None
        video_path = output_dir / "tracks.mp4" if save else None

        if text_path is not None:
            run.save(text_path)
        if video_path is not None:
            self._save_video(run, video_path, fps=self._resolve_output_fps(source))

        summary = run.summary()
        timings = dict(summary["timings_ms"])
        timings["fps"] = (1000.0 / timings["avg_total"]) if timings.get("avg_total") else 0.0

        return TrackRunResult(
            source=source,
            results=run,
            video_path=video_path,
            text_path=text_path,
            timings=timings,
            summary=summary,
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

        for index, config in enumerate(self._iter_tune_configs(n_trials, rng), start=1):
            metrics = self._run_validation_pipeline(
                benchmark=benchmark,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                half=half,
                project=project,
                verbose=verbose,
                evolve_config=config,
            )
            score = self._score_summary(metrics.summary, maximize=maximize, minimize=minimize)
            trials.append(TuneTrialResult(index=index, config=config, metrics=metrics, score=score))

        best = trials[0]
        for trial in trials[1:]:
            if _compare_scores(trial.score, best.score):
                best = trial

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


BoxMOT = Boxmot


__all__ = (
    "BoxMOT",
    "Boxmot",
    "ExportResult",
    "Results",
    "TrackRunResult",
    "Tracks",
    "TuneResult",
    "TuneTrialResult",
    "ValidationResult",
    "evaluate",
    "track",
)