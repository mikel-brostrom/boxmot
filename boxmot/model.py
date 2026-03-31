from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from boxmot.configs import build_mode_namespace, normalize_overrides, parse_string_list
from boxmot.utils.evaluation.results import _select_plot_metrics_data

_METRIC_ALIASES = {
    "hota": "HOTA",
    "mota": "MOTA",
    "idf1": "IDF1",
    "assa": "AssA",
    "assre": "AssRe",
    "idsw": "IDSW",
    "ids": "IDs",
}


def _resolve_metric_key(metrics: Mapping[str, Any], key: str) -> str | None:
    if key in metrics:
        return key

    lowered = str(key).lower()
    if lowered in _METRIC_ALIASES and _METRIC_ALIASES[lowered] in metrics:
        return _METRIC_ALIASES[lowered]

    for metric_name in metrics:
        if metric_name.lower() == lowered:
            return metric_name

    return None


class _ResultsWrapper:
    """Shared helpers for Python API result wrappers."""

    def __init__(self, results: Mapping[str, Any] | None):
        self._results = deepcopy(dict(results or {}))

    @staticmethod
    def _path_or_none(value: Any) -> Path | None:
        return Path(value) if value is not None else None

    @property
    def raw(self) -> dict[str, Any]:
        return deepcopy(self._results)

    @property
    def results_dict(self) -> dict[str, Any]:
        return self.raw

    def to_dict(self) -> dict[str, Any]:
        return self.raw


class TrackEvalMetrics:
    """Convenience wrapper around parsed TrackEval results."""

    def __init__(self, results: Mapping[str, Any] | None):
        self._results = deepcopy(dict(results or {}))

    @staticmethod
    def _is_flat_metrics(data: Mapping[str, Any]) -> bool:
        return any(isinstance(value, (int, float)) for value in data.values())

    @property
    def raw(self) -> dict[str, Any]:
        return deepcopy(self._results)

    @property
    def results_dict(self) -> dict[str, Any]:
        return self.raw

    @property
    def summary_name(self) -> str:
        if self._is_flat_metrics(self._results):
            return "single_class"

        selected_name, _ = _select_plot_metrics_data(self._results)
        return selected_name

    @property
    def summary(self) -> dict[str, float | int]:
        if self._is_flat_metrics(self._results):
            return {
                key: value
                for key, value in self._results.items()
                if isinstance(value, (int, float))
            }

        _, selected_metrics = _select_plot_metrics_data(self._results)
        return {
            key: value
            for key, value in selected_metrics.items()
            if isinstance(value, (int, float))
        }

    @property
    def classes(self) -> dict[str, "TrackEvalMetrics"]:
        if self._is_flat_metrics(self._results):
            return {}

        return {
            name: TrackEvalMetrics(metrics)
            for name, metrics in self._results.items()
            if isinstance(metrics, Mapping)
        }

    @property
    def per_sequence(self) -> dict[str, Any]:
        if not self._is_flat_metrics(self._results):
            return {}

        per_sequence = self._results.get("per_sequence", {})
        return deepcopy(per_sequence) if isinstance(per_sequence, Mapping) else {}

    def to_dict(self) -> dict[str, Any]:
        return self.raw

    def __getitem__(self, key: str) -> Any:
        if key in self._results:
            value = self._results[key]
            if key == "per_sequence" or not isinstance(value, Mapping):
                return deepcopy(value)
            return TrackEvalMetrics(value)

        metric_key = _resolve_metric_key(self.summary, key)
        if metric_key is not None:
            return self.summary[metric_key]

        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        metric_key = _resolve_metric_key(self.summary, name)
        if metric_key is not None:
            return self.summary[metric_key]

        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __repr__(self) -> str:
        summary = ", ".join(f"{key}={value}" for key, value in self.summary.items())
        return f"{type(self).__name__}({summary})"


class TuneTrialResult:
    """Convenience wrapper around one Ray Tune trial result."""

    def __init__(self, result: Mapping[str, Any] | None):
        self._result = deepcopy(dict(result or {}))

    @property
    def raw(self) -> dict[str, Any]:
        return deepcopy(self._result)

    @property
    def trial_id(self) -> str:
        return str(self._result.get("trial_id", ""))

    @property
    def trial_dir(self) -> Path | None:
        value = self._result.get("trial_dir")
        return Path(value) if value is not None else None

    @property
    def config(self) -> dict[str, Any]:
        value = self._result.get("config", {})
        return deepcopy(dict(value)) if isinstance(value, Mapping) else {}

    @property
    def metrics(self) -> TrackEvalMetrics:
        value = self._result.get("metrics", {})
        return TrackEvalMetrics(value if isinstance(value, Mapping) else {})

    def to_dict(self) -> dict[str, Any]:
        return self.raw

    def __getattr__(self, name: str) -> Any:
        return getattr(self.metrics, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(trial_id={self.trial_id!r}, metrics={self.metrics!r})"


class TuneResults:
    """Convenience wrapper around BoxMOT tune output."""

    def __init__(self, results: Mapping[str, Any] | None):
        self._results = deepcopy(dict(results or {}))

    @staticmethod
    def _path_or_none(value: Any) -> Path | None:
        return Path(value) if value is not None else None

    @property
    def raw(self) -> dict[str, Any]:
        return deepcopy(self._results)

    @property
    def results_dict(self) -> dict[str, Any]:
        return self.raw

    @property
    def tracking_method(self) -> str:
        return str(self._results.get("tracking_method", ""))

    @property
    def benchmark(self) -> str:
        return str(self._results.get("benchmark", ""))

    @property
    def objectives(self) -> list[str]:
        return parse_string_list(self._results.get("objectives"))

    @property
    def maximize(self) -> list[str]:
        return parse_string_list(self._results.get("maximize"))

    @property
    def minimize(self) -> list[str]:
        return parse_string_list(self._results.get("minimize"))

    @property
    def tune_dir(self) -> Path | None:
        return self._path_or_none(self._results.get("tune_dir"))

    @property
    def results_csv(self) -> Path | None:
        return self._path_or_none(self._results.get("results_csv"))

    @property
    def summary_path(self) -> Path | None:
        return self._path_or_none(self._results.get("summary_path"))

    @property
    def best_yaml(self) -> Path | None:
        return self._path_or_none(self._results.get("best_yaml"))

    @property
    def best_trial_id(self) -> str:
        return str(self._results.get("best_trial_id", ""))

    @property
    def best_config(self) -> dict[str, Any]:
        value = self._results.get("best_config", {})
        return deepcopy(dict(value)) if isinstance(value, Mapping) else {}

    @property
    def best_metrics(self) -> TrackEvalMetrics:
        value = self._results.get("best_metrics", {})
        return TrackEvalMetrics(value if isinstance(value, Mapping) else {})

    @property
    def best(self) -> TuneTrialResult:
        value = self._results.get("best_trial")
        if isinstance(value, Mapping):
            return TuneTrialResult(value)

        return TuneTrialResult(
            {
                "trial_id": self.best_trial_id,
                "config": self.best_config,
                "metrics": self.best_metrics.to_dict(),
            }
        )

    @property
    def trials(self) -> list[TuneTrialResult]:
        value = self._results.get("trials", [])
        if not isinstance(value, (list, tuple)):
            return []
        return [TuneTrialResult(trial) for trial in value if isinstance(trial, Mapping)]

    def to_dict(self) -> dict[str, Any]:
        return self.raw

    def __getitem__(self, key: str) -> Any:
        if key in self._results:
            value = self._results[key]
            if key == "best_trial" and isinstance(value, Mapping):
                return TuneTrialResult(value)
            return deepcopy(value)

        return self.best_metrics[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.best_metrics, name)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tracking_method={self.tracking_method!r}, "
            f"best_trial_id={self.best_trial_id!r}, metrics={self.best_metrics!r})"
        )


class TrackResults(_ResultsWrapper):
    """Convenience wrapper around BoxMOT track output."""

    @property
    def source(self) -> str:
        return str(self._results.get("source", ""))

    @property
    def tracking_method(self) -> str:
        return str(self._results.get("tracking_method", ""))

    @property
    def detector(self) -> Path | None:
        return self._path_or_none(self._results.get("detector"))

    @property
    def reid(self) -> Path | None:
        return self._path_or_none(self._results.get("reid"))

    @property
    def save_dir(self) -> Path | None:
        return self._path_or_none(self._results.get("save_dir"))

    @property
    def video_path(self) -> Path | None:
        return self._path_or_none(self._results.get("video_path"))

    @property
    def text_path(self) -> Path | None:
        return self._path_or_none(self._results.get("text_path"))

    @property
    def frames(self) -> int:
        return int(self._results.get("frames", 0) or 0)

    @property
    def user_quit(self) -> bool:
        return bool(self._results.get("user_quit", False))

    @property
    def timings(self) -> dict[str, Any]:
        value = self._results.get("timings", {})
        return deepcopy(dict(value)) if isinstance(value, Mapping) else {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tracking_method={self.tracking_method!r}, "
            f"source={self.source!r}, frames={self.frames})"
        )


class ExportResults(_ResultsWrapper):
    """Convenience wrapper around BoxMOT export output."""

    @property
    def weights(self) -> Path | None:
        return self._path_or_none(self._results.get("weights"))

    @property
    def output_dir(self) -> Path | None:
        return self._path_or_none(self._results.get("output_dir"))

    @property
    def include(self) -> list[str]:
        return parse_string_list(self._results.get("include"))

    @property
    def elapsed_time(self) -> float:
        value = self._results.get("elapsed_time", 0.0)
        return float(value or 0.0)

    @property
    def input_shape(self) -> tuple[int, ...]:
        value = self._results.get("input_shape") or ()
        return tuple(value)

    @property
    def output_shape(self) -> tuple[int, ...]:
        value = self._results.get("output_shape") or ()
        return tuple(value)

    @property
    def files(self) -> dict[str, Path]:
        value = self._results.get("files", {})
        if not isinstance(value, Mapping):
            return {}
        return {str(name): Path(path) for name, path in value.items()}

    def __getitem__(self, key: str) -> Any:
        if key in self.files:
            return self.files[key]
        if key in self._results:
            return deepcopy(self._results[key])
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        if name in self.files:
            return self.files[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __repr__(self) -> str:
        formats = ", ".join(sorted(self.files))
        return f"{type(self).__name__}(weights={self.weights!r}, formats=[{formats}])"


class BoxMOT:
    """Stateful workflow wrapper with YOLO-style ``val()``, ``tune()``, ``track()``, and ``export()`` APIs."""

    def __init__(
        self,
        detector: str | Path | None = None,
        reid: str | Path | None = None,
        tracker: str = "bytetrack",
        benchmark: str | Path | None = None,
        classes: Any = None,
        **kwargs: Any,
    ) -> None:
        overrides: dict[str, Any] = {"tracker": tracker}
        if detector is not None:
            overrides["detector"] = detector
        if reid is not None:
            overrides["reid"] = reid
        if benchmark is not None:
            overrides["data"] = benchmark
        if classes is not None:
            overrides["classes"] = classes
        overrides.update(kwargs)

        self._overrides = normalize_overrides(overrides)
        self.args: SimpleNamespace | None = None
        self.metrics: TrackEvalMetrics | None = None
        self.tune_results: TuneResults | None = None
        self.track_results: TrackResults | None = None
        self.export_results: ExportResults | None = None

    @property
    def overrides(self) -> dict[str, Any]:
        return deepcopy(self._overrides)

    def _merge_overrides(self, incoming: Mapping[str, Any]) -> dict[str, Any]:
        merged = deepcopy(self._overrides)
        merged.update(normalize_overrides(incoming))
        return merged

    def val(self, benchmark: str | Path | None = None, **kwargs: Any) -> TrackEvalMetrics:
        """Run benchmark evaluation and return parsed TrackEval metrics."""
        incoming = dict(kwargs)
        if benchmark is not None:
            incoming["data"] = benchmark

        merged = self._merge_overrides(incoming)
        merged.setdefault("tracking_backend", "thread")
        if merged.get("data") in (None, ""):
            raise ValueError(
                "BoxMOT.val() requires a benchmark config. "
                "Pass data='mot17-mini' or benchmark='mot17-mini'."
            )

        args = build_mode_namespace("eval", merged, explicit_keys=set(merged))

        from boxmot.engine.evaluator import main as run_eval

        results = run_eval(args)
        self._overrides = merged
        self.args = args
        self.metrics = TrackEvalMetrics(results)
        return self.metrics

    def tune(self, benchmark: str | Path | None = None, **kwargs: Any) -> TuneResults:
        """Run hyperparameter tuning and return structured best-trial results."""
        incoming = dict(kwargs)
        if benchmark is not None:
            incoming["data"] = benchmark

        merged = self._merge_overrides(incoming)
        merged.setdefault("tracking_backend", "thread")
        if merged.get("data") in (None, ""):
            raise ValueError(
                "BoxMOT.tune() requires a benchmark config. "
                "Pass data='mot17-mini' or benchmark='mot17-mini'."
            )

        args = build_mode_namespace("tune", merged, explicit_keys=set(merged))

        from boxmot.engine.tuner import main as run_tune

        results = run_tune(args)
        self._overrides = merged
        self.args = args
        self.tune_results = TuneResults(results)
        return self.tune_results

    def track(self, source: str | Path | None = None, **kwargs: Any) -> TrackResults:
        """Run the BoxMOT tracking pipeline and return structured run metadata."""
        incoming = dict(kwargs)
        if source is not None:
            incoming["source"] = source

        merged = self._merge_overrides(incoming)
        if merged.get("source") in (None, ""):
            raise ValueError(
                "BoxMOT.track() requires a tracking source. "
                "Pass source='video.mp4' or source=0."
            )

        args = build_mode_namespace("track", merged, explicit_keys=set(merged))

        from boxmot.engine.tracker import main as run_track

        results = run_track(args)
        self._overrides = merged
        self.args = args
        self.track_results = TrackResults(results)
        return self.track_results

    def export(self, weights: str | Path | None = None, **kwargs: Any) -> ExportResults:
        """Export a ReID model and return structured artifact metadata."""
        incoming = dict(kwargs)
        if weights is not None:
            incoming["weights"] = weights

        merged = self._merge_overrides(incoming)
        if merged.get("weights") is None and merged.get("reid") is None:
            raise ValueError(
                "BoxMOT.export() requires model weights. "
                "Pass weights='osnet_x0_25_msmt17.pt' or configure reid='osnet_x0_25_msmt17'."
            )

        args = build_mode_namespace("export", merged, explicit_keys=set(merged))

        from boxmot.engine.export import main as run_export

        results = run_export(args)
        self._overrides = merged
        self.args = args
        self.export_results = ExportResults(results)
        return self.export_results


def boxmot(*args: Any, **kwargs: Any) -> BoxMOT:
    """Return a ``BoxMOT`` instance through a lowercase factory."""
    return BoxMOT(*args, **kwargs)
