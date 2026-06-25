from __future__ import annotations

from collections.abc import Mapping, Sequence

from .constants import RESEARCH_METRICS


def _aggregate_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {metric: 0.0 for metric in RESEARCH_METRICS}
    count = float(len(rows))
    return {
        metric: sum(float(row.get(metric, 0.0)) for row in rows) / count
        for metric in RESEARCH_METRICS
    }

def _metric_delta(
    current: Mapping[str, int | float],
    baseline: Mapping[str, int | float] | None,
) -> dict[str, float]:
    if not baseline:
        return {}

    delta: dict[str, float] = {}
    for key, value in current.items():
        baseline_value = baseline.get(key)
        if isinstance(value, bool) or isinstance(baseline_value, bool):
            continue
        if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta[key] = float(value) - float(baseline_value)
    return delta


def _nested_metric_delta(
    current: Mapping[str, Mapping[str, int | float]],
    baseline: Mapping[str, Mapping[str, int | float]] | None,
) -> dict[str, dict[str, float]]:
    if not baseline:
        return {}

    deltas: dict[str, dict[str, float]] = {}
    for name, metrics in current.items():
        metric_delta = _metric_delta(metrics, baseline.get(name, {}))
        if metric_delta:
            deltas[name] = metric_delta
    return deltas

