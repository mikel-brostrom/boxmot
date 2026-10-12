"""Publish calibration artifacts and attach subsequent evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from boxmot.engine.eval.results import ValidationResult


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish one complete calibration report."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a readable configuration without reordering its fields."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class KalmanCalibrationResult:
    """Reusable calibrated settings and counts supporting their estimates."""

    config_path: Path
    report_path: Path
    matched_detections: int
    gt_transitions: int
    fitted_parameters: tuple[str, ...]
    parameter_count: int = 5

    @property
    def description(self) -> str:
        """Describe the calibration evidence alongside the one final evaluation."""
        return (
            f"KF calibration: {self.matched_detections} matched detections, {self.gt_transitions} GT transitions; "
            f"{len(self.fitted_parameters)}/{self.parameter_count} covariance scales fitted.\n"
            f"Saved tracker configuration: {self.config_path}"
        )

    def record_final(self, result: ValidationResult) -> None:
        """Attach the final evaluation score without using it to select parameters."""
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        report["final_summary"] = result.summary
        report["final_output_dir"] = str(result.exp_dir)
        write_json(self.report_path, report)
