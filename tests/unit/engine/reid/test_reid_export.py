from types import SimpleNamespace

import torch

from boxmot.engine.commands.reid import export as reid_export
from boxmot.engine.ui.reporters import export as export_reporter
from boxmot.reid.exporters import workflow as export_workflow


class _Pipeline:
    def __init__(self) -> None:
        self.updates: list[str] = []
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, message: str) -> None:
        self.updates.append(message)

    def advance(self, message: str) -> None:
        self.updates.append(message)

    def finish(self) -> None:
        self.finished = True


def test_main_reports_checkpoint_size_before_export(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x" * 1_000_000)
    args = SimpleNamespace(
        weights=weights,
        include=(),
        verbose=False,
        half=False,
    )
    dummy_input = torch.zeros(1, 3, 8, 4)
    pipeline = _Pipeline()
    reporter = SimpleNamespace(pipeline=lambda: pipeline)

    monkeypatch.setattr(export_reporter, "ExportWorkflowReporter", lambda _args: reporter)
    monkeypatch.setattr(export_workflow, "prepare_export", lambda _args: (torch.nn.Identity(), dummy_input))
    monkeypatch.setattr(export_workflow, "execute_export", lambda *_args: {})

    result = reid_export.main(args)

    assert result.weights == weights
    assert any(
        "Input shape:" in update and "Output shape:" in update and "(1.0 MB)" in update
        for update in pipeline.updates
    )
    assert pipeline.finished is True
