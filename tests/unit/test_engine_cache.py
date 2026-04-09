from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import boxmot.engine.cache as cache_module
import boxmot.engine.evaluator as evaluator_module


def test_cache_main_runs_generation_pipeline(monkeypatch):
    generated = []
    printed = []

    monkeypatch.setattr(cache_module, "run_generate_dets_embs", lambda args, timing_stats=None: generated.append((args, timing_stats)))
    monkeypatch.setattr(cache_module.TimingStats, "print_summary", lambda self: printed.append(self.frames))

    args = SimpleNamespace()
    cache_module.main(args)

    assert generated and generated[0][0] is args
    assert len(printed) == 0


def test_evaluator_reexports_cache_generation_helpers():
    assert evaluator_module.generate_dets_embs_batched is cache_module.generate_dets_embs_batched
    assert evaluator_module.run_generate_dets_embs is cache_module.run_generate_dets_embs


def test_run_generate_dets_embs_logs_only_when_verbose(monkeypatch, tmp_path):
    logged = []
    generated = []

    monkeypatch.setattr(cache_module, "_configure_benchmark_runtime", lambda args: None)
    monkeypatch.setattr(
        cache_module,
        "generate_dets_embs_batched",
        lambda args, yolo_model, source_root, timing_stats=None: generated.append(
            (args.verbose, args.show_progress, yolo_model, source_root)
        ),
    )
    monkeypatch.setattr(cache_module.LOGGER, "info", lambda message: logged.append(message))

    quiet_args = SimpleNamespace(
        project=tmp_path,
        source=tmp_path / "benchmark",
        data=None,
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        batch_size=16,
        n_threads=1,
        auto_batch=True,
        resume=True,
        verbose=False,
        show_progress=True,
    )
    cache_module.run_generate_dets_embs(quiet_args)

    assert logged == []
    assert len(generated) == 1
    assert generated[0][0] is False
    assert generated[0][1] is True
    assert generated[0][2].name == "det.pt"
    assert generated[0][3] == tmp_path / "benchmark"

    verbose_args = SimpleNamespace(
        project=tmp_path,
        source=tmp_path / "benchmark",
        data=None,
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        batch_size=16,
        n_threads=1,
        auto_batch=True,
        resume=True,
        verbose=True,
        show_progress=True,
    )
    cache_module.run_generate_dets_embs(verbose_args)

    assert logged == ["Generating dets+embs (batched single-process): det.pt"]
    assert generated[-1][0] is True
    assert generated[-1][1] is True
    assert generated[-1][2].name == "det.pt"
    assert generated[-1][3] == tmp_path / "benchmark"
