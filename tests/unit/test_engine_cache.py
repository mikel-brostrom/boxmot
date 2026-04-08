from __future__ import annotations

from types import SimpleNamespace

import boxmot.engine.cache as cache_module
import boxmot.engine.evaluator as evaluator_module


def test_cache_workflow_runner_delegates_to_main(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setattr(cache_module, "main", fake_main)

    args = SimpleNamespace()
    cache_module.DetectionsEmbeddingsGenerator(args).run()

    assert captured["args"] is args


def test_evaluator_reexports_cache_generation_helpers():
    assert evaluator_module.generate_dets_embs_batched is cache_module.generate_dets_embs_batched
    assert evaluator_module.run_generate_dets_embs is cache_module.run_generate_dets_embs