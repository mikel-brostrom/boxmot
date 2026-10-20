"""Tuning outputs and replay share one resolved configuration for each trial."""

from types import SimpleNamespace

import pytest

from boxmot.engine.eval import evaluator
from boxmot.engine.eval.replay import ReplayResult
from boxmot.engine.materialization import fingerprint


@pytest.mark.parametrize("guided", [False, True])
def test_trial_resolves_settings_once_and_keeps_output_identity(monkeypatch, tmp_path, guided: bool) -> None:
    from boxmot.segmentors.propagation import weights

    profile = tmp_path / "profile.yaml"
    profile.write_text("track_thresh: 0.4\nuse_cmc: false\n", encoding="utf-8")
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.write_bytes(b"fixture model")
    options = SimpleNamespace(
        _build_validated=True,
        build_path=tmp_path / "build",
        dataset_id="fixture",
        experiment_id=None,
        geometry="aabb",
        sequence_workers=1,
        project=tmp_path / "runs",
        seq_info={"sequence": 2},
        sequence_names=None,
        split="validation",
        tracker="bytetrack",
        tracker_config=profile,
        tracker_class_ids=(1,),
        tracker_class_names=((1, "person"),),
        edgetam=guided,
        mask_guidance_weights=checkpoint,
    )
    resolved = []
    artifacts = []
    replayed = []
    resolve_options = evaluator.resolve_tracker_options
    resolve_artifact = weights.resolve_edgetam_artifact

    def resolve(args, overrides):
        result = resolve_options(args, overrides)
        resolved.append(result)
        return result

    def artifact(reference):
        artifacts.append(reference)
        return resolve_artifact(reference)

    def replay(build, spec, *, output_dir, **kwargs):
        replayed.append((spec, output_dir))
        assert kwargs.get("mask_guidance_weights") == (checkpoint if guided else None)
        return ReplayResult(build=build, output_dir=output_dir, sequence_files=(), frames=2, track_rows=0)

    monkeypatch.setattr(evaluator, "resolve_tracker_options", resolve)
    monkeypatch.setattr(weights, "resolve_edgetam_artifact", artifact)
    monkeypatch.setattr(evaluator, "replay_build", replay)
    monkeypatch.setattr(evaluator, "run_motmetrics", lambda *_args, **_kwargs: {"HOTA": 50.0})

    results = [
        evaluator.run_eval(options, evolve_config={"track_thresh": threshold}, setup=False, show_progress=False)
        for threshold in (0.3, 0.7)
    ]

    assert len(resolved) == 2
    assert len(artifacts) == (2 if guided else 0)
    assert [spec.option_dict["track_thresh"] for spec, _ in replayed] == [0.3, 0.7]
    assert results[0].exp_dir != results[1].exp_dir
    assert [result.raw for result in results] == [{"HOTA": 50.0}, {"HOTA": 50.0}]
    for (spec, output), result in zip(replayed, results):
        assert output == result.exp_dir
        assert output.parent.name == "trials"
        assert output.name == fingerprint(dict(spec.options))[:16]
        assert (output / "mask-guidance.json").exists() is guided
