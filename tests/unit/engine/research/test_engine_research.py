import json
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import boxmot.engine.research.benchmarks as benchmarks_module
import boxmot.engine.research.paths as paths_module
import boxmot.engine.research.proposal as proposal_module
import boxmot.engine.research.runner as runner_module
from boxmot.engine.research.benchmarks import (
    _discover_sequences,
    _resolve_experiment_runtime,
    _select_examples,
    _split_examples,
)
from boxmot.engine.research.candidates import (
    _build_reflection_prompt_templates,
    _make_checked_candidate_proposer,
    _normalize_editable_files,
    _normalize_proposed_text,
    _proposal_log_summary,
    _ProposalLogText,
    _raw_text,
    _validate_candidate_keys,
)
from boxmot.engine.research.metrics import _metric_delta, _nested_metric_delta
from boxmot.engine.research.models import RegressionPenalties, ResearchConfig
from boxmot.engine.research.proposal import (
    _build_reflection_lm,
    _ensure_not_local_gepa_path,
    _run_instruction_proposal_signature,
)
from boxmot.engine.research.runner import TrackerResearcher
from boxmot.utils import ROOT


def test_regression_penalties_reject_negative_values():
    with pytest.raises(ValueError) as exc:
        RegressionPenalties(idf1_penalty=-1.0)
    assert "must be non-negative" in str(exc.value)


def test_validate_candidate_keys_rejects_missing_or_unexpected():
    with pytest.raises(ValueError) as exc:
        _validate_candidate_keys(
            {"boxmot/trackers/box/strongsort/tracker.py": "pass", "unexpected.py": "pass"},
            ("boxmot/trackers/box/strongsort/tracker.py",),
        )
    assert "unexpected keys" in str(exc.value)


def test_validate_candidate_keys_preserves_underlying_proposal_text():
    candidate = {
        "boxmot/trackers/box/bytetrack/tracker.py": _ProposalLogText(
            "print('ok')",
            "[summary]",
        )
    }

    validated = _validate_candidate_keys(
        candidate,
        ("boxmot/trackers/box/bytetrack/tracker.py",),
    )

    assert validated["boxmot/trackers/box/bytetrack/tracker.py"] == "print('ok')"


def test_ensure_not_local_gepa_path_rejects_repo_checkout():
    with pytest.raises(RuntimeError) as exc:
        _ensure_not_local_gepa_path(ROOT / "gepa" / "src" / "gepa")
    assert "local `./gepa` checkout" in str(exc.value)


def test_ensure_not_local_gepa_path_accepts_site_packages():
    _ensure_not_local_gepa_path(Path("/tmp/site-packages/gepa"))


@pytest.mark.parametrize(
    ("tracker", "expected_file"),
    (
        ("strongsort", "boxmot/trackers/box/strongsort/tracker.py"),
        ("maf_hda", "boxmot/trackers/multimodal/maf_hda/tracker.py"),
        ("eagermot", "boxmot/trackers/multimodal/eagermot/tracker.py"),
    ),
)
def test_normalize_editable_files_defaults_to_registered_tracker_source(tracker, expected_file):
    files = _normalize_editable_files(tracker, None)

    assert files == (expected_file,)
    assert all(not Path(path).is_absolute() for path in files)


def test_normalize_editable_files_rejects_unregistered_tracker():
    with pytest.raises(ValueError, match="Unknown tracker type"):
        _normalize_editable_files("unknown", None)


def test_split_examples_creates_holdout_when_requested():
    examples = [
        {"sequence": "a", "sequence_dir": "/tmp/a"},
        {"sequence": "b", "sequence_dir": "/tmp/b"},
        {"sequence": "c", "sequence_dir": "/tmp/c"},
        {"sequence": "d", "sequence_dir": "/tmp/d"},
    ]
    train, val = _split_examples(examples, validation_split=0.25)
    assert [row["sequence"] for row in train] == ["a", "b", "c"]
    assert [row["sequence"] for row in val] == ["d"]


def test_discover_sequences_ignores_appledouble_only_directories(tmp_path):
    valid = tmp_path / "valid" / "img1"
    valid.mkdir(parents=True)
    (valid / "000001.jpg").write_bytes(b"candidate frame")
    sidecar_only = tmp_path / "sidecar-only" / "img1"
    sidecar_only.mkdir(parents=True)
    (sidecar_only / "._000001.jpg").write_bytes(b"AppleDouble metadata")

    examples = _discover_sequences(tmp_path)

    assert [example["sequence"] for example in examples] == ["valid"]


def test_select_examples_uses_union_of_requested_sequences():
    examples = [
        {"sequence": "a", "sequence_dir": "/tmp/a"},
        {"sequence": "b", "sequence_dir": "/tmp/b"},
        {"sequence": "c", "sequence_dir": "/tmp/c"},
    ]
    selected = _select_examples(examples, train_sequences=("a", "b"), val_sequences=("b", "c"))
    assert [row["sequence"] for row in selected] == ["a", "b", "c"]


def test_research_config_from_namespace_uses_experiment_selector():
    config = ResearchConfig.from_namespace(
        SimpleNamespace(
            tracker="bytetrack",
            experiment="mot17-mini",
            build="build-id",
            source=None,
            detector=[Path("yolov8n.pt")],
            reid=[Path("osnet_x0_25_msmt17.pt")],
            detector_explicit=False,
            reid_explicit=False,
        )
    )
    assert config.experiment == "mot17-mini"
    assert config.build == "build-id"
    assert config.progress_bar is True
    assert not hasattr(config, "detector")
    assert not hasattr(config, "reid")


def test_research_config_from_namespace_ignores_removed_model_overrides():
    config = ResearchConfig.from_namespace(
        SimpleNamespace(
            tracker="bytetrack",
            experiment="mot17-mini",
            build="build-id",
            source=None,
            detector=[Path("custom_detector.pt")],
            reid=[Path("custom_reid.pt")],
            detector_explicit=True,
            reid_explicit=True,
        )
    )

    assert config.build == "build-id"
    assert not hasattr(config, "detector")
    assert not hasattr(config, "reid")


def test_research_config_from_namespace_captures_proposal_api_key_settings():
    config = ResearchConfig.from_namespace(
        SimpleNamespace(
            tracker="bytetrack",
            experiment="mot17-mini",
            source=None,
            detector=[Path("yolov8n.pt")],
            reid=[Path("osnet_x0_25_msmt17.pt")],
            detector_explicit=False,
            reid_explicit=False,
            proposal_model="anthropic/claude-sonnet-4-20250514",
            proposal_api_key="anthropic-secret",
            proposal_api_key_env="ANTHROPIC_API_KEY",
        )
    )

    assert config.proposal_model == "anthropic/claude-sonnet-4-20250514"
    assert config.proposal_model_kwargs["api_key"] == "anthropic-secret"
    assert config.proposal_model_kwargs["api_key_env"] == "ANTHROPIC_API_KEY"
    assert config.proposal_model_kwargs["reasoning_effort"] == "medium"


def test_resolve_experiment_runtime_resolves_only_dataset_identity(monkeypatch, tmp_path):
    source_dir = tmp_path / "data"
    source_dir.mkdir()

    monkeypatch.setattr(
        benchmarks_module,
        "resolve_experiment_config",
        lambda *_args, **_kwargs: {
            "id": "mot17-mini",
            "dataset": {"id": "mot17", "split_path": "data"},
            "benchmark": {},
        },
    )
    monkeypatch.setattr(benchmarks_module, "resolve_dataset_root", lambda *_args: source_dir.parent)

    source_root, experiment_id, dataset_id, benchmark, cfg = _resolve_experiment_runtime("mot17-mini")

    assert source_root == source_dir
    assert experiment_id == "mot17-mini"
    assert dataset_id == "mot17"
    assert benchmark == "mot17"
    assert cfg["benchmark"] == {}


def test_build_reflection_prompt_templates_embed_objective_and_background():
    templates = _build_reflection_prompt_templates(
        ("boxmot/trackers/box/bytetrack/tracker.py",),
        objective="Improve HOTA.",
        background="Detector: /tmp/yolox.pt\nReID: /tmp/lmbn.pt",
    )

    template = templates["boxmot/trackers/box/bytetrack/tracker.py"]
    assert "Improve HOTA." in template
    assert "Detector: /tmp/yolox.pt" in template
    assert "Prefer algorithmic tracking improvements" in template
    assert "Do not spend a proposal on isolated single-variable" in template
    assert "Do not wrap the response in Markdown fences" in template
    assert "<curr_param>" in template
    assert "<side_info>" in template


def test_normalize_proposed_text_strips_wrapping_code_fence():
    proposed = "```python\nprint('ok')\n```\n"

    normalized = _normalize_proposed_text(proposed, "module.py")

    assert normalized == "print('ok')"


def test_normalize_proposed_text_extracts_code_block_from_chatty_response():
    proposed = (
        "Here is the updated file.\n\n"
        "```python\n"
        "from x import y\n"
        "print('ok')\n"
        "```\n\n"
        "This version keeps the API stable.\n"
    )

    normalized = _normalize_proposed_text(proposed, "module.py")

    assert normalized == "from x import y\nprint('ok')"


def test_normalize_proposed_text_recovers_unfenced_python_from_chatty_response():
    proposed = (
        "Updated file below.\n\n"
        "from x import y\n"
        "\n"
        "def main():\n"
        "    return 1\n"
        "\n"
        "main()\n"
        "\n"
        "Explanation: I kept the API stable.\n"
    )

    normalized = _normalize_proposed_text(proposed, "module.py")

    assert normalized == "from x import y\n\ndef main():\n    return 1\n\nmain()"


def test_raw_text_extracts_underlying_proposal_value():
    wrapped = _ProposalLogText("print('ok')", "[summary]")

    assert _raw_text(wrapped) == "print('ok')"


def test_run_instruction_proposal_signature_uses_published_run_api():
    calls = []

    class _Signature:
        @staticmethod
        def run(*, lm, input_dict):
            calls.append((lm, input_dict))
            return {"new_instruction": "updated"}

    result = _run_instruction_proposal_signature(
        _Signature,
        lm="lm",
        input_dict={"current_instruction_doc": "x"},
    )

    assert result == {"new_instruction": "updated"}
    assert calls == [("lm", {"current_instruction_doc": "x"})]


def test_run_instruction_proposal_signature_prefers_run_with_metadata():
    calls = []

    class _Signature:
        @staticmethod
        def run_with_metadata(*, lm, input_dict):
            calls.append((lm, input_dict))
            return {"new_instruction": "updated"}, {"prompt": "p"}, {"raw": "r"}

    result = _run_instruction_proposal_signature(
        _Signature,
        lm="lm",
        input_dict={"current_instruction_doc": "x"},
    )

    assert result == {"new_instruction": "updated"}
    assert calls == [("lm", {"current_instruction_doc": "x"})]


def test_proposal_log_text_keeps_full_value_but_renders_compact_summary():
    summary = _proposal_log_summary(
        "boxmot/trackers/box/bytetrack/tracker.py",
        "line1\nline2\n",
        "line1\nline2 changed\nline3\n",
    )
    wrapped = _ProposalLogText("line1\nline2 changed\nline3\n", summary)

    assert str(wrapped).startswith("[applying code modification to tracker.py:")
    assert wrapped == "line1\nline2 changed\nline3\n"


def test_build_reflection_lm_uses_published_gepa_factory_when_available(monkeypatch):
    calls = []

    def fake_make_litellm_lm(model_name):
        calls.append(model_name)
        return lambda prompt: "ok"

    monkeypatch.setattr(proposal_module, "_load_gepa_litellm_factory", lambda: fake_make_litellm_lm)

    lm = _build_reflection_lm("openai/gpt-5.4", {"reasoning_effort": "medium"})

    assert callable(lm)
    assert calls == ["openai/gpt-5.4"]


def test_build_reflection_lm_injects_inferred_provider_api_key_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(proposal_module, "_load_gepa_litellm_factory", lambda: lambda model_name: model_name)

    lm = _build_reflection_lm("openai/gpt-5.4", {"api_key": "sk-test"})

    assert lm == "openai/gpt-5.4"
    assert os.environ["OPENAI_API_KEY"] == "sk-test"


def test_build_reflection_lm_requires_env_name_for_unknown_provider_api_keys(monkeypatch):
    monkeypatch.setattr(proposal_module, "_load_gepa_litellm_factory", lambda: lambda model_name: model_name)

    with pytest.raises(ValueError, match="--proposal-api-key-env"):
        _build_reflection_lm("custom/provider-model", {"api_key": "secret"})


def test_run_eval_subprocess_streams_stderr_when_progress_bar_enabled(monkeypatch):
    popen_kwargs = {}

    class _FakePopen:
        def __init__(self, *_args, **kwargs):
            popen_kwargs.update(kwargs)
            self.returncode = 0
            self.pid = 1234

        def communicate(self, timeout=None):
            _ = timeout
            return ('{"ok": true, "summary": {"HOTA": 1.0}}', None)

    monkeypatch.setattr(runner_module.subprocess, "Popen", _FakePopen)

    researcher = TrackerResearcher.__new__(TrackerResearcher)
    researcher.config = SimpleNamespace(eval_timeout=1.0, progress_bar=True)
    researcher.workspace_dir = Path(".")

    result = researcher._run_eval_subprocess(Path("payload.json"))

    assert result["ok"] is True
    assert result["stderr"] == ""
    assert popen_kwargs["stderr"] is None


def test_build_eval_payload_uses_explicit_materialized_build():
    researcher = TrackerResearcher.__new__(TrackerResearcher)
    researcher.config = SimpleNamespace(
        tracker="bytetrack",
        tracker_backend="python",
        experiment="mot17-mini",
        data_root=None,
        progress_bar=True,
    )
    researcher.build_path = Path("/tmp/build-id")
    researcher._evaluation_context = {
        "seq_paths": [Path("/tmp/source/a/img1"), Path("/tmp/source/b/img1")],
        "seq_info": {"a": 10, "b": 12},
        "dataset_id": "mot17",
    }
    researcher.boxmot_project_dir = Path("/tmp/research-run/boxmot_runs")

    payload = researcher._build_eval_payload(("a",), "candidate_all_sequences")
    assert payload["experiment"] == "mot17-mini"
    assert payload["dataset_id"] == "mot17"
    assert payload["build"] == "/tmp/build-id"
    assert payload["sequence_names"] == ["a"]
    assert payload["seq_info"] == {"a": 10}
    assert payload["project"] == Path("/tmp/research-run/boxmot_runs")


def test_reset_gepa_run_dir_removes_stale_state(tmp_path):
    researcher = TrackerResearcher.__new__(TrackerResearcher)
    researcher.gepa_run_dir = tmp_path / "gepa"
    researcher.gepa_run_dir.mkdir(parents=True)
    stale_file = researcher.gepa_run_dir / "gepa_state.bin"
    stale_file.write_text("stale", encoding="utf-8")

    researcher._reset_gepa_run_dir()

    assert researcher.gepa_run_dir.exists()
    assert not stale_file.exists()


def test_checked_candidate_proposer_retries_invalid_candidate_before_returning():
    attempts = []

    def fake_runner(_candidate, reflective_dataset, _components_to_update):
        attempts.append(reflective_dataset)
        if len(attempts) == 1:
            return {"boxmot/trackers/box/bytetrack/tracker.py": "def broken(:\n"}
        return {"boxmot/trackers/box/bytetrack/tracker.py": "def fixed():\n    return 1\n"}

    proposer = _make_checked_candidate_proposer(
        fake_runner,
        expected_keys=("boxmot/trackers/box/bytetrack/tracker.py",),
        candidate_checker=lambda candidate: [],
        max_attempts=2,
    )

    updates = proposer(
        {"boxmot/trackers/box/bytetrack/tracker.py": "def seed():\n    return 0\n"},
        {"boxmot/trackers/box/bytetrack/tracker.py": [{"Feedback": "improve tracking"}]},
        ["boxmot/trackers/box/bytetrack/tracker.py"],
    )

    assert updates["boxmot/trackers/box/bytetrack/tracker.py"] == "def fixed():\n    return 1\n"
    assert len(attempts) == 2
    retry_feedback = attempts[1]["boxmot/trackers/box/bytetrack/tracker.py"][-1]
    assert "Rejected Proposal Errors" in retry_feedback


def test_run_eval_subprocess_timeout_returns_failure(monkeypatch):
    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.pid = 123
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(cmd="python", timeout=timeout)
            return ("", "")

    monkeypatch.setattr(runner_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(paths_module.os, "killpg", lambda pid, sig: None)

    researcher = TrackerResearcher.__new__(TrackerResearcher)
    researcher.config = SimpleNamespace(eval_timeout=0.1)
    researcher.workspace_dir = Path(".")

    result = researcher._run_eval_subprocess(Path("payload.json"))
    assert result["ok"] is False
    assert "timed out" in result["error"]
    assert result["summary"] == {"HOTA": 0.0, "IDF1": 0.0, "MOTA": 0.0}
    assert result["summary_label"] == ""
    assert result["per_sequence_metrics"] == {}
    assert result["per_class_metrics"] == {}


def test_run_eval_subprocess_keyboard_interrupt_terminates_process_group(monkeypatch):
    signals = []

    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.pid = 456
            self.calls = 0

        def communicate(self, timeout=None):
            _ = timeout
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return ("", "")

    monkeypatch.setattr(runner_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(paths_module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    researcher = TrackerResearcher.__new__(TrackerResearcher)
    researcher.config = SimpleNamespace(eval_timeout=1.0, progress_bar=False)
    researcher.workspace_dir = Path(".")

    with pytest.raises(KeyboardInterrupt):
        researcher._run_eval_subprocess(Path("payload.json"))

    assert signals == [(456, signal.SIGTERM)]


def test_run_eval_subprocess_preserves_rich_mot_feedback(monkeypatch):
    payload = {
        "ok": True,
        "summary_label": "all",
        "summary": {"HOTA": 61.0, "IDF1": 62.0, "MOTA": 63.0, "CLR_TP": 120},
        "per_sequence_metrics": {"MOT17-02": {"HOTA": 60.0, "IDSW": 4}},
        "per_class_metrics": {"all": {"HOTA": 61.0, "CLR_TP": 120}},
    }

    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.pid = 123
            self.returncode = 0

        def communicate(self, timeout=None):
            _ = timeout
            return (json.dumps(payload), "")

    monkeypatch.setattr(runner_module.subprocess, "Popen", _FakePopen)

    researcher = TrackerResearcher.__new__(TrackerResearcher)
    researcher.config = SimpleNamespace(eval_timeout=1.0)
    researcher.workspace_dir = Path(".")

    result = researcher._run_eval_subprocess(Path("payload.json"))

    assert result["ok"] is True
    assert result["summary_label"] == "all"
    assert result["summary"]["HOTA"] == 61.0
    assert result["summary"]["CLR_TP"] == 120
    assert result["per_sequence_metrics"]["MOT17-02"]["IDSW"] == 4
    assert result["per_class_metrics"]["all"]["CLR_TP"] == 120


def test_score_candidate_uses_hota_minus_regression_penalties():
    researcher = TrackerResearcher.__new__(TrackerResearcher)
    researcher.penalties = RegressionPenalties(
        idf1_penalty=1.0,
        mota_penalty=1.0,
        idf1_tolerance=0.0,
        mota_tolerance=0.0,
    )
    researcher.baseline_summary = {"HOTA": 60.0, "IDF1": 70.0, "MOTA": 80.0}

    score, breakdown = researcher._score_candidate({"HOTA": 61.0, "IDF1": 69.0, "MOTA": 79.0})
    assert score == pytest.approx(59.0)
    assert breakdown["idf1_regression"] == pytest.approx(1.0)
    assert breakdown["mota_regression"] == pytest.approx(1.0)
    assert breakdown["total_penalty"] == pytest.approx(2.0)


def test_objective_targets_combined_benchmark_metrics():
    researcher = TrackerResearcher.__new__(TrackerResearcher)
    researcher.config = SimpleNamespace(tracker="bytetrack")
    researcher.experiment_id = "mot17-mini"
    researcher.benchmark = "mot17"

    objective = researcher._objective({"HOTA": 60.0, "IDF1": 70.0, "MOTA": 80.0})
    assert "combined benchmark baseline" in objective
    assert "Optimize the combined benchmark HOTA directly" in objective
    assert "penalizing regressions in combined IDF1 and MOTA" in objective


def test_metric_delta_helpers_include_nested_sequence_deltas():
    combined = _metric_delta({"HOTA": 62.0, "IDSW": 5}, {"HOTA": 60.0, "IDSW": 7})
    per_sequence = _nested_metric_delta(
        {"seq-a": {"HOTA": 61.0, "IDSW": 3}},
        {"seq-a": {"HOTA": 59.0, "IDSW": 6}},
    )

    assert combined["HOTA"] == pytest.approx(2.0)
    assert combined["IDSW"] == pytest.approx(-2.0)
    assert per_sequence["seq-a"]["HOTA"] == pytest.approx(2.0)
    assert per_sequence["seq-a"]["IDSW"] == pytest.approx(-3.0)
