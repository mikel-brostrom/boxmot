import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxmot.engine import research as research_module


def test_regression_penalties_reject_negative_values():
    with pytest.raises(ValueError) as exc:
        research_module.RegressionPenalties(idf1_penalty=-1.0)
    assert "must be non-negative" in str(exc.value)


def test_validate_candidate_keys_rejects_missing_or_unexpected():
    with pytest.raises(ValueError) as exc:
        research_module._validate_candidate_keys(
            {"boxmot/trackers/strongsort/strongsort.py": "pass", "unexpected.py": "pass"},
            ("boxmot/trackers/strongsort/strongsort.py",),
        )
    assert "unexpected keys" in str(exc.value)


def test_validate_candidate_keys_preserves_underlying_proposal_text():
    candidate = {
        "boxmot/trackers/bytetrack/bytetrack.py": research_module._ProposalLogText(
            "print('ok')",
            "[summary]",
        )
    }

    validated = research_module._validate_candidate_keys(
        candidate,
        ("boxmot/trackers/bytetrack/bytetrack.py",),
    )

    assert validated["boxmot/trackers/bytetrack/bytetrack.py"] == "print('ok')"


def test_ensure_not_local_gepa_path_rejects_repo_checkout():
    with pytest.raises(RuntimeError) as exc:
        research_module._ensure_not_local_gepa_path(research_module.ROOT / "gepa" / "src" / "gepa")
    assert "local `./gepa` checkout" in str(exc.value)


def test_ensure_not_local_gepa_path_accepts_site_packages():
    research_module._ensure_not_local_gepa_path(Path("/tmp/site-packages/gepa"))


def test_normalize_editable_files_defaults_to_primary_tracker_source():
    files = research_module._normalize_editable_files("strongsort", None)
    assert "boxmot/trackers/bbox/strongsort.py" in files
    assert "boxmot/trackers/bbox/strongsort/__init__.py" not in files
    assert "boxmot/trackers/bbox/strongsort/strongsort_kf.py" not in files
    assert "boxmot/configs/trackers/strongsort.yaml" not in files
    assert all(not Path(path).is_absolute() for path in files)


def test_split_examples_creates_holdout_when_requested():
    examples = [
        {"sequence": "a", "sequence_dir": "/tmp/a"},
        {"sequence": "b", "sequence_dir": "/tmp/b"},
        {"sequence": "c", "sequence_dir": "/tmp/c"},
        {"sequence": "d", "sequence_dir": "/tmp/d"},
    ]
    train, val = research_module._split_examples(examples, validation_split=0.25)
    assert [row["sequence"] for row in train] == ["a", "b", "c"]
    assert [row["sequence"] for row in val] == ["d"]


def test_select_examples_uses_union_of_requested_sequences():
    examples = [
        {"sequence": "a", "sequence_dir": "/tmp/a"},
        {"sequence": "b", "sequence_dir": "/tmp/b"},
        {"sequence": "c", "sequence_dir": "/tmp/c"},
    ]
    selected = research_module._select_examples(examples, train_sequences=("a", "b"), val_sequences=("b", "c"))
    assert [row["sequence"] for row in selected] == ["a", "b", "c"]


def test_research_config_from_namespace_uses_data_when_benchmark_field_is_empty():
    config = research_module.ResearchConfig.from_namespace(
        SimpleNamespace(
            tracker="bytetrack",
            benchmark="",
            data="mot17-mini",
            source=None,
            detector=[Path("yolov8n.pt")],
            reid=[Path("osnet_x0_25_msmt17.pt")],
            detector_explicit=False,
            reid_explicit=False,
        )
    )
    assert config.benchmark == "mot17-mini"
    assert config.progress_bar is True
    assert config.detector is None
    assert config.reid is None


def test_research_config_from_namespace_preserves_explicit_model_overrides():
    config = research_module.ResearchConfig.from_namespace(
        SimpleNamespace(
            tracker="bytetrack",
            benchmark="mot17-mini",
            data="",
            source=None,
            detector=[Path("custom_detector.pt")],
            reid=[Path("custom_reid.pt")],
            detector_explicit=True,
            reid_explicit=True,
        )
    )

    assert config.detector == Path("custom_detector.pt")
    assert config.reid == Path("custom_reid.pt")


def test_research_config_from_namespace_captures_proposal_api_key_settings():
    config = research_module.ResearchConfig.from_namespace(
        SimpleNamespace(
            tracker="bytetrack",
            benchmark="mot17-mini",
            data="",
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


def test_resolve_benchmark_runtime_normalizes_models_to_absolute_paths(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source_dir = tmp_path / "data"
    source_dir.mkdir()

    def fake_apply_benchmark_config(probe):
        probe.source = source_dir
        probe.benchmark_id = "mot17-mini"
        return {"benchmark": {}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(research_module, "apply_benchmark_config", fake_apply_benchmark_config)
    monkeypatch.setattr(
        research_module, "resolve_required_yolo_model", lambda _cfg: Path("models/yolox_x_MOT17_ablation.pt")
    )
    monkeypatch.setattr(research_module, "resolve_required_reid_model", lambda _cfg: Path("models/lmbn_n_duke.pt"))
    monkeypatch.setattr(research_module, "resolve_model_path", lambda path: Path(path))

    source_root, benchmark_id, detector_path, reid_path, cfg = research_module._resolve_benchmark_runtime("mot17-mini")

    assert source_root == source_dir.resolve()
    assert benchmark_id == "mot17-mini"
    assert detector_path == (tmp_path / "models" / "yolox_x_MOT17_ablation.pt").resolve()
    assert reid_path == (tmp_path / "models" / "lmbn_n_duke.pt").resolve()
    assert cfg == {"benchmark": {}}


def test_build_reflection_prompt_templates_embed_objective_and_background():
    templates = research_module._build_reflection_prompt_templates(
        ("boxmot/trackers/bytetrack/bytetrack.py",),
        objective="Improve HOTA.",
        background="Detector: /tmp/yolox.pt\nReID: /tmp/lmbn.pt",
    )

    template = templates["boxmot/trackers/bytetrack/bytetrack.py"]
    assert "Improve HOTA." in template
    assert "Detector: /tmp/yolox.pt" in template
    assert "Prefer algorithmic tracking improvements" in template
    assert "Do not spend a proposal on isolated single-variable" in template
    assert "Do not wrap the response in Markdown fences" in template
    assert "<curr_param>" in template
    assert "<side_info>" in template


def test_normalize_proposed_text_strips_wrapping_code_fence():
    proposed = "```python\nprint('ok')\n```\n"

    normalized = research_module._normalize_proposed_text(proposed, "module.py")

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

    normalized = research_module._normalize_proposed_text(proposed, "module.py")

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

    normalized = research_module._normalize_proposed_text(proposed, "module.py")

    assert normalized == "from x import y\n\ndef main():\n    return 1\n\nmain()"


def test_raw_text_extracts_underlying_proposal_value():
    wrapped = research_module._ProposalLogText("print('ok')", "[summary]")

    assert research_module._raw_text(wrapped) == "print('ok')"


def test_run_instruction_proposal_signature_uses_published_run_api():
    calls = []

    class _Signature:
        @staticmethod
        def run(*, lm, input_dict):
            calls.append((lm, input_dict))
            return {"new_instruction": "updated"}

    result = research_module._run_instruction_proposal_signature(
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

    result = research_module._run_instruction_proposal_signature(
        _Signature,
        lm="lm",
        input_dict={"current_instruction_doc": "x"},
    )

    assert result == {"new_instruction": "updated"}
    assert calls == [("lm", {"current_instruction_doc": "x"})]


def test_proposal_log_text_keeps_full_value_but_renders_compact_summary():
    summary = research_module._proposal_log_summary(
        "boxmot/trackers/bytetrack/bytetrack.py",
        "line1\nline2\n",
        "line1\nline2 changed\nline3\n",
    )
    wrapped = research_module._ProposalLogText("line1\nline2 changed\nline3\n", summary)

    assert str(wrapped).startswith("[applying code modification to bytetrack.py:")
    assert wrapped == "line1\nline2 changed\nline3\n"


def test_build_reflection_lm_uses_published_gepa_factory_when_available(monkeypatch):
    calls = []

    def fake_make_litellm_lm(model_name):
        calls.append(model_name)
        return lambda prompt: "ok"

    monkeypatch.setattr(research_module, "_load_gepa_litellm_factory", lambda: fake_make_litellm_lm)

    lm = research_module._build_reflection_lm("openai/gpt-5.4", {"reasoning_effort": "medium"})

    assert callable(lm)
    assert calls == ["openai/gpt-5.4"]


def test_build_reflection_lm_injects_inferred_provider_api_key_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(research_module, "_load_gepa_litellm_factory", lambda: lambda model_name: model_name)

    lm = research_module._build_reflection_lm("openai/gpt-5.4", {"api_key": "sk-test"})

    assert lm == "openai/gpt-5.4"
    assert os.environ["OPENAI_API_KEY"] == "sk-test"


def test_build_reflection_lm_requires_env_name_for_unknown_provider_api_keys(monkeypatch):
    monkeypatch.setattr(research_module, "_load_gepa_litellm_factory", lambda: lambda model_name: model_name)

    with pytest.raises(ValueError, match="--proposal-api-key-env"):
        research_module._build_reflection_lm("custom/provider-model", {"api_key": "secret"})


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

    monkeypatch.setattr(research_module.subprocess, "Popen", _FakePopen)

    researcher = research_module.TrackerResearcher.__new__(research_module.TrackerResearcher)
    researcher.config = SimpleNamespace(eval_timeout=1.0, progress_bar=True)
    researcher.workspace_dir = Path(".")

    result = researcher._run_eval_subprocess(Path("payload.json"))

    assert result["ok"] is True
    assert result["stderr"] == ""
    assert popen_kwargs["stderr"] is None


def test_build_eval_payload_uses_shared_cache_project():
    researcher = research_module.TrackerResearcher.__new__(research_module.TrackerResearcher)
    researcher.benchmark_id = "mot17-mini"
    researcher.config = SimpleNamespace(tracker="bytetrack")
    researcher.detector_path = Path("/tmp/yolo.pt")
    researcher.reid_path = Path("/tmp/reid.pt")
    researcher.boxmot_project_dir = Path("/tmp/research-run/boxmot_runs")
    researcher.cache_project_dir = Path("/tmp/shared-runs")

    payload = researcher._build_eval_payload(Path("/tmp/source"), "candidate_all_sequences")
    assert payload["project"] == Path("/tmp/research-run/boxmot_runs")
    assert payload["cache_project"] == Path("/tmp/shared-runs")


def test_reset_gepa_run_dir_removes_stale_state(tmp_path):
    researcher = research_module.TrackerResearcher.__new__(research_module.TrackerResearcher)
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
            return {"boxmot/trackers/bytetrack/bytetrack.py": "def broken(:\n"}
        return {"boxmot/trackers/bytetrack/bytetrack.py": "def fixed():\n    return 1\n"}

    proposer = research_module._make_checked_candidate_proposer(
        fake_runner,
        expected_keys=("boxmot/trackers/bytetrack/bytetrack.py",),
        candidate_checker=lambda candidate: [],
        max_attempts=2,
    )

    updates = proposer(
        {"boxmot/trackers/bytetrack/bytetrack.py": "def seed():\n    return 0\n"},
        {"boxmot/trackers/bytetrack/bytetrack.py": [{"Feedback": "improve tracking"}]},
        ["boxmot/trackers/bytetrack/bytetrack.py"],
    )

    assert updates["boxmot/trackers/bytetrack/bytetrack.py"] == "def fixed():\n    return 1\n"
    assert len(attempts) == 2
    retry_feedback = attempts[1]["boxmot/trackers/bytetrack/bytetrack.py"][-1]
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

    monkeypatch.setattr(research_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(research_module.os, "killpg", lambda pid, sig: None)

    researcher = research_module.TrackerResearcher.__new__(research_module.TrackerResearcher)
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

    monkeypatch.setattr(research_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(research_module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    researcher = research_module.TrackerResearcher.__new__(research_module.TrackerResearcher)
    researcher.config = SimpleNamespace(eval_timeout=1.0, progress_bar=False)
    researcher.workspace_dir = Path(".")

    with pytest.raises(KeyboardInterrupt):
        researcher._run_eval_subprocess(Path("payload.json"))

    assert signals == [(456, research_module.signal.SIGTERM)]


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

    monkeypatch.setattr(research_module.subprocess, "Popen", _FakePopen)

    researcher = research_module.TrackerResearcher.__new__(research_module.TrackerResearcher)
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
    researcher = research_module.TrackerResearcher.__new__(research_module.TrackerResearcher)
    researcher.penalties = research_module.RegressionPenalties(
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
    researcher = research_module.TrackerResearcher.__new__(research_module.TrackerResearcher)
    researcher.config = SimpleNamespace(tracker="bytetrack")
    researcher.benchmark_id = "mot17-mini"

    objective = researcher._objective({"HOTA": 60.0, "IDF1": 70.0, "MOTA": 80.0})
    assert "combined benchmark baseline" in objective
    assert "Optimize the combined benchmark HOTA directly" in objective
    assert "penalizing regressions in combined IDF1 and MOTA" in objective


def test_metric_delta_helpers_include_nested_sequence_deltas():
    combined = research_module._metric_delta({"HOTA": 62.0, "IDSW": 5}, {"HOTA": 60.0, "IDSW": 7})
    per_sequence = research_module._nested_metric_delta(
        {"seq-a": {"HOTA": 61.0, "IDSW": 3}},
        {"seq-a": {"HOTA": 59.0, "IDSW": 6}},
    )

    assert combined["HOTA"] == pytest.approx(2.0)
    assert combined["IDSW"] == pytest.approx(-2.0)
    assert per_sequence["seq-a"]["HOTA"] == pytest.approx(2.0)
    assert per_sequence["seq-a"]["IDSW"] == pytest.approx(-3.0)
