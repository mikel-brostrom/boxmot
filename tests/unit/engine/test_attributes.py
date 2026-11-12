"""Background attribute ownership, snapshots, output, and interruption contracts."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, field
from io import StringIO
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from boxmot.attributes.config import resolve_attribute_config
from boxmot.detectors import DetectorCapabilities
from boxmot.engine.cli import boxmot
from boxmot.engine.config.runtime import build_mode_namespace
from boxmot.engine.tracking.attributes import AttributeRun
from boxmot.engine.tracking.runner import TrackingRunner
from boxmot.engine.tracking.sinks import JsonLinesSink, MotSink, _attribute_label_parts, render_result
from boxmot.engine.tracking.workflow import run_track
from boxmot.pipelines import TrackingPipeline
from boxmot.structures import Boxes, Detections, Frame, Tracks
from boxmot.trackers import TrackerRequirements


@dataclass(frozen=True)
class Event:
    track_id: int = 7
    status: str = "completed"
    values: dict = field(default_factory=lambda: {"appearance": {"color": "red"}})
    display_fields: tuple[str, ...] = ("appearance.color",)
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class Enricher:
    def __init__(self, *, ready: bool = False) -> None:
        self.event = Event()
        self.ready = ready
        self.events = []
        self.calls = []
        self.pending_count = 0
        self.closed = False
        self.cancelled = False

    def update(self, frame, tracks):
        self.calls.append((frame.sample_id, tracks.detection_indices.tolist()))
        self.pending_count += 1
        return {7: self.event} if self.ready else {}

    def reset(self):
        self.calls.append("reset")

    def poll(self, limit=100):
        events, self.events = self.events[:limit], self.events[limit:]
        return events

    def finish(self):
        self.calls.append("finish")
        self.pending_count = 0
        yield self.event

    def cancel(self):
        self.cancelled = True
        self.pending_count = 0
        self.events.append(Event(status="cancelled"))

    def close(self):
        self.closed = True


def _frame(index=0, sequence="video"):
    return Frame(torch.zeros((3, 80, 100), dtype=torch.uint8), f"{sequence}/{index}", sequence, index)


def _detections(frame):
    return Detections(
        Boxes(torch.tensor([[10, 20, 70, 75]], dtype=torch.float32)),
        torch.tensor([0.9]), torch.tensor([0]), frame.sample_id,
    )


class Detector:
    capabilities = DetectorCapabilities()

    def predict(self, frames):
        return [_detections(frame) for frame in frames]


class Tracker:
    name = "fixture"
    supports_obb = False
    requirements = TrackerRequirements()

    def reset(self):
        pass

    def update(self, detections, frame=None):
        return Tracks(
            detections.geometry, torch.tensor([7]), detections.scores, detections.class_ids,
            torch.tensor([0]), detections.sample_id,
        )


class Source:
    def __init__(self, frames):
        self.frames = frames
        self.closed = False

    def __iter__(self):
        return iter(self.frames)

    def close(self):
        self.closed = True


def test_pipeline_attributes_are_detached_and_reset_with_tracking() -> None:
    attributes = Enricher(ready=True)
    pipeline = TrackingPipeline(Detector(), Tracker(), attributes=attributes)
    pipeline.reset()
    result = pipeline.step(_frame())
    attributes.event.values["appearance"]["color"] = "blue"
    assert result.attributes[7].values["appearance"]["color"] == "red"
    with pytest.raises(TypeError):
        result.attributes[8] = attributes.event
    assert attributes.calls == ["reset", ("video/0", [0])]
    assert result.tracks.to_aabb_rows().shape == (1, 8)


def test_workflow_persists_results_arriving_after_final_video_frame(tmp_path) -> None:
    attributes = Enricher()
    source = Source([_frame()])
    run = run_track(
        detector=Detector(), tracker=Tracker(), source=source, attributes=attributes,
        project=tmp_path, name="late", geometry="aabb", save_json=True,
    )
    assert run.attributes_path == tmp_path / "late" / "attributes.jsonl"
    assert json.loads(run.attributes_path.read_text())["values"]["appearance"]["color"] == "red"
    assert "attributes" not in json.loads(run.json_path.read_text())["tracks"][0]
    assert attributes.closed and source.closed and not attributes.cancelled
    assert attributes.calls[-1] == "finish"


@pytest.mark.parametrize("reference_kind", ("profile", "yaml", "mapping", "config"))
def test_attribute_host_override_preserves_authored_profile(tmp_path, monkeypatch, reference_kind) -> None:
    """A deployment endpoint replaces only the host passed to the attribute factory."""
    original = resolve_attribute_config("person-appearance-vllm-cpu")
    authored = asdict(original)
    references = {"profile": "person-appearance-vllm-cpu", "mapping": authored, "config": original}
    path = tmp_path / "attributes.yaml"
    path.write_text(json.dumps(authored), encoding="utf-8")
    references["yaml"] = path
    captured = []

    def create(reference, **kwargs):
        captured.append(reference)
        return Enricher()

    monkeypatch.setattr("boxmot.attributes.create_attribute_enricher", create)
    run_track(
        detector=Detector(), tracker=Tracker(), source=Source([_frame()]),
        attributes=references[reference_kind], attributes_host="http://vllm-cpu:8000",
        project=tmp_path, name="override", geometry="aabb",
    )
    assert len(captured) == 1
    assert asdict(captured[0]) == {**authored, "host": "http://vllm-cpu:8000"}
    assert captured[0] is not original
    assert asdict(original) == authored
    assert original.host == "http://127.0.0.1:8001"
    assert json.loads(path.read_text()) == json.loads(json.dumps(authored))


def test_prepared_attribute_host_reaches_vllm_request(tmp_path, monkeypatch, attribute_database) -> None:
    """The CLI namespace endpoint is used by the real backend without a network server."""
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    client_type = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": '{"color":"red"}'},
        }]})

    monkeypatch.setattr("boxmot.attributes.backends.http.require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs),
    )
    args = build_mode_namespace("track", {
        "attributes": {
            "backend": "vllm", "model": "fixture", "prompt": "Describe color.",
            "schema": {"type": "object"}, "host": "http://original:8000", "batch_size": 1,
        },
        "attributes_host": "http://vllm-cpu:8000/v1",
        "project": tmp_path, "name": "request", "geometry": "aabb",
    })
    run = run_track(args, detector=Detector(), tracker=Tracker(), source=Source([_frame()]))
    assert len(requests) == 1
    assert str(requests[0].url) == "http://vllm-cpu:8000/v1/chat/completions"
    assert json.loads(run.attributes_path.read_text())["values"] == {"color": "red"}
    assert args.attributes["host"] == "http://original:8000"


@pytest.mark.parametrize("reference", (None, Enricher()))
def test_attribute_host_requires_a_reconfigurable_profile(reference, tmp_path) -> None:
    """Reject unusable overrides before opening a camera or loading models."""
    with pytest.raises(ValueError, match="attributes_host"):
        run_track(attributes=reference, attributes_host="http://vllm-cpu:8000", project=tmp_path)
    assert list(tmp_path.iterdir()) == []
    if reference is not None:
        assert reference.calls == [] and not reference.closed


@pytest.mark.parametrize("host", ("", "vllm-cpu:8000", "http://vllm-cpu:bad", "ftp://vllm-cpu:8000"))
def test_attribute_host_override_uses_config_validation(host, tmp_path) -> None:
    """The same URL contract applies to authored and deployment endpoints."""
    with pytest.raises(ValueError, match="host"):
        run_track(attributes="person-appearance-vllm-cpu", attributes_host=host, project=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_attribute_status_distinguishes_pending_batches_and_journaled_outcomes(tmp_path) -> None:
    """Report inference progress without treating queued work as completed attributes."""
    attributes = Enricher()
    attributes.pending_count = 3
    with AttributeRun(attributes, tmp_path) as owner:
        assert owner.status_summary == "Attributes • completed analyses: 0 • pending batches: 3 • failed analyses: 0"
        attributes.events.extend([
            Event(), Event(status="failed"), Event(status="cancelled"), Event(status="ineligible"),
        ])
        attributes.pending_count = 1
        owner.poll()
        owner.poll()
        assert owner.status_summary == "Attributes • completed analyses: 1 • pending batches: 1 • failed analyses: 1"
        owner.finish()
        assert owner.status_summary == "Attributes • completed analyses: 2 • pending batches: 0 • failed analyses: 1"


def test_workflow_reports_pending_attributes_and_terminal_outcomes_while_tracking(tmp_path) -> None:
    """Keep extraction progress visible before the first result and on later frames."""

    class ProgressEnricher(Enricher):
        def update(self, frame: Frame, tracks: Tracks) -> dict[int, Event]:
            result = super().update(frame, tracks)
            if frame.frame_index == 1:
                self.pending_count = 1
                self.events.extend([Event(), Event(status="failed")])
            return result

    updates: list[str] = []
    run_track(
        detector=Detector(), tracker=Tracker(), source=Source([_frame(), _frame(1)]),
        attributes=ProgressEnricher(), project=tmp_path, name="progress", geometry="aabb",
        ui_pipeline=SimpleNamespace(update=updates.append, advance=lambda _detail: None),
    )
    frame_updates = [message for message in updates if message.startswith("Processed ")]
    assert frame_updates == [
        "Processed 1 frame(s) • video/0\nAttributes • completed analyses: 0 • pending batches: 1 • failed analyses: 0",
        "Processed 2 frame(s) • video/1\nAttributes • completed analyses: 1 • pending batches: 1 • failed analyses: 1",
    ]


@pytest.mark.parametrize("delivery", ("poll", "finish"))
@pytest.mark.parametrize("error", ("ReadError: server closed the connection", None))
def test_attribute_failures_warn_with_track_and_journal_path(tmp_path, monkeypatch, delivery, error) -> None:
    """Expose backend failures both during preview and while draining late results."""
    warnings = []
    monkeypatch.setattr(
        "boxmot.engine.tracking.attributes.LOGGER.warning",
        lambda message, *args: warnings.append(message % args),
    )
    attributes = Enricher()
    event = Event(status="failed", error=error)
    with AttributeRun(attributes, tmp_path) as owner:
        if delivery == "poll":
            attributes.events.append(event)
            owner.poll()
        else:
            attributes.event = event
            owner.finish()
        assert len(warnings) == 1
        assert "track 7" in warnings[0]
        assert str(owner.path) in warnings[0]
        assert (error or "Attribute backend did not provide an error message.") in warnings[0]
        assert json.loads(owner.path.read_text())["status"] == "failed"


def test_attribute_failure_warnings_suppress_repeats_until_error_changes_or_recovers(tmp_path, monkeypatch) -> None:
    """Keep every outcome while warning once per unchanged failure across tracks."""
    warnings = []
    monkeypatch.setattr(
        "boxmot.engine.tracking.attributes.LOGGER.warning",
        lambda message, *args: warnings.append(message % args),
    )
    attributes = Enricher()
    events = [
        Event(status="failed", error="ReadError"),
        Event(track_id=8, status="failed", error="ReadError"),
        Event(status="failed", error="Request timed out"),
        Event(),
        Event(status="failed", error="Request timed out"),
        Event(track_id=8, status="failed", error="Request timed out"),
    ]
    with AttributeRun(attributes, tmp_path) as owner:
        for event in events:
            attributes.events.append(event)
            owner.poll()
        assert len(warnings) == 3
        assert "ReadError" in warnings[0]
        assert all("Request timed out" in message for message in warnings[1:])
        assert [json.loads(line) for line in owner.path.read_text().splitlines()] == [
            event.to_dict() | {"display_fields": list(event.display_fields)} for event in events
        ]


def test_real_background_session_drains_short_tracks_across_sequences(tmp_path, attribute_database) -> None:
    """Exercise the real PostgreSQL/worker boundary without model servers or weights."""
    from boxmot.attributes import AttributeConfig, create_attribute_enricher

    class Backend:
        closed = False

        async def extract(self, track_id, samples, images):
            assert track_id == 7 and len(samples) == len(images) == 1
            return {"color": "red"}

        async def aclose(self):
            self.closed = True

    backend = Backend()
    attributes = create_attribute_enricher(
        AttributeConfig(model="fixture", prompt="Describe color", schema={"type": "object"}),
        backend=backend,
    )
    run = run_track(
        detector=Detector(), tracker=Tracker(), source=Source([_frame(sequence="first"), _frame(sequence="second")]),
        attributes=attributes, project=tmp_path, name="real", geometry="aabb",
    )
    outcomes = [json.loads(line) for line in run.attributes_path.read_text().splitlines()]
    assert len(outcomes) == 2
    assert {row["sequence_id"] for row in outcomes} == {"first", "second"}
    assert len({row["generation"] for row in outcomes}) == 2
    assert all(row["status"] == "completed" and row["values"] == {"color": "red"} for row in outcomes)
    assert backend.closed


def test_closing_runner_generator_cancels_and_journals_pending_work(tmp_path) -> None:
    attributes = Enricher()
    owner = AttributeRun(attributes, tmp_path)
    source = Source([_frame(), _frame(1)])
    runner = TrackingRunner(
        source, TrackingPipeline(Detector(), Tracker(), attributes=attributes), attribute_run=owner,
    )
    iterator = runner.run()
    next(iterator)
    iterator.close()
    assert attributes.cancelled and attributes.closed and source.closed
    assert runner.summary.interrupted
    assert json.loads(owner.path.read_text())["status"] == "cancelled"
    assert "finish" not in attributes.calls


def test_attribute_failure_closes_session_and_source(tmp_path) -> None:
    attributes = Enricher()

    def fail(*args):
        raise OSError("spool full")

    attributes.update = fail
    source = Source([_frame()])
    with pytest.raises(OSError, match="spool full"):
        run_track(
            detector=Detector(), tracker=Tracker(), source=source, attributes=attributes,
            project=tmp_path, name="failed", geometry="aabb",
        )
    assert attributes.cancelled and attributes.closed and source.closed


def test_journal_open_failure_closes_session_even_when_cancellation_fails(tmp_path, monkeypatch) -> None:
    """A full output disk must not prevent cleanup of the background worker."""
    attributes = Enricher()

    def fail_open(*args, **kwargs):
        raise OSError("attribute journal could not be opened")

    def fail_cancel():
        attributes.cancelled = True
        raise OSError("cancellation journal could not be written")

    monkeypatch.setattr(type(tmp_path), "open", fail_open)
    monkeypatch.setattr(attributes, "cancel", fail_cancel)
    with pytest.raises(OSError, match="cancellation journal"):
        AttributeRun(attributes, tmp_path)
    assert attributes.cancelled and attributes.closed


def test_json_snapshots_carry_attributes_without_changing_mot_rows() -> None:
    frame = _frame()
    pipeline = TrackingPipeline(Detector(), Tracker(), attributes=Enricher(ready=True))
    result = pipeline.step(frame)
    json_output, mot_output = StringIO(), StringIO()
    JsonLinesSink(json_output).write(frame, result)
    MotSink(mot_output).write(frame, result)
    assert json.loads(json_output.getvalue())["tracks"][0]["attributes"]["values"] == {
        "appearance": {"color": "red"}
    }
    assert len(mot_output.getvalue().strip().split(",")) == 9


def test_selected_label_fields_and_show_labels(monkeypatch) -> None:
    frame = _frame()
    result = TrackingPipeline(Detector(), Tracker(), attributes=Enricher(ready=True)).step(frame)
    assert _attribute_label_parts(result.attributes[7]) == ["appearance.color: red"]
    assert _attribute_label_parts(Event(status="failed")) == []
    assert _attribute_label_parts(Event(display_fields=("missing", "appearance"))) == []
    labels = []
    monkeypatch.setattr("boxmot.engine.tracking.sinks._draw_label", lambda *args, **kwargs: labels.append(kwargs))
    render_result(frame, result, show_labels=False)
    assert labels == []
    render_result(frame, result)
    assert labels[0]["attribute"].values == result.attributes[7].values


@pytest.mark.parametrize("tracker", ("bytetrack", "edgetam"))
@pytest.mark.parametrize("host", (None, "http://vllm-cpu:8000"))
def test_cli_dispatches_attribute_profile_for_both_tracking_workflows(monkeypatch, tracker, host) -> None:
    module = importlib.import_module("boxmot.engine.commands.track")
    captured = {}
    monkeypatch.setattr(module, "_dispatch_cli_workflow", lambda ctx, mode, target, payload: captured.update(payload))
    command = ["track", "--tracker", tracker, "--attributes", "custom.yaml"]
    if host is not None:
        command.extend(["--attributes-host", host])
    result = CliRunner().invoke(boxmot, command)
    assert result.exit_code == 0, result.output
    assert captured["attributes"] == "custom.yaml"
    assert captured["attributes_host"] == host


def test_cli_rejects_attribute_host_without_profile_and_documents_override(monkeypatch) -> None:
    """An ineffective override is rejected before dispatch and its dependency appears in help."""
    module = importlib.import_module("boxmot.engine.commands.track")
    dispatched = []
    monkeypatch.setattr(module, "_dispatch_cli_workflow", lambda *args: dispatched.append(args))
    result = CliRunner().invoke(boxmot, ["track", "--attributes-host", "http://vllm-cpu:8000"])
    assert result.exit_code == 2
    assert "--attributes-host requires --attributes" in result.output
    assert dispatched == []
    help_result = CliRunner().invoke(boxmot, ["track", "--help"])
    assert help_result.exit_code == 0
    assert "--attributes-host URL" in help_result.output
