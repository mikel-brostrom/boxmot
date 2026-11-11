"""Interactive workflow lifecycle and prompt ordering with model-free doubles."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from boxmot.engine.tracking import prompted
from boxmot.engine.ui.prompts import ObjectPrompt, PromptController, PromptViewport, PromptWindow
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, Tracks


def _frame(
    index: int, *, sequence: str = "video", width: int = 20, height: int = 16, source_uri: str | None = None
) -> Frame:
    return Frame(
        torch.zeros((3, height, width), dtype=torch.uint8),
        sample_id=f"{sequence}/{index}",
        sequence_id=sequence,
        frame_index=index,
        source_uri=source_uri,
    )


class Source:
    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames
        self.closed = False
        self.yielded: list[str] = []

    def __iter__(self):
        for frame in self.frames:
            self.yielded.append(frame.sample_id)
            yield frame

    def close(self) -> None:
        self.closed = True


class Tracker:
    config = SimpleNamespace(max_objects=8)

    def __init__(self) -> None:
        self.objects: set[int] = set()
        self.events: list[tuple] = []
        self.fail = False

    def reset(self) -> None:
        self.events.append(("reset",))
        self.objects.clear()

    def add_box(self, box: np.ndarray, *, object_id: int) -> int:
        self.events.append(("box", object_id, tuple(box)))
        self.objects.add(object_id)
        return object_id

    def add_points(self, points: np.ndarray, *, labels: np.ndarray, object_id: int) -> int:
        self.events.append(("points", object_id, points.tolist(), labels.tolist()))
        self.objects.add(object_id)
        return object_id

    def remove_object(self, object_id: int) -> None:
        self.events.append(("remove", object_id))
        self.objects.remove(object_id)

    def update(self, detections: Detections, *, frame: Frame) -> Tracks:
        self.events.append(("update", frame.sample_id))
        assert len(detections) == 0
        assert detections.sample_id == frame.sample_id
        if self.fail:
            raise RuntimeError("inference failed")
        ids = sorted(self.objects)
        masks = torch.zeros((len(ids), frame.height, frame.width), dtype=torch.bool)
        masks[:, 2:8, 2:8] = True
        return Tracks(
            Boxes(torch.tensor([[2, 2, 8, 8]] * len(ids), dtype=torch.float32).reshape(-1, 4)),
            torch.tensor(ids, dtype=torch.int64),
            torch.ones(len(ids), dtype=torch.float32),
            torch.zeros(len(ids), dtype=torch.int64),
            torch.full((len(ids),), -1, dtype=torch.int64),
            frame.sample_id,
            masks=MaskBatch(masks),
        )


class Sink:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.results: list[Tracks] = []
        self.closed = False

    def write(self, *_args) -> None:
        pytest.fail("rendered output should be reused")

    def write_rendered(self, frame, result, image) -> None:
        assert result.tracks.masks is not None
        self.images.append(image)
        self.results.append(result.tracks)

    def close(self) -> None:
        self.closed = True


class Window:
    def __init__(self) -> None:
        self.controller = PromptController()
        self.images: list[np.ndarray] = []
        self.prepared: list[str] = []
        self.closed = False
        self.script = lambda _frame: None

    def open(self) -> None:
        return None

    def prepare_frame(self, frame: Frame) -> bool:
        self.prepared.append(frame.sample_id)
        self.script(frame)
        return not self.controller.quit

    def show_result(self, image: np.ndarray) -> None:
        self.images.append(image)

    def close(self) -> None:
        self.closed = True


def test_prompts_precede_updates_and_correct_existing_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    window, tracker, sink = Window(), Tracker(), Sink()
    source = Source([_frame(0), _frame(1), _frame(2)])

    def edits(frame: Frame) -> None:
        if frame.frame_index == 0:
            window.controller.pending = {
                1: ObjectPrompt(box=(2, 2, 8, 8)),
                2: ObjectPrompt(points=((10, 8), (12, 9)), labels=(1, 0)),
            }
        elif frame.frame_index == 1:
            window.controller.pending = {1: ObjectPrompt(points=((4, 4),), labels=(1,))}
        else:
            window.controller.removed.add(1)

    window.script = edits
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    updates: list[str] = []
    advances: list[str] = []
    pipeline = SimpleNamespace(update=updates.append, advance=advances.append)
    run = prompted.run_prompted_track(
        SimpleNamespace(),
        tracker=tracker,
        source=source,
        sinks=(sink,),
        ui_pipeline=pipeline,
    )

    assert [event[:2] for event in tracker.events] == [
        ("reset",),
        ("box", 1),
        ("points", 2),
        ("update", "video/0"),
        ("points", 1),
        ("update", "video/1"),
        ("remove", 1),
        ("update", "video/2"),
        ("reset",),
    ]
    assert run.summary.frames == 3 and run.summary.track_rows == 5
    assert run.summary.detections == 0 and run.summary.unique_track_ids == 2
    assert run.summary.interrupted is False
    assert source.closed and sink.closed and window.closed
    assert len(window.images) == len(sink.images) == 3
    assert all(first is second for first, second in zip(window.images, sink.images))
    assert advances == ["Select objects in the video window…"]
    assert updates[-1] == "Processed 3 frame(s) • video/2"


def test_boxed_object_resumes_after_only_exclude_clicks_in_real_prompt_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, tracker, sink = PromptWindow(), Tracker(), Sink()
    source = Source([_frame(0), _frame(1), _frame(2)])
    delays: list[int] = []
    closed_windows: list[str] = []
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    monkeypatch.setattr(window, "open", lambda: setattr(window, "_opened", True))
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(cv2, "destroyWindow", closed_windows.append)

    def mouse(event: int, point: tuple[int, int]) -> None:
        viewport = window.controller.viewport
        assert viewport is not None
        window._mouse(event, *viewport.display_point(point), 0, None)

    def key(delay: int) -> int:
        delays.append(delay)
        expected_delays = [30, 1, 30, 1, 1]
        assert len(delays) <= len(expected_delays), "The correction never left the paused frame"
        assert delay == expected_delays[len(delays) - 1], "Resume must leave the correction editor"
        if len(delays) == 1:
            assert source.yielded == ["video/0"]
            mouse(cv2.EVENT_LBUTTONDOWN, (2, 2))
            mouse(cv2.EVENT_LBUTTONUP, (8, 8))
            mouse(cv2.EVENT_RBUTTONDOWN, (3, 3))
            assert window.controller.mode == "box"
            assert window.controller.pending[1].box == (2, 2, 8, 8)
            assert window.controller.pending[1].labels == (0,)
            return 13
        if len(delays) == 2:
            assert window.controller.active_ids == {1}
            assert not window.controller.pending
            return 32
        if len(delays) == 3:
            assert source.yielded == ["video/0", "video/1"]
            window.controller.handle_key(ord("p"))
            mouse(cv2.EVENT_RBUTTONDOWN, (4, 4))
            mouse(cv2.EVENT_RBUTTONDOWN, (5, 5))
            assert window.controller.pending[1].box is None
            assert window.controller.pending[1].labels == (0, 0)
            return 13
        return -1

    monkeypatch.setattr(cv2, "waitKey", key)
    run = prompted.run_prompted_track(SimpleNamespace(), tracker=tracker, source=source, sinks=(sink,))

    assert delays == [30, 1, 30, 1, 1]
    assert tracker.events == [
        ("reset",),
        ("box", 1, (2, 2, 8, 8)),
        ("points", 1, [[3, 3]], [0]),
        ("update", "video/0"),
        ("points", 1, [[4, 4], [5, 5]], [0, 0]),
        ("update", "video/1"),
        ("update", "video/2"),
        ("reset",),
    ]
    assert run.summary.frames == 3 and not run.summary.interrupted
    assert all(result.track_ids.tolist() == [1] for result in sink.results)
    assert window.controller.active_ids == {1} and not window.controller.pending
    assert source.closed and sink.closed and closed_windows == [window.name]


def test_saved_points_are_not_replayed_and_only_moved_points_reach_the_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, tracker, sink = PromptWindow(), Tracker(), Sink()
    source = Source([_frame(index, width=240, height=120) for index in range(4)])
    delays: list[int] = []
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    monkeypatch.setattr(window, "open", lambda: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)

    def mouse(event: int, point: tuple[int, int]) -> None:
        viewport = window.controller.viewport
        assert viewport is not None
        window._mouse(event, *viewport.display_point(point), 0, None)

    def key(delay: int) -> int:
        delays.append(delay)
        assert len(delays) <= 8, "Point editing must resume on each selected frame"
        assert delay == (30 if len(delays) % 2 else 1)
        index = source.frames[len(source.yielded) - 1].frame_index
        controller = window.controller
        if delay == 1:
            return 32 if index < 3 else -1
        if index == 0:
            mouse(cv2.EVENT_LBUTTONDOWN, (20, 20))
            mouse(cv2.EVENT_LBUTTONUP, (100, 100))
            mouse(cv2.EVENT_RBUTTONDOWN, (40, 40))
            mouse(cv2.EVENT_RBUTTONDOWN, (70, 70))
            controller.handle_key(ord("n"))
            mouse(cv2.EVENT_LBUTTONDOWN, (130, 20))
            mouse(cv2.EVENT_LBUTTONUP, (220, 100))
            mouse(cv2.EVENT_RBUTTONDOWN, (170, 60))
        elif index == 1:
            for identity in (1, 2):
                controller.select_object(identity)
                assert controller.point_editor.visible(identity)
                assert not controller.pending
        elif index == 2:
            controller.select_object(1)
            mouse(cv2.EVENT_LBUTTONDOWN, (40, 40))
            mouse(cv2.EVENT_MOUSEMOVE, (60, 40))
            mouse(cv2.EVENT_LBUTTONUP, (60, 40))
            assert controller.pending[1].points == ((60, 40),)
            assert set(controller.pending) == {1}
        else:
            controller.select_object(1)
            mouse(cv2.EVENT_LBUTTONDOWN, (70, 70))
            mouse(cv2.EVENT_LBUTTONUP, (70, 70))
            controller.handle_key(ord("d"))
            assert not controller.pending
        return 13

    monkeypatch.setattr(cv2, "waitKey", key)
    run = prompted.run_prompted_track(SimpleNamespace(), tracker=tracker, source=source, sinks=(sink,))

    assert run.summary.frames == 4 and not run.summary.interrupted
    assert tracker.events == [
        ("reset",),
        ("box", 1, (20, 20, 100, 100)),
        ("points", 1, [[40, 40], [70, 70]], [0, 0]),
        ("box", 2, (130, 20, 220, 100)),
        ("points", 2, [[170, 60]], [0]),
        ("update", "video/0"),
        ("update", "video/1"),
        ("points", 1, [[60, 40]], [0]),
        ("update", "video/2"),
        ("update", "video/3"),
        ("reset",),
    ]
    assert [point.position for point in window.controller.point_editor.visible(1)] == [(60, 40)]
    assert [point.position for point in window.controller.point_editor.visible(2)] == [(170, 60)]
    assert source.closed and sink.closed


@pytest.mark.parametrize("change", ["sequence", "resolution"])
def test_sequence_or_resolution_change_collects_fresh_prompts(monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    window, tracker = Window(), Tracker()
    second = _frame(0, sequence="next") if change == "sequence" else _frame(1, width=24)
    source = Source([_frame(0), second])

    def edits(_frame: Frame) -> None:
        assert not window.controller.active_ids
        assert window.controller.paused
        window.controller.pending[1] = ObjectPrompt(box=(2, 2, 8, 8))

    window.script = edits
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    run = prompted.run_prompted_track(SimpleNamespace(), tracker=tracker, source=source, sinks=())
    assert run.summary.frames == run.summary.sequences == 2
    assert tracker.events.count(("reset",)) == 3


def test_quit_before_first_update_and_headless_failure_avoid_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    window, tracker = Window(), Tracker()
    source = Source([_frame(0)])
    window.script = lambda _frame: setattr(window.controller, "quit", True)
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    run = prompted.run_prompted_track(SimpleNamespace(), tracker=tracker, source=source, sinks=())
    assert run.summary.frames == 0 and run.summary.interrupted
    assert not any(event[0] == "update" for event in tracker.events)
    assert source.closed and window.closed

    source, sink = Source([_frame(0)]), Sink()

    def no_display() -> None:
        raise RuntimeError("no desktop")

    window.open = no_display
    monkeypatch.setattr(
        prompted, "_create_prompted_tracker", lambda _args: pytest.fail("model loaded before GUI check")
    )
    with pytest.raises(RuntimeError, match="no desktop"):
        prompted.run_prompted_track(SimpleNamespace(), source=source, sinks=(sink,))
    assert source.closed and sink.closed and window.closed


def test_inference_error_closes_all_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    window, tracker, sink = Window(), Tracker(), Sink()
    source = Source([_frame(0)])
    tracker.fail = True
    window.script = lambda _frame: window.controller.pending.update({1: ObjectPrompt(box=(2, 2, 8, 8))})
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    with pytest.raises(RuntimeError, match="inference failed"):
        prompted.run_prompted_track(SimpleNamespace(), tracker=tracker, source=source, sinks=(sink,))
    assert source.closed and sink.closed and window.closed
    assert tracker.events[-1] == ("reset",)


@pytest.mark.parametrize("option", [{"geometry": "obb"}, {"tracker_backend": "cpp"}, {"per_class": True}])
def test_incompatible_modes_rejected_before_opening_window(monkeypatch: pytest.MonkeyPatch, option: dict) -> None:
    monkeypatch.setattr(prompted, "PromptWindow", lambda: pytest.fail("window opened for invalid mode"))
    with pytest.raises(ValueError, match="EdgeTAM"):
        prompted.run_prompted_track(SimpleNamespace(**option))


def test_controller_drives_real_edgetam_tracker_with_injected_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    from boxmot import EdgeTAM

    class Backend:
        def __init__(self) -> None:
            self.calls = []
            self.masks = {}

        def process(self, frame_index, frame, prompts=()):
            self.calls.append((frame_index, prompts))
            for prompt in prompts:
                mask = torch.zeros(frame.shape[:2], dtype=torch.bool)
                if prompt.box is not None:
                    x1, y1, x2, y2 = map(int, prompt.box)
                    mask[y1:y2, x1:x2] = True
                for (x, y), label in zip(prompt.points, prompt.labels):
                    mask[int(y), int(x)] = bool(label)
                self.masks[prompt.object_id] = mask
            return self.masks

        def remove_object(self, object_id):
            self.masks.pop(object_id, None)

        def reset(self):
            self.masks.clear()

    backend = Backend()
    tracker = EdgeTAM(propagator=backend)
    window, sink = Window(), Sink()

    def box(x: int) -> None:
        window.controller.handle_key(ord("b"))
        window.controller.mouse("left_down", x, 2)
        window.controller.mouse("left_up", x + 6, 8)

    def edits(frame: Frame) -> None:
        controller = window.controller
        controller.viewport = PromptViewport(frame.width, frame.height, frame.width, frame.height)
        if not controller.paused:
            controller.handle_key(32)
        controller.editable = True
        if frame.frame_index == 0:
            box(2)
            controller.mouse("right_down", 4, 4)
            assert controller.mode == "box"
            controller.handle_key(ord("n"))
            box(10)
        elif frame.frame_index == 1:
            controller.handle_key(9)
            assert controller.selected_id == 1
            controller.handle_key(ord("p"))
            controller.mouse("left_down", 18, 14)
            controller.mouse("right_down", 4, 4)
        else:
            controller.handle_key(ord("x"))
            controller.handle_key(9)
            controller.handle_key(ord("x"))
        controller.handle_key(13)
        assert not controller.paused

    window.script = edits
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    run = prompted.run_prompted_track(
        SimpleNamespace(), tracker=tracker, source=Source([_frame(0), _frame(1), _frame(2)]), sinks=(sink,)
    )
    assert run.summary.frames == 3
    assert sink.results[0].track_ids.tolist() == sink.results[1].track_ids.tolist() == [1, 2]
    assert len(sink.results[2]) == 0
    assert backend.calls[0][1][0].box == (2, 2, 8, 8)
    assert backend.calls[0][1][0].object_id == 1
    assert backend.calls[0][1][0].points == ((4, 4),)
    assert backend.calls[0][1][0].labels == (0,)
    assert [prompt.object_id for prompt in backend.calls[1][1]] == [1]
    assert backend.calls[1][1][0].points == ((18, 14), (4, 4))
    assert backend.calls[1][1][0].labels == (1, 0)
    torch.testing.assert_close(sink.results[0].masks.values[1], sink.results[1].masks.values[1])


def _select_camera_frame_and_draw_box(window: Window, frame: Frame, *, x: int = 2) -> None:
    """Use real controls to freeze the displayed frame and submit a prompt."""
    controller = window.controller
    controller.viewport = PromptViewport(frame.width, frame.height, frame.width, frame.height)
    controller.handle_key(32)
    assert not controller.choosing_frame and controller.paused and controller.editable
    controller.mouse("left_down", x, 2)
    controller.mouse("left_up", x + 6, 8)
    controller.handle_key(13)
    assert not controller.paused


def test_camera_preview_preserves_tracker_memory_and_uses_the_selected_frame_for_first_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from boxmot import EdgeTAM

    window, sink = Window(), Sink()
    source = Source([_frame(index, source_uri="camera:0") for index in range(4)])
    for frame in source.frames:
        frame.image.fill_(frame.frame_index)

    class Backend:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str, int, int]] = []
            self.mask = torch.zeros((16, 20), dtype=torch.bool)
            self.mask[2:8, 2:8] = True

        def process(self, frame_index, frame, prompts=()):
            self.calls.append((frame_index, source.yielded[-1], len(source.yielded), int(frame[0, 0, 0])))
            if frame_index == 0:
                assert len(prompts) == 1 and prompts[0].object_id == 1
                assert prompts[0].box == (2.0, 2.0, 8.0, 8.0)
            else:
                assert not prompts
            return {1: self.mask}

        def remove_object(self, object_id):
            pytest.fail("preview unexpectedly removed an object")

        def reset(self):
            pass

    backend = Backend()
    tracker = EdgeTAM(propagator=backend)

    def choose(frame: Frame) -> None:
        if frame.frame_index <= 2:
            assert window.controller.choosing_frame
            assert tracker.frame_count == 0 and tracker.object_ids == ()
            assert backend.calls == [] and sink.results == []
            if frame.frame_index == 2:
                _select_camera_frame_and_draw_box(window, frame)
        else:
            assert not window.controller.choosing_frame
            assert tracker.frame_count == 1 and tracker.object_ids == (1,)

    window.script = choose
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    run = prompted.run_prompted_track(SimpleNamespace(), tracker=tracker, source=source, sinks=(sink,))

    assert backend.calls == [(0, "video/2", 3, 2), (1, "video/3", 4, 3)]
    assert [tracks.sample_id for tracks in sink.results] == ["video/2", "video/3"]
    assert all(tracks.track_ids.tolist() == [1] for tracks in sink.results)
    assert len(window.images) == 2
    assert run.summary.frames == run.summary.track_rows == 2
    assert run.summary.unique_track_ids == 1 and run.summary.detections == 0
    assert [timing.sample_id for timing in run.summary.timings] == ["video/2", "video/3"]
    assert not run.summary.interrupted
    assert source.closed and sink.closed and window.closed


def test_quitting_camera_preview_closes_resources_without_tracking_or_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, tracker, sink = Window(), Tracker(), Sink()
    tracker.fail = True
    source = Source([_frame(index, source_uri="camera:0") for index in range(3)])

    def quit_preview(frame: Frame) -> None:
        assert window.controller.choosing_frame
        if frame.frame_index == 1:
            window.controller.handle_key(ord("q"))

    window.script = quit_preview
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    run = prompted.run_prompted_track(SimpleNamespace(), tracker=tracker, source=source, sinks=(sink,))

    assert tracker.events == [("reset",), ("reset",)]
    assert source.yielded == ["video/0", "video/1"]
    assert not sink.results and not window.images
    assert run.summary.frames == run.summary.track_rows == run.summary.unique_track_ids == 0
    assert not run.summary.timings and run.summary.interrupted
    assert source.closed and sink.closed and window.closed


def test_camera_reselection_skips_draft_frame_and_keeps_committed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, tracker, sink = Window(), Tracker(), Sink()
    source = Source([_frame(index, source_uri="camera:0") for index in range(4)])

    def choose_again(frame: Frame) -> None:
        controller = window.controller
        if frame.frame_index == 0:
            _select_camera_frame_and_draw_box(window, frame)
        elif frame.frame_index == 1:
            controller.handle_key(32)
            controller.editable = True
            controller.handle_key(ord("n"))
            controller.mouse("left_down", 10, 2)
            controller.mouse("left_up", 16, 8)
            assert 2 in controller.pending
            controller.handle_key(ord("r"))
            assert controller.choosing_frame and not controller.pending
            assert controller.active_ids == {1}
        elif frame.frame_index == 2:
            assert controller.choosing_frame and tracker.objects == {1}
            assert [event for event in tracker.events if event[0] == "update"] == [("update", "video/0")]
        else:
            controller.selected_id = 1
            _select_camera_frame_and_draw_box(window, frame, x=3)

    window.script = choose_again
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    run = prompted.run_prompted_track(SimpleNamespace(), tracker=tracker, source=source, sinks=(sink,))

    assert [event for event in tracker.events if event[0] == "update"] == [
        ("update", "video/0"),
        ("update", "video/3"),
    ]
    assert [event[1] for event in tracker.events if event[0] == "box"] == [1, 1]
    assert all(tracks.track_ids.tolist() == [1] for tracks in sink.results)
    assert run.summary.frames == 2 and run.summary.unique_track_ids == 1


def test_prompted_camera_source_keeps_latest_frames_without_fixed_startup_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, tracker = Window(), Tracker()
    source = Source([_frame(0, source_uri="camera:0")])
    calls = []

    def create_source(value, **kwargs):
        calls.append((value, kwargs))
        return source

    window.script = lambda _frame: window.controller.handle_key(ord("q"))
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    monkeypatch.setattr(prompted, "create_frame_source", create_source)
    run = prompted.run_prompted_track(SimpleNamespace(source="0", vid_stride=2), tracker=tracker, sinks=())

    assert calls == [("0", {"stride": 2, "camera_warmup_s": 0, "latest_camera_frame": True})]
    assert run.summary.frames == 0 and run.summary.interrupted
    assert source.closed and window.closed


def test_camera_keeps_capturing_during_prompt_edits_and_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process the frozen prompt frame, then the freshest capture after inference."""
    from boxmot.engine.tracking import sources

    editing = Event()
    captured_during_edits = Event()
    inference = Event()
    captured_during_inference = Event()

    class Camera:
        def __init__(self) -> None:
            self.index = 0
            self.image = np.zeros((16, 20, 3), dtype=np.uint8)
            self.closed = False

        def isOpened(self) -> bool:
            return True

        def set(self, _property: int, _value: float) -> bool:
            return False  # Fresh capture must not depend on backend buffer-size support.

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index == 1:
                assert editing.wait(5), "workflow did not enter prompt editing"
            elif self.index == 3:
                captured_during_edits.set()  # Frames 1 and 2 have been published.
                assert inference.wait(5), "workflow did not start inference"
            elif self.index == 5:
                captured_during_inference.set()  # Frames 3 and 4 have been published.
                return False, None
            self.image.fill(self.index)
            self.index += 1
            return True, self.image

        def get(self, property_id: int) -> float:
            if property_id == sources.cv2.CAP_PROP_FPS:
                return 30.0
            return (self.index - 1) * 1000 / 30

        def release(self) -> None:
            self.closed = True

    class SlowTracker(Tracker):
        def update(self, detections: Detections, *, frame: Frame) -> Tracks:
            if not any(event[0] == "update" for event in self.events):
                inference.set()
                assert captured_during_inference.wait(5), "camera stalled during inference"
                assert torch.all(frame.image == 0), "capture replaced the selected prompt frame"
            return super().update(detections, frame=frame)

    camera, window, tracker, sink = Camera(), Window(), SlowTracker(), Sink()

    def select(frame: Frame) -> None:
        if len(window.prepared) == 1:
            editing.set()
            assert captured_during_edits.wait(5), "camera stalled during prompt editing"
            assert torch.all(frame.image == 0), "capture mutated the displayed frame"
            _select_camera_frame_and_draw_box(window, frame)
        else:
            assert torch.all(frame.image == 4), "workflow replayed an obsolete camera frame"
            assert not window.controller.paused

    window.script = select
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda _source: camera)
    run = prompted.run_prompted_track(SimpleNamespace(source="0"), tracker=tracker, sinks=(sink,))

    assert run.summary.frames == 2 and run.summary.unique_track_ids == 1
    assert [result.sample_id.rsplit(":", 1)[-1] for result in sink.results] == ["000000000000", "000000000004"]
    assert [event[0] for event in tracker.events] == ["reset", "box", "update", "update", "reset"]
    assert camera.closed and sink.closed and window.closed
