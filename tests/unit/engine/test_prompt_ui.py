"""Prompt interaction contracts without a desktop or model checkpoint."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from boxmot.engine.ui import prompts
from boxmot.engine.ui.prompts import PromptController, PromptViewport, PromptWindow
from boxmot.structures import Frame


def _controller() -> PromptController:
    controller = PromptController()
    controller.viewport = PromptViewport(2000, 1000, 1000, 500, left=20)
    controller.editable = True
    return controller


def _box(controller: PromptController, start: tuple[int, int], end: tuple[int, int]) -> None:
    controller.mouse("left_down", *start)
    controller.mouse("move", *end)
    controller.mouse("left_up", *end)


def _window_mouse(window: PromptWindow, event: int, point: tuple[int, int]) -> None:
    """Send a native mouse event at an original-image point on this canvas."""
    viewport = window.controller.viewport
    assert viewport is not None
    x, y = viewport.display_point(point)
    window._mouse(event, x, y, 0, None)


def _window_box(window: PromptWindow, start: tuple[int, int], end: tuple[int, int]) -> None:
    _window_mouse(window, cv2.EVENT_LBUTTONDOWN, start)
    _window_mouse(window, cv2.EVENT_LBUTTONUP, end)


def _video_region(window: PromptWindow, canvas: np.ndarray) -> np.ndarray:
    """Exclude controls and margins when checking the frame's displayed pixels."""
    viewport = window.controller.viewport
    assert viewport is not None
    return canvas[
        viewport.top : viewport.top + viewport.display_height,
        viewport.left : viewport.left + viewport.display_width,
    ]


def _button_center(window: PromptWindow, *, key: int | None = None, object_id: int | None = None) -> tuple[int, int]:
    """Find a displayed control without depending on toolbar layout."""
    button = next(
        button
        for button in window._buttons
        if (button.object_id == object_id if object_id is not None else button.object_id is None and button.key == key)
    )
    x1, y1, x2, y2 = button.bounds
    return (x1 + x2) // 2, (y1 + y2) // 2


def _click(window: PromptWindow, position: tuple[int, int]) -> None:
    """Complete a native click so releases cannot accidentally finish a box."""
    window._mouse(cv2.EVENT_LBUTTONDOWN, *position, 0, None)
    window._mouse(cv2.EVENT_LBUTTONUP, *position, 0, None)


def test_box_drag_maps_scaled_and_offset_image_coordinates() -> None:
    controller = _controller()
    _box(controller, (420, 250), (120, 50))

    assert controller.pending[1].box == (200, 100, 800, 500)
    controller.handle_key(13)
    assert not controller.paused
    controller.commit()
    assert controller.active_ids == {1}
    assert controller.pending == {}


def test_clicks_outside_image_and_tiny_boxes_are_ignored() -> None:
    controller = _controller()
    _box(controller, (10, 10), (300, 300))
    assert not controller.pending
    _box(controller, (100, 100), (100, 101))
    assert not controller.pending
    assert "too small" in controller.message
    _box(controller, (100, 100), (2000, 2000))
    assert controller.pending[1].box == (160, 200, 1999, 999)


def test_point_labels_validation_and_stable_ids_across_corrections() -> None:
    controller = _controller()
    controller.handle_key(ord("p"))
    controller.mouse("right_down", 120, 50)
    controller.handle_key(13)
    assert controller.paused
    assert "positive point" in controller.message
    controller.mouse("left_down", 140, 80)
    assert controller.pending[1].points == ((200, 100), (240, 160))
    assert controller.pending[1].labels == (0, 1)
    controller.handle_key(13)
    controller.commit()

    controller.handle_key(32)
    controller.editable = True
    controller.mouse("right_down", 130, 60)
    controller.handle_key(13)
    assert not controller.paused
    assert controller.pending[1].box is None
    assert controller.pending[1].labels == (0,)
    controller.commit()
    assert controller.active_ids == {1}


def test_new_object_still_requires_foreground_when_another_object_is_active() -> None:
    controller = _controller()
    _box(controller, (120, 50), (420, 250))
    controller.handle_key(13)
    controller.commit()
    controller.handle_key(32)
    controller.editable = True
    controller.handle_key(ord("p"))
    controller.handle_key(ord("n"))
    controller.mouse("right_down", 500, 100)
    controller.handle_key(13)

    assert controller.paused and controller.editable
    assert controller.active_ids == {1}
    assert controller.selected_id == 2
    assert controller.pending[2].labels == (0,)
    assert "New object 2" in controller.message and "positive point" in controller.message


def test_object_selection_undo_remove_and_clear_edits() -> None:
    controller = _controller()
    _box(controller, (120, 50), (420, 250))
    controller.handle_key(ord("n"))
    assert controller.selected_id == 2
    controller.handle_key(ord("p"))
    controller.mouse("left_down", 500, 100)
    controller.mouse("right_down", 520, 120)
    controller.handle_key(ord("u"))
    assert controller.pending[2].labels == (1,)
    controller.commit()
    controller.handle_key(9)
    assert controller.selected_id == 1
    controller.handle_key(ord("x"))
    assert controller.object_ids == [2]
    controller.handle_key(ord("c"))
    assert controller.object_ids == [1, 2]
    controller.handle_key(ord("x"))
    controller.commit()
    assert controller.active_ids == {2}


def test_object_limit_and_unprocessed_frame_boundary() -> None:
    controller = _controller()
    controller.max_objects = 1
    _box(controller, (120, 50), (420, 250))
    controller.handle_key(ord("n"))
    _box(controller, (500, 50), (700, 250))
    assert set(controller.pending) == {1}
    assert "limit" in controller.message
    controller.handle_key(13)
    controller.commit()
    controller.handle_key(ord("p"))  # Pauses the next frame, not this processed frame.
    assert controller.paused and not controller.editable
    controller.mouse("left_down", 100, 100)
    assert not controller.pending
    controller.reset()
    assert controller.object_ids == [] and controller.selected_id == 1


def test_run_requires_prompt_and_quit_always_works() -> None:
    controller = _controller()
    controller.handle_key(32)
    assert controller.paused
    assert "at least one" in controller.message
    controller.handle_key(27)
    assert controller.quit


@pytest.mark.parametrize("use_points_tool", [False, True])
def test_box_and_points_combine_and_undo_preserves_box_context(use_points_tool: bool) -> None:
    controller = _controller()
    _box(controller, (120, 50), (420, 250))
    if use_points_tool:
        controller.handle_key(ord("p"))
    controller.mouse("right_down", 130, 60)
    assert controller.mode == ("points" if use_points_tool else "box")
    assert controller.pending[1].box == (200, 100, 800, 500)
    assert controller.pending[1].labels == (0,)
    controller.handle_key(ord("u"))
    assert controller.pending[1].box is not None and not controller.pending[1].points
    controller.mouse("right_down", 130, 60)
    controller.handle_key(13)
    assert not controller.paused  # The box supplies foreground context.


def test_box_tool_shows_exclusion_dots_before_resuming(monkeypatch: pytest.MonkeyPatch) -> None:
    window = PromptWindow()
    frame = Frame(torch.full((3, 120, 240), 64, dtype=torch.uint8), "selected")
    shown: list[np.ndarray] = []
    steps = iter(("draw", "undo", "redraw", "resume"))
    monkeypatch.setattr(cv2, "imshow", lambda _name, canvas: shown.append(canvas.copy()))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)

    def key(delay: int) -> int:
        assert delay == 30
        assert window.controller.mode == "box"
        step = next(steps)
        if step == "draw":
            _window_box(window, (20, 20), (180, 100))
            _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (120, 70))
            return -1
        prompt = window.controller.pending[1]
        assert prompt.box == (20, 20, 180, 100)
        assert window.controller.selected_id == 1
        pixel = _video_region(window, shown[-1])[74, 124]
        if step == "redraw":
            assert not prompt.points and not prompt.labels
            np.testing.assert_array_equal(pixel, (64, 64, 64))
            _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (120, 70))
            return -1
        assert prompt.points == ((120, 70),) and prompt.labels == (0,)
        assert pixel[2] > 200 and pixel[0] < 100 and pixel[1] < 100
        return ord("u") if step == "undo" else 13

    monkeypatch.setattr(cv2, "waitKey", key)
    assert window.prepare_frame(frame)
    assert len(shown) == 4
    assert not window.controller.paused
    assert window.controller.mode == "box"
    assert window.controller.pending[1].box == (20, 20, 180, 100)
    assert window.controller.pending[1].labels == (0,)


def test_removing_last_object_can_be_applied_by_resuming() -> None:
    controller = _controller()
    _box(controller, (120, 50), (420, 250))
    controller.commit()
    controller.handle_key(ord("x"))
    controller.handle_key(13)
    assert not controller.paused
    assert controller.removed == {1}
    controller.commit()
    assert not controller.object_ids


def test_window_waits_for_prompts_and_maps_canvas_clicks(monkeypatch: pytest.MonkeyPatch) -> None:
    window = PromptWindow()
    frame = Frame(torch.zeros((3, 10, 20), dtype=torch.uint8), "sample")
    shown: list[np.ndarray] = []
    delays: list[int] = []
    monkeypatch.setattr(cv2, "imshow", lambda _name, image: shown.append(image))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)

    def key(delay: int) -> int:
        delays.append(delay)
        if delay == 30:
            _window_box(window, (2, 2), (8, 8))
            return 13
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", key)
    assert window.prepare_frame(frame)
    assert window.controller.pending[1].box == (2, 2, 8, 8)
    window.controller.commit()
    window.show_result(np.zeros((10, 20, 3), dtype=np.uint8))
    assert delays == [30, 1]
    assert _video_region(window, shown[0]).shape == (10, 20, 3)
    assert window.controller.quit


def test_headless_check_does_not_attempt_native_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args: pytest.fail("native window opened without a display"))

    with pytest.raises(RuntimeError, match="desktop display"):
        PromptWindow().open()


def test_window_closes_if_mouse_callback_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(prompts, "require_display", lambda: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args: calls.append("open"))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args: calls.append("close"))

    def fail_callback(*_args: object) -> None:
        raise cv2.error("no callback")

    monkeypatch.setattr(cv2, "setMouseCallback", fail_callback)
    with pytest.raises(RuntimeError, match="prompt window"):
        PromptWindow().open()
    assert calls == ["open", "close"]


def test_closed_window_stops_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    window = PromptWindow()
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 0)
    monkeypatch.setattr(cv2, "waitKey", lambda *_args: pytest.fail("closed window polled"))
    window._poll(30)
    assert window.controller.quit


def test_live_preview_ignores_edits_until_a_frame_is_chosen() -> None:
    controller = _controller()
    controller.reset(live=True)
    assert controller.live_source and controller.choosing_frame and not controller.paused
    for key in (ord("b"), ord("p"), ord("n"), 9, ord("u"), ord("x"), ord("c"), ord("r")):
        controller.handle_key(key)
    controller.editable = True  # Preview remains protected even with a stale edit flag.
    _box(controller, (120, 50), (420, 250))
    assert not controller.pending and not controller.drag_start
    assert controller.selected_id == 1 and controller.mode == "box"

    controller.handle_key(32)
    assert not controller.choosing_frame and controller.paused and controller.editable
    controller.handle_key(13)
    assert controller.paused  # Choosing the frame does not bypass prompt validation.
    _box(controller, (120, 50), (420, 250))
    controller.handle_key(13)
    assert not controller.paused and not controller.choosing_frame


def test_reselecting_live_frame_discards_edits_but_preserves_objects() -> None:
    controller = _controller()
    controller.reset(live=True)
    controller.handle_key(13)
    _box(controller, (120, 50), (420, 250))
    controller.commit()
    controller.handle_key(ord("x"))
    controller.handle_key(ord("n"))
    _box(controller, (500, 50), (700, 250))
    controller.mouse("left_down", 600, 100)
    assert controller.pending and controller._history and controller.removed
    assert controller.drag_start is not None

    controller.handle_key(ord("r"))

    assert controller.active_ids == {1}
    assert controller.object_ids == [1]
    assert controller.choosing_frame and not controller.paused and not controller.editable
    assert not controller.pending and not controller._history and not controller.removed
    assert controller.drag_start is controller.drag_end is None
    assert controller.selected_id == 2  # Unused IDs are not silently reassigned.


def test_camera_frame_selection_edits_the_current_displayed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    window = PromptWindow()
    window.controller.reset(live=True)
    displayed: list[int] = []
    delays: list[int] = []
    actions = iter((-1, 32, 13))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(
        cv2, "imshow", lambda _name, canvas: displayed.append(int(_video_region(window, canvas)[0, 0, 0]))
    )

    def key(delay: int) -> int:
        delays.append(delay)
        if delay == 30:
            _window_box(window, (2, 2), (8, 8))
        return next(actions)

    monkeypatch.setattr(cv2, "waitKey", key)
    preview = Frame(torch.full((3, 10, 20), 17, dtype=torch.uint8), "preview")
    selected = Frame(torch.full((3, 10, 20), 63, dtype=torch.uint8), "selected")

    assert window.prepare_frame(preview)
    assert window.controller.choosing_frame
    assert window.prepare_frame(selected)

    assert displayed == [17, 63, 63]
    assert delays == [1, 1, 30]
    assert not window.controller.choosing_frame and not window.controller.paused
    assert window.controller.pending[1].box == (2, 2, 8, 8)


def test_reselecting_from_frame_editor_returns_to_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    window = PromptWindow()
    window.controller.reset(live=True)
    frame = Frame(torch.zeros((3, 10, 20), dtype=torch.uint8), "frame")
    keys = iter((13, ord("r"), -1))
    delays: list[int] = []
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)

    def key(delay: int) -> int:
        delays.append(delay)
        if delay == 30:
            _window_box(window, (2, 2), (8, 8))
        return next(keys)

    monkeypatch.setattr(cv2, "waitKey", key)
    assert window.prepare_frame(frame)
    assert window.controller.choosing_frame and not window.controller.pending
    assert window.prepare_frame(frame)
    assert window.controller.choosing_frame
    assert delays == [1, 30, 1]


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_live_preview_quit_returns_without_edit_loop(monkeypatch: pytest.MonkeyPatch, key: int) -> None:
    window = PromptWindow()
    window.controller.reset(live=True)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: key if delay == 1 else pytest.fail("entered editor"))

    assert not window.prepare_frame(Frame(torch.zeros((3, 10, 20), dtype=torch.uint8), "frame"))
    assert window.controller.quit and not window.controller.pending


def test_non_camera_sources_remain_paused_and_ignore_reselect() -> None:
    controller = _controller()
    assert not controller.live_source and not controller.choosing_frame
    _box(controller, (120, 50), (420, 250))
    expected = dict(controller.pending)
    controller.handle_key(ord("r"))
    assert controller.paused and controller.pending == expected
    controller.handle_key(13)
    assert not controller.paused


def test_live_hud_explains_frame_selection_and_reselection(monkeypatch: pytest.MonkeyPatch) -> None:
    window = PromptWindow()
    window.controller.reset(live=True)
    window.controller.viewport = PromptViewport(20, 10, 20, 10, top=window._HEADER_HEIGHT)
    text: list[str] = []
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "putText", lambda _canvas, value, *_args: text.append(value))
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    window._draw(image)
    assert "LIVE PREVIEW" in text and "Choose frame" in text
    window.controller.handle_key(13)
    text.clear()
    window._draw(image)
    assert "PAUSED" in text and "New frame" in text and "R" in text


def test_submitted_point_markers_stay_on_the_prompted_result_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submitting prompts must not hide their dots or paint them onto later frames."""
    window = PromptWindow()
    frame = Frame(torch.full((3, 120, 240), 64, dtype=torch.uint8), "selected")
    image = np.full((120, 240, 3), 64, dtype=np.uint8)
    shown: list[np.ndarray] = []
    monkeypatch.setattr(cv2, "imshow", lambda _name, canvas: shown.append(canvas.copy()))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)
    actions = iter((ord("p"), 13))

    def key(delay: int) -> int:
        if delay == 1:
            return -1
        action = next(actions)
        if action == 13:
            _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (40, 40))
            _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (120, 70))
        return action

    monkeypatch.setattr(cv2, "waitKey", key)
    assert window.prepare_frame(frame)
    assert window.controller.pending[1].labels == (1, 0)
    window.controller.commit()
    assert not window.controller.pending
    window.show_result(image)

    result = _video_region(window, shown[-1])
    np.testing.assert_array_equal(result[44, 44], (100, 230, 100))
    np.testing.assert_array_equal(result[74, 124], (80, 80, 255))
    # The positive dot has a vertical stroke; the negative dot has only a dash.
    assert result[37, 40].max() < 100
    assert result[67, 120, 2] > 200
    assert result[40, 48].min() > 200  # Light outline remains visible on dark video.
    np.testing.assert_array_equal(image, np.full_like(image, 64))

    assert window.prepare_frame(Frame(frame.image.clone(), "next"))
    window.show_result(image)
    np.testing.assert_array_equal(_video_region(window, shown[-1]), image)


@pytest.mark.parametrize("key", [ord("u"), ord("c"), ord("x")])
def test_editing_removes_visible_point_markers_immediately(monkeypatch: pytest.MonkeyPatch, key: int) -> None:
    window = PromptWindow()
    window.controller.viewport = PromptViewport(240, 120, 240, 120, left=380, top=window._HEADER_HEIGHT)
    window.controller.editable = True
    window.controller.handle_key(ord("p"))
    _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (40, 40))
    _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (120, 70))
    shown: list[np.ndarray] = []
    monkeypatch.setattr(cv2, "imshow", lambda _name, canvas: shown.append(canvas.copy()))
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    window._draw(image)
    assert _video_region(window, shown[-1])[74, 124, 2] > 200

    window.controller.handle_key(key)
    window._draw(image)

    result = _video_region(window, shown[-1])
    assert not result[60:81, 110:131].any()
    if key == ord("u"):
        assert result[44, 44, 1] > 200
    else:
        assert not result.any()


def test_toolbar_selects_frame_and_tools_without_creating_image_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    window = PromptWindow()
    window.controller.reset(live=True)
    frame = Frame(torch.zeros((3, 120, 240), dtype=torch.uint8), "camera-frame")
    delays: list[int] = []
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)

    def interact(delay: int) -> int:
        delays.append(delay)
        if delay == 1:
            assert window.controller.choosing_frame
            assert not any(button.key == ord("p") for button in window._buttons)
            _click(window, _button_center(window, key=13))
            assert not window.controller.choosing_frame
            assert window.controller.paused and not window.controller.pending
        else:
            points = _button_center(window, key=ord("p"))
            window._mouse(cv2.EVENT_RBUTTONDOWN, *points, 0, None)
            assert window.controller.mode == "box" and not window.controller.pending
            _click(window, points)
            assert window.controller.mode == "points" and not window.controller.pending
            assert window.controller.drag_start is None
            # Header and surrounding margins must never become image points.
            _click(window, (1, 1))
            assert not window.controller.pending
            _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (40, 40))
            _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (120, 70))
            _click(window, _button_center(window, key=13))
        return -1

    monkeypatch.setattr(cv2, "waitKey", interact)
    assert window.prepare_frame(frame)
    assert delays == [1, 30]
    assert not window.controller.paused and not window.controller.choosing_frame
    assert window.controller.pending[1].points == ((40.0, 40.0), (120.0, 70.0))
    assert window.controller.pending[1].labels == (1, 0)
    assert window.controller.viewport.top > 0


def test_object_buttons_select_correction_target_only_while_editing(monkeypatch: pytest.MonkeyPatch) -> None:
    window = PromptWindow()
    controller = window.controller
    controller.viewport = PromptViewport(240, 120, 240, 120, left=380, top=window._HEADER_HEIGHT)
    controller.active_ids = {1, 2}
    controller.editable = True
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    window._draw(image)
    second_object = _button_center(window, object_id=2)

    _click(window, second_object)
    assert controller.selected_id == 2 and not controller.pending
    assert controller.drag_start is None
    controller.handle_key(ord("p"))
    _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (60, 40))
    assert set(controller.pending) == {2}
    controller.commit()
    controller.handle_key(13)
    window._draw(image)

    assert not any(button.object_id is not None for button in window._buttons)
    _click(window, second_object)
    assert controller.selected_id == 2 and not controller.pending


def test_queued_run_then_points_click_edits_the_next_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued tool click cannot trap the submitted frame in a disabled editor."""
    window = PromptWindow()
    frame = Frame(torch.zeros((3, 120, 240), dtype=torch.uint8), "submitted")
    editing_polls: list[str] = []
    current_frame = "submitted"
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)

    def queued_clicks(delay: int) -> int:
        if delay == 1:
            return -1
        assert current_frame not in editing_polls, "submitted frame reopened its disabled editor"
        editing_polls.append(current_frame)
        assert window.controller.editable
        if current_frame == "submitted":
            _window_box(window, (20, 20), (80, 80))
            _click(window, _button_center(window, key=13))
            _click(window, _button_center(window, key=ord("p")))
            assert window.controller.paused and not window.controller.editable
        else:
            assert window.controller.mode == "points" and not window.controller.pending
            _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (40, 40))
            _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (60, 60))
            _click(window, _button_center(window, key=13))
        return -1

    monkeypatch.setattr(cv2, "waitKey", queued_clicks)
    assert window.prepare_frame(frame)
    assert editing_polls == ["submitted"]
    assert window.controller.pending[1].box == (20.0, 20.0, 80.0, 80.0)
    window.controller.commit()
    window.show_result(np.zeros((120, 240, 3), dtype=np.uint8))

    current_frame = "next"
    assert window.prepare_frame(Frame(frame.image.clone(), current_frame))
    assert editing_polls == ["submitted", "next"]
    assert window.controller.pending[1].box is None
    assert window.controller.pending[1].points == ((40.0, 40.0), (60.0, 60.0))
    assert window.controller.pending[1].labels == (0, 1)
    assert not window.controller.paused


def _point_window() -> PromptWindow:
    """Create an editable canvas with enough room to distinguish point handles."""
    window = PromptWindow()
    window.controller.viewport = PromptViewport(240, 120, 240, 120, left=380, top=window._HEADER_HEIGHT)
    window.controller.editable = True
    return window


def _save_points(window: PromptWindow) -> None:
    """Commit two exclusions and reopen editing without replaying their coordinates."""
    _window_box(window, (20, 20), (180, 100))
    _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (40, 40))
    _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (120, 70))
    window.controller.handle_key(13)
    window.controller.commit()
    window.controller.handle_key(32)
    window.controller.editable = True


def test_saved_points_reappear_only_for_the_selected_paused_object(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _point_window()
    controller = window.controller
    _save_points(window)
    first_points = controller.point_editor.visible(1)
    assert [point.position for point in first_points] == [(40, 40), (120, 70)]
    assert not controller.pending

    controller.handle_key(ord("n"))
    _window_box(window, (140, 20), (220, 100))
    _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (180, 60))
    controller.handle_key(13)
    controller.commit()
    controller.handle_key(32)
    controller.editable = True
    image = np.full((120, 240, 3), 64, dtype=np.uint8)
    drawn: list[tuple[tuple[int, int], int, bool]] = []
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(window, "_poll", lambda _delay: None)
    monkeypatch.setattr(
        window,
        "_draw_point",
        lambda _canvas, position, label, **kwargs: drawn.append((position, label, kwargs.get("historical", False))),
    )

    for identity, positions in ((1, [(40, 40), (120, 70)]), (2, [(180, 60)])):
        controller.select_object(identity)
        drawn.clear()
        window._draw(image)
        viewport = controller.viewport
        assert viewport is not None
        assert [
            (position, label, historical)
            for position, label, historical in drawn
            if viewport.top <= position[1] < viewport.top + viewport.display_height
        ] == [(viewport.display_point(position), 0, True) for position in positions]
        assert not controller.pending and not controller.point_editor.changed(identity)

    controller.handle_key(13)
    controller.commit()
    drawn.clear()
    window.show_result(image)
    assert not [position for position, _label, _historical in drawn if 52 <= position[1] < 172]
    assert controller.point_editor.visible(1) == first_points


def test_clicking_a_scaled_saved_handle_without_dragging_does_not_queue_rounded_coordinates() -> None:
    window = _point_window()
    controller = window.controller
    controller.viewport = PromptViewport(1920, 1080, 903, 508, left=48, top=window._HEADER_HEIGHT)
    controller.active_ids.add(1)
    point = controller.point_editor.put(1, (123.5, 60.5), 0)
    controller.point_editor.commit()
    position = controller.viewport.display_point(point.position)

    _click(window, position)

    assert controller.selected_point == point.id
    assert controller.point_editor.visible(1) == (point,)
    assert not controller.point_editor.changed(1)
    assert not controller.pending


def test_moving_a_saved_point_submits_only_the_edited_handle() -> None:
    window = _point_window()
    controller = window.controller
    _save_points(window)
    original = controller.point_editor.visible(1)

    _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (40, 40))
    assert controller.selected_point == original[0].id
    assert not controller.pending
    _window_mouse(window, cv2.EVENT_MOUSEMOVE, (70, 40))
    _window_mouse(window, cv2.EVENT_LBUTTONUP, (70, 40))

    assert controller.mode == "box" and controller.drag_start is None
    assert controller.pending[1].box is None
    assert controller.pending[1].points == ((70, 40),)
    assert controller.pending[1].labels == (0,)
    assert [point.id for point in controller.point_editor.changed(1)] == [original[0].id]
    assert controller.point_editor.visible(1)[1] == original[1]
    controller.handle_key(ord("u"))
    assert controller.point_editor.visible(1) == original
    assert not controller.pending


@pytest.mark.parametrize("delete_key", [ord("d"), 127])
def test_deleting_a_saved_point_removes_only_its_marker_and_can_be_undone(delete_key: int) -> None:
    window = _point_window()
    controller = window.controller
    _save_points(window)
    original = controller.point_editor.visible(1)
    _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (40, 40))
    _window_mouse(window, cv2.EVENT_LBUTTONUP, (40, 40))

    controller.handle_key(delete_key)

    assert controller.point_editor.visible(1) == original[1:]
    assert not controller.pending and not controller.point_editor.changed(1)
    assert controller.object_ids == [1]
    assert "include point" in controller.message.lower() and "redraw" in controller.message.lower()
    controller.handle_key(ord("u"))
    assert controller.point_editor.visible(1) == original
    assert not controller.pending


@pytest.mark.parametrize("key, label", [(ord("i"), 1), (ord("e"), 0)])
def test_relabeling_a_saved_point_queues_only_that_point_and_clear_restores_saved(key: int, label: int) -> None:
    window = _point_window()
    controller = window.controller
    _save_points(window)
    if label == 0:
        _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (40, 40))
        _window_mouse(window, cv2.EVENT_LBUTTONUP, (40, 40))
        controller.handle_key(ord("i"))
        controller.handle_key(13)
        controller.commit()
        controller.handle_key(32)
        controller.editable = True
    original = controller.point_editor.visible(1)
    _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (40, 40))
    _window_mouse(window, cv2.EVENT_LBUTTONUP, (40, 40))

    controller.handle_key(key)

    assert controller.pending[1].points == ((40, 40),)
    assert controller.pending[1].labels == (label,)
    controller.handle_key(ord("c"))
    assert controller.point_editor.visible(1) == original
    assert not controller.pending


def test_reselecting_camera_frame_discards_point_edits_without_losing_saved_markers() -> None:
    window = _point_window()
    controller = window.controller
    _save_points(window)
    controller.live_source = True
    original = controller.point_editor.visible(1)
    _window_mouse(window, cv2.EVENT_LBUTTONDOWN, (40, 40))
    _window_mouse(window, cv2.EVENT_MOUSEMOVE, (70, 40))
    _window_mouse(window, cv2.EVENT_LBUTTONUP, (70, 40))
    assert controller.pending

    controller.handle_key(ord("r"))

    assert controller.choosing_frame
    assert controller.point_editor.visible(1) == original
    assert not controller.pending


def test_removing_objects_and_reset_clear_their_saved_points() -> None:
    window = _point_window()
    controller = window.controller
    _save_points(window)
    original = controller.point_editor.visible(1)
    controller.handle_key(ord("x"))
    controller.handle_key(ord("c"))
    assert controller.point_editor.visible(1) == original
    controller.handle_key(ord("x"))
    controller.handle_key(13)
    controller.commit()
    assert not controller.point_editor.visible(1)
    assert not controller.active_ids
    controller.handle_key(32)
    controller.editable = True
    _window_box(window, (20, 20), (180, 100))
    _window_mouse(window, cv2.EVENT_RBUTTONDOWN, (40, 40))
    controller.commit()
    assert controller.point_editor.visible(1)
    controller.reset()
    assert not controller.point_editor.visible(1)
