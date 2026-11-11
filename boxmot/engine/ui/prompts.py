"""Mouse and keyboard prompting for the native tracking preview window."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files

import cv2
import numpy as np

from boxmot.engine.ui.point_editor import PointEditor, PromptPoint
from boxmot.structures import Frame


@dataclass(frozen=True, slots=True)
class ObjectPrompt:
    """One object's replacement box or point correction on this frame."""

    box: tuple[float, float, float, float] | None = None
    points: tuple[tuple[float, float], ...] = ()
    labels: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptViewport:
    """Map a displayed image rectangle to its original pixel coordinates."""

    width: int
    height: int
    display_width: int
    display_height: int
    left: int = 0
    top: int = 0

    def point(self, x: int, y: int, *, clip: bool = False) -> tuple[float, float] | None:
        """Ignore clicks outside the image; clip drag endpoints to its edges."""
        x, y = x - self.left, y - self.top
        if not clip and not (0 <= x < self.display_width and 0 <= y < self.display_height):
            return None
        return (
            min(max(x * self.width / self.display_width, 0.0), self.width - 1.0),
            min(max(y * self.height / self.display_height, 0.0), self.height - 1.0),
        )

    def display_point(self, point: tuple[float, float]) -> tuple[int, int]:
        """Return the canvas position of an original-image point."""
        return (
            self.left + round(point[0] * self.display_width / self.width),
            self.top + round(point[1] * self.display_height / self.height),
        )


@dataclass(frozen=True, slots=True)
class _PromptButton:
    """A displayed control routed through the same actions as keyboard input."""

    bounds: tuple[int, int, int, int]
    key: int
    object_id: int | None = None


@dataclass(frozen=True, slots=True)
class _PromptEdit:
    """An object's draft prompt and point edits before one undoable action."""

    prompt: ObjectPrompt | None
    point_changes: dict[int, PromptPoint | None]


class PromptController:
    """Pure interaction state; no model loading, windows, or frame advancement."""

    def __init__(self, *, max_objects: int = 32) -> None:
        self.max_objects = max_objects
        self.viewport: PromptViewport | None = None
        self.reset()

    def reset(self, *, live: bool = False) -> None:
        """Begin a sequence, previewing live sources until the user chooses a frame."""
        self.active_ids: set[int] = set()
        self.pending: dict[int, ObjectPrompt] = {}
        self.point_editor = PointEditor()
        self.selected_point: int | None = None
        self._drag_point: PromptPoint | None = None
        self._point_drag_position: tuple[float, float] | None = None
        self._history: dict[int, list[_PromptEdit]] = {}
        self.removed: set[int] = set()
        self.selected_id = 1
        self._next_id = 2
        self.mode = "box"
        self.live_source = live
        self.choosing_frame = live
        self.paused = not live
        self.editable = False
        self.quit = False
        self.message = (
            "Space or Enter freezes the camera frame currently shown."
            if live
            else "Draw a box, or press P and click points, then press Enter."
        )
        self.drag_start: tuple[float, float] | None = None
        self.drag_end: tuple[float, float] | None = None

    @property
    def object_ids(self) -> list[int]:
        """List identities that will exist after the pending edits are applied."""
        return sorted((self.active_ids - self.removed) | self.pending.keys())

    def _can_edit_object(self) -> bool:
        """Check capacity before changing either prompts or their editable handles."""
        if self.selected_id not in self.object_ids and len(self.object_ids) >= self.max_objects:
            self.message = f"Object limit {self.max_objects} reached. Select an object with Tab and remove it with X."
            return False
        return True

    def _remember(self) -> None:
        """Capture one object's current draft, including deleted saved handles."""
        self._history.setdefault(self.selected_id, []).append(
            _PromptEdit(self.pending.get(self.selected_id), self.point_editor.snapshot(self.selected_id))
        )

    def _put(self, prompt: ObjectPrompt, *, remember: bool = True) -> None:
        """Store an edit and its undo state after checking the object budget."""
        if not self._can_edit_object():
            return
        if remember:
            self._remember()
        self.removed.discard(self.selected_id)
        self.pending[self.selected_id] = prompt
        hint = "Right-click excludes; " if self.mode == "box" else ""
        self.message = f"Object {self.selected_id}: prompt ready. {hint}Enter runs; N adds another object."

    def _cancel_drag(self) -> None:
        """Cancel unfinished box and point gestures without submitting either."""
        self.drag_start = self.drag_end = None
        self._drag_point = None
        self._point_drag_position = None

    @property
    def selected_handle(self) -> PromptPoint | None:
        """Return the selected object's current point, if it still exists."""
        return next(
            (point for point in self.point_editor.visible(self.selected_id) if point.id == self.selected_point), None
        )

    def _hit_point(self, x: int, y: int) -> PromptPoint | None:
        """Find the closest handle for the selected object in display pixels."""
        assert self.viewport is not None
        if self.selected_id in self.removed:
            return None
        closest = None
        distance = 11**2
        for point in self.point_editor.visible(self.selected_id):
            px, py = self.viewport.display_point(point.position)
            candidate = (px - x) ** 2 + (py - y) ** 2
            if candidate <= distance:
                closest, distance = point, candidate
        return closest

    def _sync_point_prompt(self) -> None:
        """Submit only deliberate current-frame changes, keeping saved positions out."""
        previous = self.pending.get(self.selected_id, ObjectPrompt())
        changed = self.point_editor.changed(self.selected_id)
        if previous.box is None and not changed:
            self.pending.pop(self.selected_id, None)
            return
        self._put(
            ObjectPrompt(
                box=previous.box,
                points=tuple(point.position for point in changed),
                labels=tuple(point.label for point in changed),
            ),
            remember=False,
        )

    def _edit_point(self, position: tuple[float, float], label: int, *, point_id: int | None = None) -> None:
        """Create or revise one point as an undoable current-frame correction."""
        if not self._can_edit_object():
            return
        self._remember()
        point = self.point_editor.put(self.selected_id, position, label, point_id=point_id)
        self._sync_point_prompt()
        self.selected_point = point.id

    def _delete_point(self) -> None:
        """Delete a handle without claiming to undo an earlier model update."""
        point = self.selected_handle
        if point is None:
            self.message = "Click a point to select it, then D to delete."
            return
        was_saved = point.id in self.point_editor.saved.get(self.selected_id, {})
        self._remember()
        self.point_editor.delete(self.selected_id, point.id)
        self._sync_point_prompt()
        self.selected_point = None
        self._cancel_drag()
        self.message = (
            "Saved marker removed. To restore excluded pixels, add an include point or redraw the box."
            if was_saved
            else "Draft point removed. U restores it."
        )

    def select_object(self, object_id: int) -> None:
        """Select an existing identity while editing the current frame."""
        if self.choosing_frame or not self.paused or not self.editable or object_id not in self.object_ids:
            return
        self.selected_id = object_id
        self.selected_point = None
        self._cancel_drag()
        count = len(self.point_editor.visible(object_id))
        self.message = (
            f"Object {object_id}: {count} point(s). 'prev' marks last saved positions. Click a point to edit."
            if count
            else f"Selected object {object_id}. Draw a replacement box or add points."
        )

    def mouse(self, event: str, x: int, y: int) -> None:
        """Apply a mouse event expressed in displayed-image coordinates."""
        if self.choosing_frame or not self.paused or not self.editable or self.viewport is None:
            return
        point = self.viewport.point(x, y, clip=event in {"move", "left_up"})
        if point is None:
            return
        if self._drag_point is not None and event in {"move", "left_up"}:
            self._point_drag_position = point
            if event == "left_up":
                if self.viewport.display_point(point) != self.viewport.display_point(self._drag_point.position):
                    self._edit_point(point, self._drag_point.label, point_id=self._drag_point.id)
                self._cancel_drag()
            return
        if event == "left_down":
            handle = self._hit_point(x, y)
            if handle is not None:
                self._cancel_drag()
                self.selected_point = handle.id
                self._drag_point = handle
                self._point_drag_position = handle.position
                self.message = "Point selected: drag to move; I includes, E excludes, D deletes."
                return
            self.selected_point = None
        if event == "right_down" or (self.mode == "points" and event == "left_down"):
            self._edit_point(point, int(event == "left_down"))
        elif self.mode == "box":
            if event == "left_down":
                self.drag_start = self.drag_end = point
            elif self.drag_start is not None and event == "move":
                self.drag_end = point
            elif self.drag_start is not None and event == "left_up":
                x1, x2 = sorted((self.drag_start[0], point[0]))
                y1, y2 = sorted((self.drag_start[1], point[1]))
                self.drag_start = self.drag_end = None
                if x2 - x1 < 2 or y2 - y1 < 2:
                    self.message = "Box is too small. Drag a box at least 2 pixels wide and tall."
                else:
                    previous = self.pending.get(self.selected_id, ObjectPrompt())
                    self._put(ObjectPrompt(box=(x1, y1, x2, y2), points=previous.points, labels=previous.labels))

    def handle_key(self, key: int) -> None:
        """Handle OpenCV key codes without performing inference or I/O."""
        if key < 0:
            return
        if key in (27, ord("q"), ord("Q")):
            self.quit = True
            return
        if self.choosing_frame:
            if key in (10, 13, 32):
                self.choosing_frame = False
                self.paused = self.editable = True
                self.message = "Frame selected. Draw a box or points, then Enter to track. R chooses another frame."
            return
        if self.live_source and self.paused and key in (ord("r"), ord("R")):
            self._clear_edits()
            self.choosing_frame = True
            self.paused = self.editable = False
            self.message = "Frame edits discarded. Space or Enter chooses another camera frame."
            return
        if key in (10, 13, 32):
            if self.paused:
                if not self.object_ids and not self.active_ids:
                    self.message = "Add at least one box or positive point before running."
                    return
                for object_id, prompt in self.pending.items():
                    if prompt.box is None and 1 not in prompt.labels and object_id not in self.active_ids:
                        self.message = f"New object {object_id} needs a box or positive point (left click)."
                        return
                self.paused = self.editable = False
                self.message = "Tracking. Space pauses on the next frame for edits."
            else:
                self.paused = True
                self.message = "Paused on the next frame. Tab selects an object to correct."
            return
        if key in (ord("b"), ord("B"), ord("p"), ord("P"), ord("n"), ord("N")):
            self.paused = True
            self._cancel_drag()
            if key in (ord("n"), ord("N")):
                self.selected_id = self._next_id
                self._next_id += 1
                self.selected_point = None
                self.message = f"New object {self.selected_id}: draw a box or add positive points."
            else:
                self.mode = "box" if key in (ord("b"), ord("B")) else "points"
                self.message = (
                    "Drag a rectangle; right-click excludes."
                    if self.mode == "box"
                    else "Left click includes; right click excludes."
                )
            return
        if not self.paused or not self.editable:
            return
        if key == 9 and self.object_ids:
            ids = self.object_ids
            index = ids.index(self.selected_id) + 1 if self.selected_id in ids else 0
            self.select_object(ids[index % len(ids)])
        elif key in (127, ord("d"), ord("D")):
            self._delete_point()
        elif key in (ord("i"), ord("I"), ord("e"), ord("E")):
            point = self.selected_handle
            if point is not None:
                self._edit_point(point.position, int(key in (ord("i"), ord("I"))), point_id=point.id)
                self._cancel_drag()
        elif key in (8, ord("u"), ord("U")):
            history = self._history.get(self.selected_id, [])
            if history:
                previous = history.pop()
                self.point_editor.restore(self.selected_id, previous.point_changes)
                if previous.prompt is None:
                    self.pending.pop(self.selected_id, None)
                else:
                    self.pending[self.selected_id] = previous.prompt
            self.selected_point = None
            self._cancel_drag()
            self.message = "Undid the latest draft prompt."
        elif key in (ord("x"), ord("X")):
            self.pending.pop(self.selected_id, None)
            self._history.pop(self.selected_id, None)
            self.point_editor.restore(self.selected_id, {})
            self.selected_point = None
            self._cancel_drag()
            if self.selected_id in self.active_ids:
                self.removed.add(self.selected_id)
            self.message = f"Object {self.selected_id} will be removed when tracking resumes."
        elif key in (ord("c"), ord("C")):
            self._clear_edits()
            self.message = "Cleared this frame's edits; tracked objects are unchanged."

    def _clear_edits(self) -> None:
        """Discard changes for this frame without changing committed object IDs."""
        self.pending.clear()
        self.point_editor.discard()
        self._history.clear()
        self.removed.clear()
        self.selected_point = None
        self._cancel_drag()

    def commit(self) -> None:
        """Keep editable point records separate from already submitted inference inputs."""
        self.active_ids.difference_update(self.removed)
        self.active_ids.update(self.pending)
        self.point_editor.commit(self.removed)
        self._clear_edits()


def require_display() -> None:
    """Reject missing desktop/OpenCV GUI support before loading any model."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("Interactive EdgeTAM tracking needs a desktop display. Run it in a graphical session.")
    gui = re.search(r"^\s*GUI:\s*(.*)$", cv2.getBuildInformation(), re.MULTILINE)
    if gui is not None and gui.group(1).strip().upper() in {"NONE", "NO"}:
        raise RuntimeError("Interactive EdgeTAM tracking needs OpenCV GUI support; use a desktop OpenCV build.")


class PromptWindow:
    """OpenCV adapter with a fixed canvas and explicit image-to-pixel mapping."""

    _HEADER_HEIGHT = 52
    _HUD_HEIGHT = 160
    _BACKGROUND = (30, 23, 18)
    _SURFACE = (42, 33, 26)
    _BORDER = (65, 54, 43)
    _TEXT = (245, 241, 233)
    _MUTED = (171, 156, 139)
    _ACCENT = (196, 231, 110)

    def __init__(self, controller: PromptController | None = None) -> None:
        self.controller = controller or PromptController()
        self.name = "BoxMOT EdgeTAM - boxes and points"
        self._logo = self._load_logo()
        self._opened = False
        self._frame_prompts: dict[int, ObjectPrompt] = {}
        self._buttons: list[_PromptButton] = []
        self._pointer: tuple[int, int] | None = None

    @classmethod
    def _load_logo(cls) -> np.ndarray:
        """Prepare the bundled logo once, preserving its proportions and transparency."""
        data = files("boxmot.engine.ui").joinpath("assets", "logo_white.png").read_bytes()
        logo = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if logo is None or logo.ndim != 3 or logo.shape[2] != 4:
            raise RuntimeError("Could not decode the bundled BoxMOT logo.")
        x, y, width, height = cv2.boundingRect(logo[:, :, 3])
        logo = logo[y : y + height, x : x + width]
        alpha = logo[:, :, 3:4].astype(np.float32) / 255.0
        background = np.asarray(cls._SURFACE, dtype=np.float32)
        blended = np.rint(logo[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
        display_height = cls._HEADER_HEIGHT - 16
        return cv2.resize(
            blended, (round(width * display_height / height), display_height), interpolation=cv2.INTER_AREA
        )

    def open(self) -> None:
        """Create the interactive window before constructing a tracker."""
        require_display()
        try:
            flags = cv2.WINDOW_AUTOSIZE
            if cv2.currentUIFramework() == "QT":
                # Qt's expanded controls add a right-click menu over negative-point clicks.
                # Keep this Qt-only: Cocoa requires the exact AUTOSIZE flag value.
                flags |= cv2.WINDOW_GUI_NORMAL
            cv2.namedWindow(self.name, flags)
            self._opened = True
            cv2.setMouseCallback(self.name, self._mouse)
        except cv2.error as exc:
            self.close()
            raise RuntimeError(
                "Could not open the EdgeTAM prompt window; a working desktop display is required."
            ) from exc

    def _mouse(self, event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        """Route toolbar clicks before forwarding image gestures to the controller."""
        self._pointer = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for button in self._buttons:
                x1, y1, x2, y2 = button.bounds
                if x1 <= x < x2 and y1 <= y < y2:
                    self.controller._cancel_drag()
                    if button.object_id is not None:
                        self.controller.select_object(button.object_id)
                    else:
                        self.controller.handle_key(button.key)
                    return
        names = {
            cv2.EVENT_LBUTTONDOWN: "left_down",
            cv2.EVENT_LBUTTONUP: "left_up",
            cv2.EVENT_MOUSEMOVE: "move",
            cv2.EVENT_RBUTTONDOWN: "right_down",
        }
        if event in names:
            self.controller.mouse(names[event], x, y)

    def _poll(self, delay: int) -> None:
        try:
            if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                self.controller.quit = True
                return
            key = cv2.waitKey(delay)
        except cv2.error:
            self.controller.quit = True
            return
        self.controller.handle_key(key & 0xFF if key >= 0 else -1)

    def prepare_frame(self, frame: Frame) -> bool:
        """Preview or edit this frame without reading ahead; return false on quit.

        A true return with ``controller.choosing_frame`` still set means the
        caller should acquire another frame without tracking or saving this one.
        """
        self._frame_prompts.clear()
        scale = min(1.0, 1280 / frame.width, (720 - self._HEADER_HEIGHT - self._HUD_HEIGHT) / frame.height)
        width, height = max(1, round(frame.width * scale)), max(1, round(frame.height * scale))
        self.controller.viewport = PromptViewport(
            frame.width, frame.height, width, height, (max(1000, width) - width) // 2, self._HEADER_HEIGHT
        )
        self.controller.editable = self.controller.paused and not self.controller.choosing_frame
        if self.controller.paused or self.controller.choosing_frame:
            image = np.ascontiguousarray(frame.image.permute(1, 2, 0).numpy()[..., ::-1])
            if self.controller.choosing_frame:
                self._draw(image)
                self._poll(1)
            while (
                self.controller.paused
                and self.controller.editable
                and not self.controller.choosing_frame
                and not self.controller.quit
            ):
                self._draw(image)
                self._poll(30)
        self.controller.editable = False
        if not self.controller.quit and not self.controller.choosing_frame:
            self._frame_prompts = dict(self.controller.pending)
        return not self.controller.quit

    @classmethod
    def _draw_point(
        cls,
        canvas: np.ndarray,
        position: tuple[int, int],
        label: int,
        *,
        historical: bool = False,
        selected: bool = False,
    ) -> None:
        """Distinguish saved handles from fresh signed dots and highlight selection."""
        color = (100, 230, 100) if label else (80, 80, 255)
        if selected:
            cv2.circle(canvas, position, 13, cls._ACCENT, 2, cv2.LINE_AA)
        cv2.circle(canvas, position, 10, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, position, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, position, 7, color, -1, cv2.LINE_AA)
        if historical:
            cv2.circle(canvas, position, 5, cls._BACKGROUND, -1, cv2.LINE_AA)
        stroke = color if historical else (20, 20, 20)
        x, y = position
        cv2.line(canvas, (x - 4, y), (x + 4, y), stroke, 2, cv2.LINE_AA)
        if label:
            cv2.line(canvas, (x, y - 4), (x, y + 4), stroke, 2, cv2.LINE_AA)

    def _draw_prompt_point(
        self,
        canvas: np.ndarray,
        object_id: int,
        point: tuple[float, float],
        label: int,
        *,
        historical: bool = False,
        selected: bool = False,
    ) -> None:
        """Draw a point and a readable object/age label in video coordinates."""
        viewport = self.controller.viewport
        assert viewport is not None
        position = viewport.display_point(point)
        self._draw_point(canvas, position, label, historical=historical, selected=selected)
        text = f"{object_id}{'+' if label else '-'}" + (" prev" if historical else "")
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        origin = (
            min(max(position[0] + 16, 2), canvas.shape[1] - text_width - 2),
            min(
                max(position[1] - 10, viewport.top + 16),
                max(viewport.top + 16, viewport.top + viewport.display_height - 4),
            ),
        )
        for text_color, thickness in (((20, 20, 20), 3), ((255, 255, 255), 1)):
            self._text(canvas, text, origin, color=text_color, scale=0.5, thickness=thickness)

    @staticmethod
    def _panel(
        canvas: np.ndarray, bounds: tuple[int, int, int, int], color: tuple[int, int, int], radius: int = 6
    ) -> None:
        """Paint a small rounded surface without allocating another image."""
        x1, y1, x2, y2 = bounds
        cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for x in (x1 + radius, x2 - radius):
            for y in (y1 + radius, y2 - radius):
                cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)

    @classmethod
    def _text(
        cls,
        canvas: np.ndarray,
        text: str,
        origin: tuple[int, int],
        *,
        color: tuple[int, int, int] | None = None,
        scale: float = 0.43,
        thickness: int = 1,
        max_width: int | None = None,
    ) -> None:
        """Draw readable text, truncating long status messages within their region."""
        if (
            max_width is not None
            and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_width
        ):
            while text and cv2.getTextSize(text + "...", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_width:
                text = text[:-1]
            text += "..."
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color or cls._TEXT, thickness, cv2.LINE_AA)

    def _button(
        self,
        canvas: np.ndarray,
        bounds: tuple[int, int, int, int],
        title: str,
        shortcut: str,
        key: int,
        *,
        active: bool = False,
        primary: bool = False,
        enabled: bool = True,
        object_id: int | None = None,
    ) -> None:
        """Draw an actionable control with its keyboard equivalent and hit region."""
        x1, y1, x2, y2 = bounds
        hovered = self._pointer is not None and x1 <= self._pointer[0] < x2 and y1 <= self._pointer[1] < y2
        fill = self._ACCENT if primary else (63, 71, 43) if active else self._SURFACE
        foreground = self._BACKGROUND if primary else self._ACCENT if active else self._TEXT if enabled else self._MUTED
        border = self._ACCENT if active or primary else (109, 98, 79) if hovered and enabled else self._BORDER
        self._panel(canvas, bounds, border)
        self._panel(canvas, (x1 + 1, y1 + 1, x2 - 1, y2 - 1), fill, radius=5)
        baseline = (y1 + y2) // 2 + 5
        self._text(canvas, title, (x1 + 11, baseline), color=foreground)
        if shortcut:
            key_width = cv2.getTextSize(shortcut, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0] + 12
            key_x = x2 - key_width - 8
            self._panel(canvas, (key_x, y1 + 7, x2 - 8, y2 - 7), (155, 191, 80) if primary else self._BACKGROUND, 3)
            self._text(canvas, shortcut, (key_x + 6, baseline - 1), color=foreground, scale=0.32)
        if enabled:
            self._buttons.append(_PromptButton(bounds, key, object_id))

    def _draw_header(self, canvas: np.ndarray) -> None:
        """Keep the tracker identity and current interaction state easy to scan."""
        width = canvas.shape[1]
        cv2.rectangle(canvas, (0, 0), (width, self._HEADER_HEIGHT - 1), self._SURFACE, -1)
        logo_height, logo_width = self._logo.shape[:2]
        logo_top = (self._HEADER_HEIGHT - logo_height) // 2
        canvas[logo_top : logo_top + logo_height, 16 : 16 + logo_width] = self._logo
        divider = 16 + logo_width + 20
        cv2.line(canvas, (divider, 16), (divider, 36), self._BORDER, 1)
        self._text(canvas, "EdgeTAM", (divider + 17, 33), scale=0.56)
        self._text(canvas, "INTERACTIVE TRACKING", (divider + 128, 31), color=self._MUTED, scale=0.34)
        state, color = (
            ("LIVE PREVIEW", self._ACCENT)
            if self.controller.choosing_frame
            else (("PAUSED", (120, 206, 249)) if self.controller.paused else ("TRACKING", (151, 222, 113)))
        )
        self._panel(canvas, (width - 175, 11, width - 16, 41), self._BACKGROUND)
        cv2.circle(canvas, (width - 159, 26), 4, color, -1, cv2.LINE_AA)
        self._text(canvas, state, (width - 145, 31), color=color, scale=0.42)

    def _draw_toolbar(self, canvas: np.ndarray) -> None:
        """Group tools, object selection, and feedback in a compact control dock."""
        viewport = self.controller.viewport
        assert viewport is not None
        y = viewport.top + viewport.display_height
        width = canvas.shape[1]
        controller = self.controller
        cv2.line(canvas, (0, y), (width, y), self._BORDER, 1)
        can_prompt = not controller.choosing_frame
        editing = can_prompt and controller.paused and controller.editable
        self._button(
            canvas, (16, y + 16, 120, y + 50), "Box", "B", ord("b"), active=controller.mode == "box", enabled=can_prompt
        )
        self._button(
            canvas,
            (128, y + 16, 248, y + 50),
            "Points",
            "P",
            ord("p"),
            active=controller.mode == "points",
            enabled=can_prompt,
        )
        self._button(canvas, (256, y + 16, 404, y + 50), "New object", "N", ord("n"), enabled=can_prompt)
        include_hint = "Left click" if controller.mode == "points" else "P + left click"
        for x, label, title, hint in ((430, 1, "Include", include_hint), (568, 0, "Exclude", "Right click")):
            self._draw_point(canvas, (x, y + 32), label)
            self._text(canvas, title, (x + 18, y + 29), scale=0.40)
            self._text(canvas, hint, (x + 18, y + 45), color=self._MUTED, scale=0.32)
        action = (
            "Choose frame" if controller.choosing_frame else "Start tracking" if controller.paused else "Pause / edit"
        )
        self._button(canvas, (width - 208, y + 16, width - 16, y + 50), action, "Enter", 13, primary=True)
        self._text(canvas, "OBJECTS", (16, y + 87), color=self._MUTED, scale=0.35)
        if controller.choosing_frame:
            self._text(canvas, "Choose a frame to edit", (94, y + 87), color=self._MUTED, scale=0.40)
        else:
            ids = sorted(set(controller.object_ids) | {controller.selected_id})
            visible_ids = ids[:5]
            if controller.selected_id not in visible_ids:
                visible_ids[-1] = controller.selected_id
            for index, object_id in enumerate(visible_ids):
                x = 94 + index * 48
                self._button(
                    canvas,
                    (x, y + 66, x + 40, y + 100),
                    f"{object_id:02d}",
                    "",
                    9,
                    active=object_id == controller.selected_id,
                    enabled=editing and object_id in controller.object_ids,
                    object_id=object_id,
                )
            x = 94 + len(visible_ids) * 48
            self._button(
                canvas,
                (x, y + 66, x + 94, y + 100),
                f"+{len(ids) - 5}" if len(ids) > 5 else "Next",
                "Tab",
                9,
                enabled=editing and bool(controller.object_ids),
            )
        if controller.live_source:
            self._button(
                canvas, (width - 474, y + 66, width - 328, y + 100), "New frame", "R", ord("r"), enabled=editing
            )
        for offset, button_width, title, shortcut in (
            (320, 90, "Undo", "U"),
            (222, 90, "Clear", "C"),
            (124, 108, "Remove", "X"),
        ):
            self._button(
                canvas,
                (width - offset, y + 66, width - offset + button_width, y + 100),
                title,
                shortcut,
                ord(shortcut.lower()),
                enabled=editing,
            )
        cv2.line(canvas, (16, y + 113), (width - 16, y + 113), self._BORDER, 1)
        selected_point = editing and controller.selected_handle is not None
        self._text(
            canvas,
            controller.message,
            (16, y + 139),
            color=self._MUTED,
            scale=0.41,
            max_width=width - (340 if selected_point else 180),
        )
        if selected_point:
            self._button(canvas, (width - 300, y + 121, width - 140, y + 151), "Delete point", "D", ord("d"))
        self._button(canvas, (width - 132, y + 121, width - 16, y + 151), "Quit", "Esc", 27)

    def _draw_box(self, canvas: np.ndarray, box: tuple[float, float, float, float], object_id: int) -> None:
        """Use a compact identity tag and corner accents for each prompt box."""
        viewport = self.controller.viewport
        assert viewport is not None
        p1, p2 = viewport.display_point(box[:2]), viewport.display_point(box[2:])
        color = self._ACCENT if object_id == self.controller.selected_id else self._MUTED
        cv2.rectangle(canvas, p1, p2, color, 1, cv2.LINE_AA)
        length = min(14, max(1, (p2[0] - p1[0]) // 3), max(1, (p2[1] - p1[1]) // 3))
        for x, dx in ((p1[0], length), (p2[0], -length)):
            for y, dy in ((p1[1], length), (p2[1], -length)):
                cv2.line(canvas, (x, y + dy), (x, y), color, 2, cv2.LINE_AA)
                cv2.line(canvas, (x, y), (x + dx, y), color, 2, cv2.LINE_AA)
        title = f"Object {object_id:02d}"
        tag_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0] + 16
        x = min(p1[0], max(0, canvas.shape[1] - tag_width - 1))
        y = max(viewport.top, p1[1] - 27)
        self._panel(canvas, (x, y, x + tag_width, y + 23), self._BACKGROUND, 4)
        self._text(canvas, title, (x + 8, y + 16), color=color, scale=0.38)

    def _draw(self, image: np.ndarray, *, frame_prompts: Mapping[int, ObjectPrompt] | None = None) -> None:
        """Render the native editor without changing frames sent to output sinks."""
        viewport = self.controller.viewport
        assert viewport is not None
        canvas = np.full(
            (viewport.top + viewport.display_height + self._HUD_HEIGHT, max(1000, viewport.display_width), 3),
            self._BACKGROUND,
            np.uint8,
        )
        resized = cv2.resize(image, (viewport.display_width, viewport.display_height))
        canvas[
            viewport.top : viewport.top + viewport.display_height,
            viewport.left : viewport.left + viewport.display_width,
        ] = resized
        visible_prompts = self.controller.pending if frame_prompts is None else frame_prompts
        editing = frame_prompts is None and self.controller.editable and not self.controller.choosing_frame
        for object_id, prompt in visible_prompts.items():
            if prompt.box is not None:
                self._draw_box(canvas, prompt.box, object_id)
            if editing and object_id == self.controller.selected_id:
                continue  # Render this object's current and saved handles together below.
            for point, label in zip(prompt.points, prompt.labels):
                self._draw_prompt_point(canvas, object_id, point, label)
        if editing and self.controller.selected_id not in self.controller.removed:
            object_id = self.controller.selected_id
            changed_ids = {point.id for point in self.controller.point_editor.changed(object_id)}
            for point in self.controller.point_editor.visible(object_id):
                position = point.position
                historical = point.id not in changed_ids
                if self.controller._drag_point is not None and self.controller._drag_point.id == point.id:
                    position = self.controller._point_drag_position or position
                    historical = historical and position == point.position
                self._draw_prompt_point(
                    canvas,
                    object_id,
                    position,
                    point.label,
                    historical=historical,
                    selected=point.id == self.controller.selected_point,
                )
        if self.controller.drag_start is not None and self.controller.drag_end is not None:
            x1, x2 = sorted((self.controller.drag_start[0], self.controller.drag_end[0]))
            y1, y2 = sorted((self.controller.drag_start[1], self.controller.drag_end[1]))
            self._draw_box(canvas, (x1, y1, x2, y2), self.controller.selected_id)
        self._buttons.clear()
        self._draw_header(canvas)
        self._draw_toolbar(canvas)
        cv2.imshow(self.name, canvas)

    def show_result(self, image: np.ndarray) -> None:
        """Display tracked masks and poll controls without editing a processed frame."""
        self.controller.editable = False
        self._draw(image, frame_prompts=self._frame_prompts)
        self._poll(1)

    def close(self) -> None:
        """Release this window, including when inference or source iteration fails."""
        if self._opened:
            self._opened = False
            try:
                cv2.destroyWindow(self.name)
            except cv2.error:
                pass  # A user may have already closed the native window.
