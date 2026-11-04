"""Readable track labels for axis-aligned and oriented image overlays."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest
import torch

import boxmot.engine.tracking.sinks as sink_module
from boxmot.engine.tracking.sinks import render_result
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes, Tracks


def _render(
    rows: list[list[float]],
    *,
    is_obb: bool = False,
    class_label: str = "heavy vehicle",
    line_width: int = 2,
) -> tuple[np.ndarray, Frame]:
    """Render visible identities against a blank frame without detector inference."""
    frame = Frame(torch.zeros((3, 180, 320), dtype=torch.uint8), "labels")
    values = torch.tensor(rows, dtype=torch.float32)
    geometry = OrientedBoxes(values) if is_obb else Boxes(values)
    scores = torch.full((len(rows),), 0.9, dtype=torch.float32)
    classes = torch.zeros(len(rows), dtype=torch.int64)
    detections = Detections(geometry, scores, classes, frame.sample_id)
    tracks = Tracks(
        geometry,
        torch.arange(1, len(rows) + 1, dtype=torch.int64),
        scores,
        classes,
        torch.arange(len(rows), dtype=torch.int64),
        frame.sample_id,
    )
    image = render_result(
        frame,
        PipelineResult(detections, tracks),
        class_names={0: class_label},
        line_width=line_width,
    )
    return image, frame


def _record_drawing(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Observe actual draw calls while preserving OpenCV's pixel rendering."""
    sink_module._label_text_mask.cache_clear()
    calls: list[dict[str, Any]] = []
    rectangle = cv2.rectangle
    put_text = cv2.putText
    polylines = cv2.polylines

    def record_rectangle(image, pt1, pt2, color, thickness=1, *args, **kwargs):
        calls.append({"kind": "badge" if thickness < 0 else "box", "start": pt1, "end": pt2, "color": color})
        return rectangle(image, pt1, pt2, color, thickness, *args, **kwargs)

    def record_text(image, text, org, fontFace, fontScale, color, thickness=1, *args, **kwargs):
        calls.append({
            "kind": "text",
            "content": text,
            "font_scale": fontScale,
        })
        return put_text(image, text, org, fontFace, fontScale, color, thickness, *args, **kwargs)

    def record_polylines(image, pts, isClosed, color, *args, **kwargs):
        calls.append({"kind": "box", "color": color})
        return polylines(image, pts, isClosed, color, *args, **kwargs)

    monkeypatch.setattr(cv2, "rectangle", record_rectangle)
    monkeypatch.setattr(cv2, "putText", record_text)
    monkeypatch.setattr(cv2, "polylines", record_polylines)
    return calls


def _visible_ink(
    image: np.ndarray, badge: dict[str, Any]
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Locate the actual glyph pixels independently of OpenCV's text origin."""
    x1, y1 = badge["start"]
    x2, y2 = badge["end"]
    pixels = image[y1 : y2 + 1, x1 : x2 + 1]
    ink = np.any(pixels != np.asarray(badge["color"]), axis=2)
    ys, xs = np.nonzero(ink)
    assert len(xs), "The complete label must remain visible."
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return (
        (x1 + left, y1 + top, x1 + right, y1 + bottom),
        ink[top : bottom + 1, left : right + 1],
        pixels[top : bottom + 1, left : right + 1],
    )


def _reference_ink(label: str, font_scale: float) -> np.ndarray:
    """Rasterize a full unclipped reference, including faint anti-aliased pixels."""
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    guard = 8
    mask = np.zeros((height + baseline + 2 * guard, width + 2 * guard), dtype=np.uint8)
    cv2.putText(
        mask,
        label,
        (guard, guard + height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        255,
        1,
        cv2.LINE_AA,
    )
    ys, xs = np.nonzero(mask)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


@pytest.mark.parametrize("is_obb", (False, True))
@pytest.mark.parametrize(("line_width", "stroke_radius"), enumerate((0, 1, 2, 2, 3, 3, 4, 4), start=1))
def test_label_badge_attaches_to_box_top_and_contains_text_baseline(
    monkeypatch: pytest.MonkeyPatch,
    is_obb: bool,
    line_width: int,
    stroke_radius: int,
) -> None:
    """Attach a matching identity badge to a real corner with all text inside."""
    calls = _record_drawing(monkeypatch)
    if is_obb:
        rows = [[150.0, 120.0, 90.0, 40.0, 0.6]]
        corners = cv2.boxPoints(((150.0, 120.0), (90.0, 40.0), float(np.degrees(0.6))))
        corners = np.rint(corners).astype(int)
        anchor = tuple(corners[np.argmin(corners[:, 1])])
    else:
        rows = [[70.0, 100.0, 200.0, 160.0]]
        anchor = (70, 100)

    image, frame = _render(rows, is_obb=is_obb, line_width=line_width)

    badge = next(call for call in calls if call["kind"] == "badge")
    box = next(call for call in calls if call["kind"] == "box")
    ink_bounds, _ink, _pixels = _visible_ink(image, badge)
    assert badge["start"][0] == anchor[0] - stroke_radius
    assert badge["end"][1] == anchor[1]
    assert ink_bounds[0] == anchor[0] + stroke_radius + 1
    assert ink_bounds[1] >= badge["start"][1] + 2
    assert ink_bounds[2] <= badge["end"][0] - 2
    assert ink_bounds[3] <= badge["end"][1] - 2
    assert badge["color"] == box["color"]
    assert np.any(image)
    assert not bool(frame.image.any())


@pytest.mark.parametrize("class_label", ("person", "heavy vehicle", "0", "W", "j", " person", "   j"))
@pytest.mark.parametrize("line_width", range(1, 9))
def test_first_visible_glyph_starts_immediately_inside_box_border(
    monkeypatch: pytest.MonkeyPatch,
    class_label: str,
    line_width: int,
) -> None:
    """Account for glyph bearings without losing AA fringes or descenders."""
    reference = np.zeros((180, 320), dtype=np.uint8)
    cv2.rectangle(reference, (80, 100), (240, 165), 255, line_width)
    left_stroke = np.flatnonzero(reference[140, :160])
    first_interior_pixel = int(left_stroke[-1]) + 1
    calls = _record_drawing(monkeypatch)

    image, _frame = _render(
        [[80.0, 100.0, 240.0, 165.0]],
        class_label=class_label,
        line_width=line_width,
    )

    badge = next(call for call in calls if call["kind"] == "badge")
    bounds, ink, pixels = _visible_ink(image, badge)
    expected_mask = _reference_ink(f"{class_label} #1 0.90", 0.5)
    assert bounds[0] == first_interior_pixel
    np.testing.assert_array_equal(ink, expected_mask > 0)
    # Intermediate intensities prove the mask's antialiasing was retained.
    assert np.any((pixels > 5) & (pixels < np.asarray(badge["color"]) - 5))


@pytest.mark.parametrize("line_width", range(1, 9))
@pytest.mark.parametrize("top", (0, 10, 100))
def test_badge_left_pixels_match_box_stroke_and_join_without_a_notch(
    monkeypatch: pytest.MonkeyPatch,
    line_width: int,
    top: int,
) -> None:
    """Compare the join against OpenCV's raster, including odd/even stroke widths."""
    reference = np.zeros((180, 320), dtype=np.uint8)
    cv2.rectangle(reference, (80, top), (240, 165), 255, line_width)
    box_left = int(np.flatnonzero(reference[140])[0])
    box_top = int(np.flatnonzero(reference[:, 160])[0])
    calls = _record_drawing(monkeypatch)

    image, _frame = _render([[80.0, float(top), 240.0, 165.0]], line_width=line_width)

    badge = next(call for call in calls if call["kind"] == "badge")
    color = np.asarray(badge["color"])
    badge_row = np.flatnonzero(np.all(image[badge["start"][1]] == color, axis=1))
    side_row = np.flatnonzero(np.all(image[140] == color, axis=1))
    assert int(badge_row[0]) == int(side_row[0]) == box_left
    if top == 100:
        assert badge["end"][1] == top
    else:
        assert badge["start"][1] == box_top
    # This full column catches both a horizontal offset and a rounded-corner gap.
    join = image[badge["start"][1] : 141, box_left]
    np.testing.assert_array_equal(join, np.broadcast_to(color, join.shape))
    x1, y1 = badge["start"]
    x2, y2 = badge["end"]
    text_pixels = image[y1 : y2 + 1, x1 : x2 + 1].astype(int)
    assert np.max(np.abs(text_pixels - color)) > 100


@pytest.mark.parametrize("left", (-12.0, 300.0))
@pytest.mark.parametrize("is_obb", (False, True))
def test_top_edge_label_falls_inside_image_and_clamps_horizontally(monkeypatch, left: float, is_obb: bool) -> None:
    """A box at either image corner keeps its entire label badge visible."""
    calls = _record_drawing(monkeypatch)

    if is_obb:
        half_height = float((35.0 * np.sin(0.6) + 20.0 * np.cos(0.6)) / 2)
        rows = [[left + 17.5, half_height, 35.0, 20.0, 0.6]]
    else:
        rows = [[left, 0.0, left + 35.0, 100.0]]
    image, _frame = _render(rows, is_obb=is_obb)

    badge = next(call for call in calls if call["kind"] == "badge")
    ink_bounds, ink, _pixels = _visible_ink(image, badge)
    height, width = image.shape[:2]
    assert 0 <= badge["start"][0] < badge["end"][0] < width
    assert badge["start"][1] == 0
    assert 0 < badge["end"][1] < height
    assert badge["start"][0] < ink_bounds[0]
    assert ink_bounds[2] <= badge["end"][0] - 2
    assert ink_bounds[1] >= badge["start"][1] + 2
    assert ink_bounds[3] <= badge["end"][1] - 2
    np.testing.assert_array_equal(ink, _reference_ink("heavy vehicle #1 0.90", 0.5) > 0)


def test_later_box_border_does_not_overwrite_an_earlier_label_badge(monkeypatch) -> None:
    """Render overlapping outlines first so labels remain legible on top."""
    calls = _record_drawing(monkeypatch)
    first_box = [80.0, 100.0, 240.0, 165.0]
    _render([first_box])
    first_badge = next(call for call in calls if call["kind"] == "badge")
    crossing_x = first_badge["start"][0] + 1
    crossing_y = (first_badge["start"][1] + first_badge["end"][1]) // 2
    calls.clear()

    image, _frame = _render([first_box, [float(crossing_x), 20.0, 300.0, 175.0]])

    kinds = [call["kind"] for call in calls]
    assert max(index for index, kind in enumerate(kinds) if kind == "box") < kinds.index("badge")
    badge = next(call for call in calls if call["kind"] == "badge")
    # The second outline crosses this pixel, safely underneath the first badge.
    np.testing.assert_array_equal(image[crossing_y, crossing_x], badge["color"])


@pytest.mark.parametrize(
    "class_label",
    ("heavy vehicle with an extended classification name", "person " * 20),
)
def test_full_long_label_fits_at_top_right_edge(monkeypatch: pytest.MonkeyPatch, class_label: str) -> None:
    """Fit and reposition the complete class, identity, and score without clipping."""
    calls = _record_drawing(monkeypatch)
    image, _frame = _render([[300.0, 0.0, 319.0, 100.0]], class_label=class_label)

    badge = next(call for call in calls if call["kind"] == "badge")
    text = [call for call in calls if call["kind"] == "text"][-1]
    ink_bounds, ink, _pixels = _visible_ink(image, badge)
    assert text["content"] == f"{class_label} #1 0.90"
    assert 0 < text["font_scale"] < 0.5
    assert 0 <= badge["start"][0] < badge["end"][0] == image.shape[1] - 1
    assert badge["start"][1] == 0
    assert badge["start"][0] < ink_bounds[0]
    assert ink_bounds[2] <= badge["end"][0] - 2
    assert ink_bounds[1] >= badge["start"][1] + 2
    assert ink_bounds[3] <= badge["end"][1] - 2
    np.testing.assert_array_equal(ink, _reference_ink(text["content"], text["font_scale"]) > 0)


@pytest.mark.parametrize(
    ("identity_color", "text_color"),
    (((240, 240, 240), (0, 0, 0)), ((15, 15, 15), (255, 255, 255))),
)
def test_label_text_contrasts_with_light_and_dark_identity_colors(
    monkeypatch: pytest.MonkeyPatch,
    identity_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    """Both bright and dark track badges keep their text readable."""
    calls = _record_drawing(monkeypatch)
    monkeypatch.setattr(sink_module, "_track_color", lambda _track_id: identity_color)

    image, _frame = _render([[70.0, 100.0, 200.0, 160.0]])

    badge = next(call for call in calls if call["kind"] == "badge")
    assert badge["color"] == identity_color
    x1, y1 = badge["start"]
    x2, y2 = badge["end"]
    pixels = image[y1 : y2 + 1, x1 : x2 + 1]
    contrast = np.abs(pixels.astype(int) - np.array(identity_color))
    assert np.any(np.all(contrast > 150, axis=2))
    _bounds, _ink, glyph_pixels = _visible_ink(image, badge)
    alpha = _reference_ink("heavy vehicle #1 0.90", 0.5).astype(float)[..., None] / 255
    expected_pixels = np.rint(
        np.asarray(identity_color) * (1 - alpha) + np.asarray(text_color) * alpha
    ).astype(np.uint8)
    np.testing.assert_array_equal(glyph_pixels, expected_pixels)
