"""Select one ground-truth prompt per person for causal mask propagation."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path


@dataclass(frozen=True)
class PersonPrompt:
    """A person's first eligible box, with a zero-based frame and original ID."""

    frame_index: int
    object_id: int
    box: tuple[float, float, float, float]


def _integer_field(value: str, *, name: str, minimum: int) -> int:
    """Read an integral annotation field without rounding large identities."""
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if not number.is_finite() or number != number.to_integral_value() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(number)


def load_first_visible_prompts(
    path: Path,
    *,
    image_sizes: Mapping[int, tuple[int, int]],
    person_class_id: int,
    min_visibility: float = 1.0,
) -> tuple[PersonPrompt, ...]:
    """Read the earliest fully contained, sufficiently visible MOT box per ID.

    MOT rows contain ``frame,id,x,y,w,h,mark,class,visibility``; input frames
    start at one. ``image_sizes`` maps selected zero-based frames to their
    actual ``(height, width)``. Unselected frames, unmarked rows, other classes,
    nonpositive boxes, and boxes crossing an image boundary cannot seed a
    person. Earlier ineligible observations never consume an identity.

    Visibility defaults to one (fully visible). Boxes are never clipped, and
    later observations never correct or reseed an existing prompt. The result
    is sorted by frame and ID even when annotations are out of order.
    """
    if not math.isfinite(min_visibility) or not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be finite and within [0, 1].")
    if isinstance(person_class_id, bool) or not isinstance(person_class_id, Integral) or person_class_id < 0:
        raise ValueError("person_class_id must be a non-negative integer.")
    for frame_index, size in image_sizes.items():
        if isinstance(frame_index, bool) or not isinstance(frame_index, Integral) or frame_index < 0:
            raise ValueError("image_sizes keys must be non-negative integer frame indices.")
        if len(size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in size
        ):
            raise ValueError("image_sizes values must contain positive integer (height, width) pairs.")

    prompts: dict[int, PersonPrompt] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            for fields in reader:
                if not fields or (len(fields) == 1 and not fields[0].strip()):
                    continue
                if len(fields) < 9:
                    raise ValueError("MOT ground truth requires at least 9 columns, including visibility.")
                values = [float(value) for value in fields]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("MOT ground truth contains non-finite values.")
                frame_index = _integer_field(fields[0], name="MOT frame ID", minimum=1) - 1
                if frame_index not in image_sizes or values[6] <= 0 or values[7] != person_class_id:
                    continue
                object_id = _integer_field(fields[1], name="MOT object ID", minimum=0)
                if object_id > 2**63 - 1:
                    raise ValueError("MOT object ID must be within the nonnegative int64 range.")
                visibility = values[8]
                if not 0.0 <= visibility <= 1.0:
                    raise ValueError("MOT person visibility must be within [0, 1].")
                x, y, width, height = values[2:6]
                image_height, image_width = image_sizes[frame_index]
                if visibility < min_visibility or width <= 0 or height <= 0:
                    continue
                if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
                    continue
                previous = prompts.get(object_id)
                if previous is None or frame_index < previous.frame_index:
                    prompts[object_id] = PersonPrompt(frame_index, object_id, (x, y, x + width, y + height))
        except (ValueError, csv.Error) as exc:
            raise ValueError(f"Invalid MOT prompt ground truth at {path}:{reader.line_num}: {exc}") from exc
    return tuple(sorted(prompts.values(), key=lambda prompt: (prompt.frame_index, prompt.object_id)))
