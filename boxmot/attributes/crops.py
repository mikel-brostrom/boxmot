"""Bounded image encoding and same-track chronological storyboards."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.models import AttributeSample
from boxmot.structures import Frame, Tracks
from boxmot.trackers.common.geometry.obb import xywha_to_xyxy


def encode_track_crop(frame: Frame, tracks: Tracks, index: int, config: AttributeConfig) -> bytes | None:
    """Encode current visible evidence as JPEG; prediction-only rows are skipped."""
    if tracks.sample_id != frame.sample_id:
        raise ValueError("Attribute frame and track sample IDs must match.")
    has_mask = False
    if tracks.masks is not None:
        if tracks.masks.image_size != frame.image_size:
            raise ValueError("Attribute masks must have the same image size as the frame.")
        has_mask = bool(tracks.masks.values[index].any())
    if int(tracks.detection_indices[index]) < 0:
        if not has_mask:
            return None
        # Propagation provides current pixels even when box geometry is stale.
        # Inclusive mask extrema become exclusive upper crop bounds.
        ys, xs = tracks.masks.values[index].nonzero(as_tuple=True)
        bounds = np.array([[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]])
    else:
        bounds = tracks.geometry.values[index : index + 1].numpy()
        if tracks.is_obb:
            bounds = xywha_to_xyxy(bounds)
    clipped = np.clip(bounds[0], (0, 0, 0, 0), (frame.width, frame.height, frame.width, frame.height))
    x1, y1 = np.floor(clipped[:2]).astype(int)
    x2, y2 = np.ceil(clipped[2:]).astype(int)
    if x2 <= x1 or y2 <= y1:
        return None
    # Only the bounded crop is copied from canonical RGB to OpenCV BGR.
    rgb = frame.image[:, y1:y2, x1:x2].permute(1, 2, 0).numpy()
    crop = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = crop.shape[:2]
    scale = min(1.0, config.max_crop_side / max(height, width))
    if scale < 1.0:
        crop = cv2.resize(
            crop, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
    return encoded.tobytes() if success else None


def build_storyboard(track_id: int, samples: Sequence[AttributeSample], images: Sequence[bytes]) -> bytes:
    """Keep a single bounded crop or combine multiple crops in a labeled JPEG."""
    if not 1 <= len(samples) <= 5 or len(samples) != len(images):
        raise ValueError("Attribute batches require one to five aligned samples and images.")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("Attribute batches cannot contain duplicate sample IDs.")
    for name in ("frame_index", "timestamp_s"):
        values = [getattr(sample, name) for sample in samples if getattr(sample, name) is not None]
        if values != sorted(values):
            raise ValueError("Attribute samples must be chronological.")
    if len(samples) == 1:
        # Padding or upscaling adds vision tokens without adding evidence.
        crop = _decode_crop(images[0])
        if images[0].startswith(b"\xff\xd8\xff"):
            return images[0]
        # The HTTP transport declares image/jpeg, including for caller-supplied crops.
        success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Could not encode attribute crop.")
        return encoded.tobytes()
    tile_width, tile_height, header_height = 320, 480, 42
    columns = min(3, len(samples))
    rows = (len(samples) + columns - 1) // columns
    storyboard = np.full((rows * tile_height, columns * tile_width, 3), 24, dtype=np.uint8)
    for position, (sample, encoded) in enumerate(zip(samples, images)):
        crop = _decode_crop(encoded)
        row, column = divmod(position, columns)
        tile = storyboard[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width]
        scale = min(tile_width / crop.shape[1], (tile_height - header_height) / crop.shape[0])
        width, height = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
        resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        x0, y0 = (tile_width - width) // 2, header_height + (tile_height - header_height - height) // 2
        tile[y0 : y0 + height, x0 : x0 + width] = resized
        cv2.rectangle(tile, (0, 0), (tile_width - 1, header_height - 1), (60, 140, 60), -1)
        frame_label = f"FRAME {sample.frame_index}" if sample.frame_index is not None else f"SAMPLE {position + 1}"
        cv2.putText(
            tile, f"TRACK {track_id} | {frame_label}", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )
    success, encoded = cv2.imencode(".jpg", storyboard, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Could not encode attribute storyboard.")
    return encoded.tobytes()


def _decode_crop(encoded: bytes) -> np.ndarray:
    """Validate encoded image evidence before forwarding or composing it."""
    if not encoded:
        raise ValueError("Attribute crop is not a readable encoded image.")
    crop = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if crop is None:
        raise ValueError("Attribute crop is not a readable encoded image.")
    return crop
