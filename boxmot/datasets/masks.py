"""Lossless bit-packed storage for full-frame boolean masks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

MASK_CODEC = "bitpack-row-major-v1"


class MaskCodecError(ValueError):
    """Raised when a materialized mask payload violates the storage contract."""


def _as_boolean_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        if mask.device.type != "cpu":
            raise MaskCodecError("Masks must be on CPU before materialization.")
        if mask.dtype is not torch.bool:
            raise MaskCodecError(f"Masks must have boolean torch.bool dtype, got {mask.dtype}.")
        array = mask.detach().numpy()
    else:
        array = np.asarray(mask)
        if array.dtype != np.bool_:
            raise MaskCodecError(f"Masks must have boolean dtype, got {array.dtype}.")
    if array.ndim != 2:
        raise MaskCodecError(f"A mask must have shape [H, W], got {array.shape}.")
    return np.ascontiguousarray(array)


def packed_mask_size(height: int, width: int) -> int:
    """Return the exact byte count for a bit-packed ``height`` by ``width`` mask."""

    if height < 0 or width < 0:
        raise MaskCodecError("Mask dimensions must be non-negative.")
    return (height * width + 7) // 8


def pack_mask(mask: torch.Tensor | np.ndarray) -> bytes:
    """Encode one full-frame boolean mask without changing its resolution."""

    array = _as_boolean_mask(mask)
    return np.packbits(array.reshape(-1), bitorder="little").tobytes()


def _validate_mask_payload(data: bytes | bytearray | memoryview, height: int, width: int) -> bytes:
    """Return canonical bytes after validating length and zero padding bits."""

    expected = packed_mask_size(height, width)
    payload = bytes(data)
    if len(payload) != expected:
        raise MaskCodecError(f"Mask payload has {len(payload)} bytes; expected {expected} for {height}x{width}.")
    remainder = (height * width) % 8
    if payload and remainder and payload[-1] & ~((1 << remainder) - 1):
        raise MaskCodecError("Mask payload has non-zero bits outside the row-major image extent.")
    return payload


def unpack_mask(data: bytes | bytearray | memoryview, height: int, width: int) -> torch.Tensor:
    """Decode one bit-packed payload into a contiguous CPU boolean tensor."""

    payload = _validate_mask_payload(data, height, width)
    expected = len(payload)
    if expected == 0:
        return torch.empty((height, width), dtype=torch.bool)

    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little", count=height * width)
    return torch.from_numpy(unpacked.reshape(height, width).astype(np.bool_, copy=True)).contiguous()


def pack_mask_batch(masks: torch.Tensor | np.ndarray) -> list[bytes]:
    """Encode a canonical ``[N,H,W]`` boolean mask batch."""

    if isinstance(masks, torch.Tensor):
        if masks.device.type != "cpu" or masks.dtype is not torch.bool:
            raise MaskCodecError("Mask batches must be CPU torch.bool tensors.")
        array = masks.detach().numpy()
    else:
        array = np.asarray(masks)
        if array.dtype != np.bool_:
            raise MaskCodecError("Mask batches must have boolean dtype.")
    if array.ndim != 3:
        raise MaskCodecError(f"A mask batch must have shape [N, H, W], got {array.shape}.")
    return [pack_mask(mask) for mask in array]


def unpack_mask_batch(payloads: Iterable[bytes], height: int, width: int) -> torch.Tensor:
    """Decode payloads into a contiguous ``[N,H,W]`` boolean tensor."""

    decoded = [unpack_mask(payload, height, width) for payload in payloads]
    if not decoded:
        return torch.empty((0, height, width), dtype=torch.bool)
    return torch.stack(decoded).contiguous()


__all__ = (
    "MASK_CODEC",
    "MaskCodecError",
    "pack_mask",
    "pack_mask_batch",
    "packed_mask_size",
    "unpack_mask",
    "unpack_mask_batch",
)
