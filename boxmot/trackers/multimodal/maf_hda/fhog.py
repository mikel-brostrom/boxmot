"""NumPy port of MAF-HDA's OpenCV latent-SVM FHOG descriptor.

Derived from ``GMPHD_MAF/fhog.cpp``. Copyright (C) 2010-2013,
University of Nizhny Novgorod. See ``APPEARANCE_LICENSE.txt`` for terms.
"""

from __future__ import annotations

import numpy as np


def fhog(image: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """Return 31-channel FHOG, excluding the outermost cell on each side.

    ``image`` is an HWC BGR patch whose dimensions are multiples of
    ``cell_size`` and contain at least four cells. Spatial voting, the four
    block normalizations, 0.2 clipping, and channel reduction follow the
    upstream ``getFeatureMaps``, ``normalizeAndTruncate``, and ``PCAFeatureMaps``.
    """
    height, width = image.shape[:2]
    if cell_size < 2 or cell_size % 2 or min(height, width) < 4 * cell_size:
        raise ValueError("FHOG requires an even cell size and at least four cells per image dimension.")
    if height % cell_size or width % cell_size:
        raise ValueError("FHOG image dimensions must be multiples of the cell size.")

    pixels = np.asarray(image, dtype=np.float32)
    dx = pixels[1:-1, 2:] - pixels[1:-1, :-2]
    dy = pixels[2:, 1:-1] - pixels[:-2, 1:-1]
    magnitude_squared = dx * dx + dy * dy
    channel = np.argmax(magnitude_squared, axis=2)[..., None]
    dx = np.take_along_axis(dx, channel, axis=2)[..., 0]
    dy = np.take_along_axis(dy, channel, axis=2)[..., 0]
    magnitude = np.sqrt(np.take_along_axis(magnitude_squared, channel, axis=2)[..., 0])

    best_dot = dx.copy()
    orientation = np.zeros(dx.shape, dtype=np.intp)
    for sector in range(9):
        angle = np.float32(sector * np.pi / 9)
        dot = np.cos(angle) * dx + np.sin(angle) * dy
        positive = dot > best_dot
        negative = (~positive) & (-dot > best_dot)
        best_dot = np.where(positive, dot, np.where(negative, -dot, best_dot))
        orientation = np.where(positive, sector, np.where(negative, sector + 9, orientation))

    yy, xx = np.indices(magnitude.shape)
    yy, xx = yy + 1, xx + 1
    cy, cx = yy // cell_size, xx // cell_size
    dy_cell = np.where(yy % cell_size < cell_size // 2, -1, 1)
    dx_cell = np.where(xx % cell_size < cell_size // 2, -1, 1)
    wy = 1.0 - np.abs((yy % cell_size + 0.5) / cell_size - 0.5)
    wx = 1.0 - np.abs((xx % cell_size + 0.5) / cell_size - 0.5)
    cells_y, cells_x = height // cell_size, width // cell_size
    histogram = np.zeros(cells_y * cells_x * 27, dtype=np.float32)
    for neighbor_y in (False, True):
        for neighbor_x in (False, True):
            target_y = cy + (dy_cell if neighbor_y else 0)
            target_x = cx + (dx_cell if neighbor_x else 0)
            valid = (target_y >= 0) & (target_y < cells_y) & (target_x >= 0) & (target_x < cells_x)
            vote = magnitude * (1.0 - wy if neighbor_y else wy) * (1.0 - wx if neighbor_x else wx)
            base = (target_y[valid] * cells_x + target_x[valid]) * 27
            for bins in (orientation % 9, orientation + 9):
                histogram += np.bincount(base + bins[valid], weights=vote[valid], minlength=histogram.size).astype(
                    np.float32
                )
    histogram = histogram.reshape(cells_y, cells_x, 27)

    energy = np.sum(histogram[..., :9] ** 2, axis=2)
    blocks = energy[:-1, :-1] + energy[:-1, 1:] + energy[1:, :-1] + energy[1:, 1:]
    norms = np.stack((blocks[1:, 1:], blocks[:-1, 1:], blocks[1:, :-1], blocks[:-1, :-1]), axis=2)
    norms = np.sqrt(norms) + np.finfo(np.float32).eps
    normalized = np.minimum(histogram[1:-1, 1:-1, None, :] / norms[..., None], 0.2)
    signed = normalized[..., 9:]
    features = np.concatenate(
        (
            signed.sum(axis=2) * 0.5,
            normalized[..., :9].sum(axis=2) * 0.5,
            signed.sum(axis=3) / np.sqrt(18.0),
        ),
        axis=2,
    )
    return np.ascontiguousarray(features.transpose(2, 0, 1), dtype=np.float32)
