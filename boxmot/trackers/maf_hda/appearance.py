"""Masked HOG+Lab kernelized correlation-filter appearance for MAF-HDA.

Ported from GMPHD_MAF's ``kcftracker.cpp``, ``ffttools.hpp``, and
``labdata.hpp``. Original KCF authors: Joao Faro, Christian Bailer, and
Joao F. Henriques; MAF modifications: Young-min Song. The redistributed
notices and BSD licenses are in ``APPEARANCE_LICENSE.txt``.
"""

from __future__ import annotations

import cv2
import numpy as np

from boxmot.trackers.maf_hda.fhog import fhog

_LAB_CENTROIDS = np.array(
    [
        [161.317504, 127.223401, 128.609333],
        [142.922425, 128.666965, 127.532319],
        [67.879757, 127.721830, 135.903311],
        [92.705062, 129.965717, 137.399500],
        [120.172257, 128.279647, 127.036493],
        [195.470568, 127.857070, 129.345415],
        [41.257102, 130.059468, 132.675336],
        [12.014861, 129.480555, 127.064714],
        [226.567086, 127.567831, 136.345727],
        [154.664210, 131.676606, 156.481669],
        [121.180447, 137.020793, 153.433743],
        [87.042204, 137.211742, 98.614874],
        [113.809537, 106.577104, 157.818094],
        [81.083293, 170.051905, 148.904079],
        [45.015485, 138.543124, 102.402528],
    ],
    dtype=np.float32,
)


def _lab_histograms(image: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """Quantize BGR pixels to the source's 15 Lab centroids per inner cell."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = lab[cell_size:-cell_size, cell_size:-cell_size]
    distances = np.sum((lab[:, :, None, :] - _LAB_CENTROIDS) ** 2, axis=3)
    labels = distances.argmin(axis=2)
    cells_y, cells_x = labels.shape[0] // cell_size, labels.shape[1] // cell_size
    yy, xx = np.indices(labels.shape)
    cells = yy // cell_size * cells_x + xx // cell_size
    histograms = np.bincount((cells * 15 + labels).ravel(), minlength=cells_y * cells_x * 15)
    histograms = histograms.reshape(cells_y, cells_x, 15) / float(cell_size * cell_size)
    return np.ascontiguousarray(histograms.transpose(2, 0, 1), dtype=np.float32)


class MaskedKCF:
    """Learn and score masked object appearance without moving its geometry.

    Images are uint8 HWC BGR; boxes use image-space ``(x1, y1, x2, y2)``;
    masks are full-frame HW arrays. ``score`` returns the upstream MAF
    appearance affinity, ``1 - mean(normalized masked response)``, in [0, 1]
    and never trains. Call ``update`` only for an accepted observation.

    The source's fixed-window HOG+Lab settings are retained. A 96-pixel
    template becomes at most 104 pixels after FHOG cell rounding. Extremely
    thin patches have a minimum of four cells to keep FHOG normalization
    defined. Empty masks or boxes outside the image have zero affinity and
    do not train. Unlike the C++ visualization path, a constant response
    carries no appearance evidence and also yields zero affinity.
    """

    def __init__(
        self,
        image: np.ndarray,
        bbox_xyxy: np.ndarray,
        full_frame_mask: np.ndarray,
        *,
        template_size: int = 96,
        padding: float = 2.5,
        interp_factor: float = 0.005,
        sigma: float = 0.4,
        regularization: float = 1e-4,
        output_sigma_factor: float = 0.1,
    ) -> None:
        if not 16 <= template_size <= 512:
            raise ValueError("template_size must lie between 16 and 512 pixels.")
        if not np.isfinite(padding) or padding < 1:
            raise ValueError("padding must be finite and at least one.")
        if not 0 < interp_factor <= 1:
            raise ValueError("interp_factor must lie in (0, 1].")
        if not all(np.isfinite(value) and value > 0 for value in (sigma, regularization, output_sigma_factor)):
            raise ValueError("KCF kernel, regularization, and output bandwidth must be finite and positive.")
        self.template_size = int(template_size)
        self.padding = float(padding)
        self.interp_factor = float(interp_factor)
        self.sigma = float(sigma)
        self.regularization = float(regularization)
        self.output_sigma_factor = float(output_sigma_factor)
        self._template: np.ndarray | None = None
        self._alphaf: np.ndarray | None = None
        self.update(image, bbox_xyxy, full_frame_mask)

    @staticmethod
    def _observation(
        image: np.ndarray, bbox_xyxy: np.ndarray, full_frame_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Validate arrays and return the visible integer box and foreground."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) == 0:
            raise ValueError("MaskedKCF requires a nonempty uint8 HWC BGR image.")
        mask = np.asarray(full_frame_mask, dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError("MaskedKCF requires a full-frame HW mask.")
        bbox = np.array(bbox_xyxy, dtype=np.float64, copy=True)
        if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
            raise ValueError("MaskedKCF requires a finite, positive-area xyxy box.")
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, image.shape[1])
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, image.shape[0])
        bbox = np.concatenate((np.floor(bbox[:2]), np.ceil(bbox[2:]))).astype(np.int64)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1 or not mask[y1:y2, x1:x2].any():
            return None
        return bbox, mask[y1:y2, x1:x2]

    def _configure(self, bbox: np.ndarray) -> None:
        """Fix the source's patch scale, FHOG grid, and Gaussian target."""
        padded = np.maximum((bbox[2:] - bbox[:2]) * self.padding, 1).astype(np.int64)
        scale = float(padded.max()) / self.template_size
        template = (np.floor(padded / scale).astype(np.int64) // 8) * 8 + 8
        self._template_wh = np.maximum(template, 16)
        self._sample_wh = np.maximum((scale * self._template_wh).astype(np.int64), 1)
        width, height = self._template_wh // 4 - 2
        self._hann = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
        yy, xx = np.indices((height, width))
        bandwidth = np.sqrt(width * height) / self.padding * self.output_sigma_factor
        target = np.exp(-0.5 * ((yy - height // 2) ** 2 + (xx - width // 2) ** 2) / bandwidth**2)
        self._prob = np.fft.fft2(target)

    def _features(self, image: np.ndarray, bbox: np.ndarray, foreground: np.ndarray) -> np.ndarray:
        """Mask the object rectangle, replicate image borders, and extract HOG+Lab."""
        center = (bbox[:2] + bbox[2:]) / 2
        start = np.trunc(center - self._sample_wh / 2).astype(np.int64)
        end = start + self._sample_wh
        left, top = np.maximum(start, 0)
        right, bottom = np.minimum(end, [image.shape[1], image.shape[0]])
        patch = image[top:bottom, left:right].copy()
        ix1, iy1 = np.maximum(bbox[:2], [left, top])
        ix2, iy2 = np.minimum(bbox[2:], [right, bottom])
        if ix2 > ix1 and iy2 > iy1:
            selection = foreground[iy1 - bbox[1] : iy2 - bbox[1], ix1 - bbox[0] : ix2 - bbox[0]]
            patch[iy1 - top : iy2 - top, ix1 - left : ix2 - left][~selection] = 0
        patch = cv2.copyMakeBorder(
            patch,
            int(top - start[1]),
            int(end[1] - bottom),
            int(left - start[0]),
            int(end[0] - right),
            cv2.BORDER_REPLICATE,
        )
        patch = cv2.resize(patch, tuple(int(value) for value in self._template_wh), interpolation=cv2.INTER_LINEAR)
        return np.concatenate((fhog(patch), _lab_histograms(patch)), axis=0) * self._hann

    def _kernel(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Evaluate the Gaussian kernel for all circular spatial shifts."""
        spectrum = np.fft.fft2(first, axes=(-2, -1)) * np.conj(np.fft.fft2(second, axes=(-2, -1)))
        correlation = np.fft.fftshift(np.fft.ifft2(spectrum.sum(axis=0)).real)
        distance = np.maximum((np.sum(first**2) + np.sum(second**2) - 2 * correlation) / first.size, 0)
        return np.exp(-distance / self.sigma**2)

    def score(self, image: np.ndarray, bbox_xyxy: np.ndarray, full_frame_mask: np.ndarray) -> float:
        """Score a candidate without changing the template or its learned filter."""
        observation = self._observation(image, bbox_xyxy, full_frame_mask)
        if observation is None or self._template is None:
            return 0.0
        bbox, foreground = observation
        features = self._features(image, bbox, foreground)
        response = np.fft.ifft2(self._alphaf * np.fft.fft2(self._kernel(features, self._template))).real
        if np.ptp(response) <= np.finfo(np.float32).eps:
            return 0.0
        # MAF converts to an 8-bit response before both normalizations.
        gray = np.clip(np.rint((response + 1.0) * 127.5), 0, 255).astype(np.uint8)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.resize(gray, (foreground.shape[1], foreground.shape[0]), interpolation=cv2.INTER_LINEAR)
        selected = gray[foreground]
        if selected.max() == selected.min():
            return 0.0
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, mask=foreground.astype(np.uint8))
        return float(np.clip(1.0 - normalized[foreground].mean() / 255.0, 0.0, 1.0))

    def update(self, image: np.ndarray, bbox_xyxy: np.ndarray, full_frame_mask: np.ndarray) -> None:
        """Train on an accepted observation; empty foreground leaves memory intact."""
        observation = self._observation(image, bbox_xyxy, full_frame_mask)
        if observation is None:
            return
        bbox, foreground = observation
        if self._template is None:
            self._configure(bbox)
        features = self._features(image, bbox, foreground)
        alphaf = self._prob / (np.fft.fft2(self._kernel(features, features)) + self.regularization)
        if self._template is None:
            self._template = features
            self._alphaf = alphaf
        else:
            rate = self.interp_factor
            self._template = (1.0 - rate) * self._template + rate * features
            self._alphaf = (1.0 - rate) * self._alphaf + rate * alphaf
