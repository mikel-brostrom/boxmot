from __future__ import annotations

import colorsys
import hashlib
from abc import ABC

import cv2 as cv
import numpy as np


class BaseVisualization(ABC):
    """
    Abstract base class for visualization methods in BaseTracker.
    """
    
    def id_to_color(
        self,
        id: int,
        saturation: float = 0.75,
        value: float = 0.95,
        state: str = "confirmed"
    ) -> tuple:
        """
        Returns green for target_id, otherwise generates a consistent unique BGR color using ID hashing.
        """
        if state == "removed":
            return (0, 0, 255)

        target_id = getattr(self, "target_id", None)
        if target_id is not None:
            return (0, 255, 0) if id == target_id else (0, 0, 0)

        # Default: consistent hashed color for other IDs
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        hue = int(hash_digest[:8], 16) / 0xFFFFFFFF

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_255 = tuple(int(component * 255) for component in rgb)

        # Convert to BGR
        return rgb_255[::-1]

    def _draw_dashed_rect(self, img, x1, y1, x2, y2, color, thickness, dash=10, gap=10):
        """Dashed rectangle for AABB (used for 'lost'/'predicted')."""
        # Top / Bottom
        for i in range(x1, x2, dash + gap):
            img = cv.line(img, (i, y1), (min(i + dash, x2), y1), color, thickness)
            img = cv.line(img, (i, y2), (min(i + dash, x2), y2), color, thickness)
        # Left / Right
        for i in range(y1, y2, dash + gap):
            img = cv.line(img, (x1, i), (x1, min(i + dash, y2)), color, thickness)
            img = cv.line(img, (x2, i), (x2, min(i + dash, y2)), color, thickness)
        return img

    @staticmethod
    def _obb_to_polygon(box: tuple) -> np.ndarray:
        arr = np.asarray(box, dtype=np.float32).reshape(-1)
        if arr.size >= 8:
            return arr[:8].reshape(4, 2)
        angle = arr[4] * 180.0 / np.pi
        box_poly = ((arr[0], arr[1]), (arr[2], arr[3]), angle)
        return cv.boxPoints(box_poly).astype(np.float32)

    def _class_label(self, cls: int) -> str:
        names = getattr(self, "names", None)
        return names.get(int(cls), str(int(cls))) if names else str(int(cls))

    @staticmethod
    def _draw_label(img, label: str, anchor: tuple[int, int], fontscale: float, color, thickness: int):
        return cv.putText(
            img,
            label,
            anchor,
            cv.FONT_HERSHEY_SIMPLEX,
            fontscale,
            color,
            thickness,
        )

    def _format_box_label(self, id: int, conf: float, cls: int, box_arr: np.ndarray | None = None) -> str:
        label = f"id: {int(id)}"
        # if self.is_obb and box_arr is not None and box_arr.size >= 5:
        #     label += f", a: {box_arr[4]:.2f}"
        return label

    def plot_box_on_img(
        self,
        img: np.ndarray,
        box: tuple,
        conf: float,
        cls: int,
        id: int,
        thickness: int = 2,
        fontscale: float = 0.5,
        state: str = "confirmed",
        style: str = "solid",  # "solid" | "dashed" (dashed only for AABB)
    ) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.
        """
        color = self.id_to_color(id, state=state)
        if self.is_obb:
            box_arr = np.asarray(box, dtype=np.float32).reshape(-1)
            box_poly = np.int_(self._obb_to_polygon(box))
            label = self._format_box_label(id=id, conf=conf, cls=cls, box_arr=box_arr)

            img = cv.polylines(
                img,
                [box_poly],
                isClosed=True,
                color=color,
                thickness=thickness,
            )

            img = self._draw_label(
                img=img,
                label=label,
                anchor=(int(box_arr[0]), int(box_arr[1]) - 10),
                fontscale=fontscale,
                color=color,
                thickness=thickness,
            )
        else:
            x1, y1, x2, y2 = map(int, (box[0], box[1], box[2], box[3]))
            if style == "dashed":
                img = self._draw_dashed_rect(img, x1, y1, x2, y2, color, thickness)
            else:
                img = cv.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness,
                )

            img = self._draw_label(
                img=img,
                label=self._format_box_label(id=id, conf=conf, cls=cls),
                anchor=(x1, max(0, y1 - 10)),
                fontscale=fontscale,
                color=color,
                thickness=thickness,
            )
        return img

    def plot_trackers_trajectories(
        self, img: np.ndarray, observations: list, id: int, state: str = "confirmed"
    ) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float(i + 1)) * 1.2)
            if self.is_obb:
                poly = self._obb_to_polygon(box)
                center = np.mean(poly, axis=0)
                img = cv.circle(
                    img,
                    (int(center[0]), int(center[1])),
                    2,
                    color=self.id_to_color(int(id), state=state),
                    thickness=trajectory_thickness,
                )
            else:
                img = cv.circle(
                    img,
                    (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)),
                    2,
                    color=self.id_to_color(int(id), state=state),
                    thickness=trajectory_thickness,
                )
        return img

    def _draw_track(self, img, track, state, style, thickness, fontscale, show_trajectories):
        history = self.get_track_history_for_display(track)
        if not history:
            return img

        box = self.get_track_box_for_display(track, state)
        if box is None:
            return img

        track_id = self.get_track_id_for_display(track)
        conf = self.get_track_conf_for_display(track)
        cls = self.get_track_cls_for_display(track)

        img = self.plot_box_on_img(
            img=img,
            box=box,
            conf=conf,
            cls=cls,
            id=track_id,
            thickness=thickness,
            fontscale=fontscale,
            state=state,
            style=style if (state == "predicted" and not self.is_obb) else "solid",
        )

        if show_trajectories:
            img = self.plot_trackers_trajectories(img, history, track_id, state=state)
        return img

    def plot_results(
        self,
        img: np.ndarray,
        show_trajectories: bool,
        thickness: int = 2,
        fontscale: float = 0.5,
        show_kf_preds: bool = False,
    ) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image.
        """
        for track, state, style in self.iter_tracks_for_display(
            show_kf_preds=show_kf_preds
        ):
            img = self._draw_track(
                img,
                track,
                state=state,
                style=style,
                thickness=thickness,
                fontscale=fontscale,
                show_trajectories=show_trajectories,
            )
        return img


class VisualizationMixin(BaseVisualization):
    """
    Mixin class for visualization methods in BaseTracker.
    """
