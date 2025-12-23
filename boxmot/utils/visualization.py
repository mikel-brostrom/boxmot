import colorsys
import hashlib
import cv2 as cv
import numpy as np

class VisualizationMixin:
    """
    Mixin class for visualization methods in BaseTracker.
    """
    
    def id_to_color(
        self, id: int, saturation: float = 0.75, value: float = 0.95
    ) -> tuple:
        """
        Returns green for target_id, otherwise generates a consistent unique BGR color using ID hashing.
        """
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

    def plot_box_on_img(
        self,
        img: np.ndarray,
        box: tuple,
        conf: float,
        cls: int,
        id: int,
        thickness: int = 2,
        fontscale: float = 0.5,
    ) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.
        """
        if self.is_obb:
            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(
                img,
                [box_poly],
                isClosed=True,
                color=self.id_to_color(id),
                thickness=thickness,
            )

            img = cv.putText(
                img,
                f"id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}, a: {box[4]:.2f}",
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness,
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),
                thickness,
            )
            img = cv.putText(
                img,
                f"id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}",
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness,
            )
        return img

    def plot_trackers_trajectories(
        self, img: np.ndarray, observations: list, id: int
    ) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float(i + 1)) * 1.2)
            if self.is_obb:
                img = cv.circle(
                    img,
                    (int(box[0]), int(box[1])),
                    2,
                    color=self.id_to_color(int(id)),
                    thickness=trajectory_thickness,
                )
            else:
                img = cv.circle(
                    img,
                    (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)),
                    2,
                    color=self.id_to_color(int(id)),
                    thickness=trajectory_thickness,
                )
        return img

    def plot_results(
        self,
        img: np.ndarray,
        show_trajectories: bool,
        thickness: int = 2,
        fontscale: float = 0.5,
    ) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image.
        """

        if self.per_class_active_tracks is None:  # dict
            active_tracks = self.active_tracks
        else:
            active_tracks = []
            for k in self.per_class_active_tracks.keys():
                active_tracks += self.per_class_active_tracks[k]

        for a in active_tracks:
            if not a.history_observations:
                continue
            if len(a.history_observations) < 3:
                continue
            box = a.history_observations[-1]
            img = self.plot_box_on_img(
                img, box, a.conf, a.cls, a.id, thickness, fontscale
            )
            if not show_trajectories:
                continue
            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        return img
