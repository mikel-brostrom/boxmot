import colorsys
import hashlib
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np

from boxmot.utils import logger as LOGGER
from boxmot.utils.iou import AssociationFunction
from boxmot.utils.visualization import VisualizationMixin


class BaseTracker(VisualizationMixin):
    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 30,
        max_obs: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        nr_classes: int = 80,
        asso_func: str = "iou",
        is_obb: bool = False,
        **kwargs,
    ):
        """
        Initialize the BaseTracker object

        Parameters:
        - det_thresh (float): Detection threshold for considering detections.
        - max_age (int): Maximum age (in frames) of a track before it is considered lost.
        - max_obs (int): Maximum number of historical observations (bounding boxes) stored for each track. max_obs is always greater than max_age by minimum 5.
        - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
        - iou_threshold (float): IOU threshold for determining match between detection and tracks.
        - per_class (bool): Enables class-separated tracking
        - nr_classes (int): Total number of object classes that the tracker will handle (for per_class=True)
        - asso_func (str): Algorithm name used for data association between detections and tracks
            Options:
                - "iou" (default): Standard Intersection over Union
                - "iou_obb": IoU for oriented bounding boxes
                - "hmiou": Height-modified IoU that incorporates vertical overlap ratio
                - "giou": Generalized IoU that penalizes non-overlapping boxes
                - "ciou": Complete IoU with center point distance and aspect ratio consistency
                - "diou": Distance IoU that considers center point distance
                - "centroid": Distance between centroids of bounding boxes
                - "centroid_obb": Centroid distance for oriented bounding boxes
        - is_obb (bool): Work with Oriented Bounding Boxes (OBB) instead of standard axis-aligned bounding boxes?
                If False (default): If True: dets.shape[1] == 6, i.e. (x1,y1,x2,y2,conf,cls)
                If True: dets.shape[1] == 7, i.e. (cx,cy,w,h,angle,conf,cls)

        Attributes:
        - frame_count (int): Counter for the frames processed.
        - active_tracks (list): List to hold active tracks, may be used differently in subclasses.
        """

        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.per_class = per_class
        self.nr_classes = nr_classes
        self.asso_func_name = asso_func + "_obb" if is_obb else asso_func
        self.is_obb = is_obb

        # Attributes
        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes

        self.per_class_active_tracks = None
        self._first_frame_processed = (
            False  # Flag to track if the first frame has been processed
        )
        self._first_dets_processed = False
        self.last_emb_size = None  # Tracks the dimensionality of embedding vectors used for re-identification during tracking.

        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []

        if self.max_age >= self.max_obs:
            LOGGER.warning(
                "Max age > max observations, increasing size of max observations..."
            )
            self.max_obs = self.max_age + 5

        # Plotting lifecycle bookkeeping
        self._plot_frame_idx = -1
        self._removed_first_seen = {}
        self._removed_expired = set()
        self.removed_display_frames = getattr(self, "removed_display_frames", 10)

        # Log all params if tracker_name provided via kwargs
        tracker_name = kwargs.pop('_tracker_name', None)
        if tracker_name:
            base_params = {
                'det_thresh': det_thresh, 'max_age': max_age, 'max_obs': max_obs,
                'min_hits': min_hits, 'iou_threshold': iou_threshold, 'per_class': per_class,
                'asso_func': asso_func,
            }
            # Filter out internal/non-config params
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                              if not k.startswith('_') and k not in ('__class__', 'reid_weights', 'device', 'half')}
            all_params = {**base_params, **filtered_kwargs}
            params_str = ", ".join(f"{k}={v}" for k, v in all_params.items())
            LOGGER.success(f"{tracker_name}: {params_str}")

    @abstractmethod
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "The update method needs to be implemented by the subclass."
        )

    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = np.empty((0, 6))
        class_embs = (
            np.empty((0, self.last_emb_size))
            if self.last_emb_size is not None
            else None
        )

        # Check if there are detections
        if dets.size == 0:
            return class_dets, class_embs

        class_indices = np.where(dets[:, 5] == cls_id)[0]
        class_dets = dets[class_indices]

        if embs is None:
            return class_dets, class_embs

        # Assert that if embeddings are provided, they have the same number of elements as detections
        assert dets.shape[0] == embs.shape[0], (
            "Detections and embeddings must have the same number of elements when both are provided"
        )
        class_embs = None
        if embs.size > 0:
            class_embs = embs[class_indices]
            self.last_emb_size = class_embs.shape[
                1
            ]  # Update the last known embedding size
        return class_dets, class_embs

    @staticmethod
    def setup_decorator(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """

        def wrapper(self, *args, **kwargs):
            # Extract detections and image from args
            dets = args[0]
            img = args[1] if len(args) > 1 else None

            # Unwrap `data` attribute if present
            if hasattr(dets, "data"):
                dets = dets.data

            # Convert memoryview to numpy array if needed
            if isinstance(dets, memoryview):
                dets = np.array(dets, dtype=np.float32)  # Adjust dtype if needed

            # First-time detection setup
            if not self._first_dets_processed and dets is not None:
                if dets.ndim == 2 and dets.shape[1] == 6:
                    self.is_obb = False
                    self._first_dets_processed = True
                elif dets.ndim == 2 and dets.shape[1] == 7:
                    self.is_obb = True
                    self._first_dets_processed = True

            # First frame image-based setup
            if not self._first_frame_processed and img is not None:
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(
                    w=self.w, h=self.h, asso_mode=self.asso_func_name
                ).asso_func
                self._first_frame_processed = True

            # Call the original method with the unwrapped `dets`
            return method(self, dets, img, *args[2:], **kwargs)

        return wrapper

    @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """

        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):
            # handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))

            if not self.per_class:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets, img=img, embs=embs)
            # else:
            # Initialize an array to store the tracks for each class
            per_class_tracks = []

            # same frame count for all classes
            frame_count = self.frame_count

            for cls_id in range(self.nr_classes):
                # Get detections and embeddings for the current class
                class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)

                LOGGER.debug(
                    f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings"
                    f" {class_embs.shape if class_embs is not None else None}"
                )

                # Activate the specific active tracks for this class id
                self.active_tracks = self.per_class_active_tracks[cls_id]

                # Reset frame count for every class
                self.frame_count = frame_count

                # Update detections using the decorated method
                tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)

                # Save the updated active tracks
                self.per_class_active_tracks[cls_id] = self.active_tracks

                if tracks.size > 0:
                    per_class_tracks.append(tracks)

            # Increase frame count by 1
            self.frame_count = frame_count + 1
            return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))

        return wrapper

    def check_inputs(self, dets, img, embs=None):
        assert isinstance(dets, np.ndarray), (
            f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        )
        assert isinstance(img, np.ndarray), (
            f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        )
        assert len(dets.shape) == 2, (
            "Unsupported 'dets' dimensions, valid number of dimensions is two"
        )

        if embs is not None:
            assert dets.shape[0] == embs.shape[0], (
                "Missmatch between detections and embeddings sizes"
            )

        if self.is_obb:
            assert dets.shape[1] == 7, (
                "Unsupported 'dets' 2nd dimension length, valid length is 7 (cx,cy,w,h,angle,conf,cls)"
            )
        else:
            assert dets.shape[1] == 6, (
                "Unsupported 'dets' 2nd dimension length, valid lengths is 6 (x1,y1,x2,y2,conf,cls)"
            )

    def reset(self):
        pass
