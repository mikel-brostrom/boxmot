import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from trackeval import _timing
from trackeval import utils
from trackeval.datasets._base_dataset import _BaseDataset
from trackeval.utils import TrackEvalException


DEFAULT_MMOT_CLASS_NAME_TO_ID = {
    "car": 0,
    "bike": 1,
    "pedestrian": 2,
    "van": 3,
    "truck": 4,
    "bus": 5,
    "tricycle": 6,
    "awning-bike": 7,
}

DEFAULT_MMOT_SUPER_CATEGORIES = {
    "HUMAN": ["pedestrian"],
    "VEHICLE": ["car", "van", "truck", "bus"],
    "BIKE": ["bike", "tricycle", "awning-bike"],
}


def _count_images(path: str) -> int:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(1 for name in os.listdir(path) if os.path.splitext(name)[1].lower() in valid_exts)


def _polygon_to_rotated_rect(polygon: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    points = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle_deg = rect
    return (float(cx), float(cy)), (float(w), float(h)), float(angle_deg)


def _rotated_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    rect_a = _polygon_to_rotated_rect(poly_a)
    rect_b = _polygon_to_rotated_rect(poly_b)
    area_a = rect_a[1][0] * rect_a[1][1]
    area_b = rect_b[1][0] * rect_b[1][1]
    if area_a <= np.finfo(float).eps or area_b <= np.finfo(float).eps:
        return 0.0

    _, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if intersection is None or len(intersection) == 0:
        inter_area = 0.0
    else:
        inter_area = float(cv2.contourArea(intersection))

    union = area_a + area_b - inter_area
    if union <= np.finfo(float).eps:
        return 0.0
    return max(0.0, min(1.0, inter_area / union))


class CustomMMOTRGB(_BaseDataset):
    """Self-contained MMOT RGB dataset compatible with the MMOT TrackEval fork."""

    @staticmethod
    def get_default_dataset_config():
        default_config = {
            "GT_FOLDER": "/data3/PublicDataset/Custom/mmot/test/mot",
            "IMG_FOLDER": "/data3/PublicDataset/Custom/mmot/test/rgb",
            "TRACKERS_FOLDER": "/data3/litianhao/workdir/mmot",
            "OUTPUT_FOLDER": None,
            "TRACKERS_TO_EVAL": None,
            "CLASSES_TO_EVAL": ["car", "bike", "pedestrian", "van", "truck", "bus", "tricycle", "awning-bike"],
            "CLASS_IDS": None,
            "SPLIT_TO_EVAL": "test",
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": True,
            "TRACKER_SUB_FOLDER": "preds",
            "OUTPUT_SUB_FOLDER": "eval",
            "TRACKER_DISPLAY_NAMES": None,
        }
        assert default_config["INPUT_AS_ZIP"] is False
        return default_config

    def __init__(self, config=None):
        super().__init__()
        cfg = {} if config is None else dict(config)
        real_classes = [cls.lower() for cls in cfg.get("CLASSES_TO_EVAL", list(DEFAULT_MMOT_CLASS_NAME_TO_ID))]
        class_ids = cfg.get("CLASS_IDS")
        if class_ids is not None:
            class_ids = [int(cid) for cid in class_ids]

        self.config = utils.init_config(cfg, self.get_default_dataset_config(), self.get_name())
        self.config["CLASSES_TO_EVAL"] = real_classes
        self.config["CLASS_IDS"] = class_ids
        self.gt_fol = self.config["GT_FOLDER"]
        self.img_fol = self.config["IMG_FOLDER"]
        self.tracker_fol = self.config["TRACKERS_FOLDER"]
        self.should_classes_combine = True

        self.output_fol = self.config["OUTPUT_FOLDER"] or self.tracker_fol
        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        if class_ids is not None:
            if len(class_ids) != len(real_classes):
                raise TrackEvalException("CLASS_IDS must have the same length as CLASSES_TO_EVAL.")
            self.valid_classes = real_classes
            self.class_list = real_classes
            self.class_name_to_class_id = {cls: int(cid) for cls, cid in zip(real_classes, class_ids)}
        else:
            self.valid_classes = list(DEFAULT_MMOT_CLASS_NAME_TO_ID)
            self.class_list = [cls if cls in self.valid_classes else None for cls in real_classes]
            if not all(self.class_list):
                raise TrackEvalException(
                    "Attempted to evaluate an invalid class without CLASS_IDS. Only classes "
                    "[car, bike, pedestrian, van, truck, bus, tricycle, awning-bike] are valid."
                )
            self.class_name_to_class_id = DEFAULT_MMOT_CLASS_NAME_TO_ID.copy()

        super_categories = {
            name: [cls for cls in members if cls in self.class_list]
            for name, members in DEFAULT_MMOT_SUPER_CATEGORIES.items()
        }
        self.super_categories = {name: members for name, members in super_categories.items() if members}
        self.use_super_categories = bool(self.super_categories)

        self.seq_lengths = {}
        self.seq_list = [seq_file.replace(".txt", "") for seq_file in os.listdir(self.gt_fol)]

        if self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        if self.config["TRACKER_DISPLAY_NAMES"] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif len(self.config["TRACKER_DISPLAY_NAMES"]) == len(self.tracker_list):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config["TRACKER_DISPLAY_NAMES"]))
        else:
            raise TrackEvalException("List of tracker files and tracker display names do not match.")

        for tracker in self.tracker_list:
            for seq in self.seq_list:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + ".txt")
                if not os.path.isfile(curr_file):
                    print("Tracker file not found: " + curr_file)
                    raise TrackEvalException(
                        "Tracker file not found: " + tracker + "/" + self.tracker_sub_fol + "/" + os.path.basename(curr_file)
                    )

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        if is_gt:
            file = os.path.join(self.gt_fol, seq + ".txt")
        else:
            file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + ".txt")

        data, _ignore_data = self._load_simple_text_file(file, is_zipped=False, zip_file=None)

        if is_gt:
            img_path = os.path.join(self.img_fol, seq)
            self.seq_lengths[seq] = _count_images(img_path)
            num_timesteps = self.seq_lengths[seq]
        else:
            num_timesteps = self.seq_lengths[seq]

        current_time_keys = [str(t + 1) for t in range(self.seq_lengths[seq])]
        extra_time_keys = [x for x in data.keys() if x not in current_time_keys]
        lack_time_keys = [x for x in current_time_keys if x not in data.keys()]

        if extra_time_keys:
            text = "Ground-truth" if is_gt else "Tracking"
            raise TrackEvalException(
                text + " data contains the following invalid timesteps in seq %s: " % seq
                + ", ".join([str(x) for x in extra_time_keys])
            )
        if lack_time_keys:
            text = "Ground-truth" if is_gt else "Tracking"
            print(
                "Warning!!!!!"
                + text
                + " data leaks the following invalid timesteps in seq %s: " % seq
                + ", ".join([str(x) for x in lack_time_keys])
            )

        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_extras", "gt_truncation"]
        else:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        for t in range(num_timesteps):
            time_key = str(t + 1)
            if time_key in data.keys():
                time_data = np.asarray(data[time_key], dtype=float)
                if time_data.ndim == 1:
                    time_data = time_data.reshape(1, -1)

                raw_data["dets"][t] = np.atleast_2d(time_data[:, 2:10])
                raw_data["ids"][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                raw_data["classes"][t] = np.atleast_1d(time_data[:, 11]).astype(int)

                if is_gt:
                    raw_data["gt_extras"][t] = {"zero_marked": np.atleast_1d(time_data[:, 10].astype(int))}
                    raw_data["gt_truncation"][t] = np.atleast_1d(time_data[:, 12])
                else:
                    raw_data["tracker_confidences"][t] = np.atleast_1d(time_data[:, 10])
            else:
                raw_data["dets"][t] = np.empty((0, 8))
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
                if is_gt:
                    raw_data["gt_extras"][t] = {"zero_marked": np.empty(0)}
                    raw_data["gt_truncation"][t] = np.empty(0)
                else:
                    raw_data["tracker_confidences"][t] = np.empty(0)

        key_map = {"ids": "gt_ids" if is_gt else "tracker_ids", "classes": "gt_classes" if is_gt else "tracker_classes", "dets": "gt_dets" if is_gt else "tracker_dets"}
        for key, value in key_map.items():
            raw_data[value] = raw_data.pop(key)
        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ["gt_ids", "tracker_ids", "gt_dets", "tracker_dets", "similarity_scores"]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data["num_timesteps"]):
            gt_class_mask = np.atleast_1d(raw_data["gt_classes"][t] == cls_id).astype(bool)
            tracker_class_mask = np.atleast_1d(raw_data["tracker_classes"][t] == cls_id).astype(bool)

            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            gt_dets = raw_data["gt_dets"][t][gt_class_mask]
            tracker_ids = raw_data["tracker_ids"][t][tracker_class_mask]
            tracker_dets = raw_data["tracker_dets"][t][tracker_class_mask]
            similarity_scores = raw_data["similarity_scores"][t][gt_class_mask, :][:, tracker_class_mask]

            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo("float").eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo("float").eps
                match_cols = match_cols[actually_matched_mask]
                _ = np.delete(np.arange(tracker_ids.shape[0]), match_cols, axis=0)

            data["tracker_ids"][t] = tracker_ids
            data["tracker_dets"][t] = tracker_dets
            data["gt_ids"][t] = gt_ids
            data["gt_dets"][t] = gt_dets
            data["similarity_scores"][t] = similarity_scores

            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][t]))
            num_tracker_dets += len(data["tracker_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])

        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(int)

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[data["tracker_ids"][t]].astype(int)

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]
        self._check_unique_ids(data)
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if len(gt_dets_t) == 0 or len(tracker_dets_t) == 0:
            return np.zeros((len(gt_dets_t), len(tracker_dets_t)), dtype=np.float32)

        scores = np.zeros((len(gt_dets_t), len(tracker_dets_t)), dtype=np.float32)
        for i, gt_det in enumerate(gt_dets_t):
            for j, tracker_det in enumerate(tracker_dets_t):
                scores[i, j] = _rotated_iou(gt_det, tracker_det)
        return scores


mmot_RGB = CustomMMOTRGB
