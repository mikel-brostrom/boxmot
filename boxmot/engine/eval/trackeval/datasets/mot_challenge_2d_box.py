import numpy as np
import trackeval._timing as _timing
from scipy.optimize import linear_sum_assignment
from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox

from boxmot.data.benchmark import COCO_CLASSES
from boxmot.engine.eval.trackeval.datasets.base import CustomMotChallengeBase

# TrackEval uses one-based class ids for MOT-style 2D boxes.
DEFAULT_CLASS_NAME_TO_ID = {class_name: index + 1 for index, class_name in enumerate(COCO_CLASSES)}


class CustomMotChallenge2DBox(CustomMotChallengeBase, MotChallenge2DBox):
    """Custom Dataset class for MOT Challenge 2D bounding box tracking with multi-class support"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        default_config = MotChallenge2DBox.get_default_dataset_config()
        default_config['CLASSES_TO_EVAL'] = ['person']
        default_config['CLASS_IDS'] = [1]
        default_config['DISTRACTOR_CLASS_IDS'] = []
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        cfg = {} if config is None else dict(config)

        real_classes, class_ids = self._normalize_class_config(cfg, ["person"])
        distractor_ids = cfg.get('DISTRACTOR_CLASS_IDS') or []

        temp_config = cfg.copy()
        temp_config['CLASSES_TO_EVAL'] = ['pedestrian']
        super().__init__(temp_config)

        self.config['CLASSES_TO_EVAL'] = real_classes
        self.config['CLASS_IDS'] = class_ids
        self.config['DISTRACTOR_CLASS_IDS'] = distractor_ids

        self._configure_class_data(
            real_classes,
            class_ids,
            DEFAULT_CLASS_NAME_TO_ID,
            validate_against_default=False,
            invalid_class_message="",
        )
        self.distractor_class_ids = [int(i) for i in distractor_ids] if distractor_ids else None

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        self._check_unique_ids(raw_data)
        cls_id = self.class_name_to_class_id[cls]
        distractor_classes = self.distractor_class_ids

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets',
                    'tracker_confidences', 'similarity_scores']
        data = {k: [None] * raw_data['num_timesteps'] for k in data_keys}

        unique_gt_ids, unique_tracker_ids = [], []
        num_gt_dets = num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):
            gt_ids            = raw_data['gt_ids'][t]
            gt_dets           = raw_data['gt_dets'][t]
            gt_classes        = raw_data['gt_classes'][t]
            gt_zero_marked    = raw_data['gt_extras'][t]['zero_marked']

            tracker_ids         = raw_data['tracker_ids'][t]
            tracker_dets        = raw_data['tracker_dets'][t]
            tracker_classes     = raw_data['tracker_classes'][t]
            tracker_confs       = raw_data['tracker_confidences'][t]
            similarity_scores   = raw_data['similarity_scores'][t]  # shape (num_gt, num_trk)

            # --- Keep only trackers of the eval class (columns) ---
            trk_keep_mask = (tracker_classes == cls_id)
            kept_tracker_ids   = tracker_ids[trk_keep_mask]
            kept_tracker_dets  = tracker_dets[trk_keep_mask, :] if tracker_dets.size else tracker_dets
            kept_tracker_confs = tracker_confs[trk_keep_mask]
            sim = similarity_scores
            sim = sim[:, trk_keep_mask]  # always slice columns; may yield (N, 0)

            # --- Remove trackers that match distractor GT (match vs ALL GT, not filtered by class) ---
            if (
                self.do_preproc
                and self.benchmark != 'MOT15'
                and gt_ids.shape[0] > 0
                and kept_tracker_ids.shape[0] > 0
                and sim.size > 0
            ):
                # threshold & Hungarian (same as original)
                matching_scores = sim.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched]
                match_cols = match_cols[actually_matched]

                # mark tracker columns to remove if matched GT is a distractor class
                is_distr = np.isin(gt_classes[match_rows], distractor_classes)
                to_remove_cols = match_cols[is_distr]

                if to_remove_cols.size > 0:
                    kept_tracker_ids   = np.delete(kept_tracker_ids,   to_remove_cols, axis=0)
                    if kept_tracker_dets.size:
                        kept_tracker_dets = np.delete(kept_tracker_dets, to_remove_cols, axis=0)
                    kept_tracker_confs = np.delete(kept_tracker_confs, to_remove_cols, axis=0)
                    sim                = np.delete(sim, to_remove_cols, axis=1)

            # --- Now keep ONLY GT of the eval class and not zero-marked (rows) ---
            if self.do_preproc and self.benchmark != 'MOT15':
                gt_keep_mask = (gt_zero_marked != 0) & (gt_classes == cls_id)
            else:
                gt_keep_mask = (gt_zero_marked != 0)

            kept_gt_ids  = gt_ids[gt_keep_mask]
            kept_gt_dets = gt_dets[gt_keep_mask, :] if gt_dets.size else gt_dets
            sim = sim[gt_keep_mask, :]

            # --- Write timestep outputs ---
            data['tracker_ids'][t]          = kept_tracker_ids
            data['tracker_dets'][t]         = kept_tracker_dets
            data['tracker_confidences'][t]  = kept_tracker_confs
            data['gt_ids'][t]               = kept_gt_ids
            data['gt_dets'][t]              = kept_gt_dets
            data['similarity_scores'][t]    = sim

            unique_gt_ids      += list(np.unique(kept_gt_ids))
            unique_tracker_ids += list(np.unique(kept_tracker_ids))
            num_tracker_dets   += len(kept_tracker_ids)
            num_gt_dets        += len(kept_gt_ids)

        self._relabel_track_ids(data, raw_data['num_timesteps'])

        # --- Stats & checks ---
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets']      = num_gt_dets
        data['num_timesteps']    = raw_data['num_timesteps']
        data['seq']              = raw_data['seq']

        self._check_unique_ids(data, after_preproc=True)
        return data
