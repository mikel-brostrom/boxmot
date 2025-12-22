import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
from trackeval.utils import TrackEvalException
import trackeval._timing as _timing

class CustomMotChallenge2DBox(MotChallenge2DBox):
    """Custom Dataset class for MOT Challenge 2D bounding box tracking with multi-class support"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        default_config = MotChallenge2DBox.get_default_dataset_config()
        default_config['CLASSES_TO_EVAL'] = ['person']
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        # Save real classes to eval
        real_classes = config.get('CLASSES_TO_EVAL', ['person']) if config else ['person']
        
        # Create a temp config with 'pedestrian' to pass super().__init__ validation
        temp_config = config.copy() if config else {}
        temp_config['CLASSES_TO_EVAL'] = ['pedestrian']
        
        # Initialize parent with temp config
        super().__init__(temp_config)
        
        # Restore real classes to eval in self.config
        self.config['CLASSES_TO_EVAL'] = real_classes
        
        # Overwrite class validation and list with real classes
        self.valid_classes = [cls.lower() for cls in self.config['CLASSES_TO_EVAL']]
        self.class_list = [cls.lower() for cls in self.config['CLASSES_TO_EVAL']]
        
        # Overwrite class map with COCO-80
        self.class_name_to_class_id = {
            'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
            'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20,
            'elephant': 21, 'bear': 22, 'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28, 'suitcase': 29, 'frisbee': 30,
            'skis': 31, 'snowboard': 32, 'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,
            'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46, 'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50,
            'broccoli': 51, 'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56, 'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60,
            'dining table': 61, 'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67, 'cell phone': 68, 'microwave': 69, 'oven': 70,
            'toaster': 71, 'sink': 72, 'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77, 'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80
        }
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        self._check_unique_ids(raw_data)
        cls_id = self.class_name_to_class_id[cls]

        # MOT distractor set (same as original; add non_mot_vehicle for MOT20)
        distractor_classes = [12, 8, 6, 7, 2] if self.benchmark == 'MOT20' else [12, 8, 7, 2]

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
            if self.do_preproc and self.benchmark != 'MOT15' and gt_ids.shape[0] > 0 and kept_tracker_ids.shape[0] > 0 and sim.size > 0:
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
                    kept_tracker_dets  = np.delete(kept_tracker_dets,  to_remove_cols, axis=0) if kept_tracker_dets.size else kept_tracker_dets
                    kept_tracker_confs = np.delete(kept_tracker_confs, to_remove_cols, axis=0)
                    sim                = np.delete(sim, to_remove_cols, axis=1)

            # --- Now keep ONLY GT of the eval class and not zero-marked (rows) ---
            if self.do_preproc and self.benchmark != 'MOT15':
                gt_keep_mask = (gt_zero_marked != 0) & (gt_classes == cls_id)
            else:
                gt_keep_mask = (gt_zero_marked != 0)

            kept_gt_ids  = gt_ids[gt_keep_mask]
            kept_gt_dets = gt_dets[gt_keep_mask, :] if gt_dets.size else gt_dets
            sim = sim[gt_keep_mask, :]  # always slice rows; ensures sim.shape == (len(kept_gt_ids), len(kept_tracker_ids))

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

        # --- Relabel to contiguous ids (no gaps) ---
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_map[data['gt_ids'][t]].astype(int)

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            trk_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            trk_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = trk_map[data['tracker_ids'][t]].astype(int)

        # --- Stats & checks ---
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets']      = num_gt_dets
        data['num_tracker_ids']  = len(np.unique(unique_tracker_ids)) if len(unique_tracker_ids) > 0 else 0
        data['num_gt_ids']       = len(np.unique(unique_gt_ids)) if len(unique_gt_ids) > 0 else 0
        data['num_timesteps']    = raw_data['num_timesteps']
        data['seq']              = raw_data['seq']

        self._check_unique_ids(data, after_preproc=True)
        return data
