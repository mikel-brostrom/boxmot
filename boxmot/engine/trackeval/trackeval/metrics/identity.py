import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils


class Identity(_BaseMetric):
    """Class which implements the ID metrics"""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a IDTP match. Default 0.5.
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ['IDTP', 'IDFN', 'IDFP']
        self.float_fields = ['IDF1', 'IDR', 'IDP']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.fields

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])

    @_timing.time
    def eval_sequence(self, data):
        """Calculates ID metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['IDFN'] = data['num_gt_dets']
            return res
        if data['num_gt_dets'] == 0:
            res['IDFP'] = data['num_tracker_dets']
            return res

        # Variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros(data['num_gt_ids'])
        tracker_id_count = np.zeros(data['num_tracker_ids'])

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the potential matches between ids in each timestep
            matches_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)
            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[tracker_ids_t] += 1

        # Calculate optimal assignment cost matrix for ID metrics
        num_gt_ids = data['num_gt_ids']
        num_tracker_ids = data['num_tracker_ids']
        fp_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
        fn_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
        fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
        fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10
        for gt_id in range(num_gt_ids):
            fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
            fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
        for tracker_id in range(num_tracker_ids):
            fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
            fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
        fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
        fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

        # Hungarian algorithm
        match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)

        # Accumulate basic statistics
        res['IDFN'] = fn_mat[match_rows, match_cols].sum().astype(int)
        res['IDFP'] = fp_mat[match_rows, match_cols].sum().astype(int)
        res['IDTP'] = (gt_id_count.sum() - res['IDFN']).astype(int)

        # Calculate final ID scores
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()
                                                if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps},
                                               field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in all_res.values()
                                      if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps], axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['IDR'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFN'])
        res['IDP'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFP'])
        res['IDF1'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + 0.5 * res['IDFP'] + 0.5 * res['IDFN'])
        return res
