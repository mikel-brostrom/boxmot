import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from collections import defaultdict
from .. import utils


class IDEucl(_BaseMetric):
    """Class which implements the ID metrics"""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.4,  # Similarity score threshold required for a IDTP match. 0.4 for IDEucl.
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.fields = ['IDEucl']
        self.float_fields = self.fields
        self.summary_fields = self.fields

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])


    @_timing.time
    def eval_sequence(self, data):
        """Calculates IDEucl metrics for all frames"""
        # Initialise results
        res = {'IDEucl' : 0}

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0 or data['num_gt_dets'] == 0.:
            return res

        data['centroid'] = []
        for t, gt_det in enumerate(data['gt_dets']):
            # import pdb;pdb.set_trace()
            data['centroid'].append(self._compute_centroid(gt_det))

        oid_hid_cent = defaultdict(list)
        oid_cent = defaultdict(list)
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            matches_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)

            # I hope the orders of ids and boxes are maintained in `data`
            for ind, gid in enumerate(gt_ids_t):
                oid_cent[gid].append(data['centroid'][t][ind])

            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            for m_gid, m_tid in zip(match_idx_gt, match_idx_tracker):
                oid_hid_cent[gt_ids_t[m_gid], tracker_ids_t[m_tid]].append(data['centroid'][t][m_gid])

        oid_hid_dist = {k : np.sum(np.linalg.norm(np.diff(np.array(v), axis=0), axis=1)) for k, v in oid_hid_cent.items()}
        oid_dist = {int(k) : np.sum(np.linalg.norm(np.diff(np.array(v), axis=0), axis=1)) for k, v in oid_cent.items()}

        unique_oid = np.unique([i[0] for i in oid_hid_dist.keys()]).tolist()
        unique_hid = np.unique([i[1] for i in oid_hid_dist.keys()]).tolist()
        o_len = len(unique_oid)
        h_len = len(unique_hid)
        dist_matrix = np.zeros((o_len, h_len))
        for ((oid, hid), dist) in oid_hid_dist.items():
            oid_ind = unique_oid.index(oid)
            hid_ind = unique_hid.index(hid)
            dist_matrix[oid_ind, hid_ind] = dist

        # opt_hyp_dist contains GT ID : max dist covered by track
        opt_hyp_dist = dict.fromkeys(oid_dist.keys(), 0.)
        cost_matrix = np.max(dist_matrix) - dist_matrix
        rows, cols = linear_sum_assignment(cost_matrix)
        for (row, col) in zip(rows, cols):
            value = dist_matrix[row, col]
            opt_hyp_dist[int(unique_oid[row])] = value

        assert len(opt_hyp_dist.keys()) == len(oid_dist.keys())
        hyp_length = np.sum(list(opt_hyp_dist.values()))
        gt_length = np.sum(list(oid_dist.values()))
        id_eucl =np.mean([np.divide(a, b, out=np.zeros_like(a), where=b!=0) for a, b in zip(opt_hyp_dist.values(), oid_dist.values())])
        res['IDEucl'] = np.divide(hyp_length, gt_length, out=np.zeros_like(hyp_length), where=gt_length!=0)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}

        for field in self.float_fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in all_res.values()
                                      if v['IDEucl'] > 0 + np.finfo('float').eps], axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.float_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res, len(all_res))
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.float_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res, len(all_res))
        return res


    @staticmethod
    def _compute_centroid(box):
        box = np.array(box)
        if len(box.shape) == 1:
            centroid = (box[0:2] + box[2:4])/2
        else:
            centroid = (box[:, 0:2] + box[:, 2:4])/2
        return  np.flip(centroid, axis=1)


    @staticmethod
    def _compute_final_fields(res, res_len):
        """
        Exists only to match signature with the original Identiy class.

        """
        return {k:v/res_len for k,v in res.items()}
