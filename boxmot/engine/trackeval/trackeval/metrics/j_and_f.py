
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from ..utils import TrackEvalException
from ._base_metric import _BaseMetric
from .. import _timing


class JAndF(_BaseMetric):
    """Class which implements the J&F metrics"""
    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ['num_gt_tracks']
        self.float_fields = ['J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay', 'J&F']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.float_fields
        self.optim_type = 'J'  # possible values J, J&F

    @_timing.time
    def eval_sequence(self, data):
        """Returns J&F metrics for one sequence"""

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils

        num_timesteps = data['num_timesteps']
        num_tracker_ids = data['num_tracker_ids']
        num_gt_ids = data['num_gt_ids']
        gt_dets = data['gt_dets']
        tracker_dets = data['tracker_dets']
        gt_ids = data['gt_ids']
        tracker_ids = data['tracker_ids']

        # get shape of frames
        frame_shape = None
        if num_gt_ids > 0:
            for t in range(num_timesteps):
                if len(gt_ids[t]) > 0:
                    frame_shape = gt_dets[t][0]['size']
                    break
        elif num_tracker_ids > 0:
            for t in range(num_timesteps):
                if len(tracker_ids[t]) > 0:
                    frame_shape = tracker_dets[t][0]['size']
                    break

        if frame_shape:
            # append all zero masks for timesteps in which tracks do not have a detection
            zero_padding = np.zeros((frame_shape), order= 'F').astype(np.uint8)
            padding_mask = mask_utils.encode(zero_padding)
            for t in range(num_timesteps):
                gt_id_det_mapping = {gt_ids[t][i]: gt_dets[t][i] for i in range(len(gt_ids[t]))}
                gt_dets[t] = [gt_id_det_mapping[index] if index in gt_ids[t] else padding_mask for index
                              in range(num_gt_ids)]
                tracker_id_det_mapping = {tracker_ids[t][i]: tracker_dets[t][i] for i in range(len(tracker_ids[t]))}
                tracker_dets[t] = [tracker_id_det_mapping[index] if index in tracker_ids[t] else padding_mask for index
                                   in range(num_tracker_ids)]
            # also perform zero padding if number of tracker IDs < number of ground truth IDs
            if num_tracker_ids < num_gt_ids:
                diff = num_gt_ids - num_tracker_ids
                for t in range(num_timesteps):
                    tracker_dets[t] = tracker_dets[t] + [padding_mask for _ in range(diff)]
                num_tracker_ids += diff

        j = self._compute_j(gt_dets, tracker_dets, num_gt_ids, num_tracker_ids, num_timesteps)

        # boundary threshold for F computation
        bound_th = 0.008

        # perform matching
        if self.optim_type == 'J&F':
            f = np.zeros_like(j)
            for k in range(num_tracker_ids):
                for i in range(num_gt_ids):
                    f[k, i, :] = self._compute_f(gt_dets, tracker_dets, k, i, bound_th)
            optim_metrics = (np.mean(j, axis=2) + np.mean(f, axis=2)) / 2
            row_ind, col_ind = linear_sum_assignment(- optim_metrics)
            j_m = j[row_ind, col_ind, :]
            f_m = f[row_ind, col_ind, :]
        elif self.optim_type == 'J':
            optim_metrics = np.mean(j, axis=2)
            row_ind, col_ind = linear_sum_assignment(- optim_metrics)
            j_m = j[row_ind, col_ind, :]
            f_m = np.zeros_like(j_m)
            for i, (tr_ind, gt_ind) in enumerate(zip(row_ind, col_ind)):
                f_m[i] = self._compute_f(gt_dets, tracker_dets, tr_ind, gt_ind, bound_th)
        else:
            raise TrackEvalException('Unsupported optimization type %s for J&F metric.' % self.optim_type)

        # append zeros for false negatives
        if j_m.shape[0] < data['num_gt_ids']:
            diff = data['num_gt_ids'] - j_m.shape[0]
            j_m = np.concatenate((j_m, np.zeros((diff, j_m.shape[1]))), axis=0)
            f_m = np.concatenate((f_m, np.zeros((diff, f_m.shape[1]))), axis=0)

        # compute the metrics for each ground truth track
        res = {
            'J-Mean': [np.nanmean(j_m[i, :]) for i in range(j_m.shape[0])],
            'J-Recall': [np.nanmean(j_m[i, :] > 0.5 + np.finfo('float').eps) for i in range(j_m.shape[0])],
            'F-Mean': [np.nanmean(f_m[i, :]) for i in range(f_m.shape[0])],
            'F-Recall': [np.nanmean(f_m[i, :] > 0.5 + np.finfo('float').eps) for i in range(f_m.shape[0])],
            'J-Decay': [],
            'F-Decay': []
        }
        n_bins = 4
        ids = np.round(np.linspace(1, data['num_timesteps'], n_bins + 1) + 1e-10) - 1
        ids = ids.astype(np.uint8)

        for k in range(j_m.shape[0]):
            d_bins_j = [j_m[k][ids[i]:ids[i + 1] + 1] for i in range(0, n_bins)]
            res['J-Decay'].append(np.nanmean(d_bins_j[0]) - np.nanmean(d_bins_j[3]))
        for k in range(f_m.shape[0]):
            d_bins_f = [f_m[k][ids[i]:ids[i + 1] + 1] for i in range(0, n_bins)]
            res['F-Decay'].append(np.nanmean(d_bins_f[0]) - np.nanmean(d_bins_f[3]))

        # count number of tracks for weighting of the result
        res['num_gt_tracks'] = len(res['J-Mean'])
        for field in ['J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']:
            res[field] = np.mean(res[field])
        res['J&F'] = (res['J-Mean'] + res['F-Mean']) / 2
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {'num_gt_tracks': self._combine_sum(all_res, 'num_gt_tracks')}
        for field in self.summary_fields:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='num_gt_tracks')
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values
        'ignore empty classes' is not yet implemented here.
        """
        res = {'num_gt_tracks': self._combine_sum(all_res, 'num_gt_tracks')}
        for field in self.float_fields:
            res[field] = np.mean([v[field] for v in all_res.values()])
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {'num_gt_tracks': self._combine_sum(all_res, 'num_gt_tracks')}
        for field in self.float_fields:
            res[field] = np.mean([v[field] for v in all_res.values()])
        return res

    @staticmethod
    def _seg2bmap(seg, width=None, height=None):
        """
        From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries.  The boundary pixels are offset by 1/2 pixel towards the
        origin from the actual segment boundary.
        Arguments:
            seg     : Segments labeled from 1..k.
            width	  :	Width of desired bmap  <= seg.shape[1]
            height  :	Height of desired bmap <= seg.shape[0]
        Returns:
            bmap (ndarray):	Binary boundary map.
         David Martin <dmartin@eecs.berkeley.edu>
         January 2003
        """

        seg = seg.astype(bool)
        seg[seg > 0] = 1

        assert np.atleast_3d(seg).shape[2] == 1

        width = seg.shape[1] if width is None else width
        height = seg.shape[0] if height is None else height

        h, w = seg.shape[:2]

        ar1 = float(width) / float(height)
        ar2 = float(w) / float(h)

        assert not (
                width > w | height > h | abs(ar1 - ar2) > 0.01
        ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

        e = np.zeros_like(seg)
        s = np.zeros_like(seg)
        se = np.zeros_like(seg)

        e[:, :-1] = seg[:, 1:]
        s[:-1, :] = seg[1:, :]
        se[:-1, :-1] = seg[1:, 1:]

        b = seg ^ e | seg ^ s | seg ^ se
        b[-1, :] = seg[-1, :] ^ e[-1, :]
        b[:, -1] = seg[:, -1] ^ s[:, -1]
        b[-1, -1] = 0

        if w == width and h == height:
            bmap = b
        else:
            bmap = np.zeros((height, width))
            for x in range(w):
                for y in range(h):
                    if b[y, x]:
                        j = 1 + math.floor((y - 1) + height / h)
                        i = 1 + math.floor((x - 1) + width / h)
                        bmap[j, i] = 1

        return bmap

    @staticmethod
    def _compute_f(gt_data, tracker_data, tracker_data_id, gt_id, bound_th):
        """
        Perform F computation for a given gt and a given tracker ID. Adapted from
        https://github.com/davisvideochallenge/davis2017-evaluation
        :param gt_data: the encoded gt masks
        :param tracker_data: the encoded tracker masks
        :param tracker_data_id: the tracker ID
        :param gt_id: the ground truth ID
        :param bound_th: boundary threshold parameter
        :return: the F value for the given tracker and gt ID
        """

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils
        from skimage.morphology import disk
        import cv2

        f = np.zeros(len(gt_data))

        for t, (gt_masks, tracker_masks) in enumerate(zip(gt_data, tracker_data)):
            curr_tracker_mask = mask_utils.decode(tracker_masks[tracker_data_id])
            curr_gt_mask = mask_utils.decode(gt_masks[gt_id])
            
            bound_pix = bound_th if bound_th >= 1 - np.finfo('float').eps else \
                np.ceil(bound_th * np.linalg.norm(curr_tracker_mask.shape))

            # Get the pixel boundaries of both masks
            fg_boundary = JAndF._seg2bmap(curr_tracker_mask)
            gt_boundary = JAndF._seg2bmap(curr_gt_mask)

            # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
            fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
            # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
            gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

            # Get the intersection
            gt_match = gt_boundary * fg_dil
            fg_match = fg_boundary * gt_dil

            # Area of the intersection
            n_fg = np.sum(fg_boundary)
            n_gt = np.sum(gt_boundary)

            # % Compute precision and recall
            if n_fg == 0 and n_gt > 0:
                precision = 1
                recall = 0
            elif n_fg > 0 and n_gt == 0:
                precision = 0
                recall = 1
            elif n_fg == 0 and n_gt == 0:
                precision = 1
                recall = 1
            else:
                precision = np.sum(fg_match) / float(n_fg)
                recall = np.sum(gt_match) / float(n_gt)

            # Compute F measure
            if precision + recall == 0:
                f_val = 0
            else:
                f_val = 2 * precision * recall / (precision + recall)

            f[t] = f_val

        return f

    @staticmethod
    def _compute_j(gt_data, tracker_data, num_gt_ids, num_tracker_ids, num_timesteps):
        """
        Computation of J value for all ground truth IDs and all tracker IDs in the given sequence. Adapted from
        https://github.com/davisvideochallenge/davis2017-evaluation
        :param gt_data: the ground truth masks
        :param tracker_data: the tracker masks
        :param num_gt_ids: the number of ground truth IDs
        :param num_tracker_ids: the number of tracker IDs
        :param num_timesteps: the number of timesteps
        :return: the J values
        """

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils

        j = np.zeros((num_tracker_ids, num_gt_ids, num_timesteps))

        for t, (time_gt, time_data) in enumerate(zip(gt_data, tracker_data)):
            # run length encoded masks with pycocotools
            area_gt = mask_utils.area(time_gt)
            time_data = list(time_data)
            area_tr = mask_utils.area(time_data)

            area_tr = np.repeat(area_tr[:, np.newaxis], len(area_gt), axis=1)
            area_gt = np.repeat(area_gt[np.newaxis, :], len(area_tr), axis=0)

            # mask iou computation with pycocotools
            ious = np.atleast_2d(mask_utils.iou(time_data, time_gt, [0]*len(time_gt)))
            # set iou to 1 if both masks are close to 0 (no ground truth and no predicted mask in timestep)
            ious[np.isclose(area_tr, 0) & np.isclose(area_gt, 0)] = 1
            assert (ious >= 0 - np.finfo('float').eps).all()
            assert (ious <= 1 + np.finfo('float').eps).all()

            j[..., t] = ious

        return j
