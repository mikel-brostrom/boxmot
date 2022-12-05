import numpy as np
import scipy.linalg
import scipy.special
import torch
import cv2
from collections import deque

from yolov5.utils.general import xyxy2xywh

from ..ocsort.association import *
from ..ocsort.kalmanfilter import KalmanFilterNew as KalmanFilter

from ..strong_sort.reid_multibackend import ReIDDetectMultiBackend
from ..strong_sort.sort import linear_assignment as strong_linear_assignment
from ..strong_sort.sort import iou_matching
from ..strong_sort.sort.detection import Detection
from ..strong_sort.sort.nn_matching import NearestNeighborDistanceMetric, _cosine_distance

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    a = bbox
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def cosine_distance_to_image_center(bbox, image_size):
    """
    Calculates the absolute cosine distance subtracted by 1, from an up vector located at the image center to
    the specified bbox.

    Inputs:
        bbox - bounding box of type (x pos, y pos, s area, r aspect ratio), shape (1, 4)
        image_size - tuple (width, height)

    Returns:
        Absolute value of cosine distance subtracted by 1 (higher values means further away from up vector located
        at image center)
    """
    up_vector = np.asarray([[0., 1., 0., 0.]])
    image_center_pos = np.asarray([[image_size[0] / 2., image_size[1] / 2., 0., 0.]])
    return np.abs(_cosine_distance(up_vector, bbox - image_center_pos).flatten() - 1)[0]


class ClassifiedDetection(Detection):
    def __init__(self, tlwh, confidence, cls, feature):
        super().__init__(tlwh, confidence, feature)
        self.cls = cls


class AssocMethod:
    FEATURE = 0
    TRAJECTORY = 1
    BYTE = 3
    REBORN = 4
    NONE = -1


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class CompatibleKalmanFilter(KalmanFilter):

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean = np.dot(self.H, mean)
        covariance = np.dot(self.H, np.dot(covariance, self.H.T)) + self.R

        measurements = measurements.copy()
        x = measurements[:, 0].copy()
        y = measurements[:, 1].copy()
        r = measurements[:, 2].copy()
        s = measurements[:, 2] * measurements[:, 3] * measurements[:, 3]   # scale is just area

        measurements[:, 0] = x
        measurements[:, 1] = y
        measurements[:, 2] = s
        measurements[:, 3] = r

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean.T

        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)

        return squared_maha


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, cls, delta_t=3, feature=None, id=-1, image_size=(0,0)):
        self.image_size = image_size
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        self.kf = CompatibleKalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]])

        """
            Motion and observation uncertainty are chosen relative to the current
            state estimate. These weights control the amount of uncertainty in
            the model. This is a bit hacky.
        """
        self._std_weight_position = 10.
        self._std_weight_size = 1. / 20
        self._std_weight_velocity = 20.

        state_space_bbox = convert_bbox_to_z(bbox)

        dist = cosine_distance_to_image_center(state_space_bbox, self.image_size)
        std = [
            2 * self._std_weight_position * dist + 1,  # the center point x
            2 * self._std_weight_position * dist + 1,  # the center point y
            2 * self._std_weight_size * state_space_bbox[2][0],  # the scale (area)
            1 * state_space_bbox[3][0],  # the ratio of width/height
            10 * self._std_weight_velocity * dist + 1,
            10 * self._std_weight_velocity * dist + 1,
            10 * self._std_weight_velocity * dist + 1]
        self.kf.P = np.diag(np.square(std).flatten())

        self.kf.x[:4] = state_space_bbox
        self.time_since_update = 0
        if id < 0:
            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1
        else:
            self.id = id
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[4]
        self.cls = cls
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

        self.state = TrackState.Tentative
        self.update_method = AssocMethod.NONE

        # Initializing trajectory queue
        self.trajectory_queue = deque(maxlen=25)

        self.features = []
        if feature is not None:
            feature = feature.copy()
            feature /= np.linalg.norm(feature)
            self.features.append(feature)

    def update(self, bbox, cls, feature=None, ema_alpha=0.9):
        """
        Updates the state vector with observed bbox.
        """

        if bbox is not None:
            self.conf = bbox[4]
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

            """
            Update feature
            """
            if feature is not None:
                feature = feature / np.linalg.norm(feature)

                smooth_feat = ema_alpha * self.features[-1] + (1 - ema_alpha) * feature
                smooth_feat /= np.linalg.norm(smooth_feat)
                self.features = [smooth_feat]

            """
            Update covariance matrices
            """
            dist = cosine_distance_to_image_center(bbox[:4].reshape(1, -1), self.image_size)
            std = [
                2. * self._std_weight_position * dist + 1,
                2. * self._std_weight_position * dist + 1,
                self._std_weight_size * self.kf.x[2][0],
                1 * self.kf.x[3][0]]
            std = [(2 - self.conf) * x for x in std]
            self.kf.R = np.diag(np.square(std).flatten())

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.occlusion_multiplier = 1.
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        # Update process noise
        dist = cosine_distance_to_image_center(self.kf.x, self.image_size)
        std_pos = [
            self._std_weight_position * dist + 1,
            self._std_weight_position * dist + 1,
            self._std_weight_size * self.kf.x[2][0],
            1 * self.kf.x[3][0]]
        std_vel = [
            self._std_weight_velocity * dist + 1,
            self._std_weight_velocity * dist + 1,
            self._std_weight_velocity * dist + 1]
        self.kf.Q = np.diag(np.square(np.r_[std_pos, std_vel]).flatten())

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    @property
    def mean(self):
        return self.kf.x

    @property
    def covariance(self):
        return self.kf.P

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = convert_x_to_bbox(self.kf.x)[:4].copy().flatten()
        ret[2:] = ret[2:] - ret[:2]
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        return convert_x_to_bbox(self.kf.x)[:4].copy().flatten()

    def camera_update(self, previous_frame, next_frame):
        warp_matrix, src_aligned = ECC(previous_frame, next_frame)
        if warp_matrix is None and src_aligned is None:
            return
        [a,b] = warp_matrix
        warp_matrix=np.array([a,b,[0,0,1]])
        warp_matrix = warp_matrix.tolist()
        matrix = get_matrix(warp_matrix)

        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
        self.kf.x[:4] = convert_bbox_to_z([x1_, y1_, x2_, y2_])


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSOC_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist
}


class StrongOCSort(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        det_thresh: float,
        max_dist: float = 0.2,
        nn_budget: int = 100,
        max_iou_distance: float = 0.7,
        ema_alpha : float = 0.9,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold : float = 0.3,
        delta_t: float = 3,
        assoc_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
        use_resurrection: bool = False
    ):
        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)

        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.assoc_func = ASSOC_FUNCS[assoc_func]
        self.inertia = inertia
        self.use_byte = use_byte
        self.use_resurrection = use_resurrection
        KalmanBoxTracker.count = 0

        self.ema_alpha = ema_alpha
        self.max_dist = max_dist
        self.metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)

        # henriksod: This is my attempt to cache features of tracks which
        # have "died". In case a detection with a similar feature shows up again
        # in a sequence (lady with red shirt in MOT17-05 for example), it should
        # create a track of the same ID as the one that previously "died".
        # Turn on/off the system with use_resurrection
        self.track_graveyard = []

        self.height, self.width = 0, 0


    def camera_update(self, previous_img, current_img):
        for track in self.trackers:
            track.camera_update(previous_img, current_img)

    def update(self, dets, ori_img):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
          ori_img - the input image from where the detections originates
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        self.frame_count += 1

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss.numpy()
        xyxys = xyxys.numpy()
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        output_results = np.column_stack((xyxys, confs, classes))

        inds_low = confs > 0.1
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = output_results[inds_second]  # detections for second matching
        remain_inds = confs > self.det_thresh
        dets = output_results[remain_inds]

        xywhs = xyxy2xywh(dets[:, :4])
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        features = self._get_features(xywhs, ori_img)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                x_c, y_c = self.trackers[t].kf.x[:2].flatten().astype(int)
                self.trackers[t].trajectory_queue.append((self.trackers[t].update_method, (x_c, y_c)))
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        """
            First association step
        """
        detections = [ClassifiedDetection(bbox_tlwh[i], dets[i, 4], dets[i, 5], features[i]) for i, _ in enumerate(dets)]

        # Run matching cascade.
        matched, unmatched_trks, unmatched_dets = self._match(detections)

        for m in matched:
            trk = self.trackers[m[0]]
            trk.update_method = m[2]
            _feature = np.asarray(features[m[1]].cpu(), dtype=np.float32)
            trk.update(dets[m[1], :5], dets[m[1], 5], _feature, self.ema_alpha)

        """
            Second association step
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)  # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update_method = AssocMethod.BYTE
                    self.trackers[trk_ind].update(dets_second[det_ind, :5], dets_second[det_ind, 5])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update_method = AssocMethod.NONE
            self.trackers[m].update(None, None)
            self.trackers[m].mark_missed()

        # create and initialise new trackers for unmatched detections
        ressurection_pool = {}
        dead_matches = {}
        for i in unmatched_dets:
            detection = detections[i]

            # We need to match the detection feature to every feature in the graveyard
            if self.track_graveyard:
                dead_ids = self.track_graveyard

                det_feature = np.asarray([detection.feature])
                match_costs = self.metric.distance(det_feature, np.asarray(dead_ids)).flatten()
                minimum_cost_index = np.argmin(match_costs)

                if match_costs[minimum_cost_index] < self.max_dist:
                    revivable_id = dead_ids[minimum_cost_index]
                    if revivable_id in dead_matches:
                        if ressurection_pool[dead_matches[revivable_id]][1] > match_costs[minimum_cost_index]:
                            del ressurection_pool[dead_matches[revivable_id]]
                            ressurection_pool[i] = (revivable_id, match_costs[minimum_cost_index])
                            dead_matches[revivable_id] = i
                    else:
                        ressurection_pool[i] = (revivable_id, match_costs[minimum_cost_index])
                        dead_matches[revivable_id] = i

        for i in unmatched_dets:
            revivable_id = -1
            update_method = AssocMethod.NONE
            detection = detections[i]
            if i in ressurection_pool:
                update_method = AssocMethod.REBORN
                revivable_id = ressurection_pool[i][0]
                self.track_graveyard.pop(self.track_graveyard.index(revivable_id))

            trk = KalmanBoxTracker(
                dets[i, :5],
                dets[i, 5],
                delta_t=self.delta_t,
                feature=detection.feature,
                id=revivable_id,
                image_size=(self.width, self.height)
            )
            trk.update_method = update_method
            self.trackers.append(trk)

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):

            d = trk.get_state()[0]

            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                if trk.state == TrackState.Tentative:
                    trk.state = TrackState.Confirmed
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1], [trk.cls])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.state == TrackState.Deleted or trk.time_since_update > self.max_age:
                if trk.state == TrackState.Confirmed and self.use_resurrection:
                    # Cache feature of dead track, it might show up again...
                    self.track_graveyard.append(trk.id)
                self.trackers.pop(i)

        # Update distance metric.
        features, targets = [], []
        for track in self.trackers:
            if not (track.is_confirmed() or (trk.time_since_update < 1)):
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        # Don't specify active targets here, since we cache dead track features for the revival system
        self.metric.partial_fit(np.asarray(features), np.asarray(targets))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = strong_linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices, detection_indices) #, only_position=True

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.trackers) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.trackers) if not t.is_confirmed()]

        """
        Matching step 1, deep assoc
        """
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            strong_linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.trackers, detections, confirmed_tracks)

        # FIXME: Probably a better way to do this, store the method used for match
        tmp = []
        for a, b in matches_a:
            m = (a, b, AssocMethod.FEATURE)
            tmp.append(m)
        matches_a = tmp

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.trackers[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.trackers[k].time_since_update != 1]

        """
        Matching step 2, trajectory assoc
        """
        matches_b = []
        unmatched_tracks_b = []
        unmatched_detections_new = unmatched_detections
        if len(iou_track_candidates) > 0 and len(unmatched_detections) > 0:
            velocities = np.array(
                [self.trackers[t].velocity if self.trackers[t].velocity is not None else np.array((0, 0)) for t in
                 iou_track_candidates])
            k_observations = np.array(
                [k_previous_obs(self.trackers[t].observations, self.trackers[t].age, self.delta_t) for t in
                 iou_track_candidates])

            xyxys = []
            confs = []
            classes = []
            for det_i in unmatched_detections:
                det = detections[det_i]
                xyxys.append(det.to_tlbr())
                confs.append(det.confidence)
                classes.append(det.cls)

            dets = np.column_stack(
                (xyxys, confs)
            )

            trks = np.zeros((len(iou_track_candidates), 5))
            for i, t in enumerate(iou_track_candidates):
                pos = self.trackers[t].history[-1][0]
                trks[i][:] = [pos[0], pos[1], pos[2], pos[3], 0]

            matched, unmatched_det, unmatched_trks = associate(
                dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)

            unmatched_tracks_b = [iou_track_candidates[i] for i in unmatched_trks]
            unmatched_detections_new = [unmatched_detections[i] for i in unmatched_det]

            matches_b = [(iou_track_candidates[t[1]], unmatched_detections[t[0]]) for t in matched]

            # FIXME: Probably a better way to do this, store the method used for match
            tmp = []
            for a, b in matches_b:
                m = (a, b, AssocMethod.TRAJECTORY)
                tmp.append(m)
            matches_b = tmp

        """
        Combine results from 2 matching steps
        """
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, np.array(unmatched_tracks), np.array(unmatched_detections_new)

    def trajectory(self, im0, color, fill=None, outline=None, width=1):
        track_id_whitelist = []  # Specify IDs to only plot those tracks
        # Add rectangle to image (PIL-only)
        for t in self.trackers:
            if track_id_whitelist:
                if t.id+1 not in track_id_whitelist:
                    continue
            # Color each point based on the association method used
            for point in t.trajectory_queue:
                if point[0] == AssocMethod.FEATURE:
                    cv2.circle(im0, point[1], 2, (255, 0, 0), 2)
                elif point[0] == AssocMethod.TRAJECTORY:
                    cv2.circle(im0, point[1], 2, (0, 255, 0), 2)
                elif point[0] == AssocMethod.BYTE:
                    cv2.circle(im0, point[1], 2, (0, 0, 255), 2)
                elif point[0] == AssocMethod.REBORN:
                    cv2.circle(im0, point[1], 2, (255, 0, 255), 2)
                else:
                    cv2.circle(im0, point[1], 2, (255, 255, 255), 2)

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    @staticmethod
    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features


def ECC(src, dst, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-5,
        max_iter=100, scale=0.1, align=False):
    """Compute the warp matrix from src to dst.
    Parameters
    ----------
    src : ndarray
        An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
    dst : ndarray
        An NxM matrix of target img(BGR or Gray).
    warp_mode: flags of opencv
        translation: cv2.MOTION_TRANSLATION
        rotated and shifted: cv2.MOTION_EUCLIDEAN
        affine(shift,rotated,shear): cv2.MOTION_AFFINE
        homography(3d): cv2.MOTION_HOMOGRAPHY
    eps: float
        the threshold of the increment in the correlation coefficient between two iterations
    max_iter: int
        the number of iterations.
    scale: float or [int, int]
        scale_ratio: float
        scale_size: [W, H]
    align: bool
        whether to warp affine or perspective transforms to the source image
    Returns
    -------
    warp matrix : ndarray
        Returns the warp matrix from src to dst.
        if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
    src_aligned: ndarray
        aligned source image of gray
    """

    # BGR2GRAY
    if src.ndim == 3:
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # make the imgs smaller to speed up
    if scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            if scale != 1:
                src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                scale = [scale, scale]
            else:
                src_r, dst_r = src, dst
                scale = None
        else:
            if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                src_r = cv2.resize(src, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
            else:
                src_r, dst_r = src, dst
                scale = None
    else:
        src_r, dst_r = src, dst

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC(src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)
    except cv2.error as e:
        print('ecc transform failed')
        return None, None

    if scale is not None:
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

    if align:
        sz = src.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        return warp_matrix, src_aligned
    else:
        return warp_matrix, None


def get_matrix(matrix):
    eye = np.eye(3)
    dist = np.linalg.norm(eye - matrix)
    if dist < 100:
        return matrix
    else:
        return eye