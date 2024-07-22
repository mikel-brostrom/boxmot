



class BaseAssocTrack(BaseTracker):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.65,
        proximity_thresh: float = 0.1,
        appearance_thresh: float = 0.25,
        cmc_method: str = "sof",
        frame_rate=30,
        with_reid: bool = True,
    ):
        super().__init__()
        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid
        if self.with_reid:
            self.model = ReidAutoBackend(
                weights=model_weights, device=device, half=fp16
            ).model

        self.cmc = SOF()

    def extract_features(self, dets_first, img, embs):
        if self.with_reid:
            if embs is not None:
                return embs
            else:
                return self.model.get_features(dets_first[:, 0:4], img)
        return None

    def prepare_detections(self, dets_first, features_high):
        if self.with_reid:
            return [STrack(det, f, max_obs=self.max_obs) for (det, f) in zip(dets_first, features_high)]
        else:
            return [STrack(det, max_obs=self.max_obs) for (det) in np.array(dets_first)]

    def update_state(self, activated_starcks, refind_stracks, lost_stracks, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        output_stracks = [track for track in self.active_tracks]
        outputs = np.array([t.xyxy + [t.id, t.conf, t.cls, t.det_ind] for t in output_stracks])
        return outputs

    def associate_tracks(self, strack_pool, detections, dists, match_thresh):
        matches, u_track, u_detection = linear_assignment(dists, thresh=match_thresh)
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        return matches, activated_starcks, refind_stracks, lost_stracks, removed_stracks
