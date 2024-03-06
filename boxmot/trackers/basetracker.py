class BaseTracker(object):
    def __init__(self, det_thresh=0.3, max_age=30, min_hits=3, iou_threshold=0.3):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes

    def update(self, dets, img, embs=None):
        raise NotImplementedError("The update method needs to be implemented by the subclass.")
