    def aiou(self, bbox, candidates):
        """
        IoU - Aspect Ratio
        """
        candidates = np.array(candidates)
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)

        iou = area_intersection / (area_bbox + area_candidates - area_intersection)

        aspect_ratio = bbox[2] / bbox[3]
        candidates_aspect_ratio = candidates[:, 2] / candidates[:, 3]
        arctan = np.arctan(aspect_ratio) - np.arctan(candidates_aspect_ratio)
        v = 1 - ((4 / np.pi ** 2) * arctan ** 2)
        alpha = v / (1 - iou + v)

        return iou, alpha

    def aiou_vectorized(self, bboxes1, bboxes2):
        """
        Vectorized implementation of AIOU (IoU with aspect ratio consideration)
        
        Args:
        bboxes1: numpy array of shape (N, 4) in format (x, y, w, h)
        bboxes2: numpy array of shape (M, 4) in format (x, y, w, h)
        
        Returns:
        ious: numpy array of shape (N, M) containing IoU values
        alphas: numpy array of shape (N, M) containing alpha values
        """
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        bboxes1_x1y1 = bboxes1[:, :2]
        bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 2:]
        bboxes2_x1y1 = bboxes2[:, :2]
        bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 2:]

        # Compute intersection
        intersect_x1y1 = np.maximum(bboxes1_x1y1[:, None], bboxes2_x1y1[None, :])
        intersect_x2y2 = np.minimum(bboxes1_x2y2[:, None], bboxes2_x2y2[None, :])
        intersect_wh = np.maximum(0., intersect_x2y2 - intersect_x1y1)

        # Compute areas
        intersect_area = intersect_wh.prod(axis=2)
        bboxes1_area = bboxes1[:, 2:].prod(axis=1)
        bboxes2_area = bboxes2[:, 2:].prod(axis=1)
        union_area = bboxes1_area[:, None] + bboxes2_area[None, :] - intersect_area

        # Compute IoU
        ious = intersect_area / union_area

        # Compute aspect ratios
        bboxes1_aspect_ratio = bboxes1[:, 2] / bboxes1[:, 3]
        bboxes2_aspect_ratio = bboxes2[:, 2] / bboxes2[:, 3]

        # Compute alpha
        arctan_diff = np.arctan(bboxes1_aspect_ratio[:, None]) - np.arctan(bboxes2_aspect_ratio[None, :])
        v = 1 - ((4 / (np.pi ** 2)) * arctan_diff ** 2)
        alphas = v / (1 - ious + v)

        return ious, alphas


    def candidates_to_detections(self, tlwh: np.ndarray, confirmed_tracks):
        tracklet_bboxes = np.array([trk.to_xywh() for trk in confirmed_tracks])

        if len(tracklet_bboxes) == 0:
            return list(range(tlwh.shape[0])), []

        print(tlwh.shape)
        print(tracklet_bboxes.shape)

        ious, alphas = self.aiou_vectorized(tlwh, tracklet_bboxes)

        matches = ious > 0.5
        match_counts = np.sum(matches, axis=1)

        risky_detections = np.where(match_counts != 1)[0].tolist()
        safe_candidates = np.where(match_counts == 1)[0]

        safe_det_trk_pairs = []
        for i in safe_candidates:
            candidate = np.argmax(ious[i])
            if alphas[i, candidate] > 0.6:
                safe_det_trk_pairs.append((i, candidate))
            else:
                risky_detections.append(i)

        return risky_detections, safe_det_trk_pairs