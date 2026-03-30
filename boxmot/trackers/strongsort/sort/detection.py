# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    __slots__ = ("tlwh", "xyah", "conf", "cls", "det_ind", "feat")

    def __init__(self, tlwh, conf, cls, det_ind, feat):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.xyah = self.tlwh.copy()
        self.xyah[:2] += self.xyah[2:] / 2.0
        self.xyah[2] /= max(self.xyah[3], 1e-6)
        self.conf = conf
        self.cls = cls
        self.det_ind = det_ind
        self.feat = None if feat is None else np.asarray(feat, dtype=np.float32)

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        return self.xyah
