# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np


class CMCInterface:

    def apply(self, im):
        raise NotImplementedError('Subclasses must implement this method.')

    def generate_mask(self, img, dets, scale):
        h, w = img.shape
        mask = np.zeros_like(img)

        mask[int(0.02 * h): int(0.98 * h), int(0.02 * w): int(0.98 * w)] = 255
        if dets is not None:
            for det in dets:
                tlbr = np.multiply(det, scale).astype(int)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        return mask
