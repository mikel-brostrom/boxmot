# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from abc import ABC, abstractmethod

import cv2
import numpy as np


class BaseCMC(ABC):

    @abstractmethod
    def apply(self, im):
        pass

    def generate_mask(self, img, dets, scale):
        h, w = img.shape
        mask = np.zeros_like(img)

        mask[int(0.02 * h): int(0.98 * h), int(0.02 * w): int(0.98 * w)] = 255
        if dets is not None:
            for det in dets:
                tlbr = np.multiply(det, scale).astype(int)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        return mask

    def preprocess(self, img):

        # bgr2gray
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize
        if self.scale is not None:
            img = cv2.resize(
                img,
                (0, 0),
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_LINEAR
            )

        return img
