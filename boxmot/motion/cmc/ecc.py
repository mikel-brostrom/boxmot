# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import time

import cv2
import numpy as np

from boxmot.motion.cmc.cmc_interface import CMCInterface
from boxmot.utils import BOXMOT
from boxmot.utils import logger as LOGGER


class ECC(CMCInterface):
    def __init__(
        self,
        warp_mode=cv2.MOTION_EUCLIDEAN,
        eps=1e-5,
        max_iter=100,
        scale=0.1,
        align=False,
        grayscale=True
    ):
        """Compute the warp matrix from src to dst.

        Parameters
        ----------
        warp_mode: opencv flag
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
        grayscale: bool
            whether to transform 3 channel RGB to single channel grayscale for faster computations

        Returns
        -------
        warp matrix : ndarray
            Returns the warp matrix from src to dst.
            if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
        src_aligned: ndarray
            aligned source image of gray
        """
        self.align = align
        self.grayscale = grayscale
        self.scale = scale
        self.warp_mode = warp_mode
        self.termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
        self.prev_img = None

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

    def apply(self, curr_img, dets):

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        if self.prev_img is None:
            self.prev_img = self.preprocess(curr_img)
            return warp_matrix

        curr_img = self.preprocess(curr_img)

        try:
            (ret_val, warp_matrix) = cv2.findTransformECC(
                self.prev_img,  # already processed
                curr_img,
                warp_matrix,
                self.warp_mode,
                self.termination_criteria,
                None,
                1
            )
        except Exception as e:
            LOGGER.warning(f'Affine matrix could not be generated: {e}. Returning identity')
            return warp_matrix

        # upscale warp matrix to original images size
        if self.scale < 1:
            warp_matrix[0, 2] /= self.scale
            warp_matrix[1, 2] /= self.scale

        if self.align:
            h, w = self.prev_img.shape
            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                self.prev_img_aligned = cv2.warpPerspective(self.prev_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                self.prev_img_aligned = cv2.warpAffine(self.prev_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
        else:
            self.prev_img_aligned = None

        self.prev_img = curr_img

        return warp_matrix  # , prev_img_aligned


def main():
    ecc = ECC(scale=0.5, align=True, grayscale=True)
    curr_img = cv2.imread('assets/MOT17-mini/train/MOT17-13-FRCNN/img1/000005.jpg')
    prev_img = cv2.imread('assets/MOT17-mini/train/MOT17-13-FRCNN/img1/000001.jpg')

    warp_matrix = ecc.apply(prev_img, None)
    warp_matrix = ecc.apply(curr_img, None)

    start = time.process_time()
    for i in range(0, 100):
        warp_matrix = ecc.apply(prev_img, None)
        warp_matrix = ecc.apply(curr_img, None)
    end = time.process_time()
    print('Total time', end - start)
    print(warp_matrix)

    if ecc.prev_img_aligned is not None:
        curr_img = ecc.preprocess(curr_img)
        prev_img = ecc.preprocess(prev_img)
        weighted_img = cv2.addWeighted(curr_img, 0.5, ecc.prev_img_aligned, 0.5, 0)
        cv2.imshow('prev_img_aligned', weighted_img)
        cv2.waitKey(0)
        cv2.imwrite(str(BOXMOT / 'motion/cmc/ecc_aligned.jpg'), weighted_img)


if __name__ == "__main__":
    main()
