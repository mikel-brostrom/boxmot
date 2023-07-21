import time

import cv2
import numpy as np

from boxmot.motion.cmc.cmc_interface import CMCInterface


class ECCStrategy(CMCInterface):

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
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            self.warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)

    def preprocess(self, curr_img, prev_img):

        # bgr2gray
        if self.grayscale:
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
            prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        # resize
        if self.scale is not None:
            curr_img = cv2.resize(
                curr_img,
                (0, 0),
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_LINEAR
            )
            prev_img = cv2.resize(
                prev_img,
                (0, 0),
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_LINEAR
            )
        return curr_img, prev_img

    def apply(self, curr_img, prev_img):

        curr_img, prev_img = self.preprocess(curr_img, prev_img)

        (ret_val, warp_matrix) = cv2.findTransformECC(
            prev_img,
            curr_img,
            self.warp_matrix,
            self.warp_mode,
            self.termination_criteria,
            None,
            1
        )

        if self.scale is not None:
            warp_matrix[0, 2] = warp_matrix[0, 2] / self.scale
            warp_matrix[1, 2] = warp_matrix[1, 2] / self.scale

        if self.align:
            h, w = prev_img.shape
            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                prev_img_aligned = cv2.warpPerspective(prev_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                prev_img_aligned = cv2.warpAffine(prev_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
            return warp_matrix, prev_img_aligned, curr_img, prev_img
        else:
            return warp_matrix, None


def main():
    ecc = ECCStrategy(scale=0.5, align=True, grayscale=True)
    curr_img = cv2.imread('assets/MOT17-mini/train/MOT17-13-FRCNN/img1/000005.jpg')
    prev_img = cv2.imread('assets/MOT17-mini/train/MOT17-13-FRCNN/img1/000001.jpg')

    start = time.process_time()
    warp_matrix, prev_img_aligned, curr_img, prev_img = ecc.apply(curr_img, prev_img)
    end = time.process_time()
    print('Total time', end - start)

    prev_img_aligned = cv2.cvtColor(prev_img_aligned, cv2.COLOR_GRAY2RGB)
    cv2.imshow('curr_img', curr_img)
    cv2.imshow('prev_img', prev_img)
    cv2.imshow('prev_img_aligned', prev_img_aligned)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
