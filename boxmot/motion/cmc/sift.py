# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import copy
import time

import cv2
import numpy as np

from boxmot.motion.cmc.cmc_interface import CMCInterface
from boxmot.utils import BOXMOT


class SIFT(CMCInterface):

    def __init__(
        self,
        warp_mode=cv2.MOTION_EUCLIDEAN,
        eps=1e-5,
        max_iter=100,
        scale=0.1,
        grayscale=True,
        draw_keypoint_matches=False,
        align=False
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
        self.grayscale = grayscale
        self.scale = scale
        self.warp_mode = warp_mode
        self.termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            self.warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)

        self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
        self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.prev_img = None
        self.minimum_features = 10
        self.prev_desc = None
        self.draw_keypoint_matches = draw_keypoint_matches
        self.align = align

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

    def apply(self, img, dets):

        H = np.eye(2, 3)

        img = self.preprocess(img)
        h, w = img.shape

        # generate dynamic object maks
        mask = self.generate_mask(img, dets, self.scale)

        # find static keypoints
        keypoints = self.detector.detect(img, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(img, keypoints)

        # handle first frame
        if self.prev_img is None:
            # Initialize data
            self.prev_dets = dets.copy()
            self.prev_img = img.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.prev_descriptors = copy.copy(descriptors)

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prev_img = img.copy()
            self.prev_keypoints = copy.copy(keypoints)
            self.prev_descriptors = copy.copy(descriptors)

            return H

        # filtered matches based on smallest spatial distance
        matches = []
        spatial_distances = []
        max_spatial_distance = 0.25 * np.array([w, h])

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prev_keypoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatial_distance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                    prevKeyPointLocation[1] - currKeyPointLocation[1])

                if (np.abs(spatial_distance[0]) < max_spatial_distance[0]) and \
                        (np.abs(spatial_distance[1]) < max_spatial_distance[1]):
                    spatial_distances.append(spatial_distance)
                    matches.append(m)

        mean_spatial_distances = np.mean(spatial_distances, 0)
        std_spatial_distances = np.std(spatial_distances, 0)

        inliesrs = (spatial_distances - mean_spatial_distances) < 2.5 * std_spatial_distances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prev_keypoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Draw the keypoint matches on the output image
        if self.draw_keypoint_matches:
            self.prev_img[:, :][mask == True] = 0  # noqa:E712
            self.matches_img = np.hstack((self.prev_img, img))
            self.matches_img = cv2.cvtColor(self.matches_img, cv2.COLOR_GRAY2BGR)

            W = np.size(self.prev_img, 1)
            for m in goodMatches:
                prev_pt = np.array(self.prev_keypoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))
                self.matches_img = cv2.line(self.matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                self.matches_img = cv2.circle(self.matches_img, prev_pt, 2, tuple(color), -1)
                self.matches_img = cv2.circle(self.matches_img, curr_pt, 2, tuple(color), -1)
            for det in dets:
                det = np.multiply(det, self.scale).astype(int)
                start = (det[0] + w, det[1])
                end = (det[2] + w, det[3])
                self.matches_img = cv2.rectangle(self.matches_img, start, end, (0, 0, 255), 2)
            for det in self.prev_dets:
                det = np.multiply(det, self.scale).astype(int)
                start = (det[0], det[1])
                end = (det[2], det[3])
                self.matches_img = cv2.rectangle(self.matches_img, start, end, (0, 0, 255), 2)
        else:
            self.matches_img = None

        # find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # upscale warp matrix to original images size
            if self.scale < 1.0:
                H[0, 2] /= self.scale
                H[1, 2] /= self.scale

            if self.align:
                self.prev_img_aligned = cv2.warpAffine(self.prev_img, H, (w, h), flags=cv2.INTER_LINEAR)
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prev_img = img.copy()
        self.prev_keypoints = copy.copy(keypoints)
        self.prev_descriptors = copy.copy(descriptors)

        return H


def main():
    sift = SIFT(scale=0.5, align=True, grayscale=True, draw_keypoint_matches=False)
    curr_img = cv2.imread('assets/MOT17-mini/train/MOT17-13-FRCNN/img1/000005.jpg')
    prev_img = cv2.imread('assets/MOT17-mini/train/MOT17-13-FRCNN/img1/000001.jpg')
    curr_dets = np.array(
        [[1083.8207,  541.5978, 1195.7952,  655.8790],  # noqa:E241
         [1635.6456,  563.8348, 1695.4153,  686.6704],  # noqa:E241
         [ 957.0879,  545.6558, 1042.6743,  611.8740],  # noqa:E241,E261,E201
         [1550.0317,  562.5705, 1600.3931,  684.7425],  # noqa:E241
         [  78.8801,  714.3307,  121.0272,  817.6857],  # noqa:E241,E261,E201
         [1382.9938,  512.2731, 1418.6012,  620.1938],  # noqa:E241
         [1459.7921,  496.2123, 1488.5767,  584.3533],  # noqa:E241
         [ 982.9818,  492.8579, 1013.6625,  517.9271],  # noqa:E241,E261,E201
         [ 496.1809,  541.3972,  531.4617,  638.0989],  # noqa:E241,E261,E201
         [1498.8512,  522.6646, 1526.1145,  587.7672],  # noqa:E241
         [ 536.4527,  548.4061,  569.2723,  635.5656],  # noqa:E241,E261,E201
         [ 247.8834,  580.8851,  287.2241,  735.3685],  # noqa:E241,E261,E201
         [ 151.4096,  572.3918,  203.5401,  731.1011],  # noqa:E241,E261,E201
         [1227.4098,  440.5505, 1252.7986,  489.5295]]  # noqa:E241
    )
    prev_dets = np.array(
        [[2.1069e-02, 6.7026e+02, 4.9816e+01, 8.8407e+02],
         [1.0765e+03, 5.4009e+02, 1.1883e+03, 6.5219e+02],
         [1.5208e+03, 5.6322e+02, 1.5711e+03, 6.7676e+02],
         [1.6111e+03, 5.5926e+02, 1.6640e+03, 6.7443e+02],
         [9.5244e+02, 5.4681e+02, 1.0384e+03, 6.1180e+02],
         [1.3691e+03, 5.1258e+02, 1.4058e+03, 6.1695e+02],
         [1.2043e+02, 7.0780e+02, 1.7309e+02, 8.0518e+02],
         [1.4454e+03, 5.0919e+02, 1.4724e+03, 5.8270e+02],
         [9.7848e+02, 4.9563e+02, 1.0083e+03, 5.1980e+02],
         [5.0166e+02, 5.4778e+02, 5.3796e+02, 6.3940e+02],
         [1.4777e+03, 5.1856e+02, 1.5105e+03, 5.9523e+02],
         [1.9540e+02, 5.7292e+02, 2.3711e+02, 7.2717e+02],
         [2.7373e+02, 5.8564e+02, 3.1335e+02, 7.3281e+02],
         [5.4038e+02, 5.4735e+02, 5.7359e+02, 6.3797e+02],
         [1.2190e+03, 4.4176e+02, 1.2414e+03, 4.9038e+02]]
    )

    warp_matrix = sift.apply(prev_img, prev_dets)
    warp_matrix = sift.apply(curr_img, curr_dets)

    start = time.process_time()
    for i in range(0, 100):
        warp_matrix = sift.apply(prev_img, prev_dets)
        warp_matrix = sift.apply(curr_img, curr_dets)
    end = time.process_time()
    print('Total time', end - start)
    print(warp_matrix)

    if sift.prev_img_aligned is not None:
        curr_img = sift.preprocess(curr_img)
        prev_img = sift.preprocess(prev_img)
        weighted_img = cv2.addWeighted(curr_img, 0.5, sift.prev_img_aligned, 0.5, 0)
        cv2.imshow('prev_img_aligned', weighted_img)
        cv2.waitKey(0)
        cv2.imwrite(str(BOXMOT / 'motion/cmc/sift_aligned.jpg'), weighted_img)


if __name__ == "__main__":
    main()
