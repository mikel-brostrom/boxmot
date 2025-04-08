import cv2
import numpy as np
import copy
import time
from boxmot.motion.cmc.base_cmc import BaseCMC


class SOF(BaseCMC):
    """
    Sparse Optical Flow (SOF) tracker for estimating a 2x3 warp (affine transformation)
    between consecutive frames. This class is modeled after a GMC implementation using
    the 'sparseOptFlow' method.
    """
    def __init__(self, scale=0.1):
        """
        Initialize the SOF object.

        Parameters
        ----------
        downscale : int, optional
            Factor by which to downscale the input frames. Defaults to 1 (no downscale).
        feature_params : dict, optional
            Parameters for cv2.goodFeaturesToTrack. Defaults to:
                {
                    maxCorners: 1000,
                    qualityLevel: 0.01,
                    minDistance: 1,
                    blockSize: 3,
                    useHarrisDetector: False,
                    k: 0.04
                }
        lk_params : dict, optional
            Lucas-Kanade optical flow parameters. Defaults to:
                {
                    winSize: (21, 21),
                    maxLevel: 3,
                    criteria: (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                }
        """
        self.scale = scale
        self.grayscale = True
        
        # Set default feature detection parameters if not provided
        self.feature_params = dict(
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=1,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04
        )

        # Set default Lucas-Kanade optical flow parameters if not provided.
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.prevFrame = None
        self.prevKeyPoints = None
        self.initializedFirstFrame = False

    def apply(self, img, detections=None):
        """
        Apply sparse optical flow tracking to estimate a warp (affine transformation)
        between the previous frame and the current raw frame.

        Parameters
        ----------
        raw_frame : np.ndarray
            The current input color image.
        detections : Any, optional
            (Not used here but provided for API compatibility.)

        Returns
        -------
        np.ndarray
            The estimated 2x3 warp matrix. If estimation fails, returns an identity matrix.
        """
        # Convert the raw frame to grayscale.
        frame_gray = self.preprocess(img)
        height, width = frame_gray.shape

        # Default transformation: identity.
        H = np.eye(2, 3, dtype=np.float32)

        # On the first frame, detect keypoints and initialize internal state.
        if not self.initializedFirstFrame:
            keypoints = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            if keypoints is None:
                return H
            # Optional subpixel refinement.
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            cv2.cornerSubPix(frame_gray, keypoints, winSize=(5, 5), zeroZone=(-1, -1), criteria=term_crit)
            self.prevFrame = frame_gray.copy()
            self.prevKeyPoints = keypoints.copy()
            self.initializedFirstFrame = True
            return H

        # Compute optical flow to track the previous keypoints into the current frame.
        nextKeypoints, status, err = cv2.calcOpticalFlowPyrLK(
            self.prevFrame, frame_gray, self.prevKeyPoints, None, **self.lk_params
        )

        # Filter out points that were not successfully tracked.
        valid_prev = []
        valid_next = []
        for i, s in enumerate(status):
            if s:
                valid_prev.append(self.prevKeyPoints[i])
                valid_next.append(nextKeypoints[i])

        if len(valid_prev) < 4:
            print("Warning: not enough matching points detected; redetecting keypoints.")
            # If too few matches, re-detect keypoints for the current frame.
            keypoints = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            self.prevFrame = frame_gray.copy()
            self.prevKeyPoints = keypoints if keypoints is not None else np.array([])
            return H

        valid_prev = np.array(valid_prev)
        valid_next = np.array(valid_next)

        # Estimate the affine warp matrix using robust RANSAC.
        H_est, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_next, method=cv2.RANSAC)
        if H_est is None:
            H_est = H
        else:
            # If the frame was downscaled, adjust the translation parameters back to original scale.
            if self.scale < 1:
                H_est[0, 2] /= self.scale
                H_est[1, 2] /= self.scale

        # Update the previous frame and keypoints for the next iteration.
        # Optionally, you might want to re-detect keypoints rather than simply tracking them.
        new_keypoints = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        if new_keypoints is None:
            # Use the tracked keypoints if new detection fails.
            new_keypoints = valid_next
        self.prevFrame = frame_gray.copy()
        self.prevKeyPoints = new_keypoints.copy()

        return H_est


# ==============================================================================
# Example Usage
# ==============================================================================

def main():
    # Create an instance of the SOF class with a downscaling factor, if desired.
    sof_tracker = SOF(scale=0.3)

    # For example purposes, load two consecutive frames.
    prev_img = cv2.imread("assets/MOT17-mini/train/MOT17-13-FRCNN/img1/000001.jpg")
    curr_img = cv2.imread("assets/MOT17-mini/train/MOT17-13-FRCNN/img1/000005.jpg")

    # Process the first frame to initialize the tracker.
    _ = sof_tracker.apply(prev_img)
    
    # Now process the next frame to compute the warp matrix.
    H = sof_tracker.apply(curr_img)
    print("Estimated warp matrix:\n", H)

    # Optionally, you can visualize the transformation (overlay, etc.)

if __name__ == "__main__":
    main()
