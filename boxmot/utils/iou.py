import numpy as np
import cv2 as cv

def iou_obb_pair(i, j, bboxes1, bboxes2):
    """
    Compute IoU for the rotated rectangles at index i and j in the batches `bboxes1`, `bboxes2` .
    """
    rect1 = bboxes1[int(i)]
    rect2 = bboxes2[int(j)]
    
    (cx1, cy1, w1, h1, angle1) = rect1[0:5]
    (cx2, cy2, w2, h2, angle2) = rect2[0:5]
    
    
    r1 = ((cx1, cy1), (w1, h1), angle1)
    r2 = ((cx2, cy2), (w2, h2), angle2)
    
    # Compute intersection
    ret, intersect = cv.rotatedRectangleIntersection(r1, r2)
    if ret == 0 or intersect is None:
        return 0.0  # No intersection
    
    # Calculate intersection area
    intersection_area = cv.contourArea(intersect)
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    # Compute IoU
    return intersection_area / union_area if union_area > 0 else 0.0

class AssociationFunction:
    def __init__(self, w, h, asso_mode="iou"):
        """
        Initializes the AssociationFunction class with the necessary parameters for bounding box operations.
        The association function is selected based on the `asso_mode` string provided during class creation.
        
        Parameters:
        w (int): The width of the frame, used for normalizing centroid distance.
        h (int): The height of the frame, used for normalizing centroid distance.
        asso_mode (str): The association function to use (e.g., "iou", "giou", "centroid", etc.).
        """
        self.w = w
        self.h = h
        self.asso_mode = asso_mode
        self.asso_func = self._get_asso_func(asso_mode)

    @staticmethod
    def iou_batch(bboxes1, bboxes2) -> np.ndarray:
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / (
            (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
            (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
            wh
        )
        return o
    
    @staticmethod
    def iou_batch_obb(bboxes1, bboxes2) -> np.ndarray:

        N, M = len(bboxes1), len(bboxes2)

        def wrapper(i, j):
            return iou_obb_pair(i, j, bboxes1, bboxes2)
        
        iou_matrix = np.fromfunction(np.vectorize(wrapper), shape=(N, M), dtype=int)
        return iou_matrix

    @staticmethod
    def hmiou_batch(bboxes1, bboxes2):
        """
        Compute a modified Intersection over Union (hIoU) between two batches of bounding boxes,
        incorporating a vertical overlap ratio.

        Parameters:
        - bboxes1: (N, 4) array of bounding boxes [x1, y1, x2, y2]
        - bboxes2: (M, 4) array of bounding boxes [x1, y1, x2, y2]

        Returns:
        - hmiou: (N, M) array where hmiou[i, j] is the modified IoU between bboxes1[i] and bboxes2[j]
        """
        # Expand dimensions for broadcasting
        bboxes1 = np.expand_dims(bboxes1, axis=1)  # Shape: (N, 1, 4)
        bboxes2 = np.expand_dims(bboxes2, axis=0)  # Shape: (1, M, 4)

        # Compute vertical overlap ratio 'o'
        intersect_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        intersect_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        intersection_height = np.maximum(0.0, intersect_y2 - intersect_y1)

        union_y1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        union_y2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        union_height = np.maximum(1e-10, union_y2 - union_y1)

        o = intersection_height / union_height

        # Compute standard IoU
        inter_x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        inter_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        inter_x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        inter_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])  # Shape: (N, 1)
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])  # Shape: (1, M)

        union_area = area1 + area2 - inter_area

        iou = inter_area / (union_area + 1e-10)

        # Modify IoU with vertical overlap ratio
        hmiou = iou * o

        return hmiou

    @staticmethod
    def giou_batch(bboxes1, bboxes2) -> np.ndarray:
        """
        :param bboxes1: predict of bbox(N,4)(x1,y1,x2,y2)
        :param bboxes2: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        """
        # Ensure predict's bbox form
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h  # Intersection area

        # Compute areas of individual boxes
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        # Union area
        union_area = area1 + area2 - wh

        iou = wh / union_area

        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        wc = xxc2 - xxc1
        hc = yyc2 - yyc1
        assert (wc > 0).all() and (hc > 0).all()
        area_enclose = wc * hc  # Area of the smallest enclosing box

        # Corrected GIoU computation
        giou = iou - (area_enclose - union_area) / area_enclose
        giou = (giou + 1.0) / 2.0  # Resize from (-1,1) to (0,1)
        return giou


    def centroid_batch(self, bboxes1, bboxes2) -> np.ndarray:
        centroids1 = np.stack(((bboxes1[..., 0] + bboxes1[..., 2]) / 2,
                               (bboxes1[..., 1] + bboxes1[..., 3]) / 2), axis=-1)
        centroids2 = np.stack(((bboxes2[..., 0] + bboxes2[..., 2]) / 2,
                               (bboxes2[..., 1] + bboxes2[..., 3]) / 2), axis=-1)

        centroids1 = np.expand_dims(centroids1, 1)
        centroids2 = np.expand_dims(centroids2, 0)

        distances = np.sqrt(np.sum((centroids1 - centroids2) ** 2, axis=-1))
        norm_factor = np.sqrt(self.w ** 2 + self.h ** 2)
        normalized_distances = distances / norm_factor

        return 1 - normalized_distances
    
    def centroid_batch_obb(self, bboxes1, bboxes2) -> np.ndarray:
        centroids1 = np.stack((bboxes1[..., 0], bboxes1[..., 1]),axis=-1)
        centroids2 = np.stack((bboxes2[..., 0], bboxes2[..., 1]),axis=-1)

        centroids1 = np.expand_dims(centroids1, 1)
        centroids2 = np.expand_dims(centroids2, 0)

        distances = np.sqrt(np.sum((centroids1 - centroids2) ** 2, axis=-1))
        norm_factor = np.sqrt(self.w ** 2 + self.h ** 2)
        normalized_distances = distances / norm_factor

        return 1 - normalized_distances
    
    
    @staticmethod
    def ciou_batch(bboxes1, bboxes2) -> np.ndarray:
        """
        Calculate Complete Intersection over Union (CIoU) for batches of bounding boxes.

        :param bboxes1: Predicted bounding boxes of shape (N, 4) as (x1, y1, x2, y2)
        :param bboxes2: Ground truth bounding boxes of shape (N, 4) as (x1, y1, x2, y2)
        :return: CIoU scores scaled between 0 and 1
        """
        epsilon = 1e-7  # Small value to prevent division by zero

        # Expand dimensions for broadcasting
        bboxes2 = np.expand_dims(bboxes2, 0)  # Shape: (1, M, 4)
        bboxes1 = np.expand_dims(bboxes1, 1)  # Shape: (N, 1, 4)

        # Calculate the intersection box
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h

        # Calculate IoU
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        iou = wh / (area1 + area2 - wh + epsilon)

        # Calculate center points
        centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
        centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
        centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
        centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

        # Calculate squared center distance
        inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

        # Calculate smallest enclosing box diagonal
        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2 + epsilon

        # Calculate aspect ratio consistency
        w1 = bboxes1[..., 2] - bboxes1[..., 0]
        h1 = bboxes1[..., 3] - bboxes1[..., 1]
        w2 = bboxes2[..., 2] - bboxes2[..., 0]
        h2 = bboxes2[..., 3] - bboxes2[..., 1]

        # Prevent division by zero
        h2 = h2 + epsilon
        h1 = h1 + epsilon
        arctan_diff = np.arctan(w2 / h2) - np.arctan(w1 / h1)
        v = (4 / (np.pi ** 2)) * (arctan_diff ** 2)

        # Calculate alpha
        S = 1 - iou
        alpha = v / (S + v + epsilon)

        # Compute CIoU
        ciou = iou - (inner_diag / outer_diag) + (alpha * v)

        # Scale CIoU to [0, 1]
        return (ciou + 1) / 2.0

    
    def diou_batch(bboxes1, bboxes2) -> np.ndarray:
        """
        :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
        :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        """
        # for details should go to https://arxiv.org/pdf/1902.09630.pdf
        # ensure predict's bbox form
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        # calculate the intersection box
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        iou = wh / (
            (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
            (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
            wh
        )

        centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
        centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
        centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
        centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

        inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

        outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
        diou = iou - inner_diag / outer_diag

        return (diou + 1) / 2.0 
    

    @staticmethod
    def run_asso_func(self, bboxes1, bboxes2):
        """
        Runs the selected association function (based on the initialization string) on the input bounding boxes.
        
        Parameters:
        bboxes1: First set of bounding boxes.
        bboxes2: Second set of bounding boxes.
        """
        return self.asso_func(bboxes1, bboxes2)

    def _get_asso_func(self, asso_mode):
        """
        Returns the corresponding association function based on the provided mode string.
        
        Parameters:
        asso_mode (str): The association function to use (e.g., "iou", "giou", "centroid", etc.).
        
        Returns:
        function: The appropriate function for the association calculation.
        """
        ASSO_FUNCS = {
            "iou": AssociationFunction.iou_batch,
            "iou_obb": AssociationFunction.iou_batch_obb,
            "hmiou": AssociationFunction.hmiou_batch,
            "giou": AssociationFunction.giou_batch,
            "ciou": AssociationFunction.ciou_batch,
            "diou": AssociationFunction.diou_batch,
            "centroid": self.centroid_batch,  # only not being staticmethod
            "centroid_obb": self.centroid_batch_obb
        }

        if self.asso_mode not in ASSO_FUNCS:
            raise ValueError(f"Invalid association mode: {self.asso_mode}. Choose from {list(ASSO_FUNCS.keys())}")

        return ASSO_FUNCS[self.asso_mode]
