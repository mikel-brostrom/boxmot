from __future__ import division, print_function, absolute_import

import sys
import os
import os.path as osp

from ..dataset import ImageDataset
# from torchreid.data import ImageDataset


class PRW(ImageDataset):
    """PRW.

    Reference:
        title={Person Re-identification in the Wild},
        author={Zheng, Liang and Zhang, Hengheng and Sun, Shaoyan and Chandraker, Manmohan and Tian, Qi},
        journal={arXiv preprint arXiv:1604.02531},
            
    URL: `<http://zheng-lab.cecs.anu.edu.au/Project/project_prw.html>`

    The package contains three folders.
    1) "frames". There are 11,816 frames in this folder.
    2) "annotations". For each frame, we provide its annotated data.
    All annotated boxes are pedestrians.
    Each MAT file records the bounding box position within the frame and its ID.
    The coordinates of each box are formatted in [x, y, w, h].
    The ID of each box takes the value of [1, 932] as well as -2.
    "-2" means that we do not know for sure the ID of the person,
    and is not used in the testing of person re-id,
    but is used in train/test of pedestrian detection (potentially used in the training of person re-identification).

    3) "query_box". It contains the query boxes of the PRW dataset.
    All togther there are 2057 queries. For naming rule,
    for example, in "479_c1s3_016471.jpg", "479" refers to the ID of the query,
    and "c1s3_016471" refers to the video frame where the query is cropped
    Note that 1) the query IDs are not included in the training set,
    2) the query images are not normalized (we typically use 128*64 for BoW extraction,
    and 224*224 for CNN feature extraction),
    3)all queries are hand-drawn boxes,
    4) we select one query image for each testing ID under each camera,
    so the maximum number of queries per ID is 6.
    In addition, we provide the bounding box information of each query in "query_info.txt",
    so one can generate the queries from the video frames through function "generate_query.m".

    In addition, we provide the train/test split of the PRW dataset.
    One do not have to perform 10-fold cross validation. In detail,
    "frame_test.mat" and "frame_train.mat" specify the train/test frames,
    and "ID_test.mat" and "ID_train.mat" specify the train/test IDs.
    Note that a small portion of IDs used in training may appear in the testing frames,
    but will not appear in the testing IDs.
    """
    
    dataset_dir = 'prw'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = osp.join(self.dataset_dir, "PRW-v16.04.20")
        print(self.data_dir)

        train = ...
        query = ...
        gallery = ...
        
        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).

        super(PRW, self).__init__(train, query, gallery, **kwargs)
    
    def process_dir(self, dirname):
        pass

import os.path as osp
import re

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset

"""
class PRW(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "PRW"
        self.img_prefix = osp.join(root, "frames")
        super(PRW, self).__init__(root, transforms, split)

    def _get_cam_id(self, img_name):
        match = re.search(r"c\d", img_name).group().replace("c", "")
        return int(match)

    def _load_queries(self):
        query_info = osp.join(self.root, "query_info.txt")
        with open(query_info, "rb") as f:
            raw = f.readlines()

        queries = []
        for line in raw:
            linelist = str(line, "utf-8").split(" ")
            pid = int(linelist[0])
            x, y, w, h = (
                float(linelist[1]),
                float(linelist[2]),
                float(linelist[3]),
                float(linelist[4]),
            )
            roi = np.array([x, y, x + w, y + h]).astype(np.int32)
            roi = np.clip(roi, 0, None)  # several coordinates are negative
            img_name = linelist[5][:-2] + ".jpg"
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([pid]),
                    "cam_id": self._get_cam_id(img_name),
                }
            )
        return queries

    def _load_split_img_names(self):
        """
        # Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        if self.split == "train":
            imgs = loadmat(osp.join(self.root, "frame_train.mat"))["img_index_train"]
        else:
            imgs = loadmat(osp.join(self.root, "frame_test.mat"))["img_index_test"]
        return [img[0][0] + ".jpg" for img in imgs]

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        annotations = []
        imgs = self._load_split_img_names()
        for img_name in imgs:
            anno_path = osp.join(self.root, "annotations", img_name)
            anno = loadmat(anno_path)
            box_key = "box_new"
            if box_key not in anno.keys():
                box_key = "anno_file"
            if box_key not in anno.keys():
                box_key = "anno_previous"

            rois = anno[box_key][:, 1:]
            ids = anno[box_key][:, 0]
            rois = np.clip(rois, 0, None)  # several coordinates are negative

            assert len(rois) == len(ids)

            rois[:, 2:] += rois[:, :2]
            ids[ids == -2] = 5555  # assign pid = 5555 for unlabeled people
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": rois.astype(np.int32),
                    # FIXME: (training pids) 1, 2,..., 478, 480, 481, 482, 483, 932, 5555
                    "pids": ids.astype(np.int32),
                    "cam_id": self._get_cam_id(img_name),
                }
            )
        return annotations
"""