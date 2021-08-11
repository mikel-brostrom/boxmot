from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

from torchreid.data import ImageDataset


class PKU_REID(ImageDataset):
    dataset_dir = 'pku-reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

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
        """PKU_REID.

        Reference:
            title={Orientation driven bag of appearances for person re-identification},
            author={Ma, Liqian and Liu, Hong and Hu, Liang and Wang, Can and Sun, Qianru},
            journal={arXiv preprint arXiv:1605.02464},

        URL: `<https://github.com/charliememory/PKU-Reid-Dataset>`

        PKU-Reid dataset: This dataset contains 114 individuals including 1824 images
        captured from two disjoint camera views. For each person,
        eight images are captured from eight different orientations
        under one camera view and are normalized to 128x48 pixels.
        This dataset is also split into two parts randomly.
        One contains 57 individuals for training, and the other contains 57 individuals for testing.
        To the best of our knowledge, PKU-Reid dataset is the first one that captures person appearance from all eight orientations.
        The image name is in the form "personId_cameraId_orientationId.png".
        
        """
        train = ...
        query = ...
        gallery = ...

        super(PKU_REID, self).__init__(train, query, gallery, **kwargs)