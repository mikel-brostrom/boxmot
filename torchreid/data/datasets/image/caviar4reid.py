from __future__ import division, print_function, absolute_import

import sys
import os
import os.path as osp

from ..dataset import ImageDataset
# from torchreid.data import ImageDataset


class CAVIAR4REID(ImageDataset):
    dataset_dir = 'caviar4reid'
    dataset_url = None

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
        """CAVIAR4REID

        Reference:
            Custom pictorial structures for re-identification
            D. S. Cheng, M. Cristani, M. Stoppa, L. Bazzani, V. Murino
            In British Machine Vision Conference (BMVC), 2011
        
        URL: `<https://lorisbaz.github.io/caviar4reid.html>`

        The zip contains the dataset as a set of images. For each person we have a set of 5 or 10 images. The filename identifies which images are associated to each person:
        XXXXYYY.jpg
        XXXX = identifier of the person
        YYY = identifier of the image for that specific person
        E.g.: 0003005.jpg is the 5th image of the 3rd person
        """
        train = ...
        query = ...
        gallery = ...

        super(CAVIAR4REID, self).__init__(train, query, gallery, **kwargs)