from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import numpy as np
import tarfile
import zipfile
import copy

import torch

from torchreid.utils import read_image, mkdir_if_missing, download_url


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = [] # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for img_path, pid, camid in other.train:
            pid += self.num_train_pids
            camid += self.num_train_cams
            train.append((img_path, pid, camid))

        ###################################
        # Things to do beforehand:
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset, setting it to True will
        #    create new IDs that should have been included
        ###################################
        if isinstance(train[0][0], str):
            return ImageDataset(
                train, self.query, self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False
            )
        else:
            return VideoDataset(
                train, self.query, self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False
            )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for _, pid, _ in self.gallery:
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError('{} dataset needs to be manually '
                               'prepared, please follow the '
                               'document to prepare this dataset'.format(self.__class__.__name__))

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print('Downloading {} dataset to "{}"'.format(self.__class__.__name__, dataset_dir))
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        extension = osp.basename(fpath).split('.')[-1]
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
              num_train_pids, len(self.train), num_train_cams,
              num_query_pids, len(self.query), num_query_cams,
              num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, img_path

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, len(self.train), num_train_cams))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, len(self.query), num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, len(self.gallery), num_gallery_cams))
        print('  ----------------------------------------')


class VideoDataset(Dataset):
    """A base class representing VideoDataset.

    All other video datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``imgs``, ``pid`` and ``camid``
    where ``imgs`` has shape (seq_len, channel, height, width). As a result,
    data in each batch has shape (batch_size, seq_len, channel, height, width).
    """

    def __init__(self, train, query, gallery, seq_len=15, sample_method='evenly', **kwargs):
        super(VideoDataset, self).__init__(train, query, gallery, **kwargs)
        self.seq_len = seq_len
        self.sample_method = sample_method

        if self.transform is None:
            raise RuntimeError('transform must not be None')

    def __getitem__(self, index):
        img_paths, pid, camid = self.data[index]
        num_imgs = len(img_paths)

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs>=self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= self.seq_len:
                num_imgs -= num_imgs % self.seq_len
                indices = np.arange(0, num_imgs, num_imgs/self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = self.seq_len - num_imgs
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num_imgs-1)])
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError('Unknown sample method: {}'.format(self.sample_method))

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0) # img must be torch.Tensor
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  -------------------------------------------')
        print('  subset   | # ids | # tracklets | # cameras')
        print('  -------------------------------------------')
        print('  train    | {:5d} | {:11d} | {:9d}'.format(num_train_pids, len(self.train), num_train_cams))
        print('  query    | {:5d} | {:11d} | {:9d}'.format(num_query_pids, len(self.query), num_query_cams))
        print('  gallery  | {:5d} | {:11d} | {:9d}'.format(num_gallery_pids, len(self.gallery), num_gallery_cams))
        print('  -------------------------------------------')