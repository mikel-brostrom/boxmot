from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

from torchreid.data.sampler import build_train_sampler
from torchreid.data.transforms import build_transforms
from torchreid.data.datasets import init_image_dataset, init_video_dataset


class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, sources=None, targets=None, height=256, width=128, transforms='random_flip',
                norm_mean=None, norm_std=None, use_gpu=False):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width

        if self.sources is None:
            raise ValueError('sources must not be None')

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if self.targets is None:
            self.targets = self.sources

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.transform_tr, self.transform_te = build_transforms(
            self.height, self.width, transforms=transforms,
            norm_mean=norm_mean, norm_std=norm_std
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    def return_dataloaders(self):
        """Returns trainloader and testloader."""
        return self.trainloader, self.testloader

    def return_testdataset_by_name(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        return self.testdataset[name]['query'], self.testdataset[name]['gallery']


class ImageDataManager(DataManager):
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        train_sampler (str, optional): sampler. Default is empty (``RandomSampler``).
        cuhk03_labeled (bool, optional): use cuhk03 labeled images.
            Default is False (defaul is to use detected images).
        cuhk03_classic_split (bool, optional): use the classic split in cuhk03.
            Default is False.
        market1501_500k (bool, optional): add 500K distractors to the gallery
            set in market1501. Default is False.

    Examples::

        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100
        )
    """
    data_type = 'image'
    def __init__(self, root='', sources=None, targets=None, height=256, width=128, transforms='random_flip', 
                 norm_mean=None, norm_std=None, use_gpu=True, split_id=0, combineall=False,
                 batch_size_train=32, batch_size_test=32, workers=4, num_instances=4, train_sampler='',
                 cuhk03_labeled=False, cuhk03_classic_split=False, market1501_500k=False):
        super(ImageDataManager, self).__init__(sources=sources, targets=targets, height=height, width=width,
                                               transforms=transforms, norm_mean=norm_mean, norm_std=norm_std,
                                               use_gpu=use_gpu)
        
        print('=> Loading train (source) dataset')
        trainset = []  
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train, train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.testloader = {name: {'query': None, 'gallery': None} for name in self.targets}
        self.testdataset = {name: {'query': None, 'gallery': None} for name in self.targets}

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.testloader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.testloader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset[name]['query'] = queryset.query
            self.testdataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train            : {}'.format(self.sources))
        print('  # train datasets : {}'.format(len(self.sources)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(trainset)))
        print('  # train cameras  : {}'.format(self.num_train_cams))
        print('  test             : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')


class VideoDataManager(DataManager):
    r"""Video data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        batch_size_train (int, optional): number of tracklets in a training batch. Default is 3.
        batch_size_test (int, optional): number of tracklets in a test batch. Default is 3.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        train_sampler (str, optional): sampler. Default is empty (``RandomSampler``).
        seq_len (int, optional): how many images to sample in a tracklet. Default is 15.
        sample_method (str, optional): how to sample images in a tracklet. Default is "evenly".
            Choices are ["evenly", "random", "all"]. "evenly" and "random" will sample ``seq_len``
            images in a tracklet while "all" samples all images in a tracklet, where the batch size
            needs to be set to 1.

    Examples::

        datamanager = torchreid.data.VideoDataManager(
            root='path/to/reid-data',
            sources='mars',
            height=256,
            width=128,
            batch_size_train=3,
            batch_size_test=3,
            seq_len=15,
            sample_method='evenly'
        )

    .. note::
        The current implementation only supports image-like training. Therefore, each image in a
        sampled tracklet will undergo independent transformation functions. To achieve tracklet-aware
        training, you need to modify the transformation functions for video reid such that each function
        applies the same operation to all images in a tracklet to keep consistency.
    """
    data_type = 'video'

    def __init__(self, root='', sources=None, targets=None, height=256, width=128, transforms='random_flip',
                 norm_mean=None, norm_std=None, use_gpu=True, split_id=0, combineall=False,
                 batch_size_train=3, batch_size_test=3, workers=4, num_instances=4, train_sampler=None,
                 seq_len=15, sample_method='evenly'):
        
        super(VideoDataManager, self).__init__(sources=sources, targets=targets, height=height, width=width,
                                               transforms=transforms, norm_mean=norm_mean, norm_std=norm_std,
                                               use_gpu=use_gpu)

        print('=> Loading train (source) dataset')
        trainset = []  
        for name in self.sources:
            trainset_ = init_video_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train, train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.testloader = {name: {'query': None, 'gallery': None} for name in self.targets}
        self.testdataset = {name: {'query': None, 'gallery': None} for name in self.targets}

        for name in self.targets:
            # build query loader
            queryset = init_video_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            self.testloader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_video_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            self.testloader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset[name]['query'] = queryset.query
            self.testdataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train             : {}'.format(self.sources))
        print('  # train datasets  : {}'.format(len(self.sources)))
        print('  # train ids       : {}'.format(self.num_train_pids))
        print('  # train tracklets : {}'.format(len(trainset)))
        print('  # train cameras   : {}'.format(self.num_train_cams))
        print('  test              : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')
