from typing import List
import time

import torch

from .OSNet import osnet_ibn_x1_0
from .utils import load_pretrained_weights, compress_feature_vector
from .image_handler import normalize, resize, ndarray_to_tensor

import numpy as np


class OsNetEncoder:

    # Encoder constants
    PRETRAINED_MODEL = False
    LOSS = 'softmax'

    def __init__(self, input_width: int, input_height: int, weight_filepath: str, batch_size: str, num_classes: int, patch_height: int, patch_width: int, norm_mean: List[float], norm_std: List[float], GPU: bool):

        self._input_width = input_width
        self._input_height = input_height
        self._weight_filepath = weight_filepath
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.patch_size = (patch_height, patch_width)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.GPU = GPU
        self._model = osnet_ibn_x1_0(
            num_classes=self.num_classes,
            loss=OsNetEncoder.LOSS,
            pretrained=OsNetEncoder.PRETRAINED_MODEL,
            use_gpu=self.GPU
        )
        self._model.eval()  # Set the torch model for evaluation
        self.weights_loaded = load_pretrained_weights(
            model=self._model,
            weight_path=self._weight_filepath
        )
        if self.GPU:
            self._model = self._model.cuda()

    def load_image(self, patch: np.ndarray):
        ''' load image involves three processes: resizing, normalising and translating
        the np.ndarray into a torch.Tensor ready for GPU.

        :param patch: single detection patch, in np.ndarray format
        :return: resized and normalised single detection tensor
        '''

        if self.GPU:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        patch = patch.cpu().numpy()

        print('\n\n\n\n', patch.size())

        resized_patch = resize(
            img=patch,
            size=self.patch_size
        )
        torch_tensor = ndarray_to_tensor(pic=resized_patch)

        normalized_tensor = normalize(
            tensor=torch_tensor,
            mean=self.norm_mean,
            std=self.norm_std,
            inplace=False
        )

        # Transforms the normalised tensor to a cuda tensor or a cpu tensor wrt which device is available
        gpu_tensor = normalized_tensor.to(device)

        return gpu_tensor

    def get_features(self, image_patches: List[np.ndarray]):

        ''' Extract the 512 features associated to each detection
        :param image_patches: List[np.ndarray] of detections
        :return features: List[np.ndarray] of features associated to each detection
        '''
        return self._model(image_patches)
