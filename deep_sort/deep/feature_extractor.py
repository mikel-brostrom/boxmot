import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

import sys
# so that init does not execute in the package
sys.path.append('deep_sort/deep/reid')
from torchreid import models


class Extractor(object):
    def __init__(self, model_type, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.input_width = 128
        self.input_height = 256

        self.model = models.build_model(name=model_type, num_classes=1000)
        self.model.to(self.device)
        self.model.eval()

        logger = logging.getLogger("root.tracker")
        logger.info("Selected model type: {}".format(model_type))
        self.size = (self.input_width, self.input_height)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("osnet_x1_0")
    feature = extr(img)
    print(feature.shape)
