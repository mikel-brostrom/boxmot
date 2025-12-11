# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class YoloInterface(ABC):

    @abstractmethod
    def __call__(self, im):
        pass

    @abstractmethod
    def preprocess(self, ims):
        pass

    @abstractmethod
    def postprocess(self, preds):
        pass

    def get_model_from_weigths(self, l, model):
        model_type = None
        for key in l:
            if Path(key).stem in str(model.name):
                model_type = str(Path(key).with_suffix(""))
                break
        return model_type
