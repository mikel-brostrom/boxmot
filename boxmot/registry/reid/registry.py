from boxmot.registry.reid.config import REID_BACKBONES, REID_URLS, REID_CLASSES_NUM
from boxmot.registry.registry import ModelRegistry


class ReIDModelRegistry(ModelRegistry):
    BACKBONES = REID_BACKBONES
    URLS = REID_URLS
    CLASSES_NUM = REID_CLASSES_NUM

    @classmethod
    def get_classes_num(cls, weights):
        # Extract dataset name from weights name,
        # then look up in the class dictionary
        dataset_key = weights.name.split("_")[1]
        return cls.CLASSES_NUM.get(dataset_key, 1)

    @classmethod
    def build(cls,
                    name: str,
                    num_classes: int,
                    loss: str = "softmax",
                    pretrained: bool = True,
                    use_gpu: bool = True):
        """
        Build a model instance.

        Args:
            name (str): Name of the model.
            num_classes (int): Number of classes in the dataset.
            loss (str, optional): Loss function to use. Defaults to "softmax".
            pretrained (bool, optional): Whether to load pretrained weights.
                                         Defaults to True.
            use_gpu (bool, optional): Whether to use GPU for inference.
                                      Defaults to True.
        """

        if name not in cls.BACKBONES:
            available = list(cls.BACKBONES.keys())
            raise KeyError(
                f"Unknown model '{name}'. Must be one of {available}")

        # Special case handling for clip model
        if "clip" in name:
            from boxmot.appearance.backbones.clip.config.defaults import _C as cfg

            return cls.BACKBONES[name](
                cfg, num_class=num_classes, camera_num=2, view_num=1
            )

        return cls.BACKBONES[name](
            num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu
        )
