from boxmot.reid.backbones.clip.make_model import make_model
from boxmot.reid.backbones.hacnn import HACNN
from boxmot.reid.backbones.lmbn.lmbn_n import LMBN_n
from boxmot.reid.backbones.mlfn import mlfn
from boxmot.reid.backbones.mobilenetv2 import (mobilenetv2_x1_0,
                                               mobilenetv2_x1_4)
from boxmot.reid.backbones.osnet import (osnet_ibn_x1_0, osnet_x0_5,
                                         osnet_x0_25, osnet_x0_75, osnet_x1_0)
from boxmot.reid.backbones.osnet_ain import (osnet_ain_x0_5, osnet_ain_x0_25,
                                             osnet_ain_x0_75, osnet_ain_x1_0)
from boxmot.reid.backbones.resnet import resnet50, resnet101

# Map model names to their respective constructors
MODEL_FACTORY = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "mobilenetv2_x1_0": mobilenetv2_x1_0,
    "mobilenetv2_x1_4": mobilenetv2_x1_4,
    "hacnn": HACNN,
    "mlfn": mlfn,
    "osnet_x1_0": osnet_x1_0,
    "osnet_x0_75": osnet_x0_75,
    "osnet_x0_5": osnet_x0_5,
    "osnet_x0_25": osnet_x0_25,
    "osnet_ibn_x1_0": osnet_ibn_x1_0,
    "osnet_ain_x1_0": osnet_ain_x1_0,
    "osnet_ain_x0_75": osnet_ain_x0_75,
    "osnet_ain_x0_5": osnet_ain_x0_5,
    "osnet_ain_x0_25": osnet_ain_x0_25,
    "lmbn_n": LMBN_n,
    "clip": make_model,
}
