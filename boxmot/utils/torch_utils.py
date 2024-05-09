# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import platform
import torch

from .. import __version__
from . import logger as LOGGER
from boxmot.utils import ROOT


def get_system_info():
    return f"Yolo Tracking v{__version__} 🚀 Python-{platform.python_version()} torch-{torch.__version__}"

def parse_device(device):
    device = str(device).lower().replace("cuda:", "").replace("none", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    return device

def assert_cuda_available(device):
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(",", ""))):
        install = ("See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no CUDA devices are seen by torch.\n" if torch.cuda.device_count() == 0 else "")
        raise ValueError(f"Invalid CUDA 'device={device}' requested. Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n" +
                         f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}" +
                         f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}" +
                         f"\nos.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}\n{install}")

def select_device(device="", batch=0):
    s = get_system_info()
    device = parse_device(device)
    mps = device == "mps"
    cpu = device == "cpu" or device == "" and not torch.cuda.is_available()

    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert_cuda_available(device)

    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(",") if device else ["0"]
        n = len(devices)
        if n > 1 and batch > 0 and batch % n != 0:
            raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}.")
        s += "\n" + "\n".join(f"CUDA:{d} ({torch.cuda.get_device_properties(i).name}, {torch.cuda.get_device_properties(i).total_memory / (1 << 20):.0f}MiB)" for i, d in enumerate(devices))
        arg = "cuda:" + devices[0]
    elif mps:
        s += "MPS"
        arg = "mps"
    else:
        s += "CPU"
        arg = "cpu"
    LOGGER.info(s)
    return torch.device(arg)
