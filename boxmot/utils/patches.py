# ultralytics_patches.py
from pathlib import Path
from typing import Union
import torch

from ultralytics.utils import plotting
from ultralytics.engine.results import Results

def my_save_txt(self, txt_file: Union[str, Path], save_conf: bool = False) -> str:
    """
    Save only “tracked” detections (where d.is_track) to a YOLO-format .txt file.
    """
    is_obb = self.obb is not None
    boxes = self.obb if is_obb else self.boxes
    masks = self.masks
    probs = self.probs
    kpts = self.keypoints
    texts = []

    if probs is not None:
        # classification
        [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
    elif boxes:
        # detection / segmentation / pose
        for j, d in enumerate(boxes):
            if not d.is_track:
                continue
            c, conf, obj_id = int(d.cls), float(d.conf), int(d.id.item())
            coords = (d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1))
            line = (c, *coords)
            if masks:
                seg = masks[j].xyn[0].copy().reshape(-1)
                line = (c, *seg)
            if kpts is not None:
                kpt = (torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2)
                       if kpts[j].has_visible else kpts[j].xyn)
                line += tuple(kpt.reshape(-1).tolist())
            # optionally append confidence and ID
            line += (conf,) * save_conf + (() if obj_id is None else (obj_id,))
            texts.append(("%g " * len(line)).rstrip() % line)

    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
        with open(txt_file, "a", encoding="utf-8") as f:
            f.writelines(t + "\n" for t in texts)

    return str(txt_file)

def apply_patches():
    # 1) Disable all plotting calls
    plotting.Annotator.box        = lambda *args, **kwargs: None
    plotting.Annotator.box_label  = lambda *args, **kwargs: None
    plotting.Annotator.line       = lambda *args, **kwargs: None

    # 2) Hook in our custom save_txt
    Results.save_txt = my_save_txt