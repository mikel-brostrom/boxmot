from io import BytesIO
import numpy as np
import torch
from ultralytics.yolo.utils import ops
import json
import datetime
from PIL import Image
import base64
import cv2
from ultralytics.yolo.utils.plotting import save_one_box

from pathlib import Path

def write_MOT_results_json(txt_path, results, frame_idx, img, save_dir, img_path):
    # Ensure img_path is a string
    if not isinstance(img_path, str):
        img_path = str(img_path)

    # Create a Path object from the string img_path
    img_path = Path(img_path)

    # Check the dimension of boxes tensor
    if len(results.boxes.xyxy.shape) == 1:
        results.boxes.xyxy.unsqueeze_(0)
        results.boxes.id.unsqueeze_(0)

    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)

    # create parent folder
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    # create json file
    json_file = txt_path.with_suffix(".json")

    # Load existing data if it exists
    if json_file.exists():
        with open(json_file, 'r') as f:
            mot_results = json.load(f)
    else:
        mot_results = []

    for j in range(nr_dets):
        bbox_tensor = results.boxes.xyxy[j]
        if len(bbox_tensor.shape) == 1:   # If bbox_tensor is 1D, make it 2D
            bbox_tensor = bbox_tensor.unsqueeze(0)

        # Save the image crop
        save_one_box(bbox_tensor, img,
                     file=(save_dir / 'crops' / img_path.stem / f"{frame_idx[0][0]}_{results.boxes.id[j]}"),
                     BGR=True)
        
        # Get the path to the saved crop
        crop_path = str(save_dir / 'crops' / img_path.stem / f"{frame_idx[0][0]}_{results.boxes.id[j]}.jpg")
        
        mot = {
            "id": results.boxes.id[j].item(),
            "bbox": ops.xyxy2ltwh(bbox_tensor).squeeze().tolist(),  # Remove the extra dimension after processing
            "timestamp": datetime.datetime.now().isoformat(),
            "crop_path": crop_path,
        }
        mot_results.append(mot)

    # write to json file
    with open(json_file, "w") as f:
        json.dump(mot_results, f, indent=4)








def write_MOT_results(txt_path, results, frame_idx, i):
    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 1), -1)
    i = torch.full((nr_dets, 1), i)
    print("Confidence scores data type: ", results.boxes.conf.dtype)
    mot = torch.cat(
        [
            frame_idx,
            results.boxes.id.unsqueeze(1).to("cpu"),
            ops.xyxy2ltwh(results.boxes.xyxy).to("cpu"),
            results.boxes.conf.unsqueeze(1).to("cpu"),
            results.boxes.cls.unsqueeze(1).to("cpu"),
            dont_care,
        ],
        dim=1,
    )

    print("Confidence scores in write_MOT_results: ", results.boxes.conf)
    print("Data to be saved: ", mot)

    # create parent folder
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    # create mot txt file
    txt_path.with_suffix(".txt").touch(exist_ok=True)

    with open(str(txt_path) + ".txt", "ab+") as f:  # append binary mode
        np.savetxt(
            f, mot.numpy().astype(np.float64), fmt="%.2f"
        )  # save as ints instead of scientific notation
