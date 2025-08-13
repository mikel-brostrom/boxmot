#!/usr/bin/env python3
import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import sam2
from sam2.build_sam import build_sam2_video_predictor
from hydra import initialize_config_dir  
from hydra.core.global_hydra import GlobalHydra  

# if you’re on Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ----------------------------------------------------------------------
# Download necessary files (edgetam.yaml and edgetam.pt)
# ----------------------------------------------------------------------

def download_file(url: str, dest_path: Path):
    """Download a file from a URL to a specified destination."""
    if not dest_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded to {dest_path}")
    else:
        print(f"{dest_path} already exists, skipping download.")

# Paths to the files
yaml_url = "https://raw.githubusercontent.com/facebookresearch/EdgeTAM/main/sam2/configs/edgetam.yaml"
pt_url = "https://raw.githubusercontent.com/facebookresearch/EdgeTAM/main/checkpoints/edgetam.pt"

yaml_path = Path("/Users/mikel.brostrom/boxmot/configs/edgetam.yaml")
pt_path = Path("checkpoints/edgetam.pt")

# Create directories if they don't exist
yaml_path.parent.mkdir(parents=True, exist_ok=True)
pt_path.parent.mkdir(parents=True, exist_ok=True)

# Download the files
download_file(yaml_url, yaml_path)
download_file(pt_url, pt_path)

# ----------------------------------------------------------------------
# MOT17 CLASS ID MAPPINGS
#
# The 8th column in the MOT17 ground-truth files ("gt.txt") encodes
# the object’s class as follows:
#
#   1  — pedestrian
#   2  — person_on_vehicle
#   3  — car
#   4  — bicycle
#   5  — motorbike
#   6  — non_mot_vehicle
#   7  — static_person
#   8  — distractor
#   9  — occluder
#  10  — occluder_on_ground
#  11  — occluder_full
#  12  — reflection
#  13  — crowd
#
# MOT_DISTRACTOR_IDS:
#   These class IDs are treated as "distractors" during evaluation—
#   they can appear in the scene and (if matched) will not count
#   against your tracker’s accuracy.
#
# MOT_IGNORE_IDS:
#   This set includes all distractor IDs plus the "crowd" class.
#   Detections of any of these classes are completely filtered out
#   before computing standard metrics (MOTA, IDF1, etc.).
# ----------------------------------------------------------------------
MOT_IGNORE_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def overlay_mask(img: np.ndarray, mask: np.ndarray, color, alpha: float = 0.6) -> np.ndarray:
    """
    Blend a single-channel boolean mask onto a BGR image,
    but *only* inside the mask, leaving background untouched.
    """
    m = mask.squeeze() if mask.ndim > 2 else mask
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, but got shape {mask.shape}")
    m = m.astype(bool)

    color_arr = np.array(color, dtype=np.uint8)
    out = img.copy()
    out[m] = (
        img[m].astype(np.float32) * (1.0 - alpha)
        + color_arr * alpha
    ).astype(np.uint8)
    return out


def main():  
    # Clear any existing Hydra instance  
    GlobalHydra.instance().clear()  
      
    # Initialize Hydra with your config directory  
    config_dir = str(yaml_path.parent.absolute())  
    with initialize_config_dir(config_dir=config_dir, version_base=None):  
        # ── 1) build your predictor  
        checkpoint = pt_path  
        predictor = build_sam2_video_predictor(  
            "edgetam",  # Just the filename without extension  
            str(checkpoint),  
            add_all_frames_to_correct_as_cond=True,  
            device="cpu"  
        )  
  
        # ── 2) load frame list  
        video_dir = Path(  
            "/Users/mikel.brostrom/boxmot/boxmot/engine/trackeval/data/MOT17-ablation/train/MOT17-09/img1/"  
        )  
        frame_paths = sorted(  
            [p for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]],  
            key=lambda p: int(p.stem)  
        )  

    # ── 3) init inference state
    inference_state = predictor.init_state(video_path=str(video_dir))

    # ── 4) extract first-appearance annotations from gt.txt
    gt_path = video_dir.parent / "gt" / "gt.txt"
    first_appear = {}  # id → (frame_idx, [x0, y0, x1, y1], class)
    img_w, img_h = 1920, 1080
    with gt_path.open("r") as f:
        for line in f:
            vals = line.strip().split(",")
            if len(vals) < 8:
                continue
            frame, obj_id, cls = int(vals[0]), int(vals[1]), int(vals[7])
            x, y, w, h = map(float, vals[2:6])
            if obj_id not in first_appear:
                frame_idx = frame - 1  # 0-based index
                xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
                xyxy[0] = np.clip(xyxy[0], 0, img_w - 1)
                xyxy[2] = np.clip(xyxy[2], 0, img_w - 1)
                xyxy[1] = np.clip(xyxy[1], 0, img_h - 1)
                xyxy[3] = np.clip(xyxy[3], 0, img_h - 1)
                first_appear[obj_id] = (frame_idx, xyxy, cls)

    # Filter out distractors, ignored classes, and non-zero initial frames
    first_appear = {
        oid: (fidx, box, cls)
        for oid, (fidx, box, cls) in first_appear.items()
        if cls not in MOT_IGNORE_IDS and fidx == 0
    }

    ann_obj_ids = sorted(first_appear)
    ann_frame_idxs = [first_appear[oid][0] for oid in ann_obj_ids]
    boxes = [first_appear[oid][1] for oid in ann_obj_ids]

    # ── 5) register those first frames as your annotations
    for fidx, oid, box_coords in zip(ann_frame_idxs, ann_obj_ids, boxes):
        _, out_ids, out_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=fidx,
            obj_id=oid,
            box=box_coords,
        )

    # ── 6) propagate through video *and display each frame immediately*
    fps = 30
    delay = int(1000 / fps)
    win = "Segmentation @ Real-Time"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    output_dir = Path("./imgs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, out_ids, out_logits in predictor.propagate_in_video(inference_state):
        img_path = frame_paths[frame_idx]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for oid, logit in zip(out_ids, out_logits):
            mask = (logit > 0.0).cpu().numpy().astype(bool)
            rng = np.random.RandomState(oid)
            color = rng.randint(0, 255, size=3).tolist()
            img = overlay_mask(img, mask, color, alpha=0.6)

            ys, xs = np.where(mask.squeeze() if mask.ndim > 2 else mask)
            if ys.size and xs.size:
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

                label = str(oid)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                y_label_top = y0 - th - baseline
                y_label_bottom = y0
                if y_label_top < 0:
                    y_label_top = y0
                    y_label_bottom = y0 + th + baseline
                cv2.rectangle(
                    img,
                    (x0, y_label_top),
                    (x0 + tw, y_label_bottom),
                    color,
                    cv2.FILLED,
                )
                text_org = (x0, y_label_bottom - baseline)
                cv2.putText(
                    img,
                    label,
                    text_org,
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        cv2.imshow(win, img)
        cv2.imwrite(str(output_dir / img_path.name), img)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
