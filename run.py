import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

from boxmot.trackers.faststrongsort.fast_strong_sort import FastStrongSORT

def get_seq_paths(dataset_path):
    imgs = {}
    seq_names = []
    for root, dirs, files in os.walk(dataset_path):
        for dire in dirs:
            if dire.startswith("MOT"):
                seq_names.append(dire)
                imgs[dire] = [os.path.join(r, file) for r, d, f in os.walk(os.path.join(root, dire)) for file in f if file.endswith(".jpg")]
                imgs[dire].sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return imgs, seq_names

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reid', type=str, default="osnet_x1_0_dukemtmcreid.pt", help='model.pt path')
    parser.add_argument('--dataset-path', type=str, default='./data/MOT17/train', help='dataset path')
    parser.add_argument('--dataset', type=str, default='train', help='dataset type')
    parser.add_argument('--device', type=str, default='cpu', help='device \'cpu\' or \'0\', \'1\', ... for gpu')
    parser.add_argument('--ot', type=float, default=0.2)
    return parser.parse_args()

def create_tracker(args):
    return FastStrongSORT(
        model_weights=Path(args.reid),
        device=args.device,
        fp16=False,
        iou_threshold=args.ot
    )

def process_detection(row):
    tlwh = row[2:6]
    return np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3], row[6], 0])

def write_result(f, frame_no, det):
    f.write(f"{frame_no + 1},{int(det[4])},{int(det[0])},{int(det[1])},{int(det[2] - det[0])},{int(det[3] - det[1])},{det[5]:.2f},-1,-1,-1\n")

if __name__ == "__main__":
    args = parse_options()
    dataset_path = "/".join(args.dataset_path.split("/")[:-1] + [args.dataset])
    imgs, seq_names = get_seq_paths(dataset_path)

    total_time = 0
    total_dets = 0
    total_frames = 0

    for seq in seq_names:
        tracker = create_tracker(args)
        print(f"Sequence: {seq}")
        seq_imgs = imgs[seq]
        output_dir = Path(f"output/{str(args.ot)}/{seq}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir.parent / f"{seq}.txt", "w") as f:
            print(f"Writing results to: {f.name}")
            seq_time = 0

            det_file = Path(f"MOT17_{args.dataset}_YOLOX+BoT/{seq}.npy")
            seq_det = np.load(det_file, allow_pickle=True)

            for frame_no, img_path in enumerate(seq_imgs):
                frame = cv2.imread(img_path)
                frame_dets = seq_det[seq_det[:, 0] == frame_no + 1]

                if len(frame_dets) < 1:
                    continue

                total_dets += len(frame_dets)
                total_frames += 1

                processed_dets = np.array([process_detection(row) for row in frame_dets])
                features = [row[10:] for row in frame_dets]

                if len(processed_dets) > 0:
                    start = datetime.now()
                    tracked_det = tracker.update(processed_dets, frame)
                    # tracked_det = tracker.update(processed_dets, frame)
                    seq_time += (datetime.now() - start).total_seconds()

                    for det in tracked_det:
                        write_result(f, frame_no, det)
                        scale = frame.shape[0] / 1080
                        cv2.putText(frame, f"id: {int(det[4])}", (int(det[0]), int(det[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)

                cv2.imwrite(str(output_dir / f"{frame_no + 1}.jpg"), frame)

        print(f"{seq} time: {seq_time:.2f}s")
        total_time += seq_time

    print(f"Total time: {total_time:.2f}s")
    print(f"FPS: {total_frames / total_time:.2f}")
    print(f"Total frames: {total_frames}")