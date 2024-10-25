import os
import cv2
import glob
import math
import numpy as np
from pathlib import Path
from PIL import Image


VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes


class LoadImagesAndVideos:
    """
    A data loader for handling both images and videos, providing batches of frames or images for processing.
    Supports various image formats, including HEIC, and handles text files with paths to images/videos.
    """

    def __init__(self, path, batch_size=1, vid_stride=1):
        self.batch_size = batch_size
        self.vid_stride = vid_stride
        self.files = self._load_files(path)
        self.video_flag = [self._is_video(f) for f in self.files]
        self.nf = len(self.files)
        self.ni = sum(not is_video for is_video in self.video_flag)
        self.mode = "image"
        
        self.cap = None
        if any(self.video_flag):
            self._start_video(self.files[self.video_flag.index(True)])
        
        if not self.files:
            raise FileNotFoundError(f"No images or videos found in {path}.")

    def _load_files(self, path):
        """Load files from a given path, which may be a directory, list, or text file."""
        if isinstance(path, str) and Path(path).suffix == ".txt":
            path = Path(path).read_text().splitlines()
        
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).absolute())
            if "*" in p:
                files.extend(glob.glob(p, recursive=True))
            elif os.path.isdir(p):
                files.extend(glob.glob(os.path.join(p, "*.*")))
            elif os.path.isfile(p):
                files.append(p)
            else:
                raise FileNotFoundError(f"{p} does not exist")
        return files

    def _is_video(self, file_path):
        """Check if a file is a video based on its extension."""
        return file_path.split('.')[-1].lower() in VID_FORMATS

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        paths, imgs, infos = [], [], []
        while len(imgs) < self.batch_size:
            if self.count >= self.nf:
                if imgs:
                    return paths, imgs, infos
                else:
                    raise StopIteration
            
            path = self.files[self.count]
            if self.video_flag[self.count]:
                self._process_video(paths, imgs, infos, path)
            else:
                self._process_image(paths, imgs, infos, path)
            self.count += 1

        return paths, imgs, infos

    def _process_image(self, paths, imgs, infos, path):
        """Process an image file and append it to the batch."""
        img = self._read_image(path)
        if img is not None:
            paths.append(path)
            imgs.append(img)
            infos.append(f"image {self.count + 1}/{self.nf} {path}")

    def _process_video(self, paths, imgs, infos, path):
        """Process a video file, reading frames as per the stride."""
        self.mode = "video"
        if not self.cap or not self.cap.isOpened():
            self._start_video(path)
        
        success = False
        for _ in range(self.vid_stride):
            success = self.cap.grab()
            if not success:
                break

        if success:
            _, frame = self.cap.retrieve()
            paths.append(path)
            imgs.append(frame)
            infos.append(f"video {self.count + 1}/{self.nf} frame {self.frame}/{self.frames} {path}")
            self.frame += 1
            if self.frame >= self.frames:
                self.cap.release()

    def _read_image(self, path):
        """Read an image from a file, handling HEIC format if necessary."""
        if path.lower().endswith("heic"):
            from pillow_heif import register_heif_opener
            register_heif_opener()
            with Image.open(path) as img:
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            return cv2.imread(path)

    def _start_video(self, path):
        """Initialize video capture for a new video file."""
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.frame = 0

    def __len__(self):
        return math.ceil(self.nf / self.batch_size)
