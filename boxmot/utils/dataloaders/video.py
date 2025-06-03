import os
import cv2
import glob
import csv
import numpy as np
from pathlib import Path
from typing import Union, Generator, List


class LazyDataLoader:
    def __init__(self, source: Union[str, int, Path]):
        self.source = str(source)
        self.generator = self._get_generator()

    def __iter__(self):
        return self.generator

    def _get_generator(self) -> Generator:
        source = self.source

        if source.endswith('.csv'):
            return self._load_from_csv(source)
        elif source.endswith('.streams'):
            return self._load_from_stream_list(source)
        elif source.startswith(('http://', 'https://')):
            if 'youtube.com' in source or 'youtu.be' in source:
                return self._load_from_youtube(source)
            else:
                return self._load_from_stream(source)
        elif source.startswith(('rtsp://', 'rtmp://', 'tcp://')):
            return self._load_from_stream(source)
        elif '*' in source:
            return self._load_from_glob(source)
        elif os.path.isdir(source):
            return self._load_from_directory(source)
        elif os.path.isfile(source):
            return self._load_from_video(source)
        elif source.isdigit():
            return self._load_from_webcam(int(source))
        elif isinstance(source, int):
            return self._load_from_webcam(source)
        else:
            raise ValueError(f"Unsupported source type: {source}")

    def _load_from_csv(self, csv_path: str) -> Generator:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    yield from iter(LazyDataLoader(row[0]))

    def _load_from_video(self, video_path: str) -> Generator:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video file: {video_path}")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

    def _load_from_directory(self, path: str) -> Generator:
        for f in sorted(Path(path).glob('*')):
            if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.mp4', '.avi']:
                yield from iter(LazyDataLoader(str(f)))

    def _load_from_glob(self, pattern: str) -> Generator:
        for f in sorted(glob.glob(pattern)):
            yield from iter(LazyDataLoader(f))

    def _load_from_youtube(self, url: str) -> Generator:
        try:
            import yt_dlp
        except ImportError as e:
            raise ImportError("yt_dlp is required for YouTube video support. Install it with `pip install yt_dlp`.") from e

        with yt_dlp.YoutubeDL({'quiet': True, 'format': 'best'}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']
            yield from self._load_from_stream(video_url)

    def _load_from_stream(self, url: str) -> Generator:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open stream: {url}")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

    def _load_from_stream_list(self, file_path: str) -> Generator[List[np.ndarray], None, None]:
        with open(file_path) as f:
            urls = [line.strip() for line in f if line.strip()]
        caps = [cv2.VideoCapture(url) for url in urls]

        for i, cap in enumerate(caps):
            if not cap.isOpened():
                print(f"[ERROR] Failed to open stream {i}: {urls[i]}")

        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                frames.append(frame if ret else None)
            if all(f is None for f in frames):
                break
            yield frames

        for cap in caps:
            cap.release()

    def _load_from_webcam(self, index: int) -> Generator:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open webcam at index {index}")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
