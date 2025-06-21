"""
lazy_dataloader.py
A general-purpose *lazy* frame generator that works with:

• CSV files of paths or URLs
• Video files (MP4, AVI, MOV, …)
• Image folders (optionally recursive)
• Glob patterns
• YouTube URLs (via yt-dlp)
• RTSP/RTMP/TCP/HTTP streams
• “*.streams” multi-stream text files
• Local webcams (by index)

Every iteration yields **batches**:  
- single-source paths → `[frame]` (list of length 1)  
- multi-stream lists  → `[frame₀, frame₁, …]` (length = #streams)

Optional kwargs let you shuffle, stride, limit, apply a transform, or recurse
through sub-directories without touching the core logic.
"""

import os
import cv2
import csv
import glob
import random
import itertools
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Generator, List, Iterable, Optional, Callable


class LazyDataLoader:
    """A lazy, iterable loader that yields lists of frames."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        source: Union[str, int, Path],
        *,
        shuffle: bool = False,
        recursive: bool = False,
        stride: int = 1,
        limit: Optional[int] = None,
        transform: Optional[Callable[[List], List]] = None,
    ):
        """
        Parameters
        ----------
        source : str | int | Path
            See list at top of file.
        shuffle : bool, default False
            Shuffle file/URL order inside CSV, glob, or directory.
        recursive : bool, default False
            Recurse into sub-directories (directory / glob modes).
        stride : int, default 1
            Return every *n*-th batch.
        limit : int | None, default None
            Stop after yielding this many batches.
        transform : Callable | None
            Function applied to each yielded *batch* before it is returned.
        """
        self.source = str(source)
        self.shuffle = shuffle
        self.recursive = recursive
        self.stride = max(1, int(stride))
        self.limit = limit
        self.transform = transform

    def __iter__(self) -> Iterable[List]:
        gen = self._dispatch()
        gen = itertools.islice(gen, 0, self.limit, self.stride)
        if self.transform:
            gen = (self.transform(batch) for batch in gen)
        return gen

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _dispatch(self) -> Generator[List, None, None]:
        """Route the source string to the appropriate loader."""
        src = self.source

        if src.endswith(".csv"):
            yield from self._from_csv(src)

        elif src.endswith(".streams"):
            yield from self._from_stream_list(src)

        elif src.startswith(("rtsp://", "rtmp://", "tcp://")):
            yield from self._from_stream(src)

        elif src.startswith(("http://", "https://")):
            if "youtube.com" in src or "youtu.be" in src:
                yield from self._from_youtube(src)
            else:
                yield from self._from_stream(src)

        elif "*" in src:
            yield from self._from_glob(src)

        elif os.path.isdir(src):
            yield from self._from_directory(src)

        elif os.path.isfile(src):
            yield from self._from_file(src)

        elif str(src).isdigit():
            yield from self._from_webcam(int(src))

        else:
            raise ValueError(f"Unsupported source type: {src}")

    # -------------------------- CSV ------------------------------------- #
    def _from_csv(self, csv_path: str) -> Generator[List, None, None]:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = [row[0] for row in reader if row and not row[0].startswith("#")]

        if self.shuffle:
            random.shuffle(rows)

        for item in rows:
            yield from LazyDataLoader(
                item,
                shuffle=self.shuffle,
                recursive=self.recursive,
                stride=1,        # stride/limit applied once in outer iterator
                limit=None,
                transform=None,
            )

    # ------------------- single file (image/video) ---------------------- #
    def _from_file(self, path: str) -> Generator[List, None, None]:
        ext = Path(path).suffix.lower()
        if ext in self.IMAGE_EXTS:
            frame = cv2.imread(path)
            if frame is not None:
                yield [frame]
            else:
                print(f"[ERROR] Could not read image: {path}")
        else:
            yield from self._from_video(path)

    # ----------------------------- directory ---------------------------- #
    def _from_directory(self, dir_path: str) -> Generator[List, None, None]:
        pattern = "**/*" if self.recursive else "*"
        items = list(Path(dir_path).glob(pattern))
        items.sort()
        if self.shuffle:
            random.shuffle(items)

        for p in items:
            if p.is_file():
                yield from self._from_file(str(p))

    # ------------------------------ glob -------------------------------- #
    def _from_glob(self, pattern: str) -> Generator[List, None, None]:
        files = glob.glob(pattern, recursive=self.recursive)
        files.sort()
        if self.shuffle:
            random.shuffle(files)

        for f in files:
            yield from self._from_file(f)

    # ----------------------------- video -------------------------------- #
    def _from_video(self, video_path: str) -> Generator[List, None, None]:
        with self._capture(video_path) as cap:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield [frame]

    # ---------------------------- YouTube ------------------------------- #
    def _from_youtube(self, url: str) -> Generator[List, None, None]:
        try:
            import yt_dlp
        except ImportError as e:
            raise ImportError(
                "yt_dlp is required for YouTube support. Run `pip install yt_dlp`."
            ) from e

        with yt_dlp.YoutubeDL({"quiet": True, "format": "best"}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info["url"]
            yield from self._from_stream(video_url)

    # ----------------------------- stream ------------------------------- #
    def _from_stream(self, url: str) -> Generator[List, None, None]:
        with self._capture(url) as cap:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield [frame]

    # -------------------------- multi-stream ---------------------------- #
    def _from_stream_list(self, file_path: str) -> Generator[List, None, None]:
        with open(file_path) as f:
            urls = [u.strip() for u in f if u.strip() and not u.startswith("#")]

        caps = [cv2.VideoCapture(u) for u in urls]
        try:
            while True:
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    frames.append(frame if ret else None)
                if all(f is None for f in frames):
                    break
                yield frames  # batch of N frames (may contain Nones)
        finally:
            for cap in caps:
                cap.release()

    # ----------------------------- webcam ------------------------------- #
    def _from_webcam(self, index: int) -> Generator[List, None, None]:
        with self._capture(index) as cap:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield [frame]

    # ------------------------- capture wrapper -------------------------- #
    @contextmanager
    def _capture(self, path_or_idx):
        """OpenCV VideoCapture with guaranteed release."""
        cap = cv2.VideoCapture(path_or_idx)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"[ERROR] Cannot open {path_or_idx}")
            yield cap
        finally:
            cap.release()
            
    # ----------------------------------------------------------------- #
    # Optional but very handy!  Allows tqdm or any other client to ask
    # “how many batches will you yield after stride/limit?”
    # ----------------------------------------------------------------- #
    def __len__(self) -> int:
        """
        Return the number of batches this loader will produce **after**
        applying `stride` and `limit`.
        
        Only defined for image folders and glob patterns, which are the
        sources you use in MOT17.  For videos/streams the length is
        unknown, so we raise `TypeError` – that’s exactly what built-ins
        like `open()` do when `len()` would be meaningless.
        """
        src = self.source
        
        # ------------- count how many *items* are available ------------
        if os.path.isdir(src):
            pattern = "**/*" if self.recursive else "*"
            items = [p for p in Path(src).glob(pattern) if p.is_file()]
        
        elif "*" in src:  # glob pattern
            items = [Path(f) for f in glob.glob(src, recursive=self.recursive)
                     if Path(f).is_file()]
        
        else:
            raise TypeError("len() is undefined for video/stream sources")
        
        items.sort()
        n_items = len(items)
        
        # ------------- convert item count → batch count ----------------
        n_batches = (n_items + self.stride - 1) // self.stride
        if self.limit is not None:
            n_batches = min(n_batches, self.limit)
        return n_batches


# --------------------------------------------------------------------- #
# Example usage (run `python lazy_dataloader.py path/or/url/or/*.jpg`)
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import cv2                     # <-- make sure cv2 is in main scope

    # Accept a CLI argument (or webcam 0 if none)
    src = sys.argv[1] if len(sys.argv) > 1 else 0

    # Create the loader (tweak flags as you like)
    loader = LazyDataLoader(src, shuffle=False, recursive=True, stride=1)

    # Iterate lazily and show frames
    for i, batch in enumerate(loader):
        print(f"Batch {i:05d} — size: {len(batch)}")

        # Display every frame in the batch (None-checks for multi-streams)
        for j, frame in enumerate(batch):
            if frame is None:
                continue
            cv2.imshow(f"stream_{j}", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()