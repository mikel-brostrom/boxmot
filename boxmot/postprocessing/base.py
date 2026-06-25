from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import queue
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from boxmot.utils import logger as LOGGER
from boxmot.utils.callbacks import safe_seq_progress_callback
from boxmot.utils.rich.workflow.progress import RichTqdm as tqdm

ProgressCallback = Callable[[str, int, int], None]
FileWorker = Callable[..., None]


class Postprocessor(ABC):
    """Base interface for MOT result postprocessing steps."""

    name = ""
    display_name = ""

    def __call__(
        self,
        mot_results_folder: Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.run(mot_results_folder, progress_callback=progress_callback)

    @abstractmethod
    def run(
        self,
        mot_results_folder: Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Apply this postprocessor in-place to MOT result files."""


class MotFilePostprocessor(Postprocessor):
    """Base class for postprocessors that process each MOT result file independently."""

    file_pattern = "*.txt"
    progress_description = "Processing files"

    def result_files(self, mot_results_folder: Path) -> list[Path]:
        """Return result files this postprocessor should process."""
        return sorted(Path(mot_results_folder).glob(self.file_pattern))

    @abstractmethod
    def worker(self) -> FileWorker:
        """Return the top-level worker function used by multiprocessing."""

    def worker_args(self) -> tuple[Any, ...]:
        """Return positional args passed to the worker after ``file_path``."""
        return ()

    def run(
        self,
        mot_results_folder: Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        progress_callback = safe_seq_progress_callback(progress_callback)
        tracking_files = self.result_files(mot_results_folder)
        total_files = len(tracking_files)
        LOGGER.debug(f"{self.display_name}: Found {total_files} file(s) to process.")

        if total_files == 0:
            LOGGER.warning(
                f"{self.display_name}: No .txt files found in results folder. Nothing to process."
            )
            return

        use_queue = progress_callback is not None
        spawn_ctx = mp.get_context("spawn")
        manager_ctx = spawn_ctx.Manager() if use_queue else nullcontext()
        failed_files: list[str] = []
        worker = self.worker()
        worker_args = self.worker_args()

        with manager_ctx as manager:
            progress_queue = manager.Queue() if use_queue else None

            with concurrent.futures.ProcessPoolExecutor(mp_context=spawn_ctx) as executor:
                futures = {
                    executor.submit(worker, file_path, *worker_args, progress_queue): file_path
                    for file_path in tracking_files
                }

                if progress_callback is not None:
                    self._watch_with_callback(futures, progress_queue, progress_callback, failed_files)
                else:
                    self._watch_with_tqdm(futures, failed_files)

        if failed_files and len(failed_files) == total_files:
            raise RuntimeError(
                f"{self.display_name} postprocessing failed for all {total_files} file(s). "
                "Check logs for details."
            )

    def _watch_with_callback(
        self,
        futures: dict[concurrent.futures.Future, Path],
        progress_queue,
        progress_callback: ProgressCallback,
        failed_files: list[str],
    ) -> None:
        seq_progress: dict[str, tuple[int, int]] = {}
        pending = set(futures)

        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=0.3,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    LOGGER.error(f"Error processing file {file_path}: {exc}")
                    failed_files.append(str(file_path))
                seq_progress[file_path.stem] = (1, 1)

            self._drain_queue(progress_queue, seq_progress)
            for name, (current, total) in seq_progress.items():
                progress_callback(name, current, total)

    def _watch_with_tqdm(
        self,
        futures: dict[concurrent.futures.Future, Path],
        failed_files: list[str],
    ) -> None:
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=self.progress_description,
        ):
            file_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                LOGGER.error(f"Error processing file {file_path}: {exc}")
                failed_files.append(str(file_path))

    @staticmethod
    def _drain_queue(progress_queue, seq_progress: dict[str, tuple[int, int]]) -> None:
        """Read all available messages from a progress queue."""
        while True:
            try:
                name, current, total = progress_queue.get_nowait()
                seq_progress[name] = (current, total)
            except (queue.Empty, OSError):
                break
