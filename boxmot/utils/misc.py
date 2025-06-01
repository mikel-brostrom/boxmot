from pathlib import Path
import sys
from boxmot.utils import logger as LOGGER
import threading

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it already exists, with an option to create the directory.

    Args:
        path (str or Path): Initial file or directory path.
        exist_ok (bool): If True, returns the original path even if it exists.
        sep (str): Separator to use between path stem and increment.
        mkdir (bool): If True, creates the directory if it doesn’t exist.

    Returns:
        Path: Incremented path, or original if exist_ok is True.

    Example:
        runs/exp --> runs/exp2, runs/exp3, etc.
    """
    path = Path(path)  # ensures OS compatibility
    if path.exists() and not exist_ok:
        base, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        
        # Increment path until a non-existing one is found
        for n in range(2, 9999):
            new_path = f"{base}{sep}{n}{suffix}"
            if not Path(new_path).exists():
                path = Path(new_path)
                break

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # creates the directory if it doesn’t exist

    return path


def prompt_overwrite(path_type: str, path: Path, ci: bool = True) -> bool:
    """
    Prompts the user to confirm overwriting an existing file, with a timeout.
    In CI mode (or if stdin isn’t interactive), always returns False.

    Args:
        path_type (str): Type of the path (e.g., 'Detections and Embeddings', 'MOT Result').
        path (Path): The path to check.
        ci (bool): If True, automatically reuse existing file without prompting (for CI environments).

    Returns:
        bool: True if user confirms to overwrite, False otherwise.
    """
    # auto-skip in CI or when there's no interactive stdin
    if ci or not sys.stdin.isatty():
        LOGGER.debug(f"{path_type} {path} already exists. Use existing due to no UI mode.")
        return False

    def input_with_timeout(prompt: str, timeout: float = 3.0) -> bool:
        print(prompt, end='', flush=True)
        result = []
        got_input = threading.Event()

        def _read():
            resp = sys.stdin.readline().strip().lower()
            result.append(resp)
            got_input.set()

        t = threading.Thread(target=_read)
        t.daemon = True
        t.start()
        t.join(timeout)

        if got_input.is_set():
            return result[0] in ('y', 'yes')
        else:
            print("\nNo response, not proceeding with overwrite...")
            return False