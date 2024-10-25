from pathlib import Path
import os

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