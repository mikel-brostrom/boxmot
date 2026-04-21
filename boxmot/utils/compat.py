from __future__ import annotations

import sys


def dataclass_slots_kwargs() -> dict[str, bool]:
    """Return dataclass keyword arguments supported by the running Python version."""
    return {"slots": True} if sys.version_info >= (3, 10) else {}