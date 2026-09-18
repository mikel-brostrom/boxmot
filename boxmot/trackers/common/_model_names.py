"""Generated model names for autocomplete. Regenerate with:

uv run --no-sync python -m tools.generate_model_names
"""

from typing import Literal, TypeAlias

TrackerName: TypeAlias = Literal[
    "boosttrack",
    "botsort",
    "bytetrack",
    "deepocsort",
    "eagermot",
    "hybridsort",
    "maf_hda",
    "occluboost",
    "ocsort",
    "sfsort",
    "strongsort",
]

__all__ = ("TrackerName",)
