"""Generated model names for autocomplete. Regenerate with:

uv run --no-sync python -m tools.generate_model_names
"""

from typing import Literal, TypeAlias

DetectorName: TypeAlias = Literal[
    "yolo11l-mmot-obb",
    "yolo26n",
    "yolox-x-dancetrack",
    "yolox-x-mot17/ablation",
    "yolox-x-mot17/test",
    "yolox-x-mot20",
    "yolox-x-sportsmot",
    "yolox-x-visdrone",
    "yolox/l",
    "yolox/m",
    "yolox/n",
    "yolox/s",
    "yolox/x",
]

__all__ = ("DetectorName",)
