# BoxMOT AGPL-3.0 license

from __future__ import annotations

from typing import Literal

BackboneFamily = Literal["cnn", "transformer", "hybrid", "legacy"]
RecipeName = Literal[
    "cnn_reid",
    "transformer_reid",
    "hybrid_reid",
    "legacy_reid",
    "lmbn_reid",
]
ImageSize = tuple[int, int]
