# BoxMOT AGPL-3.0 license

from __future__ import annotations

from typing import Any, TypeAlias

import torch
from torch import nn

ReIDInferenceOutput: TypeAlias = torch.Tensor
ReIDTrainOutput: TypeAlias = torch.Tensor | tuple[Any, torch.Tensor]


def format_reid_output(loss: str, logits: Any, features: torch.Tensor) -> Any:
    """Return the standard BoxMOT ReID training output for a loss mode."""
    loss = str(loss).lower()
    if loss == "softmax":
        return logits
    if loss in {"triplet", "ms"}:
        return logits, features
    raise ValueError(f"Unsupported loss: {loss}")


class ReIDBackbone(nn.Module):
    """Shared contract for ReID backbones.

    Backbones own spatial feature extraction. Heads own conversion from features
    to embeddings/logits. Subclasses with extra forward flags may override
    ``forward`` while still implementing ``forward_features`` and ``forward_head``.
    """

    feature_dim: int
    loss: str

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_head(self, features: torch.Tensor) -> Any:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Any:
        return self.forward_head(self.forward_features(x))

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
