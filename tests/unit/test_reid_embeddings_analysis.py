import numpy as np
import torch

from tools.analysis.reid_embeddings import (
    compute_model_macs,
    estimate_appearance_threshold,
    format_gflops,
    format_gmacs,
    resolve_per_checkpoint_arg,
)


class ConvLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, bias=False)
        self.fc = torch.nn.Linear(4 * 6 * 6, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(torch.flatten(x, 1))


def test_compute_model_macs_counts_conv_and_linear_layers() -> None:
    model = ConvLinear()

    macs = compute_model_macs(model, img_size=(8, 8), device=torch.device("cpu"))

    conv_macs = 4 * 6 * 6 * 3 * 3 * 3
    linear_macs = 10 * (4 * 6 * 6)
    assert macs == conv_macs + linear_macs


def test_format_gmacs() -> None:
    assert format_gmacs(1_234_567_890) == "1.23"
    assert format_gmacs(None) == "?"


def test_format_gflops() -> None:
    assert format_gflops(2_469_135_780) == "2.47"
    assert format_gflops(None) == "?"


def test_resolve_per_checkpoint_arg_repeats_single_value() -> None:
    assert resolve_per_checkpoint_arg(["resize"], 2, "preprocess") == ["resize", "resize"]


def test_resolve_per_checkpoint_arg_accepts_one_value_per_checkpoint() -> None:
    assert resolve_per_checkpoint_arg(["resize", "resize_pad"], 2, "preprocess") == [
        "resize",
        "resize_pad",
    ]


def test_estimate_appearance_threshold_prefers_high_tpr_before_false_accepts() -> None:
    estimate = estimate_appearance_threshold(
        np.array([0.2, 0.3, 0.5]),
        np.array([0.8, 1.0]),
        fp_penalty=10.0,
        search_min=0.1,
        search_max=1.0,
        search_steps=10,
    )

    assert np.isclose(estimate["recommended_default"], 0.5)
    assert estimate["recommended_tpr"] == 1.0
    assert estimate["recommended_fpr"] == 0.0
    assert estimate["focused_search"]["type"] == "uniform"
