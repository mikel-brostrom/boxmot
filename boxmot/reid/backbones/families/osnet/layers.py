# BoxMOT AGPL-3.0 license

from __future__ import annotations

from torch import nn

__all__ = [
    "Conv1x1",
    "Conv1x1Linear",
    "Conv3x3",
    "ConvLayer",
    "LightConv3x3",
    "LightConvStream",
]


class ConvLayer(nn.Module):
    """Convolution layer: conv + normalization + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        IN: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1(nn.Module):
    """1x1 convolution + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1Linear(nn.Module):
    """1x1 convolution + optional BN without non-linearity."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, bn: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution: 1x1 linear + depthwise 3x3."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class LightConvStream(nn.Module):
    """Stack of lightweight 3x3 convolutions used by OSNet-AIN blocks."""

    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be equal to or larger than 1, got {depth}")
        layers = [LightConv3x3(in_channels, out_channels)]
        layers.extend(LightConv3x3(out_channels, out_channels) for _ in range(depth - 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
