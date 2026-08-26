# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BNNeck(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck, self).__init__()
        self.return_f = return_f
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.reset_reid_initialization()

    def reset_reid_initialization(self):
        """Restore ReID BNNeck initialization after broader model init passes."""
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)

        if self.return_f:
            score = self.classifier(after_neck) if self.training else None
            return after_neck, score, before_neck
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
            nn.init.constant_(m.bias, 0.0)
        elif classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find("BatchNorm") != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck3(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3, self).__init__()
        self.return_f = return_f
        # self.reduction = nn.Linear(input_dim, feat_dim)
        # self.bn = nn.BatchNorm1d(feat_dim)

        self.reduction = nn.Conv2d(input_dim, feat_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.register_buffer("_inference_weight", None, persistent=False)
        self.register_buffer("_inference_bias", None, persistent=False)

        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.reset_reid_initialization()

    def reset_reid_initialization(self):
        """Restore ReID BNNeck initialization after broader model init passes."""
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)
        self.bn.bias.requires_grad_(False)

    def project(self, x):
        """Project a pooled map into the pre-BN branch descriptor."""
        projected = self.reduction(x)
        return projected.view(projected.size(0), projected.size(1))

    def forward_projected(self, before_neck):
        """Apply the existing BNNeck/classifier to a projected descriptor."""
        after_neck = self.bn(before_neck)
        if self.return_f:
            score = self.classifier(after_neck) if self.training else None
            return after_neck, score, before_neck
        return self.classifier(after_neck)

    def prepare_for_inference(self) -> None:
        """Cache the projection and BN as one affine 1x1 convolution."""
        if self.training:
            raise RuntimeError("BNNeck3 folding requires eval mode")
        if self._inference_weight is None:
            fused = nn.utils.fusion.fuse_conv_bn_eval(self.reduction, self.bn)
            self._inference_weight = fused.weight.detach()
            self._inference_bias = fused.bias.detach()

    def _invalidate_inference_cache(self) -> None:
        """Discard affine tensors derived from reduction and BN state."""
        self._inference_weight = None
        self._inference_bias = None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Invalidate deployment tensors whenever source weights are reloaded."""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._invalidate_inference_cache()

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Return the post-BN descriptor through the folded deployment path."""
        if self.training:
            raise RuntimeError("BNNeck3.forward_inference requires eval mode")
        if self._inference_weight is None:
            self.prepare_for_inference()
        projected = F.conv2d(
            x,
            self._inference_weight,
            self._inference_bias,
            stride=self.reduction.stride,
            padding=self.reduction.padding,
            dilation=self.reduction.dilation,
            groups=self.reduction.groups,
        )
        return projected.view(projected.size(0), projected.size(1))

    def train(self, mode: bool = True) -> BNNeck3:
        """Discard a deployment cache before any subsequent training."""
        super().train(mode)
        if mode:
            self._invalidate_inference_cache()
        return self

    def forward(self, x):
        return self.forward_projected(self.project(x))

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
            nn.init.constant_(m.bias, 0.0)
        elif classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find("BatchNorm") != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|


class ClassBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        class_num,
        droprate=0,
        relu=False,
        bnorm=True,
        num_bottleneck=512,
        linear=True,
        return_f=False,
    ):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(self.weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x.squeeze(3).squeeze(2))
        if self.return_f:
            f = x
            x = self.classifier(x) if self.training else None
            return f, x, f
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find("Conv") != -1:
            # For old pytorch, you may use kaiming_normal.
            nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm1d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            nn.init.constant_(m.bias.data, 0.0)
