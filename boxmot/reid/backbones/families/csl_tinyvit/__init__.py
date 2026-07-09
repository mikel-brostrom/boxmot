# BoxMOT AGPL-3.0 license

from boxmot.reid.backbones.families.csl_tinyvit.attention import Attention
from boxmot.reid.backbones.families.csl_tinyvit.blocks import (
    BasicLayer,
    Conv2d_BN,
    ConvLayer,
    DropPath,
    LayerNorm2d,
    MBConv,
    PatchEmbed,
    PatchMerging,
    ReIDResidualAdapter,
    TinyViTBlock,
    TinyViTMlp,
)
from boxmot.reid.backbones.families.csl_tinyvit.fusion import (
    CSLTinyViTFeatureFusion,
    PostFusionLocalMixer,
)
from boxmot.reid.backbones.families.csl_tinyvit.heads import (
    GPCLiteMultiBranchHead,
    LMBNStyleMultiBranchHead,
    MultiBranchHead,
)
from boxmot.reid.backbones.families.csl_tinyvit.model import CSLTinyViT
from boxmot.reid.backbones.families.csl_tinyvit.pooling import (
    ActivatedGeM,
    DSELitePool,
    GeM,
    LearnedPartTokenPool,
    PatternAdapter,
    SemanticVisibilityPartPool,
    SpatialTopDrop,
    StripeVisibilityGate,
)
from boxmot.reid.backbones.families.csl_tinyvit.variants import (
    csl_tinyvit_7m,
    csl_tinyvit_7m_lmbn,
    csl_tinyvit_11m,
    csl_tinyvit_11m_lmbn,
    csl_tinyvit_23m,
    csl_tinyvit_23m_lmbn,
    csl_tinyvit_large,
    csl_tinyvit_lmbn,
    csl_tinyvit_normal,
    csl_tinyvit_small,
)

__all__ = [
    "ActivatedGeM",
    "Attention",
    "BasicLayer",
    "CSLTinyViT",
    "CSLTinyViTFeatureFusion",
    "Conv2d_BN",
    "ConvLayer",
    "DSELitePool",
    "DropPath",
    "GPCLiteMultiBranchHead",
    "GeM",
    "LMBNStyleMultiBranchHead",
    "LayerNorm2d",
    "LearnedPartTokenPool",
    "MBConv",
    "MultiBranchHead",
    "PatchEmbed",
    "PatchMerging",
    "PatternAdapter",
    "PostFusionLocalMixer",
    "ReIDResidualAdapter",
    "SemanticVisibilityPartPool",
    "SpatialTopDrop",
    "StripeVisibilityGate",
    "TinyViTBlock",
    "TinyViTMlp",
    "csl_tinyvit_7m",
    "csl_tinyvit_7m_lmbn",
    "csl_tinyvit_11m",
    "csl_tinyvit_11m_lmbn",
    "csl_tinyvit_23m",
    "csl_tinyvit_23m_lmbn",
    "csl_tinyvit_large",
    "csl_tinyvit_lmbn",
    "csl_tinyvit_normal",
    "csl_tinyvit_small",
]
