# ReID Ablation Findings

This page records the current CSL-TinyViT recipe choices from local Market-1501 ablations.

## Locked recipe

The `csl_tinyvit_7m`, `csl_tinyvit_11m`, and `csl_tinyvit_23m` training recipes now share the same high-level choices:

- `head_pool: gem`
- `head_parts: [1, 2]`
- `inference_feature: concat_bn`
- `feature_fusion: last3`
- `branch_aware_metric: false`
- `head_warmup_epochs: 0`

## Feature fusion

`last3` fuses the final spatial map with the previous two TinyViT stages through residual 1x1 projections. It is only a small projection/interpolation cost on top of the backbone, so when `last2` and `last3` are effectively tied, the default favors the three-scale aggregation.

The 23M feature-fusion run supports that choice:

| Run | Fusion | Epochs | Best mAP | Rank-1 | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| `1_fusion/b_last3` | `last3` | 200 | 89.46 | 95.19 | 190 |
| `1_fusion/a_last2` | `last2` | 200 | 89.44 | 95.16 | 200 |
| `3_fresh_300/a_last3` | `last3` | 300 | 89.09 | 95.28 | 280 |

The completed 7M fusion runs do not justify weighted fusion as a default:

| Run | Fusion | Epochs | Best mAP | Rank-1 | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| `1_fusion/a_last2` | `last2` | 250 | 84.91 | 94.15 | 190 |
| `2_weighted/b_weighted_last3` | `weighted_last3` | 250 | 83.98 | 93.68 | 250 |

Use `last3` for the default residual path. Keep `weighted_last2` and `weighted_last3` as explicit ablation modes only.

## Head and loss

The 23M head ablation selected the simple GeM setup. `1_pool/a_gem_only` reached 88.79 mAP and 95.31 rank-1, edging out the branch-aware and head-warmup combinations while keeping the head simpler.

The loss and output-feature checks support the same defaults:

- Triplet plus center loss remained the strongest completed loss family; Circle loss was lower, and the multi-similarity run was incomplete.
- `concat_bn` stayed the inference embedding. Same-checkpoint eval from the optimized 23M control was 89.00 mAP for `concat_bn`, 88.91 for `global`, and 87.86 for `raw_mean`.
- `resize` remains the ReID preprocessing default. The `resize_pad` control was lower at 88.07 mAP versus 89.00 for `resize`.
