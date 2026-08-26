# ReID Models

Use this page to compare ReID backbones by compute, parameter count, embedding size, and available same-domain evaluation results.

## Model Reference

Same-domain scores are reported as rank-1 `(mAP)`. Rows with `-` do not yet have a documented same-domain result in this table. GFLOPs are input-size dependent and should be read together with the `Input` column; rows are not normalized to one common crop.

| Model | GFLOPs | Input | Params | Embedding | Market1501 | CUHK03-L | CUHK03-D | DukeMTMC-reID | MSMT17 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `osnet_x0_25` | 0.08 | 256x128 | 0.59M | 512 | [91.2 (75.0)](https://drive.google.com/file/d/1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj/view?usp=sharing) | - | - | [82.0 (61.4)](https://drive.google.com/file/d/1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l/view?usp=sharing) | [61.4 (29.5)](https://drive.google.com/file/d/1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF/view?usp=sharing) |
| `osnet_x0_5` | 0.27 | 256x128 | 1.02M | 512 | [92.5 (79.8)](https://drive.google.com/file/d/1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT/view?usp=sharing) | - | - | [85.1 (67.4)](https://drive.google.com/file/d/1KoUVqmiST175hnkALg9XuTi1oYpqcyTu/view?usp=sharing) | [69.7 (37.5)](https://drive.google.com/file/d/1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv/view?usp=sharing) |
| `osnet_x0_75` | 0.57 | 256x128 | 1.67M | 512 | [93.7 (81.2)](https://drive.google.com/file/d/1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer/view?usp=sharing) | - | - | [85.8 (69.8)](https://drive.google.com/file/d/1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or/view?usp=sharing) | [72.8 (41.4)](https://drive.google.com/file/d/1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc/view?usp=sharing) |
| `mobilenetv4_conv_small` | 0.72 | 384x128 | 6.14M | 1536 | - | - | - | - | - |
| `osnet_x1_0` | 0.98 | 256x128 | 2.56M | 512 | [94.2 (82.6)](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing) | - | - | [87.0 (70.2)](https://drive.google.com/file/d/1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq/view?usp=sharing) | [74.9 (43.8)](https://drive.google.com/file/d/112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M/view?usp=sharing) |
| `mobilenetv4_conv_medium` | 2.02 | 384x128 | 12.12M | 1536 | - | - | - | - | - |
| `mobilenetv4_conv_large` | 4.68 | 384x128 | 35.02M | 1536 | - | - | - | - | - |
| `hi_afa` | 2.24† | 384x128 | 12.76M† | 8192 | 97.0 (91.8)† | - | - | 91.7 (82.9)† | 87.6 (71.9)† |
| `lmbn_n` | 4.87 | 384x128 | 9.15M | 3584 | 96.3 (91.5) | 87.2 (85.1) | 84.9 (82.4) | - | - |
| `lmbn_ain_n` | 4.87 | 384x128 | 9.15M | 3584 | - | - | - | - | - |
| `csl_tinyvit_7m` | 5.18 | 384x128 | 9.79M | 1536 | - | - | - | - | - |
| `csl_tinyvit_7m_v20` | 3.29‡ | 384x128 | 6.94M‡ | 1152 | 95.6 (90.0)‡ | - | - | - | - |
| `csl_tinyvit_11m` | 8.29 | 384x128 | 15.46M | 1536 | - | - | - | - | - |
| `csl_tinyvit_11m_v20` | 5.85 | 384x128 | 13.51M | 1536 | 95.9 (91.0)§ | - | - | - | - |
| `csl_tinyvit_23m` | 15.15 | 384x128 | 25.71M | 1536 | - | - | - | - | - |

The linked OSNet scores use the published same-domain softmax setup with 256x128 input, random flip augmentation, and Euclidean distance. The `lmbn_n` scores are the LightMBN paper row reported as `Ours LightMBN` with an OSNet backbone. MobileNetV4, LMBN, Hi-AFA, and CSL-TinyViT rows use the 384x128 BoxMOT ReID crop used by their training configs.

† Hi-AFA complexity and scores are reported by the
[Hi-AFA paper](https://doi.org/10.1109/ACCESS.2024.3389698); they are not a
BoxMOT checkpoint result. The current implementation has 11.78M parameters
with Market1501's 751 classifiers. It emits the 8192-D
`balanced_unique_raw_pooled_streams` descriptor: 16 unique 512-D streams are
normalized independently before concatenation and final normalization, and the
evaluation-only DropBlock duplicate of `g4` is omitted. The paper does not
publish code and leaves attention gamma, DropBlock geometry, label smoothing,
its learning-rate schedule, and shared versus branch-specific center tables
unspecified. The stabilized `hi_afa_market1501` recipe uses zero-initialized,
trainable LDAM residual gates and disables the ambiguous shared 22-stream center
objective while recording BoxMOT's other explicit reproduction assumptions.

‡ `csl_tinyvit_7m_v20` names the promoted RGB inference topology explicitly.
Its 6,937,893 parameters count a normal 751-class training construction without
privileged anatomy; runtime deployment removes classifiers and folds BNNecks.
The full V20 training recipe enables a 227,118-parameter, training-only anatomy
teacher and therefore contains 7,165,011 parameters. The reported Market1501
result is 95.58 rank-1 / 89.96 mAP from that recipe's compact FP16 checkpoint
reloaded for RGB inference.

`csl_tinyvit_11m` is the configurable generic 11M constructor.
`csl_tinyvit_11m_v20` fixes the promoted semantic-fine, depthwise-separable,
scale-balanced topology under an explicit checkpoint-safe name.

§ The 95.90 rank-1 / 91.02 mAP Market1501 result was produced by the A11v8
multiscale EMA pose teacher, now captured by the
`csl_tinyvit_11m_v20_pose_teacher` training recipe. Its privileged branch raises
the 751-class training construction from 13,514,597 to 13,887,539 parameters,
but is pruned for deployment; inference remains the 13.51M, 1536-D RGB V20
model. The RGB-only `csl_tinyvit_11m` recipe is a separate training policy.

## Related pages

- [ReID Profiles](reid.md)
- [Train ReID](../modes/train.md)
- [Evaluate ReID](../modes/eval-reid.md)
