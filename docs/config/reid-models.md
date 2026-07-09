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
| `lmbn_n` | 4.87 | 384x128 | 9.15M | 3584 | 96.3 (91.5) | 87.2 (85.1) | 84.9 (82.4) | - | - |
| `lmbn_ain_n` | 4.87 | 384x128 | 9.15M | 3584 | - | - | - | - | - |
| `csl_tinyvit_7m` | 5.18 | 384x128 | 9.79M | 1536 | - | - | - | - | - |
| `csl_tinyvit_11m` | 8.29 | 384x128 | 15.44M | 1536 | - | - | - | - | - |
| `csl_tinyvit_23m` | 15.15 | 384x128 | 25.71M | 1536 | - | - | - | - | - |

The linked OSNet scores use the published same-domain softmax setup with 256x128 input, random flip augmentation, and Euclidean distance. The `lmbn_n` scores are the LightMBN paper row reported as `Ours LightMBN` with an OSNet backbone. MobileNetV4, LMBN, and CSL-TinyViT rows use the 384x128 BoxMOT ReID crop used by their training configs.

## Related pages

- [ReID Profiles](reid.md)
- [Train ReID](../modes/train.md)
- [Evaluate ReID](../modes/eval-reid.md)
