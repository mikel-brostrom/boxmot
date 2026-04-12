# Postprocessing

`eval` supports three postprocessing modes through `--postprocessing`:

- `none`
- `gsi`
- `gbrc`

## GSI

GSI applies Gaussian-smoothed interpolation to tracking outputs. The implementation lives in `boxmot/postprocessing/gsi.py`.

## GBRC

GBRC applies gradient-boosting-based reconnection and interpolation logic. The implementation lives in `boxmot/postprocessing/gbrc.py`.

## Example

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gsi
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gbrc
```
