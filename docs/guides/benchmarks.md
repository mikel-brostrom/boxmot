# Benchmark Workflows

BoxMOT's benchmark-driven workflows use a layered config system rather than forcing you to pass every path manually.

## Resolution model

When you run:

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
```

BoxMOT resolves:

1. the benchmark YAML under `boxmot/configs/benchmarks`
2. the associated dataset YAML under `boxmot/configs/datasets`
3. the associated detector profile under `boxmot/configs/detectors`
4. the associated ReID profile under `boxmot/configs/reid`
5. the tracker YAML under `boxmot/configs/trackers`

## Typical pipeline

```bash
boxmot generate --benchmark mot17-ablation
boxmot eval --benchmark mot17-ablation --tracker bytetrack
boxmot tune --benchmark mot17-ablation --tracker bytetrack
boxmot research --benchmark mot17-ablation --tracker bytetrack --proposal-model openai/gpt-5.4
```

## Overrides

You can override detector and ReID selections explicitly:

```bash
boxmot eval \
  --benchmark mot17-ablation \
  --detector yolo11s_obb \
  --reid lmbn_n_duke \
  --tracker boosttrack
```

## Related pages

- [Config System Overview](../config/index.md)
- [Benchmarks](../config/benchmarks.md)
- [Datasets](../config/datasets.md)
- [Detectors](../config/detectors.md)
- [ReID Profiles](../config/reid.md)
- [Tracker YAMLs](../config/trackers.md)
