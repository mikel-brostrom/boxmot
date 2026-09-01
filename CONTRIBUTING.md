# Contributing

Thank you for improving this project! Please follow these guidelines.

## Pull Requests

Proposed workflow

```bash
# Fork the repository on GitHub

# Then clone your fork locally
git clone https://github.com/your-username/boxmot.git
cd boxmot
pip install uv
# Select exactly one PyTorch profile. Use cu130 instead of cpu on CUDA 13.0 hosts.
uv sync --extra cpu --extra yolo --extra evolve --extra service \
  --group dev --group test --group docs

# Create a branch
git checkout -b feature/short-desc

# Develop
# ...

# Run functionality where changes were introduced
uv run --no-sync boxmot track --detector yolov8x --reid osnet_x0_25_msmt17 --tracker bytetrack --source my_video.mp4 --classes 0
uv run --no-sync boxmot generate --detector yolov8x --reid osnet_x0_25_msmt17 --source path/to/dataset --classes 0
uv run --no-sync boxmot eval --dataset mot17 --split ablation --tracker bytetrack
uv run --no-sync boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker bytetrack

# Run tests
uv run --no-sync pytest

# For documentation changes
uv run --no-sync mkdocs build --strict

# Commit & push
git add .
git commit -m "type: summary"
git push origin feature/short-desc

# Open a pull request
# 1. On GitHub, go to your fork: https://github.com/your-username/boxmot
# 2. Click contribute
```
