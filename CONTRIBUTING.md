# Contributing

Thank you for improving this project! Please follow these guidelines.

## Development setup

Fork the repository on GitHub, then clone your fork and install the contributor dependencies:

```bash
git clone https://github.com/your-username/boxmot.git
cd boxmot
pip install "uv==0.12.4"
uv sync --extra cpu --extra yolo --extra evolve --extra service \
  --group dev --group test --group docs
```

Replace `your-username` with your GitHub username. Use `cu130` instead of `cpu`
for CUDA 13.0. Repeat the chosen profile on later `uv sync` commands and run
commands with `uv run --no-sync`, for example `uv run --no-sync pytest`.

## Pull Requests

Proposed workflow

```bash
# Create a branch
git checkout -b feature/short-desc

# Develop
# ...

# Run functionality where changes were introduced
uv run --no-sync boxmot track --detector yolov8x --reid osnet_x0_25_msmt17 --tracker bytetrack --source my_video.mp4 --classes 0
uv run --no-sync boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml --build-root runs/materializations
uv run --no-sync boxmot eval --dataset mot17 --split ablation --build BUILD_ID --tracker bytetrack
uv run --no-sync boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker bytetrack

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
