# Contributing

Thank you for improving this project! Please follow these guidelines.

# Pull Requests

Proposed workflow

```bash
# Fork the repository on GitHub

# Then clone your fork locally
git clone https://github.com/your-username/boxmot.git
cd boxmot
pip install uv
uv sync --all-extras --all-groups  # installs boxmot in editable mode with all dependencies

# Create a branch
git checkout -b feature/short-desc

# Develop
# ...

# Run functionality where changes were introduced
uv run boxmot track --detector yolov8x --reid osnet_x0_25_msmt17 --tracker bytetrack --source my_video.mp4 --classes 0
uv run boxmot generate --detector yolov8x --reid osnet_x0_25_msmt17 --source path/to/dataset --classes 0
uv run boxmot eval --benchmark mot17 --split ablation --tracker bytetrack
uv run boxmot tune --benchmark mot17 --split ablation --tracker bytetrack

# Run tests
uv run pytest

# Commit & push
git add .
git commit -m "type: summary"
git push origin feature/short-desc

# Open a pull request
# 1. On GitHub, go to your fork: https://github.com/your-username/boxmot
# 2. Click contribute
```
