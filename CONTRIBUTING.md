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
uv sync  # builds & installs boxmot in editable mode

# Create a branch
git checkout -b feature/short-desc

# Develop
# ...

# Run functionality where changes were introduced
python boxmot/engine/cli.py track     --yolo-model yolov8x.pt --tracking-method bytetrack --source my_video.mp4 --classes 0
python boxmot/engine/cli.py generate  --yolo-model yolov8x.pt --tracking-method bytetrack --source my_video.mp4 --classes 0
python boxmot/engine/cli.py eval      --yolo-model yolov8x.pt --tracking-method bytetrack --source my_video.mp4 --classes 0
python boxmot/engine/cli.py tune      --yolo-model yolov8x.pt --tracking-method bytetrack --source my_video.mp4 --classes 0
 
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