# AGENTS.md – Guidelines for AI‑Assisted Contributions to **BoxMOT**

> **Purpose**  This document tells automated coding agents (e.g. GitHub Copilot/Codex, GPT‑based chat agents, or other LLM‑powered tools) exactly how to work inside the BoxMOT repository so that every commit is consistent, reproducible, and performance‑safe.

---

## 0 installation

```bash
# Then clone your fork locally
git clone https://github.com/your-username/boxmot.git
cd boxmot
pip install uv
uv sync --all-extras --all-groups  # builds & installs boxmot in editable mode with all its dependencies

# Create a branch
git checkout -b codex/short-desc

# Develop
# ...

# Run functionality where changes were introduced
python boxmot/engine/cli.py track     --yolo-model yolox_x_MOT17_ablation.pt --reid-model lmbn_n_duke.pt --tracking-method boosttrack --source my_video.mp4 --classes 0
python boxmot/engine/cli.py generate  --yolo-model yolox_x_MOT17_ablation.pt --reid-model lmbn_n_duke.pt --tracking-method botsort --source my_video.mp4 --classes 0
python boxmot/engine/cli.py eval      --yolo-model yolox_x_MOT17_ablation.pt --reid-model lmbn_n_duke.pt --tracking-method bytetrack --source my_video.mp4 --classes 0
python boxmot/engine/cli.py tune      --yolo-model yolox_x_MOT17_ablation.pt --reid-model lmbn_n_duke.pt --tracking-method ocsort --source my_video.mp4 --classes 0
```

## 1 Ways of working

Whenever you open a new terminal you have to activate the environment based on where it was installed

## 2  Coding Conventions

* Always add docstrings and typehints to the functionality you add
* Commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `ci:`, `perf:`.

---

*Last updated: 2025‑08‑06*
