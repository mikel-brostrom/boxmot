from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from tests._paths import REPO_ROOT


def _write_run(root: Path, run_name: str, model_name: str, best_map: float) -> None:
    run_dir = root / "runs" / "csl_tinyvit_7m_fix" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "hparams.json").write_text(
        json.dumps({"run": {"model_name": model_name}}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "dataset": "market1501",
                "best_epoch": 1,
                "best_mAP": best_map,
                "best_rank1": 0.9,
            }
        ),
        encoding="utf-8",
    )


def test_7m_context_accepts_base_and_v20_models(tmp_path: Path) -> None:
    script = tmp_path / "repotext.sh"
    shutil.copy2(REPO_ROOT / "repotext.sh", script)
    _write_run(tmp_path, "base", "csl_tinyvit_7m", 0.7)
    _write_run(tmp_path, "v20", "csl_tinyvit_7m_v20", 0.8)
    _write_run(tmp_path, "other", "csl_tinyvit_11m", 0.9)
    output = tmp_path / "snapshot.txt"

    subprocess.run(
        ["bash", str(script), "7m", str(output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    snapshot = output.read_text(encoding="utf-8")
    assert "| csl_tinyvit_7m_fix | base |" in snapshot
    assert "| csl_tinyvit_7m_fix | v20 |" in snapshot
    assert "runs/csl_tinyvit_7m_fix/base/metrics.json" in snapshot
    assert "runs/csl_tinyvit_7m_fix/v20/metrics.json" in snapshot
    assert "| csl_tinyvit_7m_fix | other |" not in snapshot
    assert "runs/csl_tinyvit_7m_fix/other/metrics.json" not in snapshot
