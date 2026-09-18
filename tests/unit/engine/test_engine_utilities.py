import logging
from pathlib import Path

from boxmot.engine.eval.output import increment_path
from boxmot.engine.tracking.timing import normalize_setup_timings_ms
from boxmot.engine.ui.logging import suppress_boxmot_logs


def test_increment_path_allocates_numbered_evaluation_directory(tmp_path: Path) -> None:
    base = tmp_path / "exp"
    base.mkdir()

    result = increment_path(base, mkdir=True)

    assert result == tmp_path / "exp2"
    assert result.is_dir()


def test_normalize_setup_timings_has_stable_nonnegative_total() -> None:
    result = normalize_setup_timings_ms({"detector_load": 2.5, "reid_adapter": -1.0})

    assert result["detector_load"] == 2.5
    assert result["reid_adapter"] == 0.0
    assert result["total"] == 2.5


def test_suppress_boxmot_logs_restores_logger_level() -> None:
    logger = logging.getLogger("boxmot")
    previous = logger.level

    with suppress_boxmot_logs(True, level="ERROR"):
        assert logger.level == logging.ERROR

    assert logger.level == previous
