"""Importing library helpers must respect the embedding application's logging."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent


def test_library_import_preserves_application_logging() -> None:
    """Exercise the first import without pytest's already-imported modules."""
    result = subprocess.run(
        [sys.executable, "-c", dedent("""\
            import logging
            import sys
            from io import StringIO

            output = StringIO()
            handler = logging.StreamHandler(output)
            configured = logging.getLogger("boxmot")
            configured.addHandler(handler)
            configured.setLevel(logging.DEBUG)
            configured.propagate = True

            from boxmot.utils import logger

            assert logger is configured
            assert logger.handlers == [handler]
            assert logger.level == logging.DEBUG
            assert logger.propagate is True
            logger.debug("application still receives debug logs")
            assert output.getvalue() == "application still receives debug logs\\n"
            assert "rich.logging" not in sys.modules
            assert "boxmot.engine.ui.logging" not in sys.modules
            """)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_library_import_leaves_unconfigured_logger_passive() -> None:
    result = subprocess.run(
        [sys.executable, "-c", dedent("""\
            import logging
            from boxmot.utils import logger

            assert logger.handlers == []
            assert logger.level == logging.NOTSET
            assert logger.propagate is True
            """)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
