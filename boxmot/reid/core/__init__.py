# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from boxmot.reid.core.config import REID_EXPORT_FORMAT_COLUMNS, REID_EXPORT_FORMAT_ROWS


def export_formats():
    """Return supported ReID export formats as the public pandas table."""
    import pandas as pd

    return pd.DataFrame(REID_EXPORT_FORMAT_ROWS, columns=REID_EXPORT_FORMAT_COLUMNS)


from .reid import ReID

__all__ = ("export_formats", "ReID")
