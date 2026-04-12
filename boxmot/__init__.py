"""BoxMOT package metadata.

The package root stays intentionally small so CLI/docs tooling can import
``boxmot`` without pulling in optional runtime dependencies. Public APIs live
under explicit modules such as ``boxmot.api``, ``boxmot.reid``,
``boxmot.trackers``, and ``boxmot.trackers.tracker_zoo``.
"""

__version__ = "17.0.0"

__all__ = ("__version__",)
