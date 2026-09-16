"""Canonical identities and lazy-import targets for built-in trackers.

Keep this module standard-library-only: package exports and CLI help consume
the manifest without loading tracker implementations or runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _TrackerManifestEntry:
    """Import paths and native geometry modes for one built-in tracker."""

    class_path: str
    config_class_path: str
    native_class_path: str | None = None
    native_geometry_modes: tuple[str, ...] = ("aabb", "obb")


_TRACKER_MANIFEST: dict[str, _TrackerManifestEntry] = {
    "boosttrack": _TrackerManifestEntry(
        "boxmot.trackers.boosttrack.tracker.BoostTrack",
        "boxmot.trackers.boosttrack.config.BoostTrackConfig",
    ),
    "botsort": _TrackerManifestEntry(
        "boxmot.trackers.botsort.tracker.BotSort",
        "boxmot.trackers.botsort.config.BotSortConfig",
        native_class_path="boxmot.trackers.botsort.native.NativeBotSortTracker",
    ),
    "bytetrack": _TrackerManifestEntry(
        "boxmot.trackers.bytetrack.tracker.ByteTrack",
        "boxmot.trackers.bytetrack.config.ByteTrackConfig",
        native_class_path="boxmot.trackers.bytetrack.native.NativeByteTrackTracker",
    ),
    "deepocsort": _TrackerManifestEntry(
        "boxmot.trackers.deepocsort.tracker.DeepOcSort",
        "boxmot.trackers.deepocsort.config.DeepOcSortConfig",
    ),
    "eagermot": _TrackerManifestEntry(
        "boxmot.trackers.eagermot.tracker.EagerMot",
        "boxmot.trackers.eagermot.config.EagerMotConfig",
    ),
    "hybridsort": _TrackerManifestEntry(
        "boxmot.trackers.hybridsort.tracker.HybridSort",
        "boxmot.trackers.hybridsort.config.HybridSortConfig",
    ),
    "maf_hda": _TrackerManifestEntry(
        "boxmot.trackers.maf_hda.tracker.MafHda",
        "boxmot.trackers.maf_hda.config.MafHdaConfig",
    ),
    "occluboost": _TrackerManifestEntry(
        "boxmot.trackers.occluboost.tracker.OccluBoost",
        "boxmot.trackers.occluboost.config.OccluBoostConfig",
        native_class_path="boxmot.trackers.occluboost.native.NativeOccluBoostTracker",
    ),
    "ocsort": _TrackerManifestEntry(
        "boxmot.trackers.ocsort.tracker.OcSort",
        "boxmot.trackers.ocsort.config.OcSortConfig",
        native_class_path="boxmot.trackers.ocsort.native.NativeOcSortTracker",
    ),
    "sfsort": _TrackerManifestEntry(
        "boxmot.trackers.sfsort.tracker.SFSORT",
        "boxmot.trackers.sfsort.config.SFSORTConfig",
        native_class_path="boxmot.trackers.sfsort.native.NativeSFSORTTracker",
    ),
    "strongsort": _TrackerManifestEntry(
        "boxmot.trackers.strongsort.tracker.StrongSort",
        "boxmot.trackers.strongsort.config.StrongSortConfig",
    ),
}
