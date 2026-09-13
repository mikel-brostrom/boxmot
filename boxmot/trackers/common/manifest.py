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
    native_class_path: str | None = None
    native_geometry_modes: tuple[str, ...] = ("aabb", "obb")


_TRACKER_MANIFEST: dict[str, _TrackerManifestEntry] = {
    "boosttrack": _TrackerManifestEntry(
        "boxmot.trackers.boosttrack.tracker.BoostTrack",
    ),
    "botsort": _TrackerManifestEntry(
        "boxmot.trackers.botsort.tracker.BotSort",
        native_class_path="boxmot.trackers.botsort.native.NativeBotSortTracker",
    ),
    "bytetrack": _TrackerManifestEntry(
        "boxmot.trackers.bytetrack.tracker.ByteTrack",
        native_class_path="boxmot.trackers.bytetrack.native.NativeByteTrackTracker",
    ),
    "deepocsort": _TrackerManifestEntry(
        "boxmot.trackers.deepocsort.tracker.DeepOcSort",
    ),
    "eagermot": _TrackerManifestEntry(
        "boxmot.trackers.eagermot.tracker.EagerMot",
    ),
    "hybridsort": _TrackerManifestEntry(
        "boxmot.trackers.hybridsort.tracker.HybridSort",
    ),
    "maf_hda": _TrackerManifestEntry(
        "boxmot.trackers.maf_hda.tracker.MafHda",
    ),
    "occluboost": _TrackerManifestEntry(
        "boxmot.trackers.occluboost.tracker.OccluBoost",
        native_class_path="boxmot.trackers.occluboost.native.NativeOccluBoostTracker",
    ),
    "ocsort": _TrackerManifestEntry(
        "boxmot.trackers.ocsort.tracker.OcSort",
        native_class_path="boxmot.trackers.ocsort.native.NativeOcSortTracker",
    ),
    "sfsort": _TrackerManifestEntry(
        "boxmot.trackers.sfsort.tracker.SFSORT",
        native_class_path="boxmot.trackers.sfsort.native.NativeSFSORTTracker",
    ),
    "strongsort": _TrackerManifestEntry(
        "boxmot.trackers.strongsort.tracker.StrongSort",
    ),
}
