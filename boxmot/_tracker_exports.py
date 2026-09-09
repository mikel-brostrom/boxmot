"""Canonical identities and lazy-import targets for built-in trackers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _TrackerManifestEntry:
    class_path: str
    native_class_path: str | None = None
    native_geometry_modes: tuple[str, ...] = ("aabb", "obb")


_TRACKER_MANIFEST: dict[str, _TrackerManifestEntry] = {
    "boosttrack": _TrackerManifestEntry(
        "boxmot.trackers.box.boosttrack.tracker.BoostTrack",
    ),
    "botsort": _TrackerManifestEntry(
        "boxmot.trackers.box.botsort.tracker.BotSort",
        native_class_path="boxmot.trackers.box.botsort.native.NativeBotSortTracker",
    ),
    "bytetrack": _TrackerManifestEntry(
        "boxmot.trackers.box.bytetrack.tracker.ByteTrack",
        native_class_path="boxmot.trackers.box.bytetrack.native.NativeByteTrackTracker",
    ),
    "deepocsort": _TrackerManifestEntry(
        "boxmot.trackers.box.deepocsort.tracker.DeepOcSort",
    ),
    "hybridsort": _TrackerManifestEntry(
        "boxmot.trackers.box.hybridsort.tracker.HybridSort",
    ),
    "maf_hda": _TrackerManifestEntry(
        "boxmot.trackers.multimodal.maf_hda.tracker.MafHda",
    ),
    "occluboost": _TrackerManifestEntry(
        "boxmot.trackers.box.occluboost.tracker.OccluBoost",
        native_class_path="boxmot.trackers.box.occluboost.native.NativeOccluBoostTracker",
    ),
    "ocsort": _TrackerManifestEntry(
        "boxmot.trackers.box.ocsort.tracker.OcSort",
        native_class_path="boxmot.trackers.box.ocsort.native.NativeOcSortTracker",
    ),
    "sam2mot": _TrackerManifestEntry(
        "boxmot.trackers.multimodal.sam2mot.tracker.Sam2Mot",
    ),
    "sfsort": _TrackerManifestEntry(
        "boxmot.trackers.box.sfsort.tracker.SFSORT",
        native_class_path="boxmot.trackers.box.sfsort.native.NativeSFSORTTracker",
    ),
    "strongsort": _TrackerManifestEntry(
        "boxmot.trackers.box.strongsort.tracker.StrongSort",
    ),
}

_TRACKER_EXPORTS = {
    entry.class_path.rsplit(".", 1)[-1]: tuple(entry.class_path.rsplit(".", 1)) for entry in _TRACKER_MANIFEST.values()
}
