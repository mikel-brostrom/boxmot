import unittest

import numpy as np

from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.tracking.track import TrackMeta, TrackState


class MockTracker(BaseTracker):
    def __init__(self):
        super().__init__()
        self.lost_tracks = []
        self.removed_tracks = []

    def _update_impl(self, dets, img, embs=None):
        return self.empty_output()


class InferredMockTracker(BaseTracker):
    def __init__(self):
        super().__init__()

    def _update_impl(self, dets, img, embs=None):
        return self.empty_output()


class MockTrack:
    def __init__(self, id, history, state="confirmed", conf=0.9, cls=0, xyxy=None):
        self.id = id
        self.history_observations = history
        self.state = state
        self.conf = conf
        self.cls = cls
        self.time_since_update = 0
        self.is_activated = True
        self.hits = 10
        self.xyxy = xyxy if xyxy is not None else (history[-1] if history else [0, 0, 10, 10])

    def get_state(self):
        return np.array(self.xyxy)


class ByteTrackStyleTrack(MockTrack):
    pass


ByteTrackStyleTrack.__module__ = "boxmot.trackers.bbox.bytetrack"


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.tracker = MockTracker()
        self.inferred_tracker = InferredMockTracker()
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_plot_results_regular(self):
        # Setup active track
        track = MockTrack(1, [[10, 10, 50, 50]])
        self.tracker.active_tracks = [track]

        # Plot without show_kf_preds
        res = self.tracker.plot_results(self.img.copy(), show_trajectories=True, show_kf_preds=False)

        # Verify something was drawn (image not all zeros)
        # The box is at 10,10 to 50,50, so check a pixel in that region
        # Note: exact pixel values depend on color hashing, but shouldn't be 0
        self.assertTrue(np.any(res[10:50, 10:50] != 0), "Should draw active track")

    def test_plot_results_show_kf_preds(self):
        # Setup lost track
        lost_track = MockTrack(2, [[20, 20, 60, 60]], state="lost")
        # Mock state inference for lost track
        lost_track.time_since_update = 5

        self.tracker.lost_tracks = [lost_track]

        # Plot with show_kf_preds=True
        res = self.tracker.plot_results(self.img.copy(), show_trajectories=True, show_kf_preds=True)

        # Verify lost track is drawn
        self.assertTrue(np.any(res[20:60, 20:60] != 0), "Should draw lost track when show_kf_preds=True")

    def test_removed_track_is_red_when_shown(self):
        removed_track = MockTrack(
            4,
            [[20, 20, 40, 40]],
            xyxy=[50, 50, 80, 80],
        )
        self.tracker.removed_tracks = [removed_track]

        res = self.tracker.plot_results(
            self.img.copy(),
            show_trajectories=False,
            show_kf_preds=True,
        )

        self.assertTrue(
            np.array_equal(res[50, 50], np.array([0, 0, 255], dtype=np.uint8)),
            "Removed tracks should render the last predicted box in red during the temporary lost-display window",
        )
        self.assertTrue(
            np.all(res[20:40, 20:40] == 0),
            "Removed tracks should not fall back to the last associated detection when a predicted box is available",
        )

    def test_plot_results_hide_kf_preds(self):
        # Setup lost track
        lost_track = MockTrack(2, [[20, 20, 60, 60]], state="lost")
        lost_track.time_since_update = 5

        self.tracker.lost_tracks = [lost_track]

        # Plot with show_kf_preds=False
        res = self.tracker.plot_results(self.img.copy(), show_trajectories=True, show_kf_preds=False)

        # Verify lost track is NOT drawn (image should remain all zeros)
        self.assertTrue(np.all(res == 0), "Should not draw lost track when show_kf_preds=False")

    def test_inferred_predicted_track_uses_tracker_display_box(self):
        predicted_track = MockTrack(
            3,
            [[10, 10, 30, 30]],
            xyxy=[40, 40, 70, 70],
        )
        predicted_track.time_since_update = 1

        self.inferred_tracker.active_tracks = [predicted_track]

        res = self.inferred_tracker.plot_results(
            self.img.copy(),
            show_trajectories=False,
            show_kf_preds=True,
        )

        self.assertTrue(
            np.any(res[40:70, 40:70] != 0),
            "Predicted inferred tracks should draw the tracker-provided box",
        )
        self.assertTrue(
            np.all(res[10:30, 10:30] == 0),
            "Predicted inferred tracks should not fall back to stale history geometry",
        )

    def test_common_track_meta_lost_state_is_displayed_as_predicted(self):
        lost_track = MockTrack(5, [[20, 20, 40, 40]])
        lost_track.meta = TrackMeta(id=5, state=TrackState.LOST)
        lost_track.time_since_update = 0

        self.assertEqual(
            self.inferred_tracker.get_track_state_for_display(lost_track),
            "predicted",
        )

    def test_bytetrack_style_integer_state_does_not_require_tracker_enum_import(self):
        lost_track = ByteTrackStyleTrack(6, [[20, 20, 40, 40]], state=2)
        del lost_track.time_since_update

        self.assertEqual(
            self.inferred_tracker.get_track_state_for_display(lost_track),
            "predicted",
        )

    def test_dashed_rect(self):
        # Test internal dashed rect drawing
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 255, 255)
        self.tracker._draw_dashed_rect(img, 10, 10, 90, 90, color, 2)

        # Check corners are drawn
        self.assertTrue(np.any(img[10, 10:20] != 0), "Top-left corner should be drawn")
        # Check gaps exist (approximate check based on dash/gap size)
        # dash=10, gap=10. So 10-20 drawn, 20-30 gap.
        self.assertTrue(np.all(img[10, 22:28] == 0), "Gap should be present in dashed line")


if __name__ == "__main__":
    unittest.main()
