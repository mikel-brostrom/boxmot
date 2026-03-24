import unittest

import numpy as np
from boxmot.trackers.basetracker import BaseTracker

class MockTracker(BaseTracker):
    def __init__(self):
        super().__init__()
        self.lost_stracks = []
        self.removed_stracks = []

    def update(self, dets, img, embs=None):
        return self.empty_output()


class InferredMockTracker(BaseTracker):
    def __init__(self):
        super().__init__()

    def update(self, dets, img, embs=None):
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

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.tracker = MockTracker()
        self.inferred_tracker = InferredMockTracker()
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)
        
    def test_plot_results_regular(self):
        # Setup active track
        track = MockTrack(1, [[10, 10, 50, 50]])
        self.tracker.active_tracks = [track]
        
        # Plot without show_lost
        res = self.tracker.plot_results(
            self.img.copy(),
            show_trajectories=True,
            show_lost=False
        )
        
        # Verify something was drawn (image not all zeros)
        # The box is at 10,10 to 50,50, so check a pixel in that region
        # Note: exact pixel values depend on color hashing, but shouldn't be 0
        self.assertTrue(np.any(res[10:50, 10:50] != 0), "Should draw active track")

    def test_plot_results_show_lost(self):
        # Setup lost track
        lost_track = MockTrack(2, [[20, 20, 60, 60]], state="lost")
        # Mock state inference for lost track
        lost_track.time_since_update = 5
        
        self.tracker.lost_stracks = [lost_track]
        
        # Plot with show_lost=True
        res = self.tracker.plot_results(
            self.img.copy(),
            show_trajectories=True,
            show_lost=True
        )
        
        # Verify lost track is drawn
        self.assertTrue(np.any(res[20:60, 20:60] != 0), "Should draw lost track when show_lost=True")
        
    def test_plot_results_hide_lost(self):
        # Setup lost track
        lost_track = MockTrack(2, [[20, 20, 60, 60]], state="lost")
        lost_track.time_since_update = 5
        
        self.tracker.lost_stracks = [lost_track]
        
        # Plot with show_lost=False
        res = self.tracker.plot_results(
            self.img.copy(),
            show_trajectories=True,
            show_lost=False
        )
        
        # Verify lost track is NOT drawn (image should remain all zeros)
        self.assertTrue(np.all(res == 0), "Should not draw lost track when show_lost=False")

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
            show_lost=True,
        )

        self.assertTrue(
            np.any(res[40:70, 40:70] != 0),
            "Predicted inferred tracks should draw the tracker-provided box",
        )
        self.assertTrue(
            np.all(res[10:30, 10:30] == 0),
            "Predicted inferred tracks should not fall back to stale history geometry",
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

if __name__ == '__main__':
    unittest.main()
