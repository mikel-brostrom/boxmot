import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from boxmot.utils.visualization import VisualizationMixin

class MockTracker(VisualizationMixin):
    def __init__(self):
        self.is_obb = False
        self.target_id = None
        self.removed_display_frames = 10
        self._plot_frame_idx = 0
        self._removed_first_seen = {}
        self._removed_expired = set()
        self.removed_tombstone_horizon = 100
        self.active_tracks = []
        self.lost_stracks = []
        self.removed_stracks = []

class MockTrack:
    def __init__(self, id, history, state="confirmed", conf=0.9, cls=0):
        self.id = id
        self.history_observations = history
        self.state = state
        self.conf = conf
        self.cls = cls
        self.time_since_update = 0
        self.is_activated = True
        self.hits = 10
        self.xyxy = history[-1] if history else [0, 0, 10, 10]
        
    def get_state(self):
        return np.array(self.xyxy)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.tracker = MockTracker()
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
