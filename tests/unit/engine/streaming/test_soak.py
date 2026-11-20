"""Synthetic CPU churn through real trackers and the streaming timing path."""

import gc
import weakref
from collections import deque

import numpy as np
import pytest

from boxmot import BotSort, BotSortConfig, ByteTrack, ByteTrackConfig
from boxmot.engine.tracking.profiling import RuntimeProfiler


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_long_cpu_stream_bounds_track_archives_and_profiler(name):
    tracker = (
        ByteTrack(config=ByteTrackConfig(max_lost_frames=2))
        if name == "bytetrack"
        else BotSort(config=BotSortConfig(max_lost_frames=2, use_embeddings=False, use_cmc=False))
    )
    profiler = RuntimeProfiler()
    window = deque(maxlen=300)
    early_retired = []
    persistent = [0, 0, 10, 10, 0.99, 0]
    visiting = [50, 50, 60, 60, 0.99, 0]
    for index in range(10000):
        detections = np.asarray([persistent, visiting] if index % 7 < 2 else [persistent], dtype=np.float32)
        sample = str(index)
        profiler.begin_pipeline(sample)
        result = tracker.update(detections)
        profiler.end_pipeline(sample)
        profiler.finish_sample(sample, 1.0)
        window.append(profiler.release_sample(sample))
        assert len(result) <= 2
        assert len(tracker.active_tracks) <= 2 and len(tracker.lost_stracks) <= 1
        assert len(tracker.removed_stracks) <= (329 if name == "botsort" else 5)
        for track in tracker.active_tracks:
            assert len(track.history_observations) <= tracker.max_obs
            if index < 70 and track.start_frame > 1:
                early_retired.append(weakref.ref(track))
    del track
    gc.collect()
    assert early_retired and all(ref() is None for ref in early_retired)
    assert len(window) == 300 and not profiler._samples
    assert profiler.totals_ms()["overall"] == 10000
    assert len(tracker._removed_first_seen) <= 329 and len(tracker._removed_expired) <= 329
