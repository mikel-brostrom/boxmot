from __future__ import annotations

from types import SimpleNamespace

import boxmot.engine.tracker as tracker_module


def test_tracking_session_consumes_finite_track_runs_without_show_or_save(monkeypatch):
    events = []

    class _FakeResults:
        def __init__(self):
            self._cache_results = True

        def __iter__(self):
            events.append(("iter", self._cache_results))

            def _gen():
                yield object()

            return _gen()

    class _FakeRun:
        def __init__(self):
            self.results = _FakeResults()

        def show(self):
            events.append(("show", None))

        def refresh(self):
            events.append(("refresh", None))

    class _FakeBoxmot:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def track(self, **kwargs):
            events.append(("track", kwargs))
            return _FakeRun()

    monkeypatch.setattr(tracker_module, "Boxmot", _FakeBoxmot)

    session = tracker_module.TrackingSession(
        SimpleNamespace(
            source="assets/DOTA8-MOT/train/P1142__1024__0___824/img1",
            yolo_model="yolo11s-obb.pt",
            reid_model="lmbn_n_duke.pt",
            tracking_method="strongsort",
            classes=None,
            project="runs/test",
            imgsz=None,
            conf=None,
            iou=0.7,
            device="cpu",
            half=False,
            save=False,
            save_txt=False,
            show=False,
            verbose=False,
        )
    )

    session.run()

    assert ("iter", False) in events
    assert ("refresh", None) in events
    assert ("show", None) not in events


def test_tracking_session_keeps_live_sources_lazy_without_show_or_save(monkeypatch):
    events = []

    class _FakeRun:
        def __init__(self):
            self.results = object()

        def show(self):
            events.append("show")

        def refresh(self):
            events.append("refresh")

    class _FakeBoxmot:
        def __init__(self, **kwargs):
            pass

        def track(self, **kwargs):
            events.append("track")
            return _FakeRun()

    monkeypatch.setattr(tracker_module, "Boxmot", _FakeBoxmot)

    session = tracker_module.TrackingSession(
        SimpleNamespace(
            source="0",
            yolo_model="yolo11s-obb.pt",
            reid_model="lmbn_n_duke.pt",
            tracking_method="strongsort",
            classes=None,
            project="runs/test",
            imgsz=None,
            conf=None,
            iou=0.7,
            device="cpu",
            half=False,
            save=False,
            save_txt=False,
            show=False,
            verbose=False,
        )
    )

    session.run()

    assert events == ["track"]