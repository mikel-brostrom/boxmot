import numpy as np
import pytest

import boxmot.postprocessing.gbrc as gbrc_module
import boxmot.postprocessing.gsi as gsi_module
import boxmot.postprocessing.gta as gta_algorithms
from boxmot.postprocessing import MotFilePostprocessor, Postprocessor, create_postprocessor, supported_postprocessors
from boxmot.postprocessing.gbrc import gradient_boosting_smooth
from boxmot.postprocessing.gbrc import linear_interpolation as gbrc_linear_interpolation
from boxmot.postprocessing.gsi import gaussian_smooth, linear_interpolation


def test_gsi():
    tracking_results = np.array(
        [
            [1, 1, 1475, 419, 75, 169, 0, 0, -1],
            [2, 1, 1475, 419, 75, 169, 0, 0, -1],
            [4, 1, 1475, 419, 75, 169, 0, 0, -1],
            [6, 1, 1475, 419, 75, 169, 0, 0, -1],
        ]
    )
    li = linear_interpolation(tracking_results, interval=20)
    gsi = gaussian_smooth(li, tau=10)
    assert len(gsi) == 6


def test_gbrc():
    tracking_results = np.array(
        [
            [1, 1, 1475, 419, 75, 169, 0, 0, -1],
            [2, 1, 1475, 419, 75, 169, 0, 0, -1],
            [4, 1, 1475, 419, 75, 169, 0, 0, -1],
            [6, 1, 1475, 419, 75, 169, 0, 0, -1],
        ]
    )
    li = gbrc_linear_interpolation(tracking_results, interval=20)
    gbrc = gradient_boosting_smooth(li)
    assert len(gbrc) == 6
    assert gbrc.shape[1] == 9


def test_postprocessor_factory_creates_file_postprocessors():
    assert supported_postprocessors() == ("gsi", "gbrc")

    gsi_postprocessor = create_postprocessor("gsi", interval=7, tau=3)
    gbrc_postprocessor = create_postprocessor("gbrc", interval=9)

    assert isinstance(gsi_postprocessor, Postprocessor)
    assert isinstance(gsi_postprocessor, MotFilePostprocessor)
    assert gsi_postprocessor.interval == 7
    assert gsi_postprocessor.tau == 3

    assert isinstance(gbrc_postprocessor, Postprocessor)
    assert isinstance(gbrc_postprocessor, MotFilePostprocessor)
    assert gbrc_postprocessor.interval == 9


def test_postprocessor_factory_rejects_unknown_step():
    with pytest.raises(ValueError, match="Unknown postprocessing step"):
        create_postprocessor("missing")


def test_legacy_gta_postprocessor_is_not_registered():
    with pytest.raises(ValueError, match="Unknown postprocessing step"):
        create_postprocessor("gta")


def test_gta_module_contains_only_in_memory_algorithms():
    assert "merge_tracklets" in gta_algorithms.__all__
    assert not hasattr(gta_algorithms, "GTAPostprocessor")
    assert not hasattr(gta_algorithms, "generate_tracklets")
    assert not hasattr(gta_algorithms, "main")


def test_gta_observation_references_survive_extract_merge_and_time_sort():
    first = gta_algorithms.Tracklet(
        1,
        frames=[3, 1],
        scores=[0.9, 0.8],
        bboxes=[[0, 0, 10, 10], [0, 0, 10, 10]],
        feats=[np.array([1, 0]), np.array([1, 0])],
        classes=[0, 0],
        observation_indices=[30, 10],
    )
    first.sort_by_time()
    assert first.observation_indices == [10, 30]
    extracted = first.extract(0, 0)
    assert extracted.observation_indices == [10]
    other = gta_algorithms.Tracklet(
        2,
        frames=[2],
        scores=[0.7],
        bboxes=[[0, 0, 10, 10]],
        feats=[np.array([1, 0])],
        classes=[0],
        observation_indices=[20],
        unmatched_times={4},
    )
    first.merge_from(other)
    assert first.times == [1, 2, 3]
    assert first.observation_indices == [10, 20, 30]
    assert first.occupied_times == {1, 2, 3, 4}


def test_gta_split_preserves_every_source_reference():
    features = [np.array([1.0, 0.0])] * 10 + [np.array([0.0, 1.0])] * 10
    tracklet = gta_algorithms.Tracklet(
        7,
        frames=list(range(20)),
        scores=[0.9] * 20,
        bboxes=[[0, 0, 10, 10]] * 20,
        feats=features,
        classes=[0] * 20,
        observation_indices=list(range(100, 120)),
        unmatched_times={21},
    )

    split = gta_algorithms.split_tracklets({7: tracklet}, len_thres=20, min_samples=3)

    assert len(split) == 2
    assert sorted(index for value in split.values() for index in value.observation_indices) == list(range(100, 120))
    assert sum(21 in value.unmatched_times for value in split.values()) == 1


def test_gta_long_track_clustering_returns_one_label_per_source_observation(monkeypatch):
    import sklearn.cluster

    class Clustering:
        def __init__(self, **kwargs):
            pass

        def fit(self, embeddings):
            self.labels_ = (embeddings[:, 0] < 0).astype(np.int64)
            return self

    monkeypatch.setattr(sklearn.cluster, "DBSCAN", Clustering)
    embeddings = np.concatenate([np.tile([1.0, 0.0], (7501, 1)), np.tile([0.0, 1.0], (7501, 1))])

    switched, labels = gta_algorithms.detect_id_switch(embeddings)

    assert switched
    assert labels.shape == (len(embeddings),)
    assert len(np.unique(labels[:7501])) == len(np.unique(labels[7501:])) == 1
    assert labels[0] != labels[-1]


def test_gta_split_accepts_empty_input():
    assert gta_algorithms.split_tracklets({}) == {}


@pytest.mark.parametrize("lengths", [(1, 7), (13, 2), (17, 31)])
def test_gta_distance_matches_explicit_pairwise_cosine_for_unequal_tracks(lengths):
    rng = np.random.default_rng(42)
    features = [rng.normal(size=(length, 16)) * rng.uniform(0.1, 100, size=(length, 1)) for length in lengths]
    original = [values.copy() for values in features]
    normalized = [
        values.astype(np.float32) / np.maximum(np.linalg.norm(values.astype(np.float32), axis=1, keepdims=True), 1e-8)
        for values in features
    ]
    expected = np.mean(1.0 - normalized[0] @ normalized[1].T)
    tracks = [gta_algorithms.Tracklet(index, feats=list(values)) for index, values in enumerate(features)]

    assert gta_algorithms.get_distance(*tracks) == pytest.approx(expected, abs=1e-6)
    assert gta_algorithms.get_distance(*reversed(tracks)) == pytest.approx(expected, abs=1e-6)
    for values, before in zip(features, original):
        np.testing.assert_array_equal(values, before)


def test_gta_distance_preserves_norm_floor_for_zero_and_tiny_features():
    first = np.array([[0.0, 0.0], [1e-10, 0.0], [4.0, 0.0]], dtype=np.float32)
    second = np.array([[-2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    normalized = [
        values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8) for values in (first, second)
    ]
    expected = np.mean(1.0 - normalized[0] @ normalized[1].T)

    distance = gta_algorithms.get_distance(
        gta_algorithms.Tracklet(1, feats=list(first)),
        gta_algorithms.Tracklet(2, feats=list(second)),
    )

    assert distance == pytest.approx(expected, abs=1e-6)


def test_gta_distance_shortcuts_do_not_require_appearance_features():
    first = gta_algorithms.Tracklet(1, frames=[2], unmatched_times={3})

    assert gta_algorithms.get_distance(first, gta_algorithms.Tracklet(1, frames=[2])) == 0.0
    assert gta_algorithms.get_distance(first, gta_algorithms.Tracklet(2, frames=[2])) == 1.0
    assert gta_algorithms.get_distance(first, gta_algorithms.Tracklet(2, frames=[3])) == 1.0


@pytest.mark.parametrize("module", (gsi_module, gbrc_module))
def test_registered_postprocessor_modules_have_no_standalone_cli(module):
    assert not hasattr(module, "main")


def _progress_tracklets():
    """Create three spatially compatible tracklets at disjoint times."""
    return {
        index: gta_algorithms.Tracklet(
            index,
            frames=[index],
            scores=[0.9],
            bboxes=[[0, 0, 10, 10]],
            feats=[np.array([1.0, 0.0])],
            classes=[0],
        )
        for index in range(1, 4)
    }


def test_gta_batched_progress_distinguishes_batches_and_global_work(capsys):
    events = []
    tracklets = _progress_tracklets()

    split = gta_algorithms.split_tracklets(tracklets, progress_fn=lambda *event: events.append(event))
    merged = gta_algorithms.merge_tracklets_batched(
        split, batch_size=2, progress_fn=lambda *event: events.append(event)
    )

    assert len(merged) == 1
    assert next(iter(merged.values())).times == [1, 2, 3]
    assert events[:4] == [("Split tracklets", current, 3) for current in range(4)]
    assert ("Batch 1/2: Compute distances", 1, 1) in events
    assert ("Batch 2/2: Compute distances", 0, 0) in events
    assert ("Global: Compute distances", 1, 1) in events
    assert ("Global: Merge candidates", 1, None) in events
    assert events[-1] == ("Global: Merge candidates", 1, 1)
    assert capsys.readouterr() == ("", "")


def test_gta_progress_counts_rejected_merge_candidates():
    tracklets = _progress_tracklets()
    del tracklets[3]
    tracklets[2].bboxes = [[100, 100, 10, 10]]
    events = []

    merged = gta_algorithms.merge_tracklets(tracklets, 0.4, 0, 0, progress_fn=lambda *event: events.append(event))

    assert len(merged) == 2
    assert events[-2:] == [("Merge candidates", 1, None), ("Merge candidates", 1, 1)]
