import numpy as np
import pytest

import trackeval


def no_confusion():
    num_timesteps = 5
    num_gt_ids = 2
    num_tracker_ids = 2

    # No overlap between pairs (0, 0) and (1, 1).
    similarity = np.zeros([num_timesteps, num_gt_ids, num_tracker_ids])
    similarity[:, 0, 1] = [0, 0, 0, 1, 1]
    similarity[:, 1, 0] = [1, 1, 0, 0, 0]
    gt_present = np.zeros([num_timesteps, num_gt_ids])
    gt_present[:, 0] = [1, 1, 1, 1, 1]
    gt_present[:, 1] = [1, 1, 1, 0, 0]
    tracker_present = np.zeros([num_timesteps, num_tracker_ids])
    tracker_present[:, 0] = [1, 1, 1, 1, 0]
    tracker_present[:, 1] = [1, 1, 1, 1, 1]

    expected = {
            'clear': {
                    'CLR_TP': 4,
                    'CLR_FN': 4,
                    'CLR_FP': 5,
                    'IDSW': 0,
                    'MOTA': 1 - 9 / 8,
            },
            'identity': {
                    'IDTP': 4,
                    'IDFN': 4,
                    'IDFP': 5,
                    'IDR': 4 / 8,
                    'IDP': 4 / 9,
                    'IDF1': 2 * 4 / 17,
            },
            'vace': {
                    'STDA': 2 / 5 + 2 / 4,
                    'ATA': (2 / 5 + 2 / 4) / 2,
            },
    }

    data = _from_dense(
            num_timesteps=num_timesteps,
            num_gt_ids=num_gt_ids,
            num_tracker_ids=num_tracker_ids,
            gt_present=gt_present,
            tracker_present=tracker_present,
            similarity=similarity,
    )
    return data, expected


def with_confusion():
    num_timesteps = 5
    num_gt_ids = 2
    num_tracker_ids = 2

    similarity = np.zeros([num_timesteps, num_gt_ids, num_tracker_ids])
    similarity[:, 0, 1] = [0, 0, 0, 1, 1]
    similarity[:, 1, 0] = [1, 1, 0, 0, 0]
    # Add some overlap between (0, 0) and (1, 1).
    similarity[:, 0, 0] = [0, 0, 1, 0, 0]
    similarity[:, 1, 1] = [0, 1, 0, 0, 0]
    gt_present = np.zeros([num_timesteps, num_gt_ids])
    gt_present[:, 0] = [1, 1, 1, 1, 1]
    gt_present[:, 1] = [1, 1, 1, 0, 0]
    tracker_present = np.zeros([num_timesteps, num_tracker_ids])
    tracker_present[:, 0] = [1, 1, 1, 1, 0]
    tracker_present[:, 1] = [1, 1, 1, 1, 1]

    expected = {
            'clear': {
                    'CLR_TP': 5,
                    'CLR_FN': 3,  # 8 - 5
                    'CLR_FP': 4,  # 9 - 5
                    'IDSW': 1,
                    'MOTA': 1 - 8 / 8,
            },
            'identity': {
                    'IDTP': 4,
                    'IDFN': 4,
                    'IDFP': 5,
                    'IDR': 4 / 8,
                    'IDP': 4 / 9,
                    'IDF1': 2 * 4 / 17,
            },
            'vace': {
                    'STDA': 2 / 5 + 2 / 4,
                    'ATA': (2 / 5 + 2 / 4) / 2,
            },
    }

    data = _from_dense(
            num_timesteps=num_timesteps,
            num_gt_ids=num_gt_ids,
            num_tracker_ids=num_tracker_ids,
            gt_present=gt_present,
            tracker_present=tracker_present,
            similarity=similarity,
    )
    return data, expected


def split_tracks():
    num_timesteps = 5
    num_gt_ids = 2
    num_tracker_ids = 5

    similarity = np.zeros([num_timesteps, num_gt_ids, num_tracker_ids])
    # Split ground-truth 0 between tracks 0, 3.
    similarity[:, 0, 0] = [1, 1, 0, 0, 0]
    similarity[:, 0, 3] = [0, 0, 0, 1, 1]
    # Split ground-truth 1 between tracks 1, 2, 4.
    similarity[:, 1, 1] = [0, 0, 1, 1, 0]
    similarity[:, 1, 2] = [0, 0, 0, 0, 1]
    similarity[:, 1, 4] = [1, 1, 0, 0, 0]
    gt_present = np.zeros([num_timesteps, num_gt_ids])
    gt_present[:, 0] = [1, 1, 0, 1, 1]
    gt_present[:, 1] = [1, 1, 1, 1, 1]
    tracker_present = np.zeros([num_timesteps, num_tracker_ids])
    tracker_present[:, 0] = [1, 1, 0, 0, 0]
    tracker_present[:, 1] = [0, 0, 1, 1, 1]
    tracker_present[:, 2] = [0, 0, 0, 0, 1]
    tracker_present[:, 3] = [0, 0, 1, 1, 1]
    tracker_present[:, 4] = [1, 1, 0, 0, 0]

    expected = {
            'clear': {
                    'CLR_TP': 9,
                    'CLR_FN': 0,  # 9 - 9
                    'CLR_FP': 2,  # 11 - 9
                    'IDSW': 3,
                    'MOTA': 1 - 5 / 9,
            },
            'identity': {
                    'IDTP': 4,
                    'IDFN': 5,  # 9 - 4
                    'IDFP': 7,  # 11 - 4
                    'IDR': 4 / 9,
                    'IDP': 4 / 11,
                    'IDF1': 2 * 4 / 20,
            },
            'vace': {
                    'STDA': 2 / 4 + 2 / 5,
                    'ATA': (2 / 4 + 2 / 5) / (0.5 * (2 + 5)),
            },
    }

    data = _from_dense(
            num_timesteps=num_timesteps,
            num_gt_ids=num_gt_ids,
            num_tracker_ids=num_tracker_ids,
            gt_present=gt_present,
            tracker_present=tracker_present,
            similarity=similarity,
    )
    return data, expected


def _from_dense(num_timesteps, num_gt_ids, num_tracker_ids, gt_present, tracker_present, similarity):
    gt_subset = [np.flatnonzero(gt_present[t, :]) for t in range(num_timesteps)]
    tracker_subset = [np.flatnonzero(tracker_present[t, :]) for t in range(num_timesteps)]
    similarity_subset = [
            similarity[t][gt_subset[t], :][:, tracker_subset[t]]
            for t in range(num_timesteps)
    ]
    data = {
            'num_timesteps': num_timesteps,
            'num_gt_ids': num_gt_ids,
            'num_tracker_ids': num_tracker_ids,
            'num_gt_dets': np.sum(gt_present),
            'num_tracker_dets': np.sum(tracker_present),
            'gt_ids': gt_subset,
            'tracker_ids': tracker_subset,
            'similarity_scores': similarity_subset,
    }
    return data


METRICS_BY_NAME = {
        'clear': trackeval.metrics.CLEAR(),
        'identity': trackeval.metrics.Identity(),
        'vace': trackeval.metrics.VACE(),
}

SEQUENCE_BY_NAME = {
        'no_confusion': no_confusion(),
        'with_confusion': with_confusion(),
        'split_tracks': split_tracks(),
}


@pytest.mark.parametrize('sequence_name,metric_name', [
        ('no_confusion', 'clear'),
        ('no_confusion', 'identity'),
        ('no_confusion', 'vace'),
        ('with_confusion', 'clear'),
        ('with_confusion', 'identity'),
        ('with_confusion', 'vace'),
        ('split_tracks', 'clear'),
        ('split_tracks', 'identity'),
        ('split_tracks', 'vace'),
])
def test_metric(sequence_name, metric_name):
    data, expected = SEQUENCE_BY_NAME[sequence_name]
    metric = METRICS_BY_NAME[metric_name]
    result = metric.eval_sequence(data)
    for key, value in expected[metric_name].items():
        assert result[key] == pytest.approx(value), key
