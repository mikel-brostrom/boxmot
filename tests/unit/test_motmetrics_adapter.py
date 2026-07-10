from __future__ import annotations

from pathlib import Path

import pytest

from boxmot.engine.eval.motmetrics import _combine_alpha_rows, evaluate_motchallenge_hota

TUD_HOTA_GOLDEN = {
    "HOTA": 0.3999570912884786,
    "DetA": 0.3976832912424188,
    "AssA": 0.4124495298453543,
    "per_sequence": {
        "TUD-Campus": {
            "HOTA": 0.3913974378451139,
            "DetA": 0.418047030142763,
            "AssA": 0.36912068120832836,
        },
        "TUD-Stadtmitte": {
            "HOTA": 0.3978490169927877,
            "DetA": 0.3922675723693166,
            "AssA": 0.4088407518112996,
        },
    },
}


def test_combine_alpha_rows_matches_legacy_sequence_aggregation():
    combined = _combine_alpha_rows(
        [
            {
                "deta_alpha": 0.5,
                "assa_alpha": 0.25,
                "num_detections": 4,
                "num_objects": 6,
                "num_false_positives": 2,
            },
            {
                "deta_alpha": 0.2,
                "assa_alpha": 0.8,
                "num_detections": 1,
                "num_objects": 2,
                "num_false_positives": 2,
            },
        ]
    )

    assert combined["deta_alpha"] == pytest.approx(5 / 12)
    assert combined["assa_alpha"] == pytest.approx(((0.25 * 4) + (0.8 * 1)) / 5)
    assert combined["hota_alpha"] == pytest.approx((combined["deta_alpha"] * combined["assa_alpha"]) ** 0.5)


def test_in_repo_motmetrics_tud_hota_matches_legacy_golden():
    repo_root = Path(__file__).resolve().parents[2]
    py_motmetrics_root = repo_root / "py-motmetrics"
    data_root = py_motmetrics_root / "motmetrics" / "data"
    if not data_root.exists():
        pytest.skip("py-motmetrics checkout with TUD fixtures is not available")

    sequence_files = {
        "TUD-Campus": (
            data_root / "TUD-Campus" / "gt.txt",
            data_root / "TUD-Campus" / "test.txt",
        ),
        "TUD-Stadtmitte": (
            data_root / "TUD-Stadtmitte" / "gt.txt",
            data_root / "TUD-Stadtmitte" / "test.txt",
        ),
    }

    motmetrics_result = evaluate_motchallenge_hota(sequence_files)

    for metric in ("HOTA", "DetA", "AssA"):
        assert motmetrics_result[metric] == pytest.approx(TUD_HOTA_GOLDEN[metric])
    for sequence_name in sequence_files:
        for metric in ("HOTA", "DetA", "AssA"):
            assert motmetrics_result["per_sequence"][sequence_name][metric] == pytest.approx(
                TUD_HOTA_GOLDEN["per_sequence"][sequence_name][metric]
            )
