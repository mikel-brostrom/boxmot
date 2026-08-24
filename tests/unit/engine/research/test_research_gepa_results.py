import hashlib
import json
import pickle
from pathlib import Path

from boxmot.engine.research import gepa_results as analysis_module


def _write_cache(run_dir: Path, candidate: dict[str, str], score: float, hota: float, idf1: float, mota: float) -> None:
    candidate_hash = hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()
    cache_path = run_dir / "fitness_cache" / f"{candidate_hash[:16]}_example.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": (candidate_hash[:16], "example"),
        "result": (
            score,
            None,
            {
                "Combined Metrics": {
                    "HOTA": hota,
                    "IDF1": idf1,
                    "MOTA": mota,
                },
                "Combined Delta vs Baseline": {
                    "HOTA": hota - 67.0,
                    "IDF1": idf1 - 75.0,
                    "MOTA": mota - 76.0,
                },
            },
        ),
    }
    cache_path.write_bytes(pickle.dumps(payload))


def test_build_gepa_report_extracts_metrics_and_headlines(tmp_path):
    run_dir = tmp_path / "gepa"
    run_dir.mkdir()

    candidates = [
        {"tracker.py": "class ByteTrack:\n    pass\n"},
        {
            "tracker.py": (
                "class ByteTrack:\n"
                "    def _track_is_confirmed(self):\n"
                "        return True\n\n"
                "    def _apply_association_priors(self):\n"
                "        return None\n"
            )
        },
        {
            "tracker.py": (
                "class ByteTrack:\n"
                "    def _track_is_confirmed(self):\n"
                "        return True\n\n"
                "    def _apply_association_priors(self):\n"
                "        return None\n\n"
                "    def _apply_ambiguity_suppression(self):\n"
                "        return None\n"
            )
        },
    ]
    (run_dir / "candidates.json").write_text(json.dumps(candidates))
    (run_dir / "run_log.json").write_text(
        json.dumps(
            [
                {"i": 0, "selected_program_candidate": 0, "new_program_idx": 1},
                {"i": 3, "selected_program_candidate": 1, "new_program_idx": 2},
            ]
        )
    )

    _write_cache(run_dir, candidates[0], score=67.0, hota=67.0, idf1=75.0, mota=76.0)
    _write_cache(run_dir, candidates[1], score=67.3, hota=67.3, idf1=75.4, mota=75.9)
    _write_cache(run_dir, candidates[2], score=67.6, hota=67.6, idf1=75.7, mota=76.1)

    report = analysis_module.build_gepa_report(run_dir)

    assert report.accepted_candidate_indices == [0, 1, 2]
    assert report.best_candidate_idx == 2
    assert report.candidate_metrics[1].hota == 67.3
    assert report.accepted_transitions[0].iteration == 1
    assert (
        report.accepted_transitions[0].headline
        == "Added lifecycle-aware association priors and delayed confirmation"
    )
    assert report.accepted_transitions[1].iteration == 4
    assert report.accepted_transitions[1].headline == "Added ambiguity suppression for dense overlap cases"


def test_write_report_artifacts_creates_plot_and_summaries(tmp_path):
    run_dir = tmp_path / "gepa"
    run_dir.mkdir()

    candidates = [
        {"tracker.py": "class ByteTrack:\n    pass\n"},
        {
            "tracker.py": (
                "class ByteTrack:\n"
                "    def _normalized_center_distance(self):\n"
                "        return 0.0\n\n"
                "    def _velocity_direction_consistency(self):\n"
                "        return 1.0\n"
            )
        },
    ]
    (run_dir / "candidates.json").write_text(json.dumps(candidates))
    (run_dir / "run_log.json").write_text(
        json.dumps([{"i": 1, "selected_program_candidate": 0, "new_program_idx": 1}])
    )

    _write_cache(run_dir, candidates[0], score=67.0, hota=67.0, idf1=75.0, mota=76.0)
    _write_cache(run_dir, candidates[1], score=67.5, hota=67.5, idf1=75.8, mota=76.2)

    report = analysis_module.build_gepa_report(run_dir)
    outputs = analysis_module.write_report_artifacts(report)

    assert outputs["plot"].exists()
    assert outputs["summary"].exists()
    assert outputs["json"].exists()
    assert "Best candidate c1" in outputs["summary"].read_text()
