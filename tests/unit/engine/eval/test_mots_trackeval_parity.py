"""Compare native MOTS preprocessing and metrics with an independent TrackEval.

The optional installed reference is used by CI. A local ``./TrackEval`` checkout
also works, including older versions that use removed NumPy scalar aliases.
Only the isolated reference modules receive those aliases; NumPy is unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from boxmot.engine.eval.motmetrics import SequenceData, _combine_bundles, _eval_bundle

_OBJECT = tuple[int, int, np.ndarray]
_FRAME = list[_OBJECT]
_SHAPE = (12, 16)
_TOLERANCE = 1e-12


class _ReferenceNumpy:
    """Expose legacy scalars only to modules from the optional old checkout."""

    def __getattr__(self, name: str) -> Any:
        aliases = {"int": int, "float": float, "bool": bool}
        return aliases[name] if name in aliases else getattr(np, name)


@pytest.fixture
def reference(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Import the installed reference, or load only needed checkout modules."""
    pytest.importorskip("pycocotools.mask", reason="MOTS parity requires the optional mots extra")
    if importlib.util.find_spec("trackeval") is not None:
        package = importlib.import_module("trackeval")
        return SimpleNamespace(dataset=package.datasets.KittiMOTS, metrics=package.metrics)

    source = Path(__file__).resolve().parents[4] / "TrackEval" / "trackeval"
    if not (source / "datasets" / "kitti_mots.py").is_file():
        pytest.skip("MOTS parity requires TrackEval installed or the local ./TrackEval checkout")

    prefix = "_boxmot_mots_reference"
    for suffix in ("", ".datasets", ".metrics"):
        package = ModuleType(prefix + suffix)
        package.__path__ = [str(source.joinpath(*suffix.strip(".").split("."))) if suffix else str(source)]
        monkeypatch.setitem(sys.modules, package.__name__, package)

    modules = {}
    for suffix in (
        "utils",
        "_timing",
        "datasets._base_dataset",
        "datasets.kitti_mots",
        "metrics._base_metric",
        "metrics.hota",
        "metrics.clear",
        "metrics.identity",
        "metrics.count",
    ):
        name = f"{prefix}.{suffix}"
        module_path = source.joinpath(*suffix.split(".")).with_suffix(".py")
        specification = importlib.util.spec_from_file_location(name, module_path)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        monkeypatch.setitem(sys.modules, name, module)
        specification.loader.exec_module(module)
        if getattr(module, "np", None) is np:
            module.np = _ReferenceNumpy()
        modules[suffix] = module
    return SimpleNamespace(
        dataset=modules["datasets.kitti_mots"].KittiMOTS,
        metrics=SimpleNamespace(
            **{
                name: getattr(modules[f"metrics.{name.lower()}"], name)
                for name in ("HOTA", "CLEAR", "Identity", "Count")
            }
        ),
    )


def _mask(*rectangles: tuple[int, int, int, int]) -> np.ndarray:
    """Build small binary masks, including disconnected shapes and holes."""
    mask = np.zeros(_SHAPE, dtype=np.uint8)
    for top, bottom, left, right in rectangles:
        mask[top:bottom, left:right] = 1
    return mask


def _cases() -> dict[str, tuple[list[_FRAME], list[_FRAME]]]:
    """Exercise matching, class filtering, ignores, identity and empty cases."""
    car = _mask((0, 4, 0, 4))
    pedestrian = _mask((0, 4, 10, 12))
    car_hole = _mask((0, 2, 0, 4), (3, 4, 0, 4))
    ignored = _mask((8, 10, 0, 4))
    gt = [
        [(1001, 1, car_hole), (2002, 2, pedestrian), (10000, 10, ignored)],
        [
            (1001, 1, _mask((0, 2, 0, 2))),
            (2002, 2, pedestrian),
            (10000, 10, _mask((0, 2, 2, 4), (8, 10, 4, 6), (8, 10, 10, 13))),
        ],
        [(1001, 1, car), (2002, 2, pedestrian)],
        [(1001, 1, car), (2002, 2, pedestrian)],
        [(10000, 10, ignored)],
        [(1001, 1, car), (2002, 2, pedestrian)],
        [],
    ]
    tracker = [
        [(41, 1, car_hole), (71, 2, pedestrian), (90, 1, ignored)],
        [
            (41, 1, _mask((0, 2, 0, 4))),
            (71, 2, pedestrian),
            (90, 1, _mask((8, 10, 4, 8))),
            (91, 1, _mask((8, 10, 10, 14))),
        ],
        [(42, 1, car), (92, 1, _mask((8, 10, 10, 14)))],
        [(71, 2, pedestrian)],
        [(93, 1, ignored), (72, 2, pedestrian)],
        [(42, 1, _mask((0, 2, 0, 2))), (73, 2, pedestrian)],
        [],
    ]
    # These masks have equal bounding boxes but mask IoU is exactly 0.5.
    stripes = _mask((0, 4, 0, 1), (0, 4, 3, 4))
    top, bottom = _mask((0, 2, 0, 4)), _mask((2, 4, 0, 4))
    left, right = _mask((0, 4, 0, 2)), _mask((0, 4, 2, 4))
    return {
        "0000": (gt, tracker),
        "0001": (
            [[(1007, 1, stripes), (2008, 2, pedestrian)], [(1007, 1, stripes)], []],
            [[(50, 1, car), (85, 2, pedestrian)], [(50, 1, stripes)], []],
        ),
        "0002": ([[(1009, 1, car)], []], [[], []]),
        "0003": ([[], []], [[(99, 2, pedestrian)], []]),
        "0004": ([[], []], [[], []]),
        # All pairwise IoUs are 1/3. Reordered result rows exercise upstream
        # Hungarian tie resolution: a parser must preserve per-frame file order.
        "0005": (
            [[(1001, 1, top), (1002, 1, bottom)]] * 2,
            [[(11, 1, left), (12, 1, right)], [(12, 1, right), (11, 1, left)]],
        ),
    }


def _rle_rows(frames: list[_FRAME]) -> Iterator[str]:
    """Encode official zero-based MOTS rows independently of BoxMOT's writer."""
    from pycocotools import mask as mask_utils

    for frame_index, objects in enumerate(frames):
        for track_id, class_id, mask in objects:
            encoded = mask_utils.encode(np.asfortranarray(mask))
            yield f"{frame_index} {track_id} {class_id} {_SHAPE[0]} {_SHAPE[1]} {encoded['counts'].decode('ascii')}\n"


@pytest.fixture
def inputs(tmp_path: Path, reference: SimpleNamespace) -> SimpleNamespace:
    """Write equivalent GT PNGs/reference RLE and shared tracker RLE files."""
    gt_folder = tmp_path / "reference_gt"
    tracker_folder = tmp_path / "tracker"
    gt_folder.mkdir()
    tracker_folder.mkdir()
    gt_frames, seq_info = {}, {}
    for sequence, (gt, tracker) in _cases().items():
        seq_info[sequence] = len(gt)
        (gt_folder / f"{sequence}.txt").write_text("".join(_rle_rows(gt)))
        (tracker_folder / f"{sequence}.txt").write_text("".join(_rle_rows(tracker)))
        png_folder = tmp_path / "png_gt" / sequence
        png_folder.mkdir(parents=True)
        frame_records = []
        for frame_index, objects in enumerate(gt):
            labels = np.zeros(_SHAPE, dtype=np.uint16)
            for track_id, _, mask in objects:
                assert not np.any(labels[mask.astype(bool)])
                labels[mask.astype(bool)] = track_id
            path = png_folder / f"{frame_index:06d}.png"
            assert cv2.imwrite(str(path), labels)
            frame_records.append((frame_index, path, *_SHAPE))
        gt_frames[sequence] = tuple(frame_records)

    dataset = reference.dataset(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(tmp_path),
            "TRACKERS_TO_EVAL": [tracker_folder.name],
            "TRACKER_SUB_FOLDER": "",
            "GT_LOC_FORMAT": "{gt_folder}/{seq}.txt",
            "SEQ_INFO": seq_info,
            "PRINT_CONFIG": False,
        }
    )
    metrics = {
        name: getattr(reference.metrics, name)({"PRINT_CONFIG": False})
        for name in ("HOTA", "CLEAR", "Identity", "Count")
    }
    return SimpleNamespace(
        gt_folder=gt_folder,
        tracker_folder=tracker_folder,
        gt_frames=gt_frames,
        seq_info=seq_info,
        dataset=dataset,
        metrics=metrics,
    )


def _assert_preprocessing(native: SequenceData, expected: dict[str, Any]) -> None:
    """Check IDs and every mask similarity, not only eventual summary scores."""
    for field in ("num_timesteps", "num_gt_ids", "num_tracker_ids", "num_gt_dets", "num_tracker_dets"):
        assert getattr(native, field) == expected[field], (native.seq, field)
    for field in ("gt_ids", "tracker_ids", "similarity_scores"):
        for frame, (actual, wanted) in enumerate(zip(getattr(native, field), expected[field], strict=True)):
            np.testing.assert_allclose(
                actual, wanted, atol=_TOLERANCE, rtol=0, err_msg=f"{native.seq}: {field}, frame {frame}"
            )


def _assert_bundle(actual: dict[str, Any], expected: dict[str, Any], context: str) -> None:
    """Compare every reference field, including all 19 HOTA thresholds."""
    for family, values in expected.items():
        for field, value in values.items():
            np.testing.assert_allclose(
                actual[family][field], value, atol=_TOLERANCE, rtol=0, err_msg=f"{context}: {family}.{field}"
            )


def test_mots_preprocessing_and_metrics_match_trackeval(inputs: SimpleNamespace) -> None:
    """Match the upstream adapter and default MOTS metric families end to end."""
    from boxmot.engine.eval.mots import _build_mots_sequence_data

    for class_name, class_id in (("car", 1), ("pedestrian", 2)):
        native_bundles, reference_bundles = {}, {}
        for sequence, length in inputs.seq_info.items():
            native = _build_mots_sequence_data(
                sequence,
                inputs.gt_frames[sequence],
                inputs.tracker_folder / f"{sequence}.txt",
                ((class_name, class_id),),
                length,
            )[class_name]
            raw = inputs.dataset.get_raw_seq_data(inputs.tracker_folder.name, sequence)
            expected = inputs.dataset.get_preprocessed_seq_data(raw, class_name)
            _assert_preprocessing(native, expected)
            native_bundles[sequence] = _eval_bundle(native)
            reference_bundles[sequence] = {
                name: metric.eval_sequence(expected) for name, metric in inputs.metrics.items()
            }
            _assert_bundle(native_bundles[sequence], reference_bundles[sequence], f"{sequence}/{class_name}")
            if class_name == "car" and sequence == "0000":
                np.testing.assert_array_equal(native.similarity_scores[1], [[0.5, 0.0]])
                assert len(native.tracker_ids[4]) == 0
                assert native_bundles[sequence]["CLEAR"]["IDSW"] > 0
            if class_name == "car" and sequence == "0001":
                np.testing.assert_array_equal(native.similarity_scores[0], [[0.5]])

        combined = {
            name: metric.combine_sequences({seq: bundle[name] for seq, bundle in reference_bundles.items()})
            for name, metric in inputs.metrics.items()
        }
        _assert_bundle(_combine_bundles(native_bundles), combined, f"all sequences/{class_name}")


def _assert_report(actual: dict[str, Any], expected: dict[str, Any], context: str) -> None:
    """Independently scale upstream ratios to the native percentage report."""
    for family, fields in {
        "HOTA": ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA"),
        "CLEAR": ("MOTA", "MOTP", "sMOTA", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR"),
        "Identity": ("IDF1", "IDR", "IDP"),
    }.items():
        for field in fields:
            assert actual[field] == pytest.approx(float(np.mean(expected[family][field])) * 100, abs=1e-10), (
                context,
                field,
            )
    for family, fields in {
        "CLEAR": ("CLR_TP", "CLR_FP", "CLR_FN", "IDSW", "MT", "PT", "ML", "Frag"),
        "Identity": ("IDTP", "IDFP", "IDFN"),
        "Count": ("Dets", "GT_Dets", "IDs", "GT_IDs"),
    }.items():
        for field in fields:
            assert actual[field] == expected[family][field], (context, field)


def test_mots_public_reports_and_class_aggregates_match_trackeval(inputs: SimpleNamespace, tmp_path: Path) -> None:
    """Check per-class, per-sequence and both class aggregation report paths."""
    from boxmot.engine.eval.mots import run_mots_metrics

    args = Namespace(exp_dir=inputs.tracker_folder, evaluation_config={"mots_gt_frames": inputs.gt_frames})
    actual = run_mots_metrics(
        args,
        [Path(name) for name in inputs.seq_info],
        tmp_path,
        inputs.gt_folder,
        seq_info=inputs.seq_info,
    )
    expected_classes = {}
    for class_name in ("car", "pedestrian"):
        sequences = {}
        for sequence in inputs.seq_info:
            raw = inputs.dataset.get_raw_seq_data(inputs.tracker_folder.name, sequence)
            data = inputs.dataset.get_preprocessed_seq_data(raw, class_name)
            sequences[sequence] = {name: metric.eval_sequence(data) for name, metric in inputs.metrics.items()}
            _assert_report(
                actual[class_name]["per_sequence"][sequence],
                sequences[sequence],
                f"{class_name}/{sequence}",
            )
        expected_classes[class_name] = {
            name: metric.combine_sequences({seq: bundle[name] for seq, bundle in sequences.items()})
            for name, metric in inputs.metrics.items()
        }
        _assert_report(actual[class_name], expected_classes[class_name], class_name)

    for label, combine in (
        ("cls_comb_cls_av", "combine_classes_class_averaged"),
        ("cls_comb_det_av", "combine_classes_det_averaged"),
    ):
        expected = {
            name: getattr(metric, combine)({cls: bundle[name] for cls, bundle in expected_classes.items()})
            for name, metric in inputs.metrics.items()
        }
        _assert_report(actual[label], expected, label)
