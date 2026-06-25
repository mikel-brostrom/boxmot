from pathlib import Path

import pytest

from boxmot.reid.core import export_formats
from boxmot.reid.core.config import REID_EXPORT_FORMAT_COLUMNS, REID_EXPORT_SUFFIXES
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.core.reid import ReID


def _model_type(path: str) -> tuple[bool, ...]:
    reid = ReID.__new__(ReID)
    return reid.model_type(Path(path))


def test_export_formats_uses_core_export_metadata():
    formats = export_formats()

    assert tuple(formats.columns) == REID_EXPORT_FORMAT_COLUMNS
    assert tuple(formats["Suffix"]) == REID_EXPORT_SUFFIXES


def test_reid_model_type_uses_exact_suffix_matching():
    assert _model_type("weights/model.pt.onnx") == (False, False, True, False, False, False)


@pytest.mark.parametrize(
    "path",
    [
        "weights/osnet_x0_25_msmt17_openvino_model",
        "weights/osnet_x0_25_msmt17.xml",
        "weights/osnet_x0_25_msmt17.bin",
    ],
)
def test_reid_model_type_accepts_openvino_artifacts(path):
    assert _model_type(path) == (False, False, False, True, False, False)


def test_registry_matches_most_specific_model_name_from_filename():
    assert (
        ReIDModelRegistry.get_model_name(Path("weights/csl_tinyvit_7m_lmbn_market1501.pt"))
        == "csl_tinyvit_7m_lmbn"
    )
    assert (
        ReIDModelRegistry.get_model_name(Path("weights/csl_tinyvit_23m_lmbn_market1501.pt"))
        == "csl_tinyvit_23m_lmbn"
    )


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ("osnet_x0_25_msmt17.pt", 1041),
        ("resnet50_fc512_market1501.pt", 751),
        ("clip_vehicleid.pt", 576),
        ("lmbn_n_cuhk03_d.pt", 767),
        ("unknown_model.pt", 1),
    ],
)
def test_registry_infers_dataset_classes_from_full_weight_name(weights, expected):
    assert ReIDModelRegistry.get_nr_classes(Path(weights)) == expected
