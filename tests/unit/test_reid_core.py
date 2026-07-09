import subprocess
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch
from torch import nn

from boxmot.reid.backbones import BACKBONE_REGISTRY, build_backbone, get_backbone_spec
from boxmot.reid.backbones.common import load_partial_state_dict
from boxmot.reid.core import export_formats
from boxmot.reid.core.config import (
    MODEL_TYPES,
    REID_EXPORT_FORMAT_COLUMNS,
    REID_EXPORT_SUFFIXES,
    TRAINED_URLS,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.core.reid import ReID


def _run_import_probe(source: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()


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


def test_boxmot_import_does_not_load_reid_runtime():
    assert _run_import_probe(
        "import sys, boxmot; "
        "print('boxmot.reid.backbones' in sys.modules); "
        "print('torch' in sys.modules); "
        "print('cv2' in sys.modules)"
    ) == ["False", "False", "False"]


def test_reid_preprocessing_import_does_not_load_backbones_or_reid_runtime():
    assert _run_import_probe(
        "import sys; "
        "import boxmot.reid.core.preprocessing; "
        "print('boxmot.reid.backbones' in sys.modules); "
        "print('boxmot.reid.core.reid' in sys.modules); "
        "print('torch' in sys.modules)"
    ) == ["False", "False", "False"]


def test_reid_config_import_does_not_load_backbone_implementations():
    assert _run_import_probe(
        "import sys; "
        "import boxmot.reid.core.config; "
        "print('boxmot.reid.backbones.families.csl_tinyvit.model' in sys.modules); "
        "print('boxmot.reid.backbones.families.osnet.model' in sys.modules); "
        "print('boxmot.reid.backbones.resnet' in sys.modules); "
        "print('torch' in sys.modules); "
        "print('cv2' in sys.modules)"
    ) == ["False", "False", "False", "False", "False"]


def test_backbone_registry_import_keeps_implementation_modules_lazy():
    assert _run_import_probe(
        "import sys; "
        "from boxmot.reid.backbones import BACKBONE_REGISTRY; "
        "print(type(BACKBONE_REGISTRY['csl_tinyvit_11m']).__name__); "
        "print('boxmot.reid.backbones.families.csl_tinyvit.model' in sys.modules); "
        "print('boxmot.reid.backbones.families.osnet.model' in sys.modules); "
        "print('boxmot.reid.backbones.resnet' in sys.modules); "
        "print('torch' in sys.modules)"
    ) == ["LazyBackboneBuilder", "False", "False", "False", "False"]


def test_registry_matches_most_specific_model_name_from_filename():
    assert ReIDModelRegistry.get_model_name(Path("weights/csl_tinyvit_7m_lmbn_market1501.pt")) == "csl_tinyvit_7m_lmbn"
    assert (
        ReIDModelRegistry.get_model_name(Path("weights/csl_tinyvit_23m_lmbn_market1501.pt")) == "csl_tinyvit_23m_lmbn"
    )


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ("osnet_x0_25_msmt17.pt", 1041),
        ("resnet50_fc512_market1501.pt", 751),
        ("vehicleid.pt", 576),
        ("lmbn_n_cuhk03_d.pt", 767),
        ("lmbn_n_market.pt", 751),
        ("unknown_model.pt", 1),
    ],
)
def test_registry_infers_dataset_classes_from_full_weight_name(weights, expected):
    assert ReIDModelRegistry.get_nr_classes(Path(weights)) == expected


def test_backbone_registry_exposes_active_models_only():
    assert "resnet50" in BACKBONE_REGISTRY
    assert "osnet_x0_25" in BACKBONE_REGISTRY
    assert "csl_tinyvit_11m" in BACKBONE_REGISTRY
    assert "mobilenetv4_conv_small" in BACKBONE_REGISTRY
    assert "clip" not in BACKBONE_REGISTRY
    assert "clip" not in MODEL_TYPES
    assert not any(name.startswith("clip_") for name in TRAINED_URLS)
    assert "vit_nano" not in BACKBONE_REGISTRY
    assert "cspreid_n" not in BACKBONE_REGISTRY


def test_public_reid_registry_names_remain_stable():
    expected = {
        "osnet_x1_0",
        "osnet_ain_x1_0",
        "lmbn_n",
        "lmbn_ain_n",
        "csl_tinyvit_11m",
        "mobilenetv4_conv_small",
    }

    assert expected.issubset(BACKBONE_REGISTRY)
    assert {get_backbone_spec(name).name for name in expected} == expected


def test_csl_tinyvit_flat_import_path_is_removed():
    assert find_spec("boxmot.reid.backbones.csl_tinyvit") is None


def test_csl_tinyvit_canonical_import_path_exposes_family_api():
    import boxmot.reid.backbones.families.csl_tinyvit as canonical

    for name in (
        "Attention",
        "CSLTinyViTFeatureFusion",
        "DSELitePool",
        "GeM",
        "GPCLiteMultiBranchHead",
        "LMBNStyleMultiBranchHead",
        "MultiBranchHead",
        "PostFusionLocalMixer",
        "ReIDResidualAdapter",
        "TinyViTBlock",
        "csl_tinyvit_7m",
        "csl_tinyvit_11m",
        "csl_tinyvit_23m",
        "csl_tinyvit_lmbn",
    ):
        assert getattr(canonical, name) is not None


def test_csl_tinyvit_aliases_stay_registered():
    expected = {
        "csl_tinyvit_7m",
        "csl_tinyvit_11m",
        "csl_tinyvit_23m",
        "csl_tinyvit_small",
        "csl_tinyvit_normal",
        "csl_tinyvit_large",
        "csl_tinyvit_7m_lmbn",
        "csl_tinyvit_11m_lmbn",
        "csl_tinyvit_23m_lmbn",
        "csl_tinyvit_lmbn",
    }

    assert expected.issubset(BACKBONE_REGISTRY)


def test_bnneck_uses_canonical_head_namespace_only():
    from boxmot.reid.backbones.heads.bnneck import BNNeck3 as canonical_bnneck

    assert find_spec("boxmot.reid.backbones.common.bnneck") is None
    assert canonical_bnneck.__name__ == "BNNeck3"


def test_backbone_package_does_not_export_model_classes_directly():
    from boxmot.reid.backbones.lmbn_ain_n import LMBN_ain_n
    from boxmot.reid.backbones.lmbn_n import LMBN_n

    with pytest.raises(AttributeError):
        getattr(import_module("boxmot.reid.backbones"), "LMBN_n")

    assert LMBN_n.__name__ == "LMBN_n"
    assert LMBN_ain_n.__name__ == "LMBN_ain_n"


def test_osnet_flat_import_paths_are_removed():
    assert find_spec("boxmot.reid.backbones.osnet") is None
    assert find_spec("boxmot.reid.backbones.osnet_ain") is None


def test_osnet_canonical_import_path_exposes_family_api():
    import boxmot.reid.backbones.families.osnet as canonical

    assert canonical.OSBlock.__name__ == "OSBlock"
    assert canonical.OSBlockAIN.__name__ == "OSBlockAIN"
    assert canonical.OSBlockINin.__name__ == "OSBlockINin"
    assert canonical.OSNet.__name__ == "OSNet"
    assert canonical.osnet_x0_25.__name__ == "osnet_x0_25"
    assert canonical.osnet_ain_x0_25.__name__ == "osnet_ain_x0_25"


def test_osnet_aliases_stay_registered():
    expected = {
        "osnet_x1_0",
        "osnet_x0_75",
        "osnet_x0_5",
        "osnet_x0_25",
        "osnet_ibn_x1_0",
        "osnet_ain_x1_0",
        "osnet_ain_x0_75",
        "osnet_ain_x0_5",
        "osnet_ain_x0_25",
    }

    assert expected.issubset(BACKBONE_REGISTRY)


def test_osnet_state_dict_layout_preserves_standard_and_ain_keys():
    from boxmot.reid.backbones.families.osnet import osnet_ain_x0_25, osnet_x0_25

    standard = osnet_x0_25(num_classes=4, pretrained=False)
    ain = osnet_ain_x0_25(num_classes=4, pretrained=False)

    standard_keys = set(standard.state_dict())
    ain_keys = set(ain.state_dict())

    assert not hasattr(standard, "pool2")
    assert hasattr(ain, "pool2")
    assert any(key.startswith("conv2.0.conv2a.") for key in standard_keys)
    assert any(key.startswith("conv2.0.conv2.0.") for key in ain_keys)
    assert not any(key.startswith("conv2.0.conv2a.") for key in ain_keys)


def test_osnet_pretrained_loader_uses_shared_gdrive_checkpoint(monkeypatch):
    from boxmot.reid.backbones.families.osnet import pretrained as osnet_pretrained

    model = nn.Linear(2, 2)
    calls = {}

    def fake_load_gdrive_checkpoint(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return {
            "module.weight": torch.ones_like(model.weight),
            "module.bias": torch.ones_like(model.bias),
        }

    monkeypatch.setattr(osnet_pretrained, "load_gdrive_checkpoint", fake_load_gdrive_checkpoint)

    osnet_pretrained.load_osnet_pretrained(model, key="osnet_x0_25")

    assert calls["url"] == osnet_pretrained.pretrained_urls["osnet_x0_25"]
    assert calls["kwargs"]["filename"] == "osnet_x0_25_imagenet.pth"
    assert calls["kwargs"]["weights_only"] is False
    assert torch.equal(model.weight, torch.ones_like(model.weight))


def test_backbone_specs_separate_training_recipe_metadata():
    csl_spec = get_backbone_spec("csl_tinyvit_11m")
    osnet_spec = get_backbone_spec("osnet_x0_25")
    hacnn_spec = get_backbone_spec("hacnn")

    assert csl_spec.family == "transformer"
    assert csl_spec.default_recipe == "transformer_reid"
    assert csl_spec.supports_layer_decay is True
    assert osnet_spec.family == "cnn"
    assert osnet_spec.default_recipe == "cnn_reid"
    assert hacnn_spec.family == "legacy"
    assert hacnn_spec.default_img_size == (160, 64)


def test_build_backbone_uses_standard_forward_contract():
    model = build_backbone(
        "resnet18",
        num_classes=3,
        loss="triplet",
        pretrained=False,
    )
    inputs = torch.randn(2, 3, 64, 32)

    model.eval()
    with torch.no_grad():
        embeddings = model(inputs)
        featuremaps = model.forward_features(inputs)
        head_embeddings = model.forward_head(featuremaps)

    assert embeddings.shape == (2, model.feature_dim)
    assert torch.allclose(embeddings, head_embeddings)
    assert torch.allclose(model.featuremaps(inputs), featuremaps)

    model.train()
    logits, features = model(inputs)

    assert logits.shape == (2, 3)
    assert features.shape == (2, model.feature_dim)


def test_partial_state_dict_loads_matching_tensors_only():
    model = nn.Linear(2, 2)
    state_dict = {
        "module.weight": torch.ones_like(model.weight),
        "module.bias": torch.ones(3),
        "module.extra": torch.ones(1),
    }

    matched, skipped = load_partial_state_dict(model, state_dict)

    assert matched == ["weight"]
    assert skipped == ["module.bias", "module.extra"]
    assert torch.equal(model.weight, torch.ones_like(model.weight))


def test_load_url_pretrained_extracts_nested_checkpoint(monkeypatch):
    import boxmot.reid.backbones.common.pretrained as common_pretrained

    model = nn.Linear(2, 2)

    def fake_load_torch_url(url, **kwargs):
        return {
            "model": {
                "weight": torch.ones_like(model.weight),
                "bias": torch.ones_like(model.bias),
            }
        }

    monkeypatch.setattr(common_pretrained, "load_torch_url", fake_load_torch_url)

    matched, skipped = common_pretrained.load_url_pretrained(
        model,
        "https://example.test/model.pt",
        strip_prefix=None,
    )

    assert matched == ["weight", "bias"]
    assert skipped == []
    assert torch.equal(model.weight, torch.ones_like(model.weight))
