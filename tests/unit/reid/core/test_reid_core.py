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
from boxmot.reid.backends.registry import BACKEND_SPECS
from boxmot.reid.core import export_formats
from boxmot.reid.core.artifacts import write_artifact_metadata
from boxmot.reid.core.catalog import TRAINED_URLS
from boxmot.reid.core.formats import (
    REID_EXPORT_FORMAT_COLUMNS,
    REID_EXPORT_SUFFIXES,
    REID_FORMATS,
    resolve_reid_format,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.core.runtime import ReID
from boxmot.reid.exporters.registry import EXPORTER_SPECS
from tests._paths import REPO_ROOT


def _run_import_probe(source: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()


def test_export_formats_uses_core_export_metadata():
    formats = export_formats()

    assert tuple(formats.columns) == REID_EXPORT_FORMAT_COLUMNS
    assert tuple(formats["Suffix"]) == REID_EXPORT_SUFFIXES


def test_reid_formats_have_stable_unique_ids():
    assert tuple(format_.id for format_ in REID_FORMATS) == (
        "pytorch",
        "torchscript",
        "onnx",
        "openvino",
        "tensorrt",
        "coreml",
        "tflite",
    )


def test_every_reid_format_has_one_backend_spec():
    assert set(BACKEND_SPECS) == {format_.id for format_ in REID_FORMATS}


def test_every_exportable_reid_format_has_one_exporter_spec():
    assert set(EXPORTER_SPECS) == {
        format_.id for format_ in REID_FORMATS if format_.argument != "-"
    }


def test_reid_format_uses_exact_suffix_matching():
    assert resolve_reid_format("weights/model.pt.onnx").id == "onnx"


@pytest.mark.parametrize(
    "path",
    [
        "weights/osnet_x0_25_msmt17_openvino_model",
        "weights/osnet_x0_25_msmt17.xml",
        "weights/osnet_x0_25_msmt17.bin",
    ],
)
def test_reid_format_accepts_openvino_artifacts(path):
    assert resolve_reid_format(path).id == "openvino"


def test_reid_format_accepts_coreml_bundle():
    assert resolve_reid_format("weights/model_coreml_model").id == "coreml"


def test_export_metadata_identifies_generic_artifact(tmp_path):
    artifact = tmp_path / "best.onnx"
    write_artifact_metadata(
        artifact,
        {
            "model_name": "csl_tinyvit_11m",
            "num_classes": 751,
            "preprocess": "resize",
            "model_kwargs": {"feat_dim": 512, "head_parts": [1, 2, 3]},
        },
    )

    assert ReIDModelRegistry.get_model_name(artifact) == "csl_tinyvit_11m"
    assert ReIDModelRegistry.get_nr_classes(artifact) == 751
    assert ReIDModelRegistry.get_checkpoint_preprocess(artifact) == "resize"
    assert ReIDModelRegistry.get_checkpoint_model_kwargs(artifact) == {
        "feat_dim": 512,
        "head_parts": (1, 2, 3),
    }


def test_export_metadata_preserves_deployed_anatomical_architecture(
    tmp_path,
):
    artifact = tmp_path / "pose_parts.onnx"
    write_artifact_metadata(
        artifact,
        {
            "model_name": "csl_tinyvit_11m",
            "num_classes": 751,
            "model_kwargs": {
                "anatomical_auxiliary": True,
                "anatomical_token_dim": 128,
                "anatomical_multiscale": True,
                "anatomical_target_type": "learned_pose_concat_ema",
                "anatomical_deployment": True,
                "anatomical_deployment_dim": 64,
                "anatomical_deployment_alpha": 0.25,
            },
        },
    )

    assert ReIDModelRegistry.get_checkpoint_model_kwargs(artifact) == {
        "anatomical_auxiliary": True,
        "anatomical_token_dim": 128,
        "anatomical_multiscale": True,
        "anatomical_target_type": "learned_pose_concat_ema",
        "anatomical_deployment": True,
        "anatomical_deployment_dim": 64,
        "anatomical_deployment_alpha": 0.25,
    }


def test_versioned_model_kwargs_reject_unknown_contract_fields(tmp_path):
    artifact = tmp_path / "model.onnx"
    write_artifact_metadata(
        artifact,
        {
            "model_name": "csl_tinyvit_7m_v20",
            "model_kwargs_schema_version": 1,
            "model_kwargs": {"img_size": [384, 128], "unknown_architecture_knob": True},
        },
    )

    with pytest.raises(ValueError, match="Unknown model_kwargs"):
        ReIDModelRegistry.get_checkpoint_model_kwargs(artifact)


class _DeploymentLoadToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 2, bias=False)
        self.classifier = nn.Linear(2, 4, bias=False)


def test_strict_deployment_load_allows_only_training_classifier_gap(tmp_path):
    source = _DeploymentLoadToy()
    state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("classifier.")
    }
    weights = tmp_path / "best.pt"
    torch.save({"state_dict": state}, weights)
    target = _DeploymentLoadToy()

    report = ReIDModelRegistry.load_deployment_weights(target, weights)

    assert report.tensor_coverage == 1.0
    assert report.numel_coverage == 1.0
    assert report.allowed_missing_keys == ("classifier.weight",)
    torch.testing.assert_close(target.backbone.weight, source.backbone.weight)


def test_strict_deployment_load_rejects_missing_inference_tensor(tmp_path):
    weights = tmp_path / "damaged.pt"
    torch.save({"state_dict": {}}, weights)

    with pytest.raises(RuntimeError, match="missing=.*backbone.weight"):
        ReIDModelRegistry.load_deployment_weights(_DeploymentLoadToy(), weights)


def test_explicit_partial_transfer_api_keeps_partial_loading(tmp_path):
    source = _DeploymentLoadToy()
    weights = tmp_path / "transfer.pt"
    torch.save({"state_dict": {"backbone.weight": source.backbone.weight.clone()}}, weights)
    target = _DeploymentLoadToy()

    report = ReIDModelRegistry.load_partial_weights(target, weights)

    assert "classifier.weight" in report.missing_keys
    assert report.tensor_coverage == 0.5
    torch.testing.assert_close(target.backbone.weight, source.backbone.weight)


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
        "print('boxmot.reid.core.runtime' in sys.modules); "
        "print('torch' in sys.modules)"
    ) == ["False", "False", "False"]


def test_reid_runtime_import_keeps_backend_implementations_lazy():
    assert _run_import_probe(
        "import sys; "
        "import boxmot.reid.core.runtime; "
        "print('boxmot.reid.backends.pytorch_backend' in sys.modules); "
        "print('boxmot.reid.backends.onnx_backend' in sys.modules); "
        "print('boxmot.reid.backends.coreml_backend' in sys.modules)"
    ) == ["False", "False", "False"]


def test_reid_exporter_registry_import_keeps_implementations_lazy():
    assert _run_import_probe(
        "import sys; "
        "import boxmot.reid.exporters.registry; "
        "print('boxmot.reid.exporters.onnx_exporter' in sys.modules); "
        "print('boxmot.reid.exporters.coreml_exporter' in sys.modules); "
        "print('boxmot.reid.exporters.tflite_exporter' in sys.modules)"
    ) == ["False", "False", "False"]


def test_reid_catalog_import_does_not_load_backbone_implementations():
    assert _run_import_probe(
        "import sys; "
        "import boxmot.reid.core.catalog; "
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
        "csl_tinyvit_7m_v20",
        "csl_tinyvit_11m",
        "csl_tinyvit_23m",
        "csl_tinyvit_lmbn",
    ):
        assert getattr(canonical, name) is not None


def test_csl_tinyvit_aliases_stay_registered():
    expected = {
        "csl_tinyvit_7m",
        "csl_tinyvit_7m_v20",
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
