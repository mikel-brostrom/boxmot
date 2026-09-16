"""Official-package loading, device precision and optional postprocessing."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from boxmot.segmentors.propagation import model


@pytest.mark.parametrize(
    "device,major,expected", [("cpu", 0, "fp32"), ("mps", 0, "fp16"), ("cuda:2", 7, "fp32"), ("cuda:2", 8, "bf16")]
)
def test_effective_precision_uses_selected_cuda_device(monkeypatch, device, major, expected):
    seen = []

    def capability(selected):
        seen.append(selected)
        return major, 0

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert model.effective_precision(device) == expected
    assert seen == ([torch.device(device)] if device.startswith("cuda") else [])


@pytest.mark.parametrize("device,precision", [("cpu", "fp16"), ("cpu", "bf16"), ("mps", "bf16"), ("mps", "unknown")])
def test_invalid_precision_fails_before_inference(device, precision):
    with pytest.raises(ValueError, match="precision|requires a? ?CUDA"):
        with model.inference_context(device, precision):
            pytest.fail("Unsupported precision must fail before entering inference")


def test_bf16_rejects_unsupported_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(ValueError, match="bfloat16 support"):
        with model.inference_context("cuda:1", "bf16"):
            pytest.fail("Unsupported bf16 must fail early")


@pytest.mark.parametrize("device,autocast_device", [("cuda:3", "cuda"), ("mps", "mps")])
def test_context_restores_grad_mode_and_uses_requested_precision(monkeypatch, device, autocast_device):
    calls = []
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: calls.append((args, kwargs)) or nullcontext())
    before = torch.is_grad_enabled()
    with model.inference_context(device, "fp16"):
        assert torch.is_inference_mode_enabled()
    assert torch.is_grad_enabled() == before and not torch.is_inference_mode_enabled()
    assert calls == [((autocast_device,), {"dtype": torch.float16})]


@pytest.mark.parametrize("device,available", [("cpu", True), ("mps", True), ("cuda:0", False), ("cuda:0", True)])
def test_postprocessing_reports_effective_extension_capability(monkeypatch, device, available):
    monkeypatch.setattr(model, "find_spec", lambda name: object() if available else None)
    monkeypatch.setattr(
        model, "import_module", lambda name: SimpleNamespace(get_connected_componnets=lambda mask: None)
    )
    metadata = model.postprocessing_metadata(device)
    assert metadata["requested_fill_hole_area"] == 8
    assert metadata["effective_fill_hole_area"] == (8 if device.startswith("cuda") and available else 0)
    assert metadata["dynamic_multimask_via_stability"] is True


def test_broken_optional_extension_is_reported_absent(monkeypatch):
    monkeypatch.setattr(model, "find_spec", lambda name: object())

    def broken(name):
        raise ImportError("incompatible optional CUDA extension ABI")

    monkeypatch.setattr(model, "import_module", broken)
    assert model.postprocessing_metadata("cuda:0")["effective_fill_hole_area"] == 0


def test_loader_uses_official_builder_without_pretrained_backbone(tmp_path, monkeypatch):
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.touch()
    sentinel, calls = SimpleNamespace(spatial_perceiver=SimpleNamespace()), []
    monkeypatch.setattr(model, "find_spec", lambda name: object())
    monkeypatch.setattr(model, "_register_model_config", lambda: "boxmot_official_edgetam")

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(model, "import_module", lambda name: SimpleNamespace(build_sam2_video_predictor=build))
    assert model.build_edgetam_predictor(checkpoint, "cpu") is sentinel
    args, kwargs = calls[0]
    assert args == ("boxmot_official_edgetam", str(checkpoint))
    assert kwargs["device"] == "cpu" and kwargs["apply_postprocessing"] is False
    overrides = kwargs["hydra_overrides_extra"]
    assert (
        "model.image_encoder.trunk._target_=boxmot.segmentors.propagation.backbone.InferenceTimmBackbone" in overrides
    )
    assert "++model.fill_hole_area=0" in overrides
    assert "++model.binarize_mask_from_pts_for_mem_enc=true" in overrides
    assert "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true" in overrides
    assert not any("local-dir" in override for override in overrides)


def test_backbone_preserves_official_forward_without_pretrained_fetch(monkeypatch):
    pytest.importorskip("sam2")
    from boxmot.segmentors.propagation import backbone

    class Features(torch.nn.Module):
        feature_info = SimpleNamespace(channels=lambda: [48, 96, 192, 384])

        def forward(self, image):
            return [image, image + 1]

    calls = []
    monkeypatch.setattr(backbone, "create_model", lambda *args, **kwargs: calls.append((args, kwargs)) or Features())
    actual = backbone.InferenceTimmBackbone("repvit_m1.dist_in1k", ["layer0", "layer1", "layer2", "layer3"])
    assert actual.channel_list == [384, 192, 96, 48]
    assert calls == [
        (
            ("repvit_m1.dist_in1k",),
            {"pretrained": False, "in_chans": 3, "features_only": True, "out_indices": (0, 1, 2, 3)},
        )
    ]
    assert [value.item() for value in actual(torch.tensor(2))] == [2, 3]


@pytest.mark.parametrize(
    "device,precision,weight_dtype,fp16_memory",
    [
        ("cpu", None, torch.float32, False),
        ("mps", None, torch.float16, True),
        ("mps", "fp32", torch.float32, False),
        ("cuda:1", "fp16", torch.float16, False),
        ("cuda:1", None, torch.float32, False),
    ],
)
def test_loader_selects_weights_memory_and_batching_without_global_patches(
    tmp_path, monkeypatch, device, precision, weight_dtype, fp16_memory
) -> None:
    """Device policies affect only the loaded model, including real tensor dtypes."""
    from boxmot.segmentors.propagation.inference import _run_memory_encoder, _run_single_frame_inference
    from boxmot.segmentors.propagation.perceiver import _forward_2d

    class Predictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial_perceiver = torch.nn.Linear(4, 4)
            self.sam_prompt_encoder = torch.nn.Embedding(4, 4)

        def _run_single_frame_inference(self):
            return None

        def _run_memory_encoder(self):
            return None

    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.touch()
    predictor, untouched = Predictor(), Predictor()
    parameter_names = set(predictor.state_dict())
    monkeypatch.setattr(model, "resolve_device", lambda value: torch.device(value))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(model, "find_spec", lambda name: object())
    monkeypatch.setattr(model, "_register_model_config", lambda: "fixture_config")
    monkeypatch.setattr(model, "postprocessing_metadata", lambda selected: {"effective_fill_hole_area": 0})
    monkeypatch.setattr(
        model, "import_module", lambda name: SimpleNamespace(build_sam2_video_predictor=lambda *a, **kw: predictor)
    )

    assert model.build_edgetam_predictor(checkpoint, device, precision=precision) is predictor
    assert {p.dtype for p in predictor.parameters()} == {weight_dtype}
    assert set(predictor.state_dict()) == parameter_names
    assert predictor.spatial_perceiver.forward_2d.__func__ is _forward_2d
    assert not hasattr(untouched.spatial_perceiver, "forward_2d")
    assert predictor._run_single_frame_inference.__func__ is (
        _run_single_frame_inference if fp16_memory else Predictor._run_single_frame_inference
    )
    assert predictor._run_memory_encoder.__func__ is (
        _run_memory_encoder if fp16_memory else Predictor._run_memory_encoder
    )
    assert untouched._run_single_frame_inference.__func__ is Predictor._run_single_frame_inference


def test_packaged_config_preserves_hydra_initialization():
    hydra = pytest.importorskip("hydra")
    global_hydra = pytest.importorskip("hydra.core.global_hydra").GlobalHydra.instance()
    scope = nullcontext() if global_hydra.is_initialized() else hydra.initialize(version_base="1.2", config_path=None)
    with scope:
        original = global_hydra.hydra
        config = hydra.compose(config_name=model._register_model_config())
        assert global_hydra.hydra is original
        assert config.model.image_encoder.trunk.name == "repvit_m1.dist_in1k"
        assert config.model.num_maskmem == 7 and config.model.image_size == 1024


def test_missing_package_is_actionable(tmp_path, monkeypatch):
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.touch()
    monkeypatch.setattr(model, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError, match="optional official EdgeTAM package"):
        model.build_edgetam_predictor(checkpoint, "cpu")


def test_missing_checkpoint_precedes_optional_import(monkeypatch):
    monkeypatch.setattr(model, "find_spec", lambda name: pytest.fail("Checkpoint validation must come first"))
    with pytest.raises(FileNotFoundError, match="EdgeTAM checkpoint"):
        model.build_edgetam_predictor("nonexistent-custom-checkpoint.pt", "cpu")
