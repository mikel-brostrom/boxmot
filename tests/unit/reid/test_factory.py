"""ReID factories resolve shorthand references and retain explicit-spec contracts."""

from __future__ import annotations

import builtins
import copy
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import boxmot.reid.config as reid_config
import boxmot.reid.factory as reid_factory
from boxmot.reid import ReIDEncoderSpec, create_reid_encoder
from boxmot.reid.protocols import EncoderRequirements


class _Encoder:
    """Record construction without loading a model runtime."""

    embedding_dim = 8
    requirements = EncoderRequirements()

    def __init__(self, spec: ReIDEncoderSpec) -> None:
        self.spec = spec

    def encode(self, frames, detections):
        raise AssertionError("Factory tests must not run inference.")


@pytest.fixture
def capture_encoder(monkeypatch):
    constructed = []

    def factory(spec: ReIDEncoderSpec) -> _Encoder:
        constructed.append(spec)
        return _Encoder(spec)

    monkeypatch.setattr(reid_factory, "_REID_ENCODER_FACTORIES", SimpleNamespace(resolve=lambda _: factory))
    return constructed


@pytest.fixture
def profile(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Resolve the documented profile name against a tiny local artifact."""
    artifact = tmp_path / "osnet_x0_25_msmt17.pt"
    artifact.write_bytes(b"fixture appearance weights")
    path = tmp_path / "osnet-x0-25-msmt17.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "osnet-x0-25-msmt17",
                "weights": {"path": artifact.name},
                "runtime": {"device": "cuda:3", "precision": "fp16"},
                "preprocessing": {"mode": "resize", "image_size": [384, 128]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(reid_config, "REID_CONFIGS_DIR", tmp_path)
    return path, artifact


@pytest.mark.parametrize("reference_kind", ("name", "yaml", "artifact", "mapping", "component_yaml"))
def test_reid_factory_resolves_reference_and_preserves_authored_defaults(
    reference_kind: str, profile: tuple[Path, Path], capture_encoder
) -> None:
    config, artifact = profile
    payload = {
        "backend": "pytorch",
        "artifact": {"path": str(artifact)},
        "device": "cuda:3",
        "precision": "fp16",
        "preprocessing": "resize",
        "options": {"image_size": [384, 128]},
    }
    references = {"name": config.stem, "yaml": config, "artifact": artifact, "mapping": payload}
    if reference_kind == "component_yaml":
        reference = config.parent / "component.yaml"
        payload["artifact"]["path"] = artifact.name
        reference.write_text(yaml.safe_dump(payload), encoding="utf-8")
    else:
        reference = references[reference_kind]

    encoder = create_reid_encoder(reference, allow_download=False)

    assert capture_encoder == [encoder.spec]
    assert encoder.spec.backend == "pytorch"
    assert encoder.spec.artifact == str(artifact)
    assert encoder.spec.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert encoder.spec.device == "cuda:3"
    assert encoder.spec.precision == "fp16"
    assert encoder.spec.preprocessing == "resize"
    assert encoder.spec.option_values() == {"image_size": (384, 128)}


@pytest.mark.parametrize("explicit_spec", (False, True))
def test_reid_factory_keyword_overrides_merge_options_without_mutation(
    explicit_spec: bool, profile: tuple[Path, Path], capture_encoder
) -> None:
    _, artifact = profile
    authored = {
        "backend": "pytorch",
        "artifact": {"path": str(artifact)},
        "device": "cuda:3",
        "precision": "fp16",
        "preprocessing": "resize",
        "options": {"image_size": [384, 128], "batch_size": 16},
    }
    original = copy.deepcopy(authored)
    reference = reid_config.resolve_reid_spec(authored, allow_download=False)[0] if explicit_spec else authored
    overrides = {"image_size": [256, 128], "session_options": {"threads": 2}}

    encoder = create_reid_encoder(
        reference,
        device="cpu",
        precision="fp32",
        preprocessing="default",
        options=overrides,
        allow_download=False,
    )

    assert encoder.spec.device == "cpu"
    assert encoder.spec.precision == "fp32"
    assert encoder.spec.preprocessing == "default"
    assert encoder.spec.option_values() == {
        "batch_size": 16,
        "image_size": (256, 128),
        "session_options": (("threads", 2),),
    }
    assert authored == original
    if explicit_spec:
        assert reference.device == "cuda:3"
        assert reference.option_values() == {"batch_size": 16, "image_size": (384, 128)}
    overrides["image_size"][0] = 1
    overrides["session_options"]["threads"] = 99
    assert encoder.spec.option_values()["image_size"] == (256, 128)
    assert encoder.spec.option_values()["session_options"] == (("threads", 2),)


def test_reid_factory_documented_name_accepts_device_keyword(profile, capture_encoder) -> None:
    encoder = create_reid_encoder("osnet-x0-25-msmt17", device="cpu", allow_download=False)
    assert encoder.spec.device == "cpu"
    assert encoder.spec.precision == "fp16"
    assert encoder.spec.preprocessing == "resize"


def test_reid_factory_explicit_spec_does_not_import_or_call_resolver(profile, capture_encoder, monkeypatch) -> None:
    _, artifact = profile
    spec = ReIDEncoderSpec(
        backend="pytorch", artifact=str(artifact), artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "boxmot.reid.config":
            raise AssertionError("An explicit spec must not import its reference resolver.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert create_reid_encoder(spec=spec).spec is spec
    assert create_reid_encoder(spec, device="cuda:1").spec.device == "cuda:1"
    assert spec.device == "cpu"


def test_reid_factory_import_does_not_load_reference_configuration() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from boxmot.reid import create_reid_encoder; import sys; "
            "assert 'boxmot.reid.config' not in sys.modules; "
            "assert 'boxmot.components.resolution' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("allow_download", (False, True))
def test_reid_factory_controls_downloads_through_real_reference_resolution(
    tmp_path: Path, capture_encoder, monkeypatch, allow_download: bool
) -> None:
    artifact = tmp_path / "missing.pt"
    downloads = []

    def download(url: str, destination: Path) -> None:
        downloads.append((url, destination))
        destination.write_bytes(b"downloaded fixture weights")

    monkeypatch.setitem(sys.modules, "boxmot.resources.download", SimpleNamespace(download_file=download))
    reference = {"backend": "pytorch", "artifact": {"path": str(artifact), "uri": "https://example.test/reid.pt"}}
    if allow_download:
        encoder = create_reid_encoder(reference)  # Downloads are allowed by default.
        assert downloads == [("https://example.test/reid.pt", artifact)]
        assert encoder.spec.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    else:
        with pytest.raises(FileNotFoundError, match="Model artifact does not exist"):
            create_reid_encoder(reference, allow_download=False)
        assert not downloads
        assert not capture_encoder
        assert not artifact.exists()


@pytest.mark.parametrize("reference", (None, 123, False, [], object()))
def test_reid_factory_rejects_invalid_reference_types(reference, capture_encoder) -> None:
    with pytest.raises(TypeError, match="spec must be a ReIDEncoderSpec"):
        create_reid_encoder(reference)
    assert not capture_encoder


@pytest.mark.parametrize("kwargs", ({"options": []}, {"allow_download": "false"}))
def test_reid_factory_rejects_invalid_keyword_types_before_resolution(kwargs, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid factory keyword types must not resolve or download weights.")

    monkeypatch.setattr(reid_config, "resolve_reid_spec", forbidden)
    with pytest.raises(TypeError):
        create_reid_encoder("osnet-x0-25-msmt17", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"device": ""}, "device"),
        ({"precision": "int8"}, "precision"),
        ({"preprocessing": "not a mode"}, "preprocessing"),
        ({"options": {"BadKey": 1}}, "key"),
        ({"options": {"threshold": float("nan")}}, "non-finite"),
    ),
)
def test_reid_factory_validates_runtime_overrides(profile, capture_encoder, kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_reid_encoder("osnet-x0-25-msmt17", allow_download=False, **kwargs)
    assert not capture_encoder


def test_reid_factory_keeps_artifact_identity_validation(profile, capture_encoder) -> None:
    _, artifact = profile
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        create_reid_encoder(ReIDEncoderSpec("pytorch", str(artifact), "0" * 64), device="cpu")
    assert not capture_encoder


def test_reid_factory_rejects_unresolved_explicit_specs(capture_encoder) -> None:
    with pytest.raises(ValueError, match="resolved local artifact"):
        create_reid_encoder(ReIDEncoderSpec("pytorch"), allow_download=True)
    assert not capture_encoder
