from types import SimpleNamespace

import numpy as np
import torch

from boxmot.engine.reid import evaluator as evaluator_module


def test_eval_reid_overrides_inference_feature_and_writes_mode_json(monkeypatch, tmp_path):
    weights = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": {},
            "model_name": "csl_tinyvit_23m",
            "num_classes": 10,
            "preprocess": "resize_pad",
            "inference_feature": "concat_bn",
            "feat_dim": 512,
        },
        weights,
    )
    (tmp_path / "hparams.json").write_text(
        '{"img_size": [384, 128], "preprocess": "resize", "flip_tta": true}'
    )
    output_dir = tmp_path / "evals"

    class _Split:
        samples = [object()]

    class _Dataset:
        query = _Split()
        gallery = _Split()
        num_train_pids = 10

    class _TorchDataset:
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return torch.zeros(3, 384, 128), 1, index

    class _Head:
        inference_feature = "concat_bn"

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = _Head()

        def load_state_dict(self, state_dict, strict=False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    built = {}
    transforms = {}
    extracted = []

    def fake_build_model(name, weights_path, **kwargs):
        model = _Model()
        model.head.inference_feature = kwargs["inference_feature"]
        built.update(name=name, weights_path=weights_path, kwargs=kwargs, model=model)
        return model

    def fake_build_test_transforms(img_size, preprocess):
        transforms.update(img_size=img_size, preprocess=preprocess)
        return object()

    def fake_extract_features(model, dataloader, device, desc="Extracting", flip_tta=False):
        extracted.append((model.head.inference_feature, flip_tta))
        return (
            np.ones((1, 512), dtype=np.float32),
            np.asarray([1]),
            np.asarray([0 if len(extracted) == 1 else 1]),
        )

    monkeypatch.setattr(evaluator_module, "build_dataset", lambda dataset, data_dir: _Dataset())
    monkeypatch.setattr(evaluator_module, "ReIDImageDataset", _TorchDataset)
    monkeypatch.setattr(evaluator_module, "build_test_transforms", fake_build_test_transforms)
    monkeypatch.setattr(evaluator_module.ReIDModelRegistry, "build_model", fake_build_model)
    monkeypatch.setattr(evaluator_module, "extract_features", fake_extract_features)

    result = evaluator_module.main(
        SimpleNamespace(
            weights=str(weights),
            model=None,
            dataset="market1501",
            data_dir=str(tmp_path),
            preprocess=None,
            imgsz=None,
            inference_feature="raw_mean",
            flip_tta=None,
            device="cpu",
            batch_size=1,
            num_workers=0,
            output=str(output_dir),
        )
    )

    assert built["kwargs"]["inference_feature"] == "raw_mean"
    assert built["model"].head.inference_feature == "raw_mean"
    assert transforms == {"img_size": (384, 128), "preprocess": "resize_pad"}
    assert extracted == [("raw_mean", True), ("raw_mean", True)]
    assert result["inference_feature"] == "raw_mean"
    assert result["feature_dim"] == 512
    assert (output_dir / "eval_csl_tinyvit_23m_market1501_raw_mean.json").exists()
