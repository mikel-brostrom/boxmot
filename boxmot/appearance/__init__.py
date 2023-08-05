# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import pandas as pd


def export_formats():
    # yolo tracking export formats
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])
