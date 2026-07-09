from __future__ import annotations

from dataclasses import dataclass

from boxmot.reid.backbones import registered_backbone_names


@dataclass(frozen=True)
class ReIDExportFormat:
    name: str
    argument: str
    suffix: str
    cpu: bool
    gpu: bool


REID_EXPORT_FORMAT_COLUMNS = ("Format", "Argument", "Suffix", "CPU", "GPU")
REID_EXPORT_FORMATS = (
    ReIDExportFormat("PyTorch", "-", ".pt", True, True),
    ReIDExportFormat("TorchScript", "torchscript", ".torchscript", True, True),
    ReIDExportFormat("ONNX", "onnx", ".onnx", True, True),
    ReIDExportFormat("OpenVINO", "openvino", "_openvino_model", True, False),
    ReIDExportFormat("TensorRT", "engine", ".engine", False, True),
    ReIDExportFormat("TensorFlow Lite", "tflite", ".tflite", True, False),
)
REID_EXPORT_FORMAT_ROWS = tuple(
    (format_.name, format_.argument, format_.suffix, format_.cpu, format_.gpu)
    for format_ in REID_EXPORT_FORMATS
)
REID_EXPORT_ARGUMENTS = tuple(format_.argument for format_ in REID_EXPORT_FORMATS)
REID_EXPORT_SUFFIXES = tuple(format_.suffix for format_ in REID_EXPORT_FORMATS)

MODEL_TYPES = [*registered_backbone_names()]

TRAINED_URLS = {
    # resnet50
    "resnet50_market1501.pt": "https://drive.google.com/uc?id=1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV",
    "resnet50_dukemtmcreid.pt": "https://drive.google.com/uc?id=17ymnLglnc64NRvGOitY3BqMRS9UWd1wg",
    "resnet50_msmt17.pt": "https://drive.google.com/uc?id=1ep7RypVDOthCRIAqDnn4_N-UhkkFHJsj",
    "resnet50_fc512_market1501.pt": "https://drive.google.com/uc?id=1kv8l5laX_YCdIGVCetjlNdzKIA3NvsSt",
    "resnet50_fc512_dukemtmcreid.pt": "https://drive.google.com/uc?id=13QN8Mp3XH81GK4BPGXobKHKyTGH50Rtx",
    "resnet50_fc512_msmt17.pt": "https://drive.google.com/uc?id=1fDJLcz4O5wxNSUvImIIjoaIF9u1Rwaud",
    # mlfn
    "mlfn_market1501.pt": "https://drive.google.com/uc?id=1wXcvhA_b1kpDfrt9s2Pma-MHxtj9pmvS",
    "mlfn_dukemtmcreid.pt": "https://drive.google.com/uc?id=1rExgrTNb0VCIcOnXfMsbwSUW1h2L1Bum",
    "mlfn_msmt17.pt": "https://drive.google.com/uc?id=18JzsZlJb3Wm7irCbZbZ07TN4IFKvR6p-",
    # hacnn
    "hacnn_market1501.pt": "https://drive.google.com/uc?id=1LRKIQduThwGxMDQMiVkTScBwR7WidmYF",
    "hacnn_dukemtmcreid.pt": "https://drive.google.com/uc?id=1zNm6tP4ozFUCUQ7Sv1Z98EAJWXJEhtYH",
    "hacnn_msmt17.pt": "https://drive.google.com/uc?id=1MsKRtPM5WJ3_Tk2xC0aGOO7pM3VaFDNZ",
    # mobilenetv2
    "mobilenetv2_x1_0_market1501.pt": "https://drive.google.com/uc?id=18DgHC2ZJkjekVoqBWszD8_Xiikz-fewp",
    "mobilenetv2_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1q1WU2FETRJ3BXcpVtfJUuqq4z3psetds",
    "mobilenetv2_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1j50Hv14NOUAg7ZeB3frzfX-WYLi7SrhZ",
    "mobilenetv2_x1_4_market1501.pt": "https://drive.google.com/uc?id=1t6JCqphJG-fwwPVkRLmGGyEBhGOf2GO5",
    "mobilenetv2_x1_4_dukemtmcreid.pt": "https://drive.google.com/uc?id=12uD5FeVqLg9-AFDju2L7SQxjmPb4zpBN",
    "mobilenetv2_x1_4_msmt17.pt": "https://drive.google.com/uc?id=1ZY5P2Zgm-3RbDpbXM0kIBMPvspeNIbXz",
    # osnet
    "osnet_x1_0_market1501.pt": "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    "osnet_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",
    "osnet_x1_0_msmt17.pt": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "osnet_x0_75_market1501.pt": "https://drive.google.com/uc?id=1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer",
    "osnet_x0_75_dukemtmcreid.pt": "https://drive.google.com/uc?id=1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or",
    "osnet_x0_75_msmt17.pt": "https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc",
    "osnet_x0_5_market1501.pt": "https://drive.google.com/uc?id=1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT",
    "osnet_x0_5_dukemtmcreid.pt": "https://drive.google.com/uc?id=1KoUVqmiST175hnkALg9XuTi1oYpqcyTu",
    "osnet_x0_5_msmt17.pt": "https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv",
    "osnet_x0_25_market1501.pt": "https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj",
    "osnet_x0_25_dukemtmcreid.pt": "https://drive.google.com/uc?id=1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l",
    "osnet_x0_25_msmt17.pt": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    # osnet_ain | osnet_ibn
    "osnet_ibn_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ",
    "osnet_ain_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
    # lmbn
    "lmbn_n_duke.pt": "https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_duke.pt",
    "lmbn_n_market.pt": "https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_market.pt",
    "lmbn_n_cuhk03_d.pt": "https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_cuhk03_d.pt",
}

NR_CLASSES_DICT = {
    "market1501": 751,
    "market": 751,
    "duke": 702,
    "cuhk03": 767,
    "msmt17": 1041,
    "msmt17_merged": 4101,
    "veri": 576,
    "vehicleid": 576,
}
