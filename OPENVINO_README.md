# BoxMOT OpenVINO 集成

本目录包含 BoxMOT 项目中 OpenVINO 模型导出和推理的完整工具链。

## 📁 文件结构

```
boxmot/
├── fix_yolov8_openvino.py          # 模型导出工具
├── verify_paths.py                  # 路径验证工具
├── compare_inference.py             # 推理对比工具
├── OPENVINO_INTEGRATION_GUIDE.md    # 完整集成指南 ⭐
├── OPENVINO_TOOLS_README.md         # 工具使用指南
└── dev_docs/                        # 开发文档
    ├── PATH_STRUCTURE.md
    ├── YOLOV8_vs_YOLO11_FIX.md
    └── ...
```

## 🚀 快速开始

### 1. 导出模型

```bash
python fix_yolov8_openvino.py
```

这将自动：
- 下载 YOLOv8n 模型（如果不存在）
- 导出 ONNX (FP32)
- 导出 OpenVINO IR (FP32)
- 验证输出一致性

### 2. 验证路径

```bash
python verify_paths.py
```

### 3. 对比推理

```bash
python compare_inference.py \
    --model yolov8n.pt \
    --openvino exports/yolov8n/yolov8n_openvino/model.xml
```

## 📖 文档导航

| 文档 | 用途 |
|------|------|
| **[OPENVINO_INTEGRATION_GUIDE.md](OPENVINO_INTEGRATION_GUIDE.md)** | **完整集成指南**（推荐阅读） |
| [OPENVINO_TOOLS_README.md](OPENVINO_TOOLS_README.md) | 工具使用说明 |
| [dev_docs/PATH_STRUCTURE.md](dev_docs/PATH_STRUCTURE.md) | 路径结构规范 |
| [dev_docs/YOLOV8_vs_YOLO11_FIX.md](dev_docs/YOLOV8_vs_YOLO11_FIX.md) | 模型版本说明 |

## 🎯 核心问题与解决方案

### 问题：OpenVINO 输出与 PyTorch 相差巨大

**根本原因**：
1. FP16 精度损失（默认导出使用 half=True）
2. 模型版本错误（使用 YOLO11n 而非 YOLOv8n）
3. 路径不一致（导出路径 ≠ 加载路径）

**解决方案**：
1. 使用 FP32 导出：`half=False`, `compress_to_fp16=False`
2. 使用正确的 YOLOv8n 模型
3. 统一路径到 `exports/{model_name}/` 结构

详见 [OPENVINO_INTEGRATION_GUIDE.md](OPENVINO_INTEGRATION_GUIDE.md)

## 📊 统一路径结构

```
boxmot/
├── yolov8n.pt                          # PyTorch 模型
└── exports/                            # 统一导出目录
    └── yolov8n/                        # 模型名称目录
        ├── yolov8n.onnx                # ONNX 模型
        └── yolov8n_openvino/           # OpenVINO 模型
            ├── model.xml
            └── model.bin
```

## ⚙️ 依赖安装

```bash
pip install torch ultralytics onnx onnxruntime openvino
```

## 📚 参考资源

- [OpenVINO YOLOv8 官方文档](https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html)
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/models/yolov8/)

---

**最后更新**: 2026-06-29
