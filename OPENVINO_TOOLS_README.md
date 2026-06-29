# OpenVINO 工具使用指南

本目录包含用于 YOLOv8 模型导出、验证和推理对比的工具脚本。

## 📦 核心脚本

### 1. fix_yolov8_openvino.py

**用途**: 自动导出 YOLOv8 模型为 ONNX 和 OpenVINO IR 格式

**功能**:
- 自动检查并下载 YOLOv8n 模型
- 导出 ONNX (FP32, opset=13)
- 导出 OpenVINO IR (FP32)
- 统一路径到 `exports/{model_name}/` 目录
- 验证输出一致性

**使用**:
```bash
python fix_yolov8_openvino.py
```

**输出**:
```
exports/yolov8n/
├── yolov8n.onnx
└── yolov8n_openvino/
    ├── model.xml
    └── model.bin
```

---

### 2. verify_paths.py

**用途**: 验证导出的模型路径是否正确

**功能**:
- 检查所有模型文件是否存在
- 验证目录结构
- 检查文件可访问性
- 提供修复建议

**使用**:
```bash
python verify_paths.py
```

**输出示例**:
```
验证模型: yolov8n
================================================================================

状态 类型            路径                                               大小      
--------------------------------------------------------------------------------
✓    PyTorch 模型    yolov8n.pt                                         6.2 MB    
✓    ONNX 模型       exports/yolov8n/yolov8n.onnx                      12.3 MB   
✓    OpenVINO XML    exports/yolov8n/yolov8n_openvino/model.xml       123.4 KB  
✓    OpenVINO BIN    exports/yolov8n/yolov8n_openvino/model.bin       12.1 MB   
--------------------------------------------------------------------------------

✓ 所有检查通过！路径配置正确。
```

---

### 3. compare_inference.py

**用途**: 对比 PyTorch、ONNX、OpenVINO 推理结果

**功能**:
- 加载并运行三种后端
- 对比输出差异（绝对差异、相对差异、余弦相似度）
- 性能基准测试（FPS、延迟）
- 可视化输出分布

**使用**:
```bash
# 对比所有后端
python compare_inference.py \
    --model yolov8n.pt \
    --onnx exports/yolov8n/yolov8n.onnx \
    --openvino exports/yolov8n/yolov8n_openvino/model.xml

# 只对比 PyTorch 和 OpenVINO
python compare_inference.py \
    --model yolov8n.pt \
    --openvino exports/yolov8n/yolov8n_openvino/model.xml

# 使用自定义图像
python compare_inference.py \
    --model yolov8n.pt \
    --openvino exports/yolov8n/yolov8n_openvino/model.xml \
    --image path/to/image.jpg
```

**输出示例**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PyTorch vs OpenVINO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Max Absolute Diff:     1.234e-04  ✓
Mean Absolute Diff:    3.456e-05  ✓
Max Relative Diff:     0.123%     ✓
Cosine Similarity:     0.99999    ✓
Status:                PASSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🚀 快速开始

### 完整工作流程

```bash
# 1. 导出模型
python fix_yolov8_openvino.py

# 2. 验证路径
python verify_paths.py

# 3. 对比推理
python compare_inference.py \
    --model yolov8n.pt \
    --onnx exports/yolov8n/yolov8n.onnx \
    --openvino exports/yolov8n/yolov8n_openvino/model.xml
```

### 常见使用场景

**场景 1: 首次导出模型**
```bash
python fix_yolov8_openvino.py
python verify_paths.py
```

**场景 2: 验证现有导出**
```bash
python verify_paths.py
python compare_inference.py --model yolov8n.pt --openvino exports/yolov8n/yolov8n_openvino/model.xml
```

**场景 3: 调试输出差异**
```bash
python compare_inference.py --model yolov8n.pt --openvino exports/yolov8n/yolov8n_openvino/model.xml --debug
```

---

## ⚙️ 依赖安装

```bash
pip install torch ultralytics onnx onnxruntime openvino
```

**版本要求**:
- Python >= 3.8
- PyTorch >= 2.0
- OpenVINO >= 2024.0
- Ultralytics >= 8.0

---

## 📋 验证标准

### 精度要求

| 指标 | 阈值 | 说明 |
|------|------|------|
| 最大绝对差异 | < 1e-3 | 输出数值的最大差异 |
| 平均绝对差异 | < 1e-4 | 输出数值的平均差异 |
| 余弦相似度 | > 0.999 | 输出向量的相似度 |

### 性能基准

| 指标 | YOLOv8n (640x640) |
|------|-------------------|
| PyTorch (CPU) | ~20 FPS |
| ONNX (CPU) | ~30 FPS |
| OpenVINO (CPU) | ~40 FPS |

---

## 🔍 故障排除

### 问题 1: FileNotFoundError

```
ONNXRuntimeError: NO_SUCHFILE
```

**解决**:
```bash
python verify_paths.py  # 检查路径
python fix_yolov8_openvino.py  # 重新导出
```

### 问题 2: 输出差异巨大

```
Max Absolute Diff: 1.234e+02  # 差异很大！
```

**可能原因**:
- FP16 精度问题
- 模型版本不匹配
- 转换参数错误

**解决**:
```bash
# 确保使用 FP32 导出
python fix_yolov8_openvino.py
```

### 问题 3: OVDict 类型错误

```
TypeError: 'OVDict' object does not support indexing
```

**原因**: OpenVINO 2024+ 返回类型变化

**解决**: 使用 `list(output.values())[0]` 而非 `output[0]`

---

## 📚 相关文档

- [OPENVINO_INTEGRATION_GUIDE.md](OPENVINO_INTEGRATION_GUIDE.md) - 完整集成指南
- [OpenVINO YOLOv8 官方文档](https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html)

---

## 💡 最佳实践

### 1. 始终使用 FP32

```python
# ONNX 导出
model.export(format='onnx', half=False)

# OpenVINO 导出
ov.save_model(model, path, compress_to_fp16=False)
```

### 2. 固定形状优于动态形状

```python
model.export(format='onnx', dynamic=False, imgsz=640)
```

### 3. 使用固定输入测试

```python
np.random.seed(42)
test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
```

### 4. 验证中间步骤

不要等到最后才发现问题：
```bash
python fix_yolov8_openvino.py  # 导出时自动验证
python verify_paths.py          # 验证路径
python compare_inference.py     # 验证输出
```

---

**最后更新**: 2026-06-29  
**维护者**: BoxMOT Contributors
