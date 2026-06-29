"""
YOLOv8 OpenVINO 模型导出工具

自动导出 YOLOv8 模型为 ONNX 和 OpenVINO IR 格式，确保使用 FP32 精度。

参考:
    https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html

功能:
    - 自动下载 YOLOv8n 模型（如果不存在）
    - 导出 ONNX 格式（FP32, opset=13）
    - 导出 OpenVINO IR 格式（FP32）
    - 统一路径到 exports/{model_name}/ 目录
    - 验证输出一致性

使用:
    python fix_yolov8_openvino.py
"""

import sys
from pathlib import Path
import shutil
import numpy as np


def check_model_availability():
    """
    检查 YOLOv8n 模型是否可用，如果不存在则自动下载。

    Returns:
        str: 模型路径，如果下载失败则返回 None
    """
    print("=" * 80)
    print("步骤 0: 检查模型可用性")
    print("=" * 80)

    model_path = Path("yolov8n.pt")

    if model_path.exists():
        print(f"  ✓ 找到模型: {model_path}")
        return str(model_path)

    print(f"  ℹ 模型不存在，尝试自动下载...")

    try:
        from ultralytics import YOLO

        print(f"  正在下载 yolov8n.pt...")
        model = YOLO("yolov8n.pt")
        print(f"  ✓ 模型下载成功: yolov8n.pt")
        return "yolov8n.pt"

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print(f"\n  请手动下载:")
        print(f"    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        return None


def backup_existing_models():
    """
    备份现有的导出模型到 exports_backup 目录。

    Returns:
        bool: 备份是否成功（失败时继续执行）
    """
    print("\n" + "=" * 80)
    print("步骤 1: 备份现有模型")
    print("=" * 80)

    backup_dir = Path("exports_backup")
    exports_dir = Path("exports/yolov8n")

    if not exports_dir.exists():
        print(f"  ℹ 导出目录不存在: {exports_dir}")
        return True

    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
            print(f"  已删除旧备份: {backup_dir}")

        shutil.copytree(exports_dir, backup_dir / "yolov8n")
        print(f"  ✓ 备份完成: {backup_dir / 'yolov8n'}")
        return True

    except Exception as e:
        print(f"  ⚠ 备份失败: {e}")
        print(f"  继续导出...")
        return True


def export_onnx_fp32(model_path):
    """
    导出 ONNX 模型，使用 FP32 精度。

    Args:
        model_path (str): PyTorch 模型路径

    Returns:
        str: 导出的 ONNX 模型路径，失败返回 False
    """
    print("\n" + "=" * 80)
    print("步骤 2: 导出 ONNX 模型 (FP32)")
    print("=" * 80)

    if not Path(model_path).exists():
        print(f"  ✗ 模型文件不存在: {model_path}")
        return False

    try:
        from ultralytics import YOLO

        print(f"  加载模型: {model_path}")
        model = YOLO(str(model_path))

        model_name = Path(model_path).stem
        export_dir = Path("exports") / model_name
        export_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = export_dir / f"{model_name}.onnx"

        print(f"  导出 ONNX...")
        print(f"    - 目标路径: {onnx_path}")
        print(f"    - format: onnx")
        print(f"    - opset: 13 (YOLOv8 推荐)")
        print(f"    - simplify: True")
        print(f"    - half: False (强制 FP32)")
        print(f"    - dynamic: False (固定形状)")
        temp_onnx_path = model.export(
            format='onnx',
            opset=13,
            simplify=True,
            half=False,
            dynamic=False,
            imgsz=640,
        )

        temp_onnx = Path(temp_onnx_path)
        if temp_onnx != onnx_path:
            if onnx_path.exists():
                onnx_path.unlink()
            shutil.move(str(temp_onnx), str(onnx_path))
            print(f"  已移动到: {onnx_path}")

        print(f"  ✓ ONNX 导出成功: {onnx_path}")

        print(f"\n  验证 ONNX 模型:")
        import onnx
        onnx_model = onnx.load(str(onnx_path))

        print(f"    输入:")
        for inp in onnx_model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else '?' for d in inp.type.tensor_type.shape.dim]
            print(f"      名称: {inp.name}, 形状: {shape}, dtype: {inp.type.tensor_type.elem_type}")

        print(f"    输出:")
        for out in onnx_model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else '?' for d in out.type.tensor_type.shape.dim]
            print(f"      名称: {out.name}, 形状: {shape}")

        print(f"    权重数据类型 (前3个):")
        for i, init in enumerate(onnx_model.graph.initializer[:3]):
            dtype_name = "FP32" if init.data_type == 1 else "FP16" if init.data_type == 10 else f"Unknown({init.data_type})"
            print(f"      [{i}] {init.name[:40]:40s}: {dtype_name}")

        return str(onnx_path)

    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_openvino_fp32(onnx_path):
    """
    从 ONNX 导出 OpenVINO IR 模型，使用 FP32 精度。

    Args:
        onnx_path (str): ONNX 模型路径

    Returns:
        str: OpenVINO IR XML 路径，失败返回 False

    参考:
        https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html
    """
    print("\n" + "=" * 80)
    print("步骤 3: 导出 OpenVINO 模型 (FP32)")
    print("=" * 80)

    if not onnx_path:
        print(f"  ✗ ONNX 路径无效")
        return False

    try:
        import openvino as ov

        print(f"  OpenVINO 版本: {ov.__version__}")
        print(f"  输入模型: {onnx_path}")

        onnx_path = Path(onnx_path)
        model_name = onnx_path.stem
        output_dir = onnx_path.parent / f"{model_name}_openvino"
        output_dir.mkdir(parents=True, exist_ok=True)
        xml_path = output_dir / "model.xml"

        print(f"  输出目录: {output_dir}")
        print(f"  参数:")
        print(f"    - input shape: [1, 3, 640, 640] (固定)")
        print(f"    - compress_to_fp16: False (强制 FP32)")

        print(f"\n  读取并转换 ONNX 模型...")
        core = ov.Core()

        model = ov.convert_model(
            str(onnx_path),
            input=[("images", [1, 3, 640, 640])],
        )

        print(f"  ✓ 模型转换成功")
        print(f"    输入: {model.inputs[0].partial_shape} (名称: {model.inputs[0].any_name})")
        print(f"    输出: {model.outputs[0].partial_shape} (名称: {model.outputs[0].any_name})")
        print(f"    输入类型: {model.inputs[0].element_type}")
        print(f"    输出类型: {model.outputs[0].element_type}")

        print(f"\n  保存 OpenVINO IR...")
        try:
            ov.save_model(model, str(xml_path), compress_to_fp16=False)
            print(f"  ✓ 已保存为 FP32 精度")
        except TypeError:
            ov.save_model(model, str(xml_path))
            print(f"  ✓ 已保存 (OpenVINO 版本不支持精度选项)")

        print(f"  ✓ OpenVINO 导出成功: {xml_path}")

        print(f"\n  验证 OpenVINO 模型:")
        loaded_model = core.read_model(str(xml_path))
        compiled_model = core.compile_model(loaded_model, "CPU")

        print(f"    输入端口: {len(compiled_model.inputs)}")
        print(f"    输出端口: {len(compiled_model.outputs)}")
        print(f"    输入类型: {loaded_model.inputs[0].element_type}")
        print(f"    输出类型: {loaded_model.outputs[0].element_type}")

        return str(xml_path)

    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_yolov8_export():
    """
    验证导出的 YOLOv8 模型，对比 PyTorch、ONNX、OpenVINO 输出。

    Returns:
        bool: 验证是否通过

    参考:
        https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html
    """
    print("\n" + "=" * 80)
    print("步骤 4: 验证导出的模型")
    print("=" * 80)

    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
        import onnxruntime as ort
        import openvino as ov

        onnx_path = Path("exports/yolov8n/yolov8n.onnx")
        openvino_path = Path("exports/yolov8n/yolov8n_openvino/model.xml")

        if not onnx_path.exists():
            print(f"\n  ✗ ONNX 模型不存在: {onnx_path}")
            return False

        if not openvino_path.exists():
            print(f"\n  ✗ OpenVINO 模型不存在: {openvino_path}")
            return False

        np.random.seed(42)
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

        print(f"\n  使用固定随机输入: {test_input.shape}")
        print(f"    范围: [{test_input.min():.3f}, {test_input.max():.3f}]")

        outputs = {}

        print(f"\n  1. PyTorch 推理:")
        yolo = YOLO("yolov8n.pt")
        model = yolo.model.eval()
        with torch.no_grad():
            torch_output = model(torch.from_numpy(test_input))

        if isinstance(torch_output, (list, tuple)):
            torch_output = torch_output[0]

        torch_np = torch_output.detach().cpu().numpy()
        outputs['pytorch'] = torch_np

        print(f"     输出形状: {torch_np.shape}")
        print(f"     输出范围: [{torch_np.min():.6f}, {torch_np.max():.6f}]")
        print(f"     输出均值: {torch_np.mean():.6f}")
        print(f"     输出标准差: {torch_np.std():.6f}")

        print(f"     YOLOv8 输出格式:")
        if len(torch_np.shape) == 3:
            batch, channels, num_predictions = torch_np.shape
            print(f"       批次: {batch}, 通道: {channels}, 预测数: {num_predictions}")
            print(f"       格式: [batch, bbox(4)+classes({channels-4}), predictions]")

        print(f"\n  2. ONNX 推理:")
        print(f"     加载模型: {onnx_path}")
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: test_input})
        onnx_np = onnx_output[0]
        outputs['onnx'] = onnx_np

        print(f"     输出形状: {onnx_np.shape}")
        print(f"     输出范围: [{onnx_np.min():.6f}, {onnx_np.max():.6f}]")
        print(f"     输出均值: {onnx_np.mean():.6f}")
        print(f"     输出标准差: {onnx_np.std():.6f}")

        print(f"\n  3. OpenVINO 推理:")
        print(f"     加载模型: {openvino_path}")
        core = ov.Core()
        ov_model = core.read_model(str(openvino_path))
        compiled_model = core.compile_model(ov_model, "CPU")
        ov_output = compiled_model(test_input)
        ov_np = list(ov_output.values())[0]
        outputs['openvino'] = ov_np

        print(f"     输出形状: {ov_np.shape}")
        print(f"     输出范围: [{ov_np.min():.6f}, {ov_np.max():.6f}]")
        print(f"     输出均值: {ov_np.mean():.6f}")
        print(f"     输出标准差: {ov_np.std():.6f}")

        print(f"\n  差异分析:")

        diff_onnx = np.abs(torch_np - onnx_np)
        rel_diff_onnx = diff_onnx / (np.abs(torch_np) + 1e-8)
        cos_sim_onnx = np.dot(torch_np.flatten(), onnx_np.flatten()) / (
            np.linalg.norm(torch_np.flatten()) * np.linalg.norm(onnx_np.flatten())
        )

        print(f"    PyTorch vs ONNX:")
        print(f"      最大绝对差异: {diff_onnx.max():.6e}")
        print(f"      平均绝对差异: {diff_onnx.mean():.6e}")
        print(f"      最大相对差异: {rel_diff_onnx.max():.6e}")
        print(f"      余弦相似度: {cos_sim_onnx:.8f}")
        status_onnx = "✓ 正常" if diff_onnx.max() < 1e-3 else "⚠ 差异较大"
        print(f"      状态: {status_onnx}")

        diff_ov = np.abs(torch_np - ov_np)
        rel_diff_ov = diff_ov / (np.abs(torch_np) + 1e-8)
        cos_sim_ov = np.dot(torch_np.flatten(), ov_np.flatten()) / (
            np.linalg.norm(torch_np.flatten()) * np.linalg.norm(ov_np.flatten())
        )

        print(f"    PyTorch vs OpenVINO:")
        print(f"      最大绝对差异: {diff_ov.max():.6e}")
        print(f"      平均绝对差异: {diff_ov.mean():.6e}")
        print(f"      最大相对差异: {rel_diff_ov.max():.6e}")
        print(f"      余弦相似度: {cos_sim_ov:.8f}")
        status_ov = "✓ 正常" if diff_ov.max() < 1e-3 else "⚠ 差异较大"
        print(f"      状态: {status_ov}")

        if diff_ov.max() < 1e-3 and cos_sim_ov > 0.999:
            print(f"\n  ✓✓✓ 修复成功！OpenVINO 输出与 PyTorch 一致")
            return True
        elif diff_ov.max() < 0.1:
            print(f"\n  ⚠ 差异在可接受范围，但不是完美一致")
            print(f"    可能原因: 数值计算的微小差异")
            return True
        else:
            print(f"\n  ⚠⚠⚠ 差异仍然较大")
            print(f"\n  可能原因:")

            if diff_onnx.max() < 1e-3:
                print(f"    - ONNX 正常，问题在 ONNX → OpenVINO 转换")
                print(f"    - 检查 OpenVINO 版本: {ov.__version__}")
                print(f"    - 尝试更新: pip install --upgrade openvino")
            else:
                print(f"    - PyTorch → ONNX 转换就有问题")
                print(f"    - 检查 Ultralytics 版本")
                print(f"    - 尝试更新: pip install --upgrade ultralytics")

            print(f"\n  详细诊断:")
            if np.allclose(ov_np, 0):
                print(f"    ✗ OpenVINO 输出全为 0（模型未正确加载）")
            if np.isnan(ov_np).any():
                print(f"    ✗ OpenVINO 输出包含 NaN")
            if np.isinf(ov_np).any():
                print(f"    ✗ OpenVINO 输出包含 Inf")

            return False

    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_yolov8_info():
    """打印 YOLOv8 相关信息"""
    print("\n" + "=" * 80)
    print("YOLOv8 相关信息")
    print("=" * 80)

    print("""
YOLOv8 输出格式:
  - 形状: [batch, 84, 8400] (COCO 数据集)
  - 84 = 4 (bbox坐标) + 80 (类别)
  - 8400 = 80x80 + 40x40 + 20x20 的 anchor points

OpenVINO 推荐设置:
  - Opset: 13
  - 精度: FP32
  - 动态形状: False

参考文档:
  https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html
    """)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     修复 YOLOv8 OpenVINO 模型导出                           ║
║                                                                              ║
║  参考: OpenVINO 官方 YOLOv8 文档                                            ║
║  https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output║
║                                                                              ║
║  本脚本将:                                                                   ║
║  1. 检查并下载 yolov8n 模型                                                 ║
║  2. 备份现有模型                                                             ║
║  3. 重新导出 ONNX (FP32, opset=13)                                          ║
║  4. 重新导出 OpenVINO (FP32)                                                 ║
║  5. 验证输出一致性                                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # 检查依赖
    print("检查依赖...")
    try:
        import torch
        import onnx
        import onnxruntime
        import openvino
        from ultralytics import YOLO
        print("  ✓ 所有依赖已安装")
        print(f"    - torch: {torch.__version__}")
        print(f"    - onnx: {onnx.__version__}")
        print(f"    - onnxruntime: {onnxruntime.__version__}")
        print(f"    - openvino: {openvino.__version__}")
        try:
            import ultralytics
            print(f"    - ultralytics: {ultralytics.__version__}")
        except:
            pass
    except ImportError as e:
        print(f"  ✗ 缺少依赖: {e}")
        print(f"\n请安装:")
        print(f"  pip install torch onnx onnxruntime openvino ultralytics")
        return 1

    # 检查模型
    model_path = check_model_availability()
    if not model_path:
        return 1

    # 执行修复流程
    if not backup_existing_models():
        print("\n备份失败，但继续执行...")

    onnx_path = export_onnx_fp32(model_path)
    if not onnx_path:
        print("\n✗ ONNX 导出失败，终止")
        return 1

    xml_path = export_openvino_fp32(onnx_path)
    if not xml_path:
        print("\n✗ OpenVINO 导出失败，终止")
        return 1

    success = verify_yolov8_export()

    print_yolov8_info()

    print("\n" + "=" * 80)
    if success:
        print("✓ 修复完成！")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 使用 compare_inference.py 进行完整对比:")
        print("     python compare_inference.py \\")
        print("         --model yolov8n.pt \\")
        print("         --onnx exports/yolov8n/yolov8n.onnx \\")
        print("         --openvino exports/yolov8n/yolov8n_openvino/model.xml")
        print("\n  2. 使用真实数据测试 E2E pipeline")
        print("\n  3. 如果使用项目代码，确保后处理逻辑与 YOLOv8 匹配")
        return 0
    else:
        print("⚠ 修复可能不完全")
        print("=" * 80)
        print("\n建议:")
        print("  1. 更新依赖:")
        print("     pip install --upgrade ultralytics openvino onnxruntime")
        print("  2. 检查 YOLOv8 后处理逻辑")
        print("  3. 参考官方文档:")
        print("     https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
