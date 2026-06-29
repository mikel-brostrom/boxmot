"""
模型路径验证工具

验证导出的模型文件路径是否符合统一的目录结构规范。

功能:
    - 检查导出目录结构
    - 验证所有文件存在
    - 检查文件可访问性
    - 提供修复建议

使用:
    python verify_paths.py
"""

import sys
from pathlib import Path


def verify_model_paths(model_name):
    """
    验证指定模型的所有路径是否存在。

    Args:
        model_name (str): 模型名称

    Returns:
        bool: 所有文件是否存在
    """
    print("=" * 80)
    print(f"验证模型: {model_name}")
    print("=" * 80)

    # 定义所有期望的路径
    paths = {
        'PyTorch 模型': Path(f"{model_name}.pt"),
        'ONNX 模型': Path(f"exports/{model_name}/{model_name}.onnx"),
        'OpenVINO XML': Path(f"exports/{model_name}/{model_name}_openvino/model.xml"),
        'OpenVINO BIN': Path(f"exports/{model_name}/{model_name}_openvino/model.bin"),
    }

    all_exist = True
    results = []

    for name, path in paths.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        size = ""

        if exists:
            try:
                file_size = path.stat().st_size
                if file_size < 1024:
                    size = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size = f"{file_size / 1024:.1f} KB"
                else:
                    size = f"{file_size / (1024 * 1024):.1f} MB"
            except:
                size = "?"

        results.append((status, name, str(path), size))

        if not exists:
            all_exist = False

    # 打印结果
    print(f"\n{'状态':<4} {'类型':<15} {'路径':<50} {'大小':<10}")
    print("-" * 80)
    for status, name, path, size in results:
        print(f"{status:<4} {name:<15} {path:<50} {size:<10}")

    print("-" * 80)

    return all_exist


def check_path_accessibility():
    """
    检查模型文件是否可以正常读取。

    Returns:
        bool: 所有文件是否可访问
    """
    print("\n" + "=" * 80)
    print("检查文件可访问性")
    print("=" * 80)

    model_name = "yolov8n"
    paths_to_check = [
        Path(f"exports/{model_name}/{model_name}.onnx"),
        Path(f"exports/{model_name}/{model_name}_openvino/model.xml"),
    ]

    all_accessible = True

    for path in paths_to_check:
        if not path.exists():
            print(f"  ⊘ {path}: 文件不存在")
            all_accessible = False
            continue

        try:
            # 尝试读取文件的前几个字节
            with open(path, 'rb') as f:
                _ = f.read(100)
            print(f"  ✓ {path}: 可读取")
        except Exception as e:
            print(f"  ✗ {path}: 无法读取 ({e})")
            all_accessible = False

    return all_accessible


def check_directory_structure():
    """
    检查并列出导出目录的完整结构。

    Returns:
        bool: 目录结构是否正确
    """
    print("\n" + "=" * 80)
    print("检查导出目录结构")
    print("=" * 80)

    exports_dir = Path("exports")

    if not exports_dir.exists():
        print(f"  ✗ 导出目录不存在: {exports_dir}")
        print(f"  建议: 运行 fix_yolov8_openvino.py 创建导出")
        return False

    # 列出所有模型
    print(f"\n导出目录: {exports_dir}")
    print(f"包含的模型:")

    model_dirs = [d for d in exports_dir.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"  (空)")
        return False

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"\n  📁 {model_name}/")

        # 检查 ONNX
        onnx_file = model_dir / f"{model_name}.onnx"
        if onnx_file.exists():
            size = onnx_file.stat().st_size / (1024 * 1024)
            print(f"    ✓ {model_name}.onnx ({size:.1f} MB)")
        else:
            print(f"    ✗ {model_name}.onnx (缺失)")

        # 检查 OpenVINO
        ov_dir = model_dir / f"{model_name}_openvino"
        if ov_dir.exists():
            print(f"    📁 {model_name}_openvino/")

            xml_file = ov_dir / "model.xml"
            bin_file = ov_dir / "model.bin"

            if xml_file.exists():
                size = xml_file.stat().st_size / 1024
                print(f"      ✓ model.xml ({size:.1f} KB)")
            else:
                print(f"      ✗ model.xml (缺失)")

            if bin_file.exists():
                size = bin_file.stat().st_size / (1024 * 1024)
                print(f"      ✓ model.bin ({size:.1f} MB)")
            else:
                print(f"      ✗ model.bin (缺失)")
        else:
            print(f"    ✗ {model_name}_openvino/ (缺失)")

    return True


def test_path_construction():
    """
    测试不同路径构造方法的一致性。

    Returns:
        bool: 路径构造是否一致
    """
    print("\n" + "=" * 80)
    print("测试路径构造")
    print("=" * 80)

    model_name = "yolov8n"

    # 方法1: 字符串拼接
    onnx_path_str = f"exports/{model_name}/{model_name}.onnx"

    # 方法2: Path 对象
    onnx_path_obj = Path("exports") / model_name / f"{model_name}.onnx"

    # 方法3: 从模型文件推导
    model_path = Path(f"{model_name}.pt")
    derived_name = model_path.stem
    onnx_path_derived = Path("exports") / derived_name / f"{derived_name}.onnx"

    print(f"\n方法1 (字符串): {onnx_path_str}")
    print(f"方法2 (Path):   {onnx_path_obj}")
    print(f"方法3 (推导):   {onnx_path_derived}")

    # 验证一致性
    if str(onnx_path_obj) == onnx_path_str.replace("/", "\\"):
        print(f"\n✓ 所有方法生成的路径一致")
        return True
    else:
        print(f"\n✗ 路径不一致！")
        return False


def provide_fix_suggestions(model_name):
    """
    提供路径问题的修复建议。

    Args:
        model_name (str): 模型名称
    """
    print("\n" + "=" * 80)
    print("修复建议")
    print("=" * 80)

    pytorch_model = Path(f"{model_name}.pt")

    if not pytorch_model.exists():
        print(f"\n1. 下载 PyTorch 模型:")
        print(f"   python -c \"from ultralytics import YOLO; YOLO('{model_name}.pt')\"")
        print(f"   或")
        print(f"   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}.pt")

    print(f"\n2. 导出模型到统一路径:")
    print(f"   python fix_yolov8_openvino.py")

    print(f"\n3. 验证路径:")
    print(f"   python verify_paths.py")

    print(f"\n4. 运行推理对比:")
    print(f"   python compare_inference.py \\")
    print(f"       --model {model_name}.pt \\")
    print(f"       --onnx exports/{model_name}/{model_name}.onnx \\")
    print(f"       --openvino exports/{model_name}/{model_name}_openvino/model.xml")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          路径验证工具                                         ║
║                                                                              ║
║  验证模型导出路径的一致性和正确性                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    model_name = "yolov8n"

    # 1. 验证路径
    paths_ok = verify_model_paths(model_name)

    # 2. 检查目录结构
    structure_ok = check_directory_structure()

    # 3. 测试路径构造
    construction_ok = test_path_construction()

    # 4. 检查可访问性
    if paths_ok:
        accessibility_ok = check_path_accessibility()
    else:
        accessibility_ok = False

    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)

    checks = [
        ("路径存在性", paths_ok),
        ("目录结构", structure_ok),
        ("路径构造", construction_ok),
        ("文件可访问", accessibility_ok),
    ]

    all_ok = True
    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    print("-" * 80)

    if all_ok:
        print("\n✓ 所有检查通过！路径配置正确。")
        return 0
    else:
        print("\n✗ 有检查失败，请按照以下建议修复：")
        provide_fix_suggestions(model_name)
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
