"""
简化版 YOLOv8 PyTorch vs OpenVINO 性能和结果对比脚本

基于 OpenVINO 官方 notebook:
https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html

功能:
1. PyTorch 推理（性能 + 结果）
2. OpenVINO 推理（性能 + 结果）
3. 性能对比（FPS、延迟）
4. 结果对比（数值一致性）
5. YOLOv8 后处理（NMS、置信度过滤）
6. 完整的检测结果可视化

使用:
    # 使用默认路径（假设已经运行 fix_yolov8_openvino.py）
    python yolov8_benchmark_simple.py

    # 自定义路径
    python yolov8_benchmark_simple.py --model yolov8n.pt --openvino exports/yolov8n/yolov8n_openvino/model.xml

    # 使用真实图片
    python yolov8_benchmark_simple.py --image path/to/image.jpg

    # 可视化结果（包含完整后处理）
    python yolov8_benchmark_simple.py --image path/to/image.jpg --visualize
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# COCO 数据集类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def postprocess_yolov8(output, conf_threshold=0.25, iou_threshold=0.45, max_det=300):
    """
    YOLOv8 后处理

    参考 OpenVINO 官方文档:
    https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html

    Args:
        output: YOLOv8 原始输出 [1, 84, 8400]
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值
        max_det: 最大检测数量

    Returns:
        detections: List of [x1, y1, x2, y2, conf, cls_id]
    """
    # 移除 batch 维度
    output = output[0]  # [84, 8400]

    # 分离 bbox 和类别分数
    boxes = output[:4, :]  # [4, 8400] - cx, cy, w, h
    scores = output[4:, :]  # [80, 8400] - class scores

    # 获取每个预测的最大类别置信度和类别 ID
    max_scores = scores.max(axis=0)  # [8400]
    max_classes = scores.argmax(axis=0)  # [8400]

    # 过滤低置信度的检测
    mask = max_scores > conf_threshold
    boxes = boxes[:, mask]  # [4, N]
    max_scores = max_scores[mask]  # [N]
    max_classes = max_classes[mask]  # [N]

    if boxes.shape[1] == 0:
        return []

    # 转换 bbox 格式: [cx, cy, w, h] -> [x1, y1, x2, y2]
    cx, cy, w, h = boxes
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=0).T  # [N, 4]

    # NMS (Non-Maximum Suppression)
    keep_indices = nms(boxes_xyxy, max_scores, iou_threshold)

    # 限制最大检测数量
    if len(keep_indices) > max_det:
        keep_indices = keep_indices[:max_det]

    # 组装最终检测结果
    detections = []
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        conf = max_scores[idx]
        cls_id = max_classes[idx]
        detections.append([x1, y1, x2, y2, conf, cls_id])

    return detections


def nms(boxes, scores, iou_threshold):
    """
    Non-Maximum Suppression (NMS)

    Args:
        boxes: [N, 4] - [x1, y1, x2, y2]
        scores: [N] - confidence scores
        iou_threshold: IoU threshold

    Returns:
        keep_indices: List of indices to keep
    """
    # 按置信度排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # 计算当前 box 与其他 box 的 IoU
        ious = compute_iou(boxes[i], boxes[order[1:]])

        # 保留 IoU 小于阈值的 box
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return keep


def compute_iou(box, boxes):
    """
    计算一个 box 与多个 boxes 的 IoU

    Args:
        box: [4] - [x1, y1, x2, y2]
        boxes: [N, 4] - [x1, y1, x2, y2]

    Returns:
        ious: [N] - IoU values
    """
    # 计算交集
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # 计算并集
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    # 计算 IoU
    ious = intersection / (union + 1e-6)

    return ious


def draw_detections(image, detections, class_names=COCO_CLASSES, input_size=(640, 640)):
    """
    在图像上绘制检测结果

    Args:
        image: 原始图像 (BGR)
        detections: List of [x1, y1, x2, y2, conf, cls_id]
        class_names: 类别名称列表
        input_size: 模型输入尺寸（用于坐标转换）

    Returns:
        image_with_boxes: 绘制了检测框的图像
    """
    img_viz = image.copy()
    img_h, img_w = img_viz.shape[:2]
    input_h, input_w = input_size

    # 计算缩放比例
    scale_x = img_w / input_w
    scale_y = img_h / input_h

    # 为不同类别生成颜色
    np.random.seed(42)
    colors = {}
    for i in range(len(class_names)):
        colors[i] = tuple(map(int, np.random.randint(0, 255, 3)))

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)

        # 转换坐标到原始图像尺寸
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # 获取类别名称和颜色
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        color = colors.get(cls_id, (0, 255, 0))

        # 绘制边界框
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_viz, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # 绘制标签文本
        cv2.putText(img_viz, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img_viz


class YOLOv8Benchmark:
    """简化版 YOLOv8 PyTorch vs OpenVINO 性能对比"""

    def __init__(self, model_path, openvino_path, input_size=(640, 640)):
        self.model_path = model_path
        self.openvino_path = openvino_path
        self.input_size = input_size

        # 模型
        self.pytorch_model = None
        self.openvino_model = None

        # 结果
        self.results = {}

    def load_models(self):
        """加载 PyTorch 和 OpenVINO 模型"""
        print("=" * 80)
        print("1. 加载模型")
        print("=" * 80)

        # PyTorch
        print(f"\n📦 加载 PyTorch 模型: {self.model_path}")
        try:
            from ultralytics import YOLO
            yolo = YOLO(self.model_path)
            self.pytorch_model = yolo.model.eval()
            print("  ✓ PyTorch 模型加载成功")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            raise

        # OpenVINO
        print(f"\n📦 加载 OpenVINO 模型: {self.openvino_path}")
        try:
            import openvino as ov
            core = ov.Core()
            model = core.read_model(self.openvino_path)
            self.openvino_model = core.compile_model(model, "CPU")
            print("  ✓ OpenVINO 模型加载成功")
            print(f"  设备: CPU")
            print(f"  输入: {model.inputs[0].partial_shape}")
            print(f"  输出: {model.outputs[0].partial_shape}")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            raise

    def prepare_input(self, image_path=None):
        """准备输入数据"""
        print("\n" + "=" * 80)
        print("2. 准备输入")
        print("=" * 80)

        if image_path and Path(image_path).exists():
            print(f"\n📷 使用真实图片: {image_path}")
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  ✗ 无法读取图片，使用随机输入")
                return self._prepare_random_input()

            img_resized = cv2.resize(img, self.input_size)

            # 转换为 RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # 归一化到 [0, 1]
            img_normalized = img_rgb.astype(np.float32) / 255.0

            # 转换为 CHW 格式
            img_chw = np.transpose(img_normalized, (2, 0, 1))

            # 添加 batch 维度
            input_tensor = np.expand_dims(img_chw, axis=0)

            print(f"  形状: {input_tensor.shape}")
            print(f"  范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

            return input_tensor, img
        else:
            return self._prepare_random_input()

    def _prepare_random_input(self):
        """准备随机输入"""
        print(f"\n🎲 使用随机输入: {self.input_size}")
        np.random.seed(42)
        input_tensor = np.random.randn(1, 3, *self.input_size).astype(np.float32)

        print(f"  形状: {input_tensor.shape}")
        print(f"  范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

        return input_tensor, None

    def benchmark_pytorch(self, input_tensor, num_runs=100, warmup=10):
        """性能测试 - PyTorch"""
        print("\n" + "=" * 80)
        print("3. PyTorch 推理")
        print("=" * 80)

        input_torch = torch.from_numpy(input_tensor)

        # Warmup
        print(f"\n预热: {warmup} 次")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.pytorch_model(input_torch)

        # Benchmark
        print(f"测试: {num_runs} 次")
        times = []
        with torch.no_grad():
            for i in range(num_runs):
                start = time.perf_counter()
                output = self.pytorch_model(input_torch)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

                if (i + 1) % 20 == 0:
                    print(f"  进度: {i+1}/{num_runs}")

        # 统计
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time

        print(f"\n📊 PyTorch 性能:")
        print(f"  平均延迟: {avg_time:.2f} ms ± {std_time:.2f} ms")
        print(f"  最小/最大: {min_time:.2f} / {max_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")

        # 获取最终输出用于对比
        with torch.no_grad():
            final_output = self.pytorch_model(input_torch)

        # YOLOv8 返回列表，取第一个元素
        if isinstance(final_output, (list, tuple)):
            final_output = final_output[0]

        final_output_np = final_output.detach().cpu().numpy()

        print(f"\n📦 输出信息:")
        print(f"  形状: {final_output_np.shape}")
        print(f"  范围: [{final_output_np.min():.6f}, {final_output_np.max():.6f}]")
        print(f"  均值: {final_output_np.mean():.6f}")

        self.results['pytorch'] = {
            'times': times,
            'avg_ms': avg_time,
            'std_ms': std_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'fps': fps,
            'output': final_output_np
        }

        return final_output_np

    def benchmark_openvino(self, input_tensor, num_runs=100, warmup=10):
        """性能测试 - OpenVINO"""
        print("\n" + "=" * 80)
        print("4. OpenVINO 推理")
        print("=" * 80)

        # Warmup
        print(f"\n预热: {warmup} 次")
        for _ in range(warmup):
            _ = self.openvino_model(input_tensor)

        # Benchmark
        print(f"测试: {num_runs} 次")
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            output = self.openvino_model(input_tensor)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{num_runs}")

        # 统计
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time

        print(f"\n📊 OpenVINO 性能:")
        print(f"  平均延迟: {avg_time:.2f} ms ± {std_time:.2f} ms")
        print(f"  最小/最大: {min_time:.2f} / {max_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")

        # 获取最终输出用于对比
        final_output = self.openvino_model(input_tensor)
        final_output_np = list(final_output.values())[0]

        print(f"\n📦 输出信息:")
        print(f"  形状: {final_output_np.shape}")
        print(f"  范围: [{final_output_np.min():.6f}, {final_output_np.max():.6f}]")
        print(f"  均值: {final_output_np.mean():.6f}")

        self.results['openvino'] = {
            'times': times,
            'avg_ms': avg_time,
            'std_ms': std_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'fps': fps,
            'output': final_output_np
        }

        return final_output_np

    def compare_outputs(self, pytorch_output, openvino_output):
        """对比输出结果"""
        print("\n" + "=" * 80)
        print("5. 结果对比")
        print("=" * 80)

        # 形状检查
        print(f"\n📏 形状对比:")
        print(f"  PyTorch:  {pytorch_output.shape}")
        print(f"  OpenVINO: {openvino_output.shape}")

        if pytorch_output.shape != openvino_output.shape:
            print("  ⚠ 形状不一致！")
            return

        # 数值对比
        print(f"\n🔢 数值对比:")

        # 绝对差异
        abs_diff = np.abs(pytorch_output - openvino_output)
        print(f"  最大绝对差异: {abs_diff.max():.6e}")
        print(f"  平均绝对差异: {abs_diff.mean():.6e}")

        # 相对差异
        rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)
        print(f"  最大相对差异: {rel_diff.max():.6e}")
        print(f"  平均相对差异: {rel_diff.mean():.6e}")

        # 余弦相似度
        pytorch_flat = pytorch_output.flatten()
        openvino_flat = openvino_output.flatten()
        cosine_sim = np.dot(pytorch_flat, openvino_flat) / (
            np.linalg.norm(pytorch_flat) * np.linalg.norm(openvino_flat)
        )
        print(f"  余弦相似度: {cosine_sim:.8f}")

        # 判断
        is_close = np.allclose(pytorch_output, openvino_output, rtol=1e-3, atol=1e-5)
        print(f"\n✅ 结果一致性: {'✓ PASS' if is_close else '✗ FAIL'}")

        if is_close:
            print("  输出在可接受误差范围内")
        else:
            print("  输出差异较大，可能需要检查:")
            print("    1. 模型导出精度设置")
            print("    2. 输入预处理方式")
            print("    3. OpenVINO 版本兼容性")

        self.results['comparison'] = {
            'max_abs_diff': float(abs_diff.max()),
            'mean_abs_diff': float(abs_diff.mean()),
            'max_rel_diff': float(rel_diff.max()),
            'mean_rel_diff': float(rel_diff.mean()),
            'cosine_similarity': float(cosine_sim),
            'is_close': is_close
        }

    def compare_performance(self):
        """性能对比总结"""
        print("\n" + "=" * 80)
        print("6. 性能对比总结")
        print("=" * 80)

        pytorch_stats = self.results['pytorch']
        openvino_stats = self.results['openvino']

        print(f"\n{'指标':<20} {'PyTorch':<20} {'OpenVINO':<20} {'加速比':<15}")
        print("-" * 80)

        # 平均延迟
        pytorch_avg = pytorch_stats['avg_ms']
        openvino_avg = openvino_stats['avg_ms']
        speedup_avg = pytorch_avg / openvino_avg
        print(f"{'平均延迟 (ms)':<20} {pytorch_avg:<20.2f} {openvino_avg:<20.2f} {speedup_avg:<15.2f}x")

        # 标准差
        pytorch_std = pytorch_stats['std_ms']
        openvino_std = openvino_stats['std_ms']
        print(f"{'标准差 (ms)':<20} {pytorch_std:<20.2f} {openvino_std:<20.2f} {'-':<15}")

        # FPS
        pytorch_fps = pytorch_stats['fps']
        openvino_fps = openvino_stats['fps']
        speedup_fps = openvino_fps / pytorch_fps
        print(f"{'FPS':<20} {pytorch_fps:<20.2f} {openvino_fps:<20.2f} {speedup_fps:<15.2f}x")

        print("\n💡 结论:")
        if speedup_avg > 1.5:
            print(f"  OpenVINO 比 PyTorch 快 {speedup_avg:.1f}x，优化效果显著")
        elif speedup_avg > 1.1:
            print(f"  OpenVINO 比 PyTorch 快 {speedup_avg:.1f}x，有一定提升")
        elif speedup_avg > 0.9:
            print(f"  PyTorch 和 OpenVINO 性能接近")
        else:
            print(f"  PyTorch 比 OpenVINO 快，可能需要检查 OpenVINO 配置")

    def visualize_detections(self, image, pytorch_output, openvino_output,
                           conf_threshold=0.25, iou_threshold=0.45,
                           save_dir="benchmark_results"):
        """可视化检测结果（包含完整后处理）"""
        if image is None:
            print("\n⚠ 未提供图片，跳过可视化")
            return

        print("\n" + "=" * 80)
        print("7. 后处理与可视化")
        print("=" * 80)

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # YOLOv8 输出格式
        print(f"\n📊 YOLOv8 输出格式:")
        print(f"  形状: {pytorch_output.shape}")
        print(f"  [batch, 84, 8400]")
        print(f"    - 84 = 4 (bbox) + 80 (COCO classes)")
        print(f"    - 8400 = 80x80 + 40x40 + 20x20 anchor points")

        # PyTorch 后处理
        print(f"\n🔷 PyTorch 后处理:")
        print(f"  置信度阈值: {conf_threshold}")
        print(f"  NMS IoU 阈值: {iou_threshold}")
        pytorch_detections = postprocess_yolov8(
            pytorch_output, conf_threshold, iou_threshold
        )
        print(f"  检测到 {len(pytorch_detections)} 个对象")

        # OpenVINO 后处理
        print(f"\n🔶 OpenVINO 后处理:")
        print(f"  置信度阈值: {conf_threshold}")
        print(f"  NMS IoU 阈值: {iou_threshold}")
        openvino_detections = postprocess_yolov8(
            openvino_output, conf_threshold, iou_threshold
        )
        print(f"  检测到 {len(openvino_detections)} 个对象")

        # 可视化 PyTorch 结果
        print(f"\n🎨 绘制 PyTorch 检测结果...")
        pytorch_viz = draw_detections(
            image, pytorch_detections, COCO_CLASSES, self.input_size
        )
        pytorch_save_path = save_dir / "pytorch_detections.jpg"
        cv2.imwrite(str(pytorch_save_path), pytorch_viz)
        print(f"  ✓ 保存: {pytorch_save_path}")

        # 可视化 OpenVINO 结果
        print(f"\n🎨 绘制 OpenVINO 检测结果...")
        openvino_viz = draw_detections(
            image, openvino_detections, COCO_CLASSES, self.input_size
        )
        openvino_save_path = save_dir / "openvino_detections.jpg"
        cv2.imwrite(str(openvino_save_path), openvino_viz)
        print(f"  ✓ 保存: {openvino_save_path}")

        # 对比可视化
        print(f"\n📊 检测结果对比:")
        print(f"  {'Backend':<15} {'检测数量':<15}")
        print("-" * 35)
        print(f"  {'PyTorch':<15} {len(pytorch_detections):<15}")
        print(f"  {'OpenVINO':<15} {len(openvino_detections):<15}")

        # 详细检测信息
        if pytorch_detections:
            print(f"\n📝 PyTorch 检测详情 (top 5):")
            for i, det in enumerate(pytorch_detections[:5]):
                x1, y1, x2, y2, conf, cls_id = det
                cls_name = COCO_CLASSES[int(cls_id)]
                print(f"  [{i+1}] {cls_name:<15} 置信度: {conf:.3f}  bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        if openvino_detections:
            print(f"\n📝 OpenVINO 检测详情 (top 5):")
            for i, det in enumerate(openvino_detections[:5]):
                x1, y1, x2, y2, conf, cls_id = det
                cls_name = COCO_CLASSES[int(cls_id)]
                print(f"  [{i+1}] {cls_name:<15} 置信度: {conf:.3f}  bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        print(f"\n📁 可视化结果保存到: {save_dir}")

        # 保存检测结果
        self.results['pytorch_detections'] = pytorch_detections
        self.results['openvino_detections'] = openvino_detections


def main():
    parser = argparse.ArgumentParser(description="简化版 YOLOv8 PyTorch vs OpenVINO 性能对比")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="PyTorch 模型路径 (默认: yolov8n.pt)")
    parser.add_argument("--openvino", type=str,
                       default="exports/yolov8n/yolov8n_openvino/model.xml",
                       help="OpenVINO 模型路径 (默认: exports/yolov8n/yolov8n_openvino/model.xml)")
    parser.add_argument("--image", type=str, default=None,
                       help="测试图片路径 (可选，默认使用随机输入)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640],
                       help="输入尺寸 (默认: 640 640)")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="性能测试次数 (默认: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                       help="预热次数 (默认: 10)")
    parser.add_argument("--visualize", action="store_true",
                       help="可视化检测结果 (需要提供 --image)")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                       help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou-threshold", type=float, default=0.45,
                       help="NMS IoU 阈值 (默认: 0.45)")
    parser.add_argument("--save-dir", type=str, default="benchmark_results",
                       help="结果保存目录 (默认: benchmark_results)")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              简化版 YOLOv8 PyTorch vs OpenVINO 性能对比                     ║
║                                                                              ║
║  基于: OpenVINO 官方 YOLOv8 Notebook                                        ║
║  https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # 检查模型文件
    if not Path(args.model).exists():
        print(f"✗ PyTorch 模型不存在: {args.model}")
        print("\n请先运行: python fix_yolov8_openvino.py")
        return 1

    if not Path(args.openvino).exists():
        print(f"✗ OpenVINO 模型不存在: {args.openvino}")
        print("\n请先运行: python fix_yolov8_openvino.py")
        return 1

    try:
        # 创建 benchmark
        benchmark = YOLOv8Benchmark(
            model_path=args.model,
            openvino_path=args.openvino,
            input_size=tuple(args.input_size)
        )

        # 1. 加载模型
        benchmark.load_models()

        # 2. 准备输入
        input_tensor, image = benchmark.prepare_input(args.image)

        # 3. PyTorch 推理
        pytorch_output = benchmark.benchmark_pytorch(
            input_tensor,
            num_runs=args.num_runs,
            warmup=args.warmup
        )

        # 4. OpenVINO 推理
        openvino_output = benchmark.benchmark_openvino(
            input_tensor,
            num_runs=args.num_runs,
            warmup=args.warmup
        )

        # 5. 结果对比
        benchmark.compare_outputs(pytorch_output, openvino_output)

        # 6. 性能对比
        benchmark.compare_performance()

        # 7. 可视化（可选）
        if args.visualize:
            benchmark.visualize_detections(
                image, pytorch_output, openvino_output,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                save_dir=args.save_dir
            )

        # 保存数值结果
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)

        np.save(save_dir / "pytorch_output.npy", pytorch_output)
        np.save(save_dir / "openvino_output.npy", openvino_output)

        print("\n" + "=" * 80)
        print("✓ 对比完成！")
        print("=" * 80)
        print(f"\n📁 结果保存到: {save_dir}")
        print(f"  - pytorch_output.npy")
        print(f"  - openvino_output.npy")
        if args.visualize and image is not None:
            print(f"  - pytorch_detections.jpg")
            print(f"  - openvino_detections.jpg")

        return 0

    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
