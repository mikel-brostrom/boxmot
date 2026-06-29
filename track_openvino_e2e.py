"""
OpenVINO E2E Tracking Pipeline

完整的基于 OpenVINO 的目标跟踪流程，与 PyTorch pipeline 对齐。

基于新的 OpenVinoDetector 实现，确保检测结果与 PyTorch 一致。

使用:
    # 基础跟踪
    python track_openvino_e2e.py \
        --openvino-model exports/yolov8n/yolov8n_openvino/model.xml \
        --source assets/MOT17-mini/train/MOT17-02-FRCNN \
        --tracker bytetrack

    # GPU 加速
    python track_openvino_e2e.py \
        --openvino-model exports/yolov8n/yolov8n_openvino/model.xml \
        --source video.mp4 \
        --tracker bytetrack \
        --ov-device GPU

    # 与 PyTorch 对比
    python track_openvino_e2e.py \
        --openvino-model exports/yolov8n/yolov8n_openvino/model.xml \
        --pytorch-model yolov8n.pt \
        --source assets/MOT17-mini/train/MOT17-02-FRCNN \
        --compare
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def load_openvino_detector(model_path, device, imgsz):
    """加载 OpenVINO 检测器"""
    print(f"📦 加载 OpenVINO 检测器: {model_path}")
    print(f"  设备: {device}")
    print(f"  输入尺寸: {imgsz}")

    try:
        from boxmot.detectors.openvino_detector import OpenVinoDetector

        detector = OpenVinoDetector(
            model=model_path,
            device=device,
            imgsz=imgsz
        )
        print("  ✓ OpenVINO 检测器加载成功")
        return detector

    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        raise


def load_pytorch_detector(model_path, device, imgsz):
    """加载 PyTorch 检测器（对比用）"""
    print(f"\n📦 加载 PyTorch 检测器: {model_path}")
    print(f"  设备: {device}")
    print(f"  输入尺寸: {imgsz}")

    try:
        from boxmot.detectors.ultralytics import UltralyticsDetector

        detector = UltralyticsDetector(
            model=model_path,
            device=device,
            imgsz=imgsz
        )
        print("  ✓ PyTorch 检测器加载成功")
        return detector

    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        raise


def load_tracker(tracker_type, reid_model=None):
    """加载跟踪器"""
    print(f"\n🔄 加载跟踪器: {tracker_type}")

    try:
        if tracker_type.lower() == "bytetrack":
            from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack
            tracker = ByteTrack()
        elif tracker_type.lower() == "botsort":
            from boxmot.trackers.bbox.botsort.botsort import BotSort
            tracker = BotSort(
                reid_weights=reid_model or "osnet_x0_25_msmt17.pt",
                device="cpu"
            )
        elif tracker_type.lower() == "strongsort":
            from boxmot.trackers.bbox.strongsort.strongsort import StrongSort
            tracker = StrongSort(
                reid_weights=reid_model or "osnet_x0_25_msmt17.pt",
                device="cpu"
            )
        else:
            raise ValueError(f"不支持的跟踪器: {tracker_type}")

        print(f"  ✓ 跟踪器加载成功")
        return tracker

    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        print(f"  支持的跟踪器: bytetrack, botsort, strongsort")
        raise


def load_source(source_path):
    """加载视频或图像序列"""
    source_path = Path(source_path)

    if source_path.is_file():
        # 视频文件
        print(f"\n📹 加载视频: {source_path}")
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  分辨率: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  总帧数: {total_frames}")

        return "video", cap, (width, height, fps, total_frames)

    elif source_path.is_dir():
        # 图像序列
        img_dir = source_path / "img1" if (source_path / "img1").exists() else source_path
        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

        if not img_files:
            raise ValueError(f"目录中没有找到图像: {img_dir}")

        print(f"\n🖼️ 加载图像序列: {img_dir}")
        print(f"  图像数量: {len(img_files)}")

        # 读取第一张图像获取尺寸
        first_img = cv2.imread(str(img_files[0]))
        height, width = first_img.shape[:2]
        print(f"  分辨率: {width}x{height}")

        return "images", img_files, (width, height, None, len(img_files))

    else:
        raise ValueError(f"无效的源路径: {source_path}")


def track_with_detector(detector, source_info, tracker, conf_threshold, iou_threshold,
                       show=False, save_vid=False, output_dir=None):
    """使用给定的检测器进行跟踪"""
    source_type, source, (width, height, fps, total_frames) = source_info

    # 统计
    stats = {
        "frames": 0,
        "detections": 0,
        "tracks": set(),
        "detection_times": [],
        "tracking_times": [],
        "total_times": []
    }

    # 视频输出
    writer = None
    if save_vid and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if source_type == "video":
            output_path = output_dir / "output_openvino.mp4"
        else:
            output_path = output_dir / "output_openvino.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps or 30,
            (width, height)
        )
        print(f"\n💾 保存视频到: {output_path}")

    print(f"\n🎬 开始跟踪...")
    print(f"  {'帧数':<10} {'检测':<10} {'跟踪':<10} {'检测时间':<15} {'跟踪时间':<15} {'总时间':<15}")
    print("=" * 90)

    frame_idx = 0

    if source_type == "video":
        # 视频源
        while True:
            ret, frame = source.read()
            if not ret:
                break

            frame_idx += 1
            process_frame(
                frame, detector, tracker, conf_threshold, iou_threshold,
                frame_idx, stats, show, writer
            )

        source.release()

    else:
        # 图像序列
        for img_path in source:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            frame_idx += 1
            process_frame(
                frame, detector, tracker, conf_threshold, iou_threshold,
                frame_idx, stats, show, writer
            )

    if writer:
        writer.release()

    if show:
        cv2.destroyAllWindows()

    return stats


def process_frame(frame, detector, tracker, conf_threshold, iou_threshold,
                 frame_idx, stats, show, writer):
    """处理单帧"""
    # 检测
    t0 = time.perf_counter()
    detections = detector(
        images=[frame],
        conf=conf_threshold,
        iou=iou_threshold,
        classes=None,
        agnostic_nms=False
    )
    t1 = time.perf_counter()
    detection_time = (t1 - t0) * 1000

    # 提取检测结果
    dets = detections[0].dets  # [N, 6] - [x1, y1, x2, y2, conf, cls]

    # 跟踪
    t2 = time.perf_counter()
    if hasattr(tracker, 'update'):
        # ByteTrack, BoTSORT, etc.
        tracks = tracker.update(dets, frame)
    else:
        tracks = []
    t3 = time.perf_counter()
    tracking_time = (t3 - t2) * 1000

    total_time = (t3 - t0) * 1000

    # 统计
    stats["frames"] += 1
    stats["detections"] += len(dets)
    stats["detection_times"].append(detection_time)
    stats["tracking_times"].append(tracking_time)
    stats["total_times"].append(total_time)

    for track in tracks:
        if len(track) >= 5:
            track_id = int(track[4])
            stats["tracks"].add(track_id)

    # 打印进度
    if frame_idx % 10 == 0 or frame_idx == 1:
        print(f"  {frame_idx:<10} {len(dets):<10} {len(tracks):<10} "
              f"{detection_time:<15.2f} {tracking_time:<15.2f} {total_time:<15.2f}")

    # 可视化
    if show or writer:
        vis_frame = draw_tracks(frame.copy(), tracks, dets)

        if show:
            cv2.imshow("Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        if writer:
            writer.write(vis_frame)


def draw_tracks(frame, tracks, detections):
    """绘制跟踪结果"""
    # 绘制检测框（灰色）
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (128, 128, 128),
            1
        )

    # 绘制跟踪框（彩色）
    for track in tracks:
        if len(track) < 5:
            continue

        x1, y1, x2, y2, track_id = track[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 为每个 ID 生成固定颜色
        np.random.seed(int(track_id))
        color = tuple(map(int, np.random.randint(0, 255, 3)))

        # 绘制框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制 ID
        label = f"ID: {int(track_id)}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    return frame


def print_summary(stats, detector_name="OpenVINO"):
    """打印统计总结"""
    print("\n" + "=" * 90)
    print(f"{detector_name} 跟踪总结")
    print("=" * 90)

    print(f"\n帧数: {stats['frames']}")
    print(f"总检测数: {stats['detections']}")
    print(f"总跟踪数: {len(stats['tracks'])}")

    if stats['detection_times']:
        avg_det = np.mean(stats['detection_times'])
        avg_track = np.mean(stats['tracking_times'])
        avg_total = np.mean(stats['total_times'])
        fps = 1000 / avg_total if avg_total > 0 else 0

        print(f"\n平均时间:")
        print(f"  检测: {avg_det:.2f} ms")
        print(f"  跟踪: {avg_track:.2f} ms")
        print(f"  总计: {avg_total:.2f} ms")
        print(f"  FPS: {fps:.2f}")


def compare_results(ov_stats, pt_stats):
    """对比 OpenVINO 和 PyTorch 结果"""
    print("\n" + "=" * 90)
    print("OpenVINO vs PyTorch 对比")
    print("=" * 90)

    ov_avg = np.mean(ov_stats['total_times'])
    pt_avg = np.mean(pt_stats['total_times'])
    speedup = pt_avg / ov_avg if ov_avg > 0 else 0

    print(f"\n{'指标':<20} {'OpenVINO':<20} {'PyTorch':<20} {'加速比':<15}")
    print("-" * 90)
    print(f"{'平均时间 (ms)':<20} {ov_avg:<20.2f} {pt_avg:<20.2f} {speedup:<15.2f}x")
    print(f"{'FPS':<20} {1000/ov_avg:<20.2f} {1000/pt_avg:<20.2f} {'-':<15}")
    print(f"{'检测数':<20} {ov_stats['detections']:<20} {pt_stats['detections']:<20} {'-':<15}")
    print(f"{'跟踪数':<20} {len(ov_stats['tracks']):<20} {len(pt_stats['tracks']):<20} {'-':<15}")

    det_diff = abs(ov_stats['detections'] - pt_stats['detections'])
    track_diff = abs(len(ov_stats['tracks']) - len(pt_stats['tracks']))

    print(f"\n精度对比:")
    print(f"  检测数差异: {det_diff} ({det_diff/pt_stats['detections']*100:.2f}%)")
    print(f"  跟踪数差异: {track_diff} ({track_diff/len(pt_stats['tracks'])*100:.2f}%)")

    if det_diff / pt_stats['detections'] < 0.05:
        print("  ✓ 检测结果基本一致")
    else:
        print("  ⚠ 检测结果差异较大")


def main():
    parser = argparse.ArgumentParser(
        description="OpenVINO E2E Tracking Pipeline"
    )

    # 模型
    parser.add_argument("--openvino-model", type=str, required=True,
                       help="OpenVINO 模型路径 (.xml)")
    parser.add_argument("--pytorch-model", type=str, default=None,
                       help="PyTorch 模型路径（对比用）")

    # 源
    parser.add_argument("--source", type=str, required=True,
                       help="视频文件或图像序列目录")

    # 跟踪器
    parser.add_argument("--tracker", type=str, default="bytetrack",
                       choices=["bytetrack", "botsort", "strongsort"],
                       help="跟踪器类型")
    parser.add_argument("--reid-model", type=str, default=None,
                       help="ReID 模型路径（用于 BoTSORT/StrongSORT）")

    # 设备和参数
    parser.add_argument("--ov-device", type=str, default="CPU",
                       choices=["CPU", "GPU", "NPU"],
                       help="OpenVINO 设备")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="输入尺寸")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                       help="检测置信度阈值")
    parser.add_argument("--iou-threshold", type=float, default=0.45,
                       help="NMS IoU 阈值")

    # 输出
    parser.add_argument("--save-vid", action="store_true",
                       help="保存输出视频")
    parser.add_argument("--output-dir", type=str, default="runs/track",
                       help="输出目录")
    parser.add_argument("--show", action="store_true",
                       help="实时显示结果")
    parser.add_argument("--compare", action="store_true",
                       help="与 PyTorch 对比（需要 --pytorch-model）")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OpenVINO E2E Tracking Pipeline                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # 检查文件
    if not Path(args.openvino_model).exists():
        print(f"✗ OpenVINO 模型不存在: {args.openvino_model}")
        return 1

    if args.compare and not args.pytorch_model:
        print("✗ 对比模式需要提供 --pytorch-model")
        return 1

    if args.compare and not Path(args.pytorch_model).exists():
        print(f"✗ PyTorch 模型不存在: {args.pytorch_model}")
        return 1

    try:
        # 1. 加载 OpenVINO 检测器
        ov_detector = load_openvino_detector(
            args.openvino_model,
            args.ov_device,
            args.imgsz
        )

        # 2. 加载跟踪器
        ov_tracker = load_tracker(args.tracker, args.reid_model)

        # 3. 加载源
        source_info = load_source(args.source)

        # 4. OpenVINO 跟踪
        ov_stats = track_with_detector(
            ov_detector,
            source_info,
            ov_tracker,
            args.conf_threshold,
            args.iou_threshold,
            show=args.show,
            save_vid=args.save_vid,
            output_dir=args.output_dir
        )

        print_summary(ov_stats, "OpenVINO")

        # 5. PyTorch 对比（可选）
        if args.compare:
            print("\n" + "=" * 90)
            print("运行 PyTorch 对比...")
            print("=" * 90)

            pt_detector = load_pytorch_detector(
                args.pytorch_model,
                "cpu",
                args.imgsz
            )
            pt_tracker = load_tracker(args.tracker, args.reid_model)

            # 重新加载源
            source_info = load_source(args.source)

            pt_stats = track_with_detector(
                pt_detector,
                source_info,
                pt_tracker,
                args.conf_threshold,
                args.iou_threshold,
                show=False,
                save_vid=False,
                output_dir=None
            )

            print_summary(pt_stats, "PyTorch")
            compare_results(ov_stats, pt_stats)

        print("\n" + "=" * 90)
        print("✓ 跟踪完成！")
        print("=" * 90)

        if args.save_vid:
            print(f"\n📁 输出保存到: {args.output_dir}")

        return 0

    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
