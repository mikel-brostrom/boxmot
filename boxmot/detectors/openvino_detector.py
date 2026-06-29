# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
OpenVINO Detector Backend

基于 yolov8_benchmark_simple.py 的验证结果，实现独立的 OpenVINO 检测器后端。

参考:
    https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html
"""

from pathlib import Path

import cv2
import numpy as np

from boxmot.detectors.base import BaseDetectorBackend, Detections
from boxmot.utils import logger as LOGGER


# COCO 数据集类别
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


class OpenVinoDetector(BaseDetectorBackend):
    """
    OpenVINO 检测器后端，支持 YOLOv8 等模型。

    实现独立的 YOLOv8 后处理逻辑，确保与 PyTorch 输出对齐。
    """

    def __init__(self, model, device, imgsz=None):
        """
        初始化 OpenVINO 检测器。

        Args:
            model: 模型路径（.xml, .bin 或目录）
            device: 设备名称（"CPU", "GPU" 等）
            imgsz: 输入尺寸（int 或 tuple）
        """
        self.model_path = Path(model)
        self.device = device if device else "CPU"
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz or (640, 640))

        LOGGER.info(f"Loading {self.model_path} for OpenVINO inference...")

        from openvino import Core, Layout

        ie = Core()

        # 查找 XML 文件
        xml_path = self._find_xml_path(self.model_path)
        LOGGER.info(f"Using model: {xml_path}")

        # 读取网络
        network = ie.read_model(model=xml_path)

        # 设置输入布局
        if network.get_parameters()[0].get_layout().empty:
            network.get_parameters()[0].set_layout(Layout("NCHW"))

        # 编译模型 - 强制 FP32 和 ACCURACY 模式（关键！）
        self.executable_network = ie.compile_model(
            network,
            device_name=self.device,
            config={
                "INFERENCE_PRECISION_HINT": "f32",
                "EXECUTION_MODE_HINT": "ACCURACY",
            },
        )

        self.output_layer = next(iter(self.executable_network.outputs))
        self.input_layer = next(iter(self.executable_network.inputs))

        # 类别名称
        self.names = {i: name for i, name in enumerate(COCO_CLASSES)}

        self.pt = False
        self.stride = 32

        # 缓存
        self._last_orig_imgs = None
        self._last_scales = None

        LOGGER.info(f"✓ OpenVINO model loaded")
        LOGGER.info(f"  Device: {self.device}")
        LOGGER.info(f"  Input shape: {network.inputs[0].partial_shape}")
        LOGGER.info(f"  Output shape: {network.outputs[0].partial_shape}")

    @staticmethod
    def _find_xml_path(model_path):
        """查找 XML 模型文件"""
        if model_path.suffix == '.bin':
            return model_path.with_suffix('.xml')
        elif model_path.suffix == '.xml':
            return model_path
        elif model_path.is_dir():
            xml_files = list(model_path.glob("*.xml"))
            if xml_files:
                return xml_files[0]
        raise ValueError(f"无法找到 XML 模型文件: {model_path}")

    def preprocess(self, images, **kwargs):
        """
        预处理图像（Letterbox resize + 归一化）。

        Args:
            images: 图像列表或单张图像

        Returns:
            preprocessed: [B, 3, H, W] numpy array
        """
        if isinstance(images, list):
            ims = images
        else:
            ims = [images]

        self._last_orig_imgs = ims

        # Letterbox resize
        preprocessed = []
        scales = []

        for img in ims:
            resized, scale_info = self._letterbox(img, self.imgsz)
            preprocessed.append(resized)
            scales.append(scale_info)

        self._last_scales = scales

        # Batch
        batch = np.stack(preprocessed, axis=0)
        return batch

    def _letterbox(self, img, new_shape):
        """
        Letterbox resize（保持宽高比）。

        Args:
            img: 原始图像 [H, W, C]
            new_shape: 目标尺寸 (H, W)

        Returns:
            img_resized: 处理后的图像 [3, H, W]
            scale_info: 缩放信息 (scale, pad_w, pad_h, new_w, new_h)
        """
        h, w = img.shape[:2]
        target_h, target_w = new_shape

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        img_padded = cv2.copyMakeBorder(
            img_resized,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # 转换为 RGB 并归一化
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # 转换为 CHW 格式
        img_chw = np.transpose(img_normalized, (2, 0, 1))

        scale_info = (scale, pad_w, pad_h, new_w, new_h)
        return img_chw, scale_info

    def process(self, preprocessed, **kwargs):
        """
        OpenVINO 推理。

        Args:
            preprocessed: [B, 3, H, W] numpy array

        Returns:
            output: [B, 84, 8400] numpy array
        """
        output = self.executable_network(preprocessed)
        return output[self.output_layer]

    def postprocess(self, raw_preds, conf=0.25, iou=0.45, classes=None, agnostic_nms=False, **kwargs):
        """
        YOLOv8 后处理（参考 yolov8_benchmark_simple.py）。

        Args:
            raw_preds: [B, 84, 8400] numpy array
            conf: 置信度阈值
            iou: NMS IoU 阈值
            classes: 类别过滤列表
            agnostic_nms: 是否使用类别无关 NMS

        Returns:
            List[Detections]: 检测结果列表
        """
        orig_imgs = kwargs.get("orig_imgs", self._last_orig_imgs) or []
        scales = kwargs.get("scales", self._last_scales) or []

        processed = []
        batch_size = raw_preds.shape[0]

        for i in range(batch_size):
            pred = raw_preds[i]  # [84, 8400]
            orig_img = orig_imgs[i] if i < len(orig_imgs) else None
            scale_info = scales[i] if i < len(scales) else None

            # YOLOv8 后处理
            detections = self._postprocess_yolov8(
                pred, conf, iou, classes, agnostic_nms
            )

            # 缩放回原图
            if orig_img is not None and scale_info is not None and len(detections) > 0:
                detections = self._scale_boxes(detections, scale_info)

            # 转换为 Detections 对象
            if len(detections) > 0:
                dets = np.array(detections, dtype=np.float32)  # [N, 6]
            else:
                dets = np.empty((0, 6), dtype=np.float32)

            processed.append(Detections(
                dets=dets,
                orig_img=orig_img,
                path="",
                names=self.names,
                masks=None,
            ))

        return processed

    def _postprocess_yolov8(self, output, conf_threshold, iou_threshold, classes, agnostic_nms):
        """
        YOLOv8 后处理核心逻辑。

        输入: [84, 8400]
          - 84 = 4 (bbox) + 80 (COCO classes)
          - 8400 = anchor points

        输出: List of [x1, y1, x2, y2, conf, cls_id]
        """
        # 分离 bbox 和类别分数
        boxes = output[:4, :]  # [4, 8400] - cx, cy, w, h
        scores = output[4:, :]  # [80, 8400] - class scores

        # 获取每个预测的最大类别置信度和类别 ID
        max_scores = scores.max(axis=0)  # [8400]
        max_classes = scores.argmax(axis=0)  # [8400]

        # 类别过滤
        if classes is not None:
            class_mask = np.isin(max_classes, classes)
            max_scores = max_scores[class_mask]
            max_classes = max_classes[class_mask]
            boxes = boxes[:, class_mask]

        # 置信度过滤
        mask = max_scores > conf_threshold
        if not mask.any():
            return []

        boxes = boxes[:, mask]
        max_scores = max_scores[mask]
        max_classes = max_classes[mask]

        # 转换 bbox 格式: [cx, cy, w, h] → [x1, y1, x2, y2]
        cx, cy, w, h = boxes
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=0).T  # [N, 4]

        # NMS
        if agnostic_nms:
            # 类别无关 NMS
            keep_indices = self._nms(boxes_xyxy, max_scores, iou_threshold)
        else:
            # 按类别 NMS
            keep_indices = []
            for cls_id in np.unique(max_classes):
                cls_mask = max_classes == cls_id
                cls_boxes = boxes_xyxy[cls_mask]
                cls_scores = max_scores[cls_mask]
                cls_indices = np.where(cls_mask)[0]

                cls_keep = self._nms(cls_boxes, cls_scores, iou_threshold)
                keep_indices.extend(cls_indices[cls_keep].tolist())

            keep_indices = sorted(keep_indices, key=lambda i: max_scores[i], reverse=True)

        # 组装结果 [x1, y1, x2, y2, conf, cls_id]
        detections = []
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            conf = float(max_scores[idx])
            cls_id = int(max_classes[idx])
            detections.append([x1, y1, x2, y2, conf, cls_id])

        return detections

    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        """
        Non-Maximum Suppression。

        Args:
            boxes: [N, 4] - [x1, y1, x2, y2]
            scores: [N] - confidence scores
            iou_threshold: IoU 阈值

        Returns:
            keep: List of indices to keep
        """
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # 计算 IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            box_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            boxes_area = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                         (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = box_area + boxes_area - intersection

            iou = intersection / (union + 1e-6)

            # 保留 IoU 小于阈值的 box
            mask = iou <= iou_threshold
            order = order[1:][mask]

        return keep

    def _scale_boxes(self, detections, scale_info):
        """
        将 boxes 从输入尺寸缩放回原图尺寸（考虑 letterbox）。

        Args:
            detections: List of [x1, y1, x2, y2, conf, cls_id]
            scale_info: (scale, pad_w, pad_h, new_w, new_h)

        Returns:
            scaled_detections: List of [x1, y1, x2, y2, conf, cls_id]
        """
        scale, pad_w, pad_h, new_w, new_h = scale_info

        scaled_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det

            # 减去 padding
            x1 = x1 - pad_w
            y1 = y1 - pad_h
            x2 = x2 - pad_w
            y2 = y2 - pad_h

            # 缩放回原图
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale

            # 裁剪到 [0, ∞)（原图边界在后续步骤处理）
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(0, x2)
            y2 = max(0, y2)

            scaled_detections.append([x1, y1, x2, y2, conf, cls_id])

        return scaled_detections

    def __call__(self, images, conf, iou, classes, agnostic_nms):
        """
        完整推理流程。

        Args:
            images: 图像列表
            conf: 置信度阈值
            iou: NMS IoU 阈值
            classes: 类别过滤
            agnostic_nms: 类别无关 NMS

        Returns:
            List[Detections]: 检测结果
        """
        preprocessed = self.preprocess(images)
        raw = self.process(preprocessed)
        return self.postprocess(
            raw,
            conf=conf,
            iou=iou,
            classes=classes,
            agnostic_nms=agnostic_nms,
            orig_imgs=self._last_orig_imgs,
            scales=self._last_scales,
        )
