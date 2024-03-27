import os
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import matplotlib.pyplot as plt
from random import shuffle, randint
import torch
import torch.nn.functional as F


class YoloV8SegPoseAPI:
    def __init__(self, host: str = 'localhost', port: int = 8001, model_name: str = 'yolov8_seg_pose_e2e', confidence_threshold: float=0.001) -> None:
        self.confidence_threshold = confidence_threshold
        self.host = host
        self.port = port
        self.model_name = f'{model_name}'
        self.url = f'{self.host}:{self.port}'
        self.triton_client = grpcclient.InferenceServerClient(
            url=self.url,
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
        config = self.triton_client.get_model_config(model_name, as_json=True)
        if len(config['config']['input'][0]['dims']) >= 3:
            self.channels = int(config['config']['input'][0]['dims'][0])
            self.height = int(config['config']['input'][0]['dims'][-2])
            self.width = int(config['config']['input'][0]['dims'][-1])
            self.input_name = config['config']['input'][0]['name']
        else:
            raise NotImplementedError()

        self.outputs = []
        for output in config['config']['output']:
            self.outputs.append(grpcclient.InferRequestedOutput(output['name']))
        self.model_dtype = "FP32"
    
    def _read_img(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return img

    def _preproc(self, image_batch):
        self.original_image_size = [img.shape for img in image_batch]
        # preproc №1
        image_list = []
        image_borders = []
        for img in image_batch:
            image, borders = resize_with_padding(img, new_shape=(self.height, self.width))
            image_list.append(image)
            image_borders.append(borders)
        # preproc №2
        self.image_borders = image_borders
        return preproc_img_for_inference(image_list)       
    
    def inference(self, img_path):
        img = self._read_img(img_path)
        x = self._preproc([img])
        predictions = self._predict(x)
        predictions = self._postproc(predictions)
        detections = []
        for bb, bs, bm, bk in zip(*predictions):
            for box, score in zip(bb, bs):
                cls = 0 # person
                detections.append([box[0], box[1], box[2]-box[0], box[3]-box[1], score[0], cls])
        # detections = np.array(detections)
        return detections

    def _predict(self, batch):
        inputs = [grpcclient.InferInput(self.input_name, list(batch.shape), self.model_dtype)]
        inputs[0].set_data_from_numpy(batch)
        results = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self.outputs,
            headers={},
            compression_algorithm=None)
        return (results.as_numpy('result_boxes').copy(), results.as_numpy('result_scores').copy(), results.as_numpy('result_masks').copy(), results.as_numpy('result_kpts').copy())

    def resize_box_to_original_image(
        self, boxes, original_img_h, original_img_w, img_h=640, img_w=640
    ):
        gain = min(img_h / original_img_h, img_w / original_img_w)  # gain  = old / new
        pad = round((img_w - original_img_w * gain) / 2 - 0.1), round(
            (img_h - original_img_h * gain) / 2 - 0.1
        )  # wh padding

        result_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 -= pad[0]
            y1 -= pad[1]
            x2 -= pad[0]
            y2 -= pad[1]
            x1 /= gain
            y1 /= gain
            x2 /= gain
            y2 /= gain
            # result_boxes.append((x1*w_ratio, y1/h_ratio, x2/w_ratio, y2*h_ratio))
            result_boxes.append((x1, y1, x2, y2))
        return result_boxes
    
    def resize_mask_to_original_image(self, masks, original_img_h, original_img_w, padding=True):
        mh, mw = masks.shape[1:]
        gain = min(mh / original_img_h, mw / original_img_w)  # gain  = old / new
        pad = [mw - original_img_w * gain, mh - original_img_h * gain]  # wh padding
        if padding:
            pad[0] /= 2
            pad[1] /= 2
        top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
        bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
        masks = masks[..., top:bottom, left:right]

        # TODO
        # print('WARNING, replace F.interpolate to cv2.resize')
        # assert 1==0
        masks = F.interpolate(torch.tensor(masks)[None], (original_img_h, original_img_w), mode='bilinear', align_corners=False).numpy()[0]  # NCHW
        return masks

    def clip_coord(self, kpts, image_shape_h, image_shape_w):
        kpts_output = kpts.copy()
        kpts_output[..., 0] = kpts[..., 0].clip(0, image_shape_w)  # x
        kpts_output[..., 1] = kpts[..., 1].clip(0, image_shape_h)  # y
        return kpts_output

    def resize_kpts_to_original_image(self, kpts, orig_img_h, orig_img_w, img_h=640, img_w=640, ratio_pad=None, normalize=False
    ):
        gain = min(img_h / orig_img_h, img_w / orig_img_w)  # gain  = old / new
        pad = round((img_w - orig_img_w * gain) / 2 - 0.1), round(
            (img_h - orig_img_h * gain) / 2 - 0.1)  # wh padding

        kpts[..., 0] -= pad[0]  # x padding
        kpts[..., 1] -= pad[1]  # y padding
        kpts[..., 0] /= gain
        kpts[..., 1] /= gain
        kpts = self.clip_coord(kpts, orig_img_h, orig_img_w)
        return kpts
    
    def decode_detection(self, detections):
        boxes = detections[0]
        scores = detections[1]
        masks = detections[2]
        kpts = detections[3]

        batch_boxes = []
        batch_scores = []
        batch_masks = []
        batch_kpts = []

        for idx_batch in range(len(boxes)):
            # skip nondetections (scores == 0)
            non_zero = np.where(scores[idx_batch] > self.confidence_threshold)[0]
            original_img_h, original_img_w = self.original_image_size[idx_batch][:2]
            
            boxes_resize = self.resize_box_to_original_image(boxes[idx_batch][non_zero], original_img_h, original_img_w)
            batch_boxes.append(boxes_resize)

            batch_scores.append(scores[idx_batch][non_zero])

            masks_resized = self.resize_mask_to_original_image(masks[idx_batch][non_zero], original_img_h, original_img_w)
            batch_masks.append(masks_resized)

            kpts_resized = self.resize_kpts_to_original_image(kpts[idx_batch][non_zero], original_img_h, original_img_w)
            batch_kpts.append(kpts_resized)

        return (batch_boxes, batch_scores, batch_masks, batch_kpts)
    
    def _postproc(self, detections):
        return self.decode_detection(detections)

# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/data/dataloaders/v5augmentations.py#L116
def resize_with_padding(img, new_shape):
    # оригинальный препроцесс, без него падает точность модели
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr to rgb
    return img, [top, bottom, left, right]


def preproc_img_for_inference(x):
    x = np.array(x).astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)  # hwc -> bhwc
    x = x.transpose((0, 3, 1, 2))  # bhwc to bchw,
    return x / 255.0  # [0,255] -> [0,1]


def read_img_bgr(img_path):
    img = cv2.imread(img_path)  # bgr
    return img


def load_images(img_dir) -> list:
    image_paths_list = os.listdir(img_dir)
    images = []
    for img_path in image_paths_list:
        img_path = os.path.join(img_dir, img_path)
        images.append(read_img_bgr(img_path))
    return images


def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img_shape, boxes, orig_image_shape):
    gain = min(
        img_shape[0] / orig_image_shape[0], img_shape[1] / orig_image_shape[1]
    )  # gain  = old / new
    pad = round((img_shape[1] - orig_image_shape[1] * gain) / 2 - 0.1), round(
        (img_shape[0] - orig_image_shape[0] * gain) / 2 - 0.1
    )  # wh padding
    print("gain", gain, img_shape, orig_image_shape)
    boxes[..., :4] /= gain
    clip_boxes(boxes, orig_image_shape)
    return boxes


def imshow(img, title="img"):
    plt.imshow(img)
    plt.title(title)
    plt.show()


def get_random_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))


def resize_box_to_original_image(
    boxes, original_img_h, original_img_w, img_h=640, img_w=640
):
    gain = min(img_h / original_img_h, img_w / original_img_w)  # gain  = old / new
    pad = round((img_w - original_img_w * gain) / 2 - 0.1), round(
        (img_h - original_img_h * gain) / 2 - 0.1
    )  # wh padding

    result_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 -= pad[0]
        y1 -= pad[1]
        x2 -= pad[0]
        y2 -= pad[1]
        x1 /= gain
        y1 /= gain
        x2 /= gain
        y2 /= gain
        # result_boxes.append((x1*w_ratio, y1/h_ratio, x2/w_ratio, y2*h_ratio))
        result_boxes.append((x1, y1, x2, y2))
    return result_boxes


def resize_mask_to_original_image(masks, original_img_h, original_img_w, padding=True):
    mh, mw = masks.shape[1:]
    gain = min(mh / original_img_h, mw / original_img_w)  # gain  = old / new
    pad = [mw - original_img_w * gain, mh - original_img_h * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    # TODO
    print("WARNING, replace F.interpolate to cv2.resize")
    # assert 1==0
    masks = F.interpolate(
        torch.tensor(masks)[None],
        (original_img_h, original_img_w),
        mode="bilinear",
        align_corners=False,
    ).numpy()[
        0
    ]  # NCHW
    return masks


def clip_coord(kpts, image_shape_h, image_shape_w):
    kpts_output = kpts.copy()
    kpts_output[..., 0] = kpts[..., 0].clip(0, image_shape_w)  # x
    kpts_output[..., 1] = kpts[..., 1].clip(0, image_shape_h)  # y
    return kpts_output


def resize_kpts_to_original_image(
    kpts, orig_img_h, orig_img_w, img_h=640, img_w=640, ratio_pad=None, normalize=False
):
    gain = min(img_h / orig_img_h, img_w / orig_img_w)  # gain  = old / new
    pad = round((img_w - orig_img_w * gain) / 2 - 0.1), round(
        (img_h - orig_img_h * gain) / 2 - 0.1
    )  # wh padding

    kpts[..., 0] -= pad[0]  # x padding
    kpts[..., 1] -= pad[1]  # y padding
    kpts[..., 0] /= gain
    kpts[..., 1] /= gain
    kpts = clip_coord(kpts, orig_img_h, orig_img_w)
    return kpts


