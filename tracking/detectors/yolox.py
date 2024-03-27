
import gdown
import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
from pathlib import Path
from boxmot.utils import logger as LOGGER

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
# import torch.backends.cudnn as cudnn

# cudnn.benchmark = True

# default model weigths for these model names
YOLOX_ZOO = {
    'yolox_n.pt': 'https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX',
    'yolox_s.pt': 'https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj',
    'yolox_m.pt': 'https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun',
    'yolox_l.pt': 'https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz',
    'yolox_x.pt': 'https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5',
}

def get_model_from_weigths(l, model):
    model_type = None
    for key in l:
        if Path(key).stem in str(model.name):
            model_type = str(Path(key).with_suffix(''))
            break
    return model_type


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

class YOLOX_wrapper:
    def __init__(self, args) -> None:
        
        self.args = args
        self.pt = False
        self.stride = 32  # max stride in YOLOX

        # model_type one of: 'yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x'
        model_type = get_model_from_weigths(YOLOX_ZOO.keys(), args.yolo_model)

        # if model_type == 'yolox_n':
        #     exp = get_exp(None, 'yolox_nano')
        # else:
        #     exp = get_exp(None, model_type)


        exp = get_exp('/home/legkovas/Projects/tracking/ByteTrack/exps/example/mot/yolox_m_mix_det.py', None)
        exp.num_classes = 1

        LOGGER.info(f'Loading {model_type} with {str(args.yolo_model)}')

        # download crowdhuman bytetrack models
        if not args.yolo_model.exists() and args.yolo_model.stem == model_type:
            LOGGER.info('Downloading pretrained weights...')
            gdown.download(
                url=YOLOX_ZOO[model_type + '.pt'],
                output=str(args.yolo_model),
                quiet=False
            )
            # needed for bytetrack yolox people models
            # update with your custom model needs
            exp.num_classes = 1
        elif args.yolo_model.stem == model_type:
            exp.num_classes = 1

        ckpt = torch.load(
            str(args.yolo_model),
            map_location=torch.device('cpu')
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = exp.get_model()
        self.model.eval()
        self.model.load_state_dict(ckpt["model"])
        # self.model = fuse_model(self.model)
        self.model.to(device)
        self.model.eval()

        self.num_classes = 1
        self.test_conf = 0.01
        self.nmsthre = 0.7

        # preproc
        self.means=(0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)
        self.swap=(2, 0, 1)
        self.test_size = (640, 640)

    def _preproc(self, img):
        img, _ = preproc(img, self.test_size, self.means, self.std, self.swap)
        img = np.expand_dims(img, 0) # c,h,w -> b,c,h,w
        return img
    
    def _postprocess(self, outputs, img_w, img_h):
        outputs = postprocess(outputs, self.num_classes, self.test_conf, self.nmsthre)

        # from original bytetrack method - update
        output_results = outputs[0]
        scores = output_results[:, 4] * output_results[:, 5]
        bboxes = output_results[:, :4]  # x1y1x2y2
        scale = min(self.test_size[0] / float(img_h), self.test_size[1] / float(img_w))
        bboxes /= scale
        cls = output_results[:, 6]

        # xyxy -> xywh
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

        outputs = torch.cat((bboxes, scores.unsqueeze(1), cls.unsqueeze(1)), 1)

        return outputs.cpu().numpy()


    def inference(self, img_path):
        img_raw = cv2.imread(img_path)
        img = self._preproc(img_raw)
        img = torch.from_numpy(img).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img)
        outputs = self._postprocess(outputs, img_h=img_raw.shape[1], img_w=img_raw.shape[2])

        return outputs
