import torch.nn as nn
import torch
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
import cv2
import gdown
from os.path import exists as file_exists
from .deep.reid_model_factory import show_downloadeable_models, get_model_url, get_model_name

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import download_url


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


class ReIDDetectMultiBackend(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(self, weights='osnet_x0_25_msmt17.pt', device=torch.device('cpu'), fp16=False):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        self.pt, self.jit, self.onnx, self.xml, self.engine, self.coreml, \
            self.saved_model, self.pb, self.tflite, self.edgetpu, self.tfjs = self.model_type(w)  # get backend
        fp16 &= (self.pt or self.jit or self.onnx or self.engine) and device.type != 'cpu'  # FP16
        self.fp16 = fp16
        if self.pt:  # PyTorch
            model_name = get_model_name(weights)
            model_url = get_model_url(weights)

            if not file_exists(weights) and model_url is not None:
                gdown.download(model_url, str(weights), quiet=False)
            elif file_exists(weights):
                pass
            elif model_url is None:
                print('No URL associated to the chosen DeepSort weights. Choose between:')
                show_downloadeable_models()
                exit()

            self.extractor = FeatureExtractor(
                # get rid of dataset information DeepSort model name
                model_name=model_name,
                model_path=weights,
                device=str(device)
            )
            
            self.extractor.model.half() if fp16 else  self.extractor.model.float()
        elif self.onnx:  # ONNX Runtime
            # LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            #check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(w, providers=providers)
        
        elif self.tflite:
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            self.interpreter = tf.lite.Interpreter(model_path=weights)
            self.interpreter.allocate_tensors()
            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Test model on random input data.
            input_data = np.array(np.random.random_sample((1,256,128,3)), dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            print(output_data.shape)
        else:
            print('This model framework is not supported yet!')
            exit()
            
        pixel_mean=[0.485, 0.456, 0.406]
        pixel_std=[0.229, 0.224, 0.225]
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(pixel_mean, pixel_std),
        ])
        self.size = (256, 128)
        self.device = device
        
        
    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs
    
    def warmup(self, imgsz=(1, 256, 128, 3)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb
        if any(warmup_types) and self.device.type != 'cpu':
            im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            im = im.cpu().numpy()
            print(im.shape)
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    def preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32), size)

        im = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        im = im.float().to(device=self.device)
        return im
    
    def forward(self, im_batch):
        im_batch = self.preprocess(im_batch)
        b, ch, h, w = im_batch.shape  # batch, channel, height, width
        features = []
        for i in range(0, im_batch.shape[0]):
            im = im_batch[i, :, :, :].unsqueeze(0)
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()  # to FP16
            if self.pt:  # PyTorch
                y = self.extractor.model(im)[0]
            elif self.jit:  # TorchScript
                y = self.model(im)[0]
            elif self.onnx:  # ONNX Runtime
                im = im.permute(0, 1, 3, 2).cpu().numpy()  # torch to numpy  # torch to numpy
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
            elif self.xml:  # OpenVINO
                im = im.cpu().numpy()  # FP32
                y = self.executable_network([im])[self.output_layer]
            else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
                im = im.permute(0, 3, 2, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = torch.tensor(self.interpreter.get_tensor(output['index']))
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, device=self.device)
            features.append(y.squeeze())

        
        return features