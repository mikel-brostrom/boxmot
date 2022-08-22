# Partial Quantization
The performance of YOLOv6s heavily degrades from 42.4% to 35.6% after traditional PTQ, which is unacceptable. To resolve this issue, we propose **partial quantization**. First we analyze the quantization sensitivity of all layers, and then we let the most sensitive layers to have full precision as a  compromise.

With partial quantization, we finally reach 42.1%, only 0.3% loss in accuracy, while the throughput of the partially quantized model is about 1.56 times that of the FP16 model at a batch size of 32. This method achieves a nice tradeoff between accuracy and throughput.

## Prerequirements
```python
pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com nvidia-pyindex
pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com pytorch_quantization
```
## Sensitivity analysis

Please use the following command to perform sensitivity analysis. Since we randomly sample 128 images from train dataset each time, the sensitivity files will be slightly different.

```python
 python3 sensitivity_analyse.py --weights yolov6s_reopt.pt \
                                --batch-size 32 \
                                --batch-number 4 \
                                --data-root train_data_path
```

## Partial quantization

With the sensitivity file at hand, we then proceed with partial quantization as follows.

```python
python3 partial_quant.py --weights yolov6s_reopt.pt \
                         --calib-weights yolov6s_repot_calib.pt \
                         --sensitivity-file yolov6s_reopt_sensivitiy_128_calib.txt \
                         --quant-boundary 55 \
                         --export-batch-size 1
```

## Deployment

Build a TRT engine

```python
trtexec --workspace=1024 --percentile=99 --streams=1 --int8 --fp16 --avgRuns=10 --onnx=yolov6s_reopt_partial_bs1.sim.onnx --saveEngine=yolov6s_reopt_partial_bs1.sim.trt
```

## Performance
| Model           | Size        | Precision        |mAP<sup>val<br/>0.5:0.95 | Speed<sup>T4<br/>trt b1 <br/>(fps) | Speed<sup>T4<br/>trt b32 <br/>(fps) |
| :-------------- | ----------- | ----------- |:----------------------- | ---------------------------------------- | -----------------------------------|
| [**YOLOv6-s-partial**] </br>[bs1](https://github.com/lippman1125/YOLOv6/releases/download/0.1.0/yolov6s_reopt_partial_bs1.sim.onnx) <br/>[bs32](https://github.com/lippman1125/YOLOv6/releases/download/0.1.0/yolov6s_reopt_partial_bs32.sim.onnx) <br/>| 640 | INT8         |42.1                     | 503                                      | 811                                |
| [**YOLOv6-s**] | 640         | FP16         |42.4                     | 373                                      | 520                                |
