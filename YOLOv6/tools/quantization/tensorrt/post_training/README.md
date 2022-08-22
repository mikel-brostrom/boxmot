# ONNX -> TensorRT INT8
These scripts were last tested using the
[NGC TensorRT Container Version 20.06-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt).
You can see the corresponding framework versions for this container [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-container-release-notes/rel_20.06.html#rel_20.06).

## Quickstart

> **NOTE**: This INT8 example is only valid for **fixed-shape** ONNX models at the moment.
>
INT8 Calibration on **dynamic-shape** models is now supported, however this example has not been updated
to reflect that yet. For more details on INT8 Calibration for **dynamic-shape** models, please
see the [documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-calib-dynamic-shapes).

### 1. Convert ONNX model to TensorRT INT8

See `./onnx_to_tensorrt.py -h` for full list of command line arguments.

```bash
./onnx_to_tensorrt.py --explicit-batch \
                      --onnx resnet50/model.onnx \
                      --fp16 \
                      --int8 \
                      --calibration-cache="caches/yolov6.cache" \
                      -o resnet50.int8.engine
```

See the [INT8 Calibration](#int8-calibration) section below for details on calibration
using your own model or different data, where you don't have an existing calibration cache
or want to create a new one.

## INT8 Calibration

See [ImagenetCalibrator.py](ImagenetCalibrator.py) for a reference implementation
of TensorRT's [IInt8EntropyCalibrator2](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html).

This class can be tweaked to work for other kinds of models, inputs, etc.

In the [Quickstart](#quickstart) section above, we made use of a pre-existing cache,
[caches/yolov6.cache](caches/yolov6.cache), to save time for the sake of an example.

However, to calibrate using different data or a different model, you can do so with the `--calibration-data` argument.

* This requires that you've mounted a dataset, such as Imagenet, to use for calibration.
    * Add something like `-v /imagenet:/imagenet` to your Docker command in Step (1)
      to mount a dataset found locally at `/imagenet`.
* You can specify your own `preprocess_func` by defining it inside of `ImageCalibrator.py`

```bash
# Path to dataset to use for calibration.
#   **Not necessary if you already have a calibration cache from a previous run.
CALIBRATION_DATA="/imagenet"

# Truncate calibration images to a random sample of this amount if more are found.
#   **Not necessary if you already have a calibration cache from a previous run.
MAX_CALIBRATION_SIZE=512

# Calibration cache to be used instead of calibration data if it already exists,
# or the cache will be created from the calibration data if it doesn't exist.
CACHE_FILENAME="caches/yolov6.cache"

# Path to ONNX model
ONNX_MODEL="model/yolov6.onnx"

# Path to write TensorRT engine to
OUTPUT="yolov6.int8.engine"

# Creates an int8 engine from your ONNX model, creating ${CACHE_FILENAME} based
# on your ${CALIBRATION_DATA}, unless ${CACHE_FILENAME} already exists, then
# it will use simply use that instead.
python3 onnx_to_tensorrt.py --fp16 --int8 -v \
        --max_calibration_size=${MAX_CALIBRATION_SIZE} \
        --calibration-data=${CALIBRATION_DATA} \
        --calibration-cache=${CACHE_FILENAME} \
        --preprocess_func=${PREPROCESS_FUNC} \
        --explicit-batch \
        --onnx ${ONNX_MODEL} -o ${OUTPUT}

```

### Pre-processing

In order to calibrate your model correctly, you should `pre-process` your data the same way
that you would during inference.
