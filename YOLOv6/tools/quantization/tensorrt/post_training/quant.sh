# Path to ONNX model
# ex: ../yolov6.onnx
ONNX_MODEL=$1

# Path to dataset to use for calibration.
#   **Not necessary if you already have a calibration cache from a previous run.
CALIBRATION_DATA=$2

# Path to Cache file to Serving
# ex: ./caches/demo.cache
CACHE_FILENAME=$3

# Path to write TensorRT engine to
OUTPUT=$4

# Creates an int8 engine from your ONNX model, creating ${CACHE_FILENAME} based
# on your ${CALIBRATION_DATA}, unless ${CACHE_FILENAME} already exists, then
# it will use simply use that instead.
python3 onnx_to_tensorrt.py --fp16 --int8 -v \
        --calibration-data=${CALIBRATION_DATA} \
        --calibration-cache=${CACHE_FILENAME} \
        --explicit-batch \
        --onnx ${ONNX_MODEL} -o ${OUTPUT}
