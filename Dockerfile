# build the image and tag it for easier later reference
#   docker build -t mikel-brostrom/yolov5_strongsort_osnet .

# Base image: Nvidia PyTorch https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Update image
RUN apt update

# Create working directory
WORKDIR /usr/src/boxmot

# Clone with submodules
RUN git clone https://github.com/mikel-brostrom/yolo_tracking.git -b master /usr/src/boxmot

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -e .
# Install custom ultralytics package which makes model from other repos loadable
RUN pip install git+https://github.com/mikel-brostrom/ultralytics.git

# ------------------------------------------------------------------------------

# A docker container exits when its main process finishes, which in this case is bash.
# This means that the containers will stop once you exit them and everything will be lost.
# To avoid this use detach mode. More on this in the next paragraph
#
#   - run interactively with all GPUs accessible:
#
#       docker run -it --gpus all mikel-brostrom/yolov5_strongsort_osnet bash
#
#   - run interactively with first and third GPU accessible:
#
#       docker run -it --gpus '"device=0, 2"' mikel-brostrom/yolov5_strongsort_osnet bash


# Run in detached mode (if you exit the container it won't stop)
#
#   -create a detached docker container from an image:
#
#       docker run -it --gpus all -d mikel-brostrom/yolov5_strongsort_osnet
#
#   - this will return a <container_id> number which makes it accessible. Access it by:
#
#       docker exec -it <container_id>
#
#   - When you are done with the container stop it by:
#
#       docker stop <container_id>
