# build the image and tag it for easier later reference
#   docker build -t mikel-brostrom/yolov5_strongsort_osnet .

# Base image: Nvidia PyTorch https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.11-py3

# Update image
RUN apt update

# Install pip packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip uninstall -y torch torchvision
RUN pip install --no-cache -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Clone with submodules
RUN git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git /usr/src/app

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
