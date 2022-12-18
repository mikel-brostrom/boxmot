# Usage example:
# build the image and tag it for easier later reference
#   docker build -t <tag> .
# run the default command
#   docker run <tag>
# run the default command in the background
#   docker run <tag> -d

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

# Default command to run when starting a container from this image
CMD ["python", "track.py", "--source", "yolov5/data/images/bus.jpg"]

# ------------------------------------------------------------------------------

# docker exec --gpus all -it <container-name> /bin/bash