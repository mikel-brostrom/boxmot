# YOLOv7 Detection in Jetson Nano

# Installation
For Jetson Nano, there are specific packages whl files from Nvidia to run both pytorch and cuda suceesfully.
[![Pytorch For Jeston Nano](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)]
For Yolov7, we are using torch version==1.8.0

``` shell
# install pytorch
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# install torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision  
cd torchvision
export BUILD_VERSION=0.9.0 
python3 setup.py install --user
```

Next, clone this repo to Jetson machine

```
git clone https://github.com/enzo-damion/tailgating_detection.git
cd ./tailgating_detection/
```

# Tracking
Depending on which camera source you are using, *0* for default while */dev/video0* for external camera

``` shell
python3 detect.py --weights yolov7-tiny.pt --source /dev/video0 --classes 0
```



