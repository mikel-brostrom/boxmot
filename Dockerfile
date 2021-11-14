FROM python:3.8

RUN apt-get update
RUN apt-get install -y \
    sudo \
    vim \
    xterm \
    git \
    zip \
    wget \
    less

RUN mkdir /app
COPY requirements.txt /app/requirements.txt
COPY yolov5 /app/yolov5
COPY deep_sort_pytorch /app/deep_sort_pytorch
COPY track.py /app/track.py
COPY main.sh /app/main.sh

RUN python -m pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN pip install -r /app/yolov5/requirements.txt

CMD bash /app/main.sh
