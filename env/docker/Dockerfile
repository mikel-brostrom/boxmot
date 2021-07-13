FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root/


RUN \
  apt update -q=2 \
  && apt install -yq=2 \
  apt-utils \
  python3-opencv \
  python3-pip


COPY py/requirements.txt .
RUN \
  pip install -r requirements.txt \
  && rm requirements.txt
  