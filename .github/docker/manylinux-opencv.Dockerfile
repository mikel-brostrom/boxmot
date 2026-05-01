# syntax=docker/dockerfile:1.7
# Custom manylinux_2_28 image with OpenCV 4 + Eigen3 preinstalled.
#
# Built and pushed by .github/workflows/build-manylinux-image.yml to
# ghcr.io/<owner>/boxmot-manylinux:opencv-<version>-<arch>.
# cibuildwheel uses it via CIBW_MANYLINUX_*_IMAGE so every wheel build
# skips the OpenCV compile (saves ~5-10 min per Python version).

ARG BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64:2024.10.07-1
FROM ${BASE_IMAGE}

ARG OPENCV_VERSION=4.10.0

RUN dnf install -y dnf-plugins-core epel-release \
    && (dnf config-manager --set-enabled powertools 2>/dev/null \
        || dnf config-manager --set-enabled crb 2>/dev/null || true) \
    && dnf install -y cmake gcc-c++ eigen3-devel zlib-devel \
    && dnf clean all

RUN set -eux; \
    curl -fsSL -o /tmp/opencv.tar.gz \
        https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz; \
    mkdir -p /tmp/opencv; \
    tar xf /tmp/opencv.tar.gz -C /tmp/opencv --strip-components=1; \
    cmake -S /tmp/opencv -B /tmp/opencv/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_LIST=calib3d,core,dnn,features2d,flann,imgcodecs,imgproc,video \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF \
        -DWITH_FFMPEG=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_GSTREAMER=OFF \
        -DWITH_V4L=OFF -DWITH_1394=OFF -DWITH_OPENEXR=OFF \
        -DWITH_PROTOBUF=ON -DBUILD_PROTOBUF=ON; \
    cmake --build /tmp/opencv/build --target install -j"$(nproc)"; \
    ldconfig; \
    rm -rf /tmp/opencv /tmp/opencv.tar.gz
