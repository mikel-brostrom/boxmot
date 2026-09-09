# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.12.4

# Copying uv from its official image avoids bootstrapping it with pip and keeps
# the resolver version identical in every build stage.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv


FROM python:${PYTHON_VERSION}-slim-bookworm AS build-base

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/boxmot/.venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/boxmot/.venv/bin:${PATH}"

WORKDIR /opt/boxmot


# Only this inexpensive stage reads version-bearing project metadata. Its
# output preserves every dependency constraint, source, and locked artifact,
# with the local project's version normalized for dependency cache reuse.
FROM build-base AS dependency-manifests

COPY docker/dependency_manifest.py /docker/dependency_manifest.py
COPY pyproject.toml uv.lock ./
RUN python /docker/dependency_manifest.py --output /opt/boxmot-dependencies


FROM build-base AS cli-build-base

# YOLOX compiles its fast_cocoeval extension when the yolo extra is selected.
# The compiler remains in CLI builders and is not copied into runtime images.
# This stage and native compilation have no project metadata ancestors, so
# Python dependency changes and release bumps do not rebuild system packages.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        g++ \
        libopencv-calib3d406 \
        libopencv-core406 \
        libopencv-dnn406 \
        libopencv-imgproc406 \
        libopencv-video406 \
    && rm -rf /var/lib/apt/lists/*


# Build every native C API once in a dedicated stage. The installed artifacts
# are merged into the Python package before its wheel is built below, so the
# final CLI images can use native trackers without carrying CMake, headers, or
# a compiler. Native ReID intentionally uses OpenCV DNN in these images.
FROM cli-build-base AS native-cli-builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libeigen3-dev \
        libopencv-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY boxmot/native/cpp ./boxmot/native/cpp

RUN cmake \
        -S boxmot/native/cpp \
        -B /tmp/boxmot-native-build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBOXMOT_INSTALL_NATIVE=ON \
        -DBOXMOT_REID_ONNXRUNTIME=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/boxmot-native \
    && cmake --build /tmp/boxmot-native-build --parallel \
    && cmake --install /tmp/boxmot-native-build --strip


FROM cli-build-base AS cli-gpu-dependencies

COPY --from=dependency-manifests /opt/boxmot-dependencies/ ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-package boxmot \
        --extra yolo \
        --extra trackeval \
        --extra cu130


FROM cli-build-base AS cli-cpu-dependencies

COPY --from=dependency-manifests /opt/boxmot-dependencies/ ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-package boxmot \
        --extra yolo \
        --extra trackeval \
        --extra cpu


FROM build-base AS service-cpu-dependencies

COPY --from=dependency-manifests /opt/boxmot-dependencies/ ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-package boxmot \
        --only-group service-runtime


FROM build-base AS service-gpu-dependencies

COPY --from=dependency-manifests /opt/boxmot-dependencies/ ./

# The root runtime plus service extras supply HTTP/tracker/ReID dependencies;
# cu130 activates the locked CUDA PyTorch source. BoxMOT runs from copied source.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-package boxmot \
        --extra service \
        --extra service-gpu \
        --extra cu130


# Both CLI profiles use the same BoxMOT wheel. Build it from the actual release
# metadata, independently of the much larger Python dependency environments.
FROM build-base AS cli-wheel-builder

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY boxmot ./boxmot
COPY --from=native-cli-builder /opt/boxmot-native/boxmot/native/cpp/ ./boxmot/native/cpp/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --no-sources --out-dir /wheels


FROM python:${PYTHON_VERSION}-slim-bookworm AS cli-runtime

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/boxmot/.venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/boxmot/.venv/bin:${PATH}"

# ffmpeg and the GL/X11 libraries cover common OpenCV video and visualization
# workflows. git and uv remain available for interactive CLI use.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libopencv-calib3d406 \
        libopencv-core406 \
        libopencv-dnn406 \
        libopencv-imgproc406 \
        libopencv-video406 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/boxmot

CMD ["bash"]


# Full CPU image: detector, ReID, CLI, and evaluation dependencies backed by
# the PyTorch CPU wheel index selected by the `cpu` extra.
FROM cli-runtime AS cli-cpu

# Preserve the dependency environment as its own runtime layer. Only BoxMOT's
# wheel installation changes when application source or its version changes.
COPY --from=cli-cpu-dependencies /opt/boxmot/.venv /opt/boxmot/.venv

RUN --mount=type=bind,from=cli-wheel-builder,source=/wheels,target=/wheels \
    uv pip install --no-cache --no-deps --python /opt/boxmot/.venv/bin/python /wheels/*.whl \
    && cd /tmp \
    && python -I -c "import ctypes; from boxmot.native import _common; names = ('reid', 'botsort', 'bytetrack', 'occluboost', 'ocsort', 'sfsort'); paths = [_common.installed_library_candidates(name, _common.library_filename(name))[0] for name in names]; assert all(path.is_file() for path in paths), paths; [ctypes.CDLL(str(path)) for path in paths]"


# Shared non-root, headless service runtime. Individual CPU/GPU targets copy
# only their locked virtual environment and the BoxMOT source tree into it.
FROM python:${PYTHON_VERSION}-slim-bookworm AS service-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/boxmot \
    UV_PROJECT_ENVIRONMENT=/opt/boxmot/.venv \
    PATH="/opt/boxmot/.venv/bin:${PATH}" \
    BOXMOT_SERVICE_PORT=8000 \
    BOXMOT_SERVICE_ASSO_FUNC=iou \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 boxmot \
    && useradd --uid 10001 --gid boxmot --create-home --shell /usr/sbin/nologin boxmot \
    && install --directory --mode=0755 /models

WORKDIR /opt/boxmot

USER boxmot
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('BOXMOT_SERVICE_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2).read()"]

CMD ["python", "-m", "boxmot.engine.service"]


# Canonical CPU detection-to-track service. It uses CPU Torch structures but
# contains no CUDA runtime and remains the image under the canonical tags.
FROM service-runtime AS service-cpu

ENV BOXMOT_SERVICE_PROFILE=cpu \
    BOXMOT_SERVICE_TRACKER=bytetrack

COPY --from=service-cpu-dependencies --chown=boxmot:boxmot /opt/boxmot/.venv /opt/boxmot/.venv
COPY --chown=boxmot:boxmot boxmot /opt/boxmot/boxmot


# CUDA ReID service. The model is not baked into the image: mount the expected
# checkpoint read-only at /models/osnet_x0_25_msmt17.pt when starting it.
FROM service-runtime AS service-gpu

ENV BOXMOT_SERVICE_PROFILE=gpu \
    BOXMOT_SERVICE_TRACKER=botsort \
    BOXMOT_SERVICE_DEVICE=0 \
    BOXMOT_SERVICE_HALF=true \
    BOXMOT_SERVICE_MAX_CONCURRENT_UPDATES=1 \
    BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt

USER root
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=service-gpu-dependencies --chown=boxmot:boxmot /opt/boxmot/.venv /opt/boxmot/.venv
COPY --chown=boxmot:boxmot boxmot /opt/boxmot/boxmot

USER boxmot


# Full CUDA image. The cu130 profile installs the CUDA 13.0 PyTorch wheels and
# their matching runtime libraries, avoiding a second CUDA stack in the base.
FROM cli-runtime AS cli-gpu

COPY --from=cli-gpu-dependencies /opt/boxmot/.venv /opt/boxmot/.venv

RUN --mount=type=bind,from=cli-wheel-builder,source=/wheels,target=/wheels \
    uv pip install --no-cache --no-deps --python /opt/boxmot/.venv/bin/python /wheels/*.whl \
    && cd /tmp \
    && python -I -c "import ctypes; from boxmot.native import _common; names = ('reid', 'botsort', 'bytetrack', 'occluboost', 'ocsort', 'sfsort'); paths = [_common.installed_library_candidates(name, _common.library_filename(name))[0] for name in names]; assert all(path.is_file() for path in paths), paths; [ctypes.CDLL(str(path)) for path in paths]"


# Preserve the familiar no-target build as an alias for the full CUDA image.
FROM cli-gpu AS default
