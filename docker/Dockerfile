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

# The Docker-only project selects CPU/CUDA variants without changing BoxMOT's
# published dependency contract. Copy metadata first for dependency-layer cache
# reuse; the package source is added only after third-party dependencies.
COPY pyproject.toml README.md LICENSE ./
COPY docker/pyproject.toml docker/uv.lock ./docker/


FROM build-base AS cli-build-base

# YOLOX compiles its fast_cocoeval extension when the yolo extra is selected.
# The compiler remains in CLI builders and is not copied into runtime images.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*


FROM cli-build-base AS cli-gpu-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project docker \
        --locked \
        --no-dev \
        --no-install-package boxmot \
        --extra cli \
        --extra cu130

COPY boxmot ./boxmot

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project docker \
        --locked \
        --no-dev \
        --no-editable \
        --extra cli \
        --extra cu130


FROM cli-build-base AS cli-cpu-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project docker \
        --locked \
        --no-dev \
        --no-install-package boxmot \
        --extra cli \
        --extra cpu

COPY boxmot ./boxmot

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project docker \
        --locked \
        --no-dev \
        --no-editable \
        --extra cli \
        --extra cpu


FROM build-base AS service-cpu-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project docker \
        --locked \
        --no-dev \
        --only-group service-runtime

COPY boxmot ./boxmot


FROM build-base AS service-gpu-builder

# The shared service group supplies the HTTP/tracker runtime. The service-gpu
# extra adds only direct ReID support, while cu130 selects the locked CUDA
# PyTorch wheels. BoxMOT itself runs from the source copied below.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project docker \
        --locked \
        --no-dev \
        --group service-runtime \
        --extra service-gpu \
        --extra cu130

COPY boxmot ./boxmot


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
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/boxmot

CMD ["bash"]


# Full CPU image: detector, ReID, CLI, and evaluation dependencies backed by
# the PyTorch CPU wheel index selected by the `cpu` extra.
FROM cli-runtime AS cli-cpu

COPY --from=cli-cpu-builder /opt/boxmot/.venv /opt/boxmot/.venv


# Shared non-root, headless service runtime. Individual CPU/GPU targets copy
# only their locked virtual environment and the BoxMOT source tree into it.
FROM python:${PYTHON_VERSION}-slim-bookworm AS service-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/boxmot \
    UV_PROJECT_ENVIRONMENT=/opt/boxmot/.venv \
    PATH="/opt/boxmot/.venv/bin:${PATH}" \
    BOXMOT_SERVICE_PORT=8000 \
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


# Geometry-only detection-to-track service. It contains neither PyTorch nor
# CUDA and remains the image published under the canonical service tags.
FROM service-runtime AS service-cpu

ENV BOXMOT_SERVICE_PROFILE=cpu \
    BOXMOT_SERVICE_TRACKER=bytetrack

COPY --from=service-cpu-builder --chown=boxmot:boxmot /opt/boxmot/.venv /opt/boxmot/.venv
COPY --from=service-cpu-builder --chown=boxmot:boxmot /opt/boxmot/boxmot /opt/boxmot/boxmot


# CUDA ReID service. The model is not baked into the image: mount the expected
# checkpoint read-only at /models/osnet_x0_25_msmt17.pt when starting it.
FROM service-runtime AS service-gpu

ENV BOXMOT_SERVICE_PROFILE=gpu \
    BOXMOT_SERVICE_TRACKER=botsort \
    BOXMOT_SERVICE_DEVICE=0 \
    BOXMOT_SERVICE_HALF=true \
    BOXMOT_SERVICE_MAX_CONCURRENT_UPDATES=1 \
    BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt

COPY --from=service-gpu-builder --chown=boxmot:boxmot /opt/boxmot/.venv /opt/boxmot/.venv
COPY --from=service-gpu-builder --chown=boxmot:boxmot /opt/boxmot/boxmot /opt/boxmot/boxmot


# Full CUDA image. The cu130 profile installs the CUDA 13.0 PyTorch wheels and
# their matching runtime libraries, avoiding a second CUDA stack in the base.
FROM cli-runtime AS cli-gpu

COPY --from=cli-gpu-builder /opt/boxmot/.venv /opt/boxmot/.venv


# Preserve the familiar no-target build as an alias for the full CUDA image.
FROM cli-gpu AS default
