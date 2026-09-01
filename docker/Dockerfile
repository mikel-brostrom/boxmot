# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.12.4

FROM python:${PYTHON_VERSION}-slim-bookworm AS build-base

ARG UV_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/boxmot/.venv \
    PATH="/opt/boxmot/.venv/bin:${PATH}"

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /opt/boxmot

# Resolve locked dependencies before copying package sources so this expensive
# layer is reused until project metadata changes.
COPY pyproject.toml uv.lock README.md LICENSE ./


FROM build-base AS cli-builder

# YOLOX ships as source and compiles its fast_cocoeval extension during sync.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --extra yolo \
        --extra trackeval \
        --no-install-project

COPY boxmot ./boxmot

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --extra yolo \
        --extra trackeval


FROM build-base AS service-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --extra service \
        --no-install-project

COPY boxmot ./boxmot

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --extra service


FROM python:${PYTHON_VERSION}-slim-bookworm AS cli

ARG UV_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/boxmot/.venv \
    PATH="/opt/boxmot/.venv/bin:${PATH}"

# ffmpeg and the GL/GLib libraries cover common OpenCV video workflows. Keep
# git and uv available for VCS-backed or first-use optional dependencies.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /opt/boxmot
COPY --from=cli-builder /opt/boxmot /opt/boxmot

CMD ["bash"]


FROM python:${PYTHON_VERSION}-slim-bookworm AS service

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/boxmot/.venv \
    PATH="/opt/boxmot/.venv/bin:${PATH}" \
    BOXMOT_SERVICE_PORT=8000

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 boxmot \
    && useradd --uid 10001 --gid boxmot --create-home --shell /usr/sbin/nologin boxmot

WORKDIR /opt/boxmot
COPY --from=service-builder --chown=boxmot:boxmot /opt/boxmot /opt/boxmot

USER boxmot
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('BOXMOT_SERVICE_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2).read()"]

CMD ["boxmot-service"]
