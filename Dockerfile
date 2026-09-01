# syntax=docker/dockerfile:1.7

# Build from the checked-out source so the image always matches the commit that
# was selected locally or by CI.
FROM python:3.11-slim-bookworm

# Keep this aligned with the uv release used by CI and the lockfile format.
ARG UV_VERSION=0.12.4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/boxmot/.venv \
    PATH="/opt/boxmot/.venv/bin:${PATH}"

# ffmpeg and the GL/GLib libraries cover the common OpenCV video runtime.
# git remains available for dependencies or workflows backed by a VCS checkout.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /opt/boxmot

# Install locked dependencies before copying the package source. This layer is
# reused until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
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

# Keep the image convenient for interactive use while allowing a command such
# as `boxmot --help` to replace the default at `docker run` time.
CMD ["bash"]
