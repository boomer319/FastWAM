# syntax=docker/dockerfile:1
# FastWAM (Wan2.2-TI2V-5B + 1B ActionDiT) - G1 Dex3 adaptation.
# Build from the repo root:  docker build -f docker/Dockerfile -t fastwam .
#
# Base image nvidia/cuda:12.8.0-devel-ubuntu24.04.
# CUDA 12.8 matches upstream's torch 2.7.1+cu128 pin (driver >= 570).
# Ubuntu 24.04 ships python 3.12, which satisfies upstream's
# requires-python ">=3.10" and every pinned wheel. The venv is baked into
# /opt; nothing is installed at runtime.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ARG tmp_UID=1000
ARG tmp_GID=1000
ARG USERNAME=user

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/fastwam \
    PATH="/opt/fastwam/bin:$PATH" \
    LD_LIBRARY_PATH=/nvidia-host

# ffmpeg: torchcodec 0.4 dynamically links the system libav* (4-7 supported;
# 24.04 ships 6.1). It is also the CLI used by the dataset converter's probe.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 images ship a uid/gid 1000 user `ubuntu`; drop it so the host
# uid can be reused for the container user.
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && (getent group "${tmp_GID}" || groupadd -g "${tmp_GID}" "${USERNAME}") \
    && useradd -m -u "${tmp_UID}" -g "${tmp_GID}" -s /bin/bash "${USERNAME}"

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip wheel "setuptools>=68"

# --- torch stack (upstream README pins) -------------------------------------
RUN pip install \
    torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# --- remaining runtime deps (exact pyproject.toml pins) ---------------------
# Installed explicitly so the editable install below can run with --no-deps
# and the torch wheels above are never re-resolved.
RUN pip install \
    accelerate==1.12.0 \
    av==15.1.0 \
    boto3==1.35.99 \
    datasets==4.8.5 \
    deepspeed==0.18.7 \
    einops==0.8.1 \
    gitpython==3.1.45 \
    huggingface-hub==0.29.2 \
    hydra-core==1.3.2 \
    imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 \
    jsonlines==4.0.0 \
    modelscope==1.34.0 \
    numpy==2.2.6 \
    omegaconf==2.3.0 \
    packaging==25.0 \
    pandas==2.2.3 \
    pillow==12.0.0 \
    pyarrow==24.0.0 \
    regex==2025.11.3 \
    rich==14.2.0 \
    safetensors==0.5.3 \
    termcolor==2.5.0 \
    torchcodec==0.4.0 \
    tqdm==4.68.3 \
    transformers==4.49.0 \
    typing-extensions==4.15.0 \
    wandb==0.23.1

# --- not in upstream pyproject, needed by scripts/g1_dex3 ------------------
#   h5py          scripts/g1_dex3/convert_wma_to_lerobot.py reads the WMA transitions
#   tensorboard   local scalar logging
#   pyzmq/msgpack scripts/g1_dex3/zmq_fastwam_server.py
RUN pip install h5py tensorboard pyzmq msgpack

# Editable install into /workspace, whose path is identical at build and run
# time, so the bind-mounted working tree is what gets imported.
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
RUN pip install --no-deps -e /workspace

RUN mkdir -p /workspace /datasets /docker_data \
    && chown -R "${tmp_UID}:${tmp_GID}" /workspace /datasets /docker_data "${VIRTUAL_ENV}"

USER ${USERNAME}
WORKDIR /workspace

CMD ["/bin/bash"]
