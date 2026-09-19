FROM vastai/base-image:cuda-12.1.1-cudnn8-devel-ubuntu22.04-py311

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    libgl1 \
    libglib2.0-0 \
    micro \
    zsh \
    xclip \
    xauth \
    htop \
    ripgrep \
    tree \
    lsof \
    psmisc \
    less \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p -m 755 /etc/apt/keyrings \
    && wget -nv -O /etc/apt/keyrings/githubcli-archive-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && mkdir -p -m 755 /etc/apt/sources.list.d \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

RUN export NVM_DIR=/root/.nvm \
    && source "$NVM_DIR/nvm.sh" \
    && npm install -g @kilocode/cli \
    && kilo --version

WORKDIR /opt/haic-env

COPY pyproject.toml uv.lock .python-version ./

RUN VIRTUAL_ENV=/venv/main uv sync --locked --inexact --active --no-install-project

RUN /venv/main/bin/python -c 'import sys, torch, gymnasium, stable_baselines3, cv2, Box2D; print("python:", sys.version); print("torch:", torch.__version__); print("torch cuda:", torch.version.cuda); print("gymnasium:", gymnasium.__version__); print("sb3:", stable_baselines3.__version__); print("opencv:", cv2.__version__); print("Box2D: OK")'

RUN env-hash > /.env_hash

WORKDIR /workspace
