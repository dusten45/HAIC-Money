FROM vastai/base-image:cuda-12.1.1-cudnn8-devel-ubuntu22.04-py311

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/haic-env

COPY pyproject.toml uv.lock .python-version ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

ENV VIRTUAL_ENV=/opt/haic-env/.venv
ENV PATH="/opt/haic-env/.venv/bin:$PATH"

WORKDIR /workspace
