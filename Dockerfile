# Use the official Debian-hosted Python image
FROM python:3.12-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential git"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive
# Python wants UTF-8 locale
ENV LANG=C.UTF-8
# Stop Python from writing .pyc files and buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell uv to copy rather than hardlink, and where to put the venv
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/home/app/.venv

# Ensure we have an up to date baseline and the packages we need
RUN set -ex; \
    for i in $(seq 1 8); do [ $i -gt 1 ] && sleep 5; \
        apt-get update -y && break || continue; done; \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock* /app/
RUN uv sync --no-install-project
COPY . /app/
RUN uv sync

ENTRYPOINT ["/bin/bash"]
CMD ["-c", "source /home/app/.venv/bin/activate && exec bash"]
