# Use the official Debian-hosted Python image
FROM python:3.12-slim-bookworm

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Tell uv to copy packages from the wheel into the site-packages, and to put the
# virtual environment at /home/app/.venv — *outside* /app. We bind-mount the repo
# over /app while developing, and a venv inside /app would be hidden by the mount.
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/home/app/.venv

# Ensure we have an up to date baseline, install uv, and create a user so we
# don't run the app as root
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && \
    chown app:app /app

# Switch to the new user
USER app
WORKDIR /app

# Install the dependencies first, on their own layer, so editing code later
# doesn't make Docker reinstall pandas and scipy.
COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-cache

# Copy the source code
COPY --chown=app:app . ./
RUN uv sync --locked --no-cache

# Entry point
ENTRYPOINT ["/bin/bash"]
# Get into the uv virtual environment shell
CMD ["-c", "source /home/app/.venv/bin/activate && exec bash"]
