#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="production-monitoring"
export BASE_DIR=$(pwd)

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container. The repo is bind-mounted over /app, so edits you make to
# monitor/probes.py in your editor are live inside the container immediately.
docker run --rm --name $IMAGE_NAME -ti \
    -v "$BASE_DIR":/app \
    $IMAGE_NAME
