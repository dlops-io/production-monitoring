#!/bin/bash
set -e

export IMAGE_NAME="production-monitoring"

docker build -t $IMAGE_NAME -f Dockerfile .
docker run --rm -ti \
    -v "$(pwd)":/app \
    $IMAGE_NAME
