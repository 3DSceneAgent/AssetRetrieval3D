#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DOCKERHUB_NAMESPACE="${DOCKERHUB_NAMESPACE:-fishwowater}"
IMAGE_NAME="${IMAGE_NAME:-assetretrieval3d}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"

IMAGE_REF="${DOCKERHUB_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

cd "$PROJECT_DIR"

echo "[1/3] Building ${IMAGE_REF}"
docker build --platform "$PLATFORM" -t "$IMAGE_REF" .

echo "[2/3] Pushing ${IMAGE_REF}"
docker push "$IMAGE_REF"

echo "[3/3] Done"
echo "Published: ${IMAGE_REF}"
