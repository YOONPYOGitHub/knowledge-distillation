#!/usr/bin/env bash
# 백엔드/프론트엔드 이미지를 ACR 클라우드 빌드로 만들고 푸시 (Docker 불필요).
#
# 사용:
#   ./scripts/build_and_push.sh <acr-name> [tag]
#
# 동작:
#   1) /tmp/kd-ctx 에 Dockerfile 이 필요한 파일만 복사 (results/, .venv/ 등 제외).
#   2) `az acr build` 로 클라우드에서 이미지 빌드 + ACR push.

set -euo pipefail

ACR="${1:?ACR 이름 필요. 예: kduiacrisfk5oaqrenwy}"
TAG="${2:-v1}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CTX_DIR="${KD_BUILD_CTX_DIR:-/tmp/kd-ctx}"

echo "==> 빌드 컨텍스트 준비: ${CTX_DIR}"
rm -rf "${CTX_DIR}"
mkdir -p "${CTX_DIR}"
rsync -a \
  --exclude='results' \
  --exclude='.venv' \
  --exclude='.git' \
  --exclude='examples' \
  --exclude='docs' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.ipynb' \
  --exclude='*.pt' \
  "${PROJECT_ROOT}/ui" \
  "${PROJECT_ROOT}/src" \
  "${PROJECT_ROOT}/.dockerignore" \
  "${CTX_DIR}/"
echo "    크기: $(du -sh "${CTX_DIR}" | cut -f1)"

echo "==> backend 빌드 + push"
az acr build \
  --registry "${ACR}" \
  --image "kd-backend:${TAG}" \
  --file ui/backend/Dockerfile \
  --platform linux/amd64 \
  "${CTX_DIR}"

echo "==> frontend 빌드 + push"
az acr build \
  --registry "${ACR}" \
  --image "kd-frontend:${TAG}" \
  --file ui/frontend/Dockerfile \
  --platform linux/amd64 \
  "${CTX_DIR}"

LOGIN_SERVER="$(az acr show -n "${ACR}" --query loginServer -o tsv)"

echo
echo "이미지 (Bicep param 에 사용):"
echo "  backendImage  = ${LOGIN_SERVER}/kd-backend:${TAG}"
echo "  frontendImage = ${LOGIN_SERVER}/kd-frontend:${TAG}"
